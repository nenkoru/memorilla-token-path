"""H40: train the per-fact rater for precision. The plain rater (test 35c, "change" wording) catches unseen health clues but says
Yes to most facts, so it ranks poorly in big profiles (test 38b). Here a LoRA (r16, q/k/v/o) on the decoder learns, listwise,
to rate a training request's known relevant fact above 15 random facts (log-odds of Yes vs No at the reply's first token).
Held-out types never appear in training. Evaluated before and after on the saved 48- and 192-fact unseen-type profiles.
  --minutes 20 -> work/runs/poc/rater_lora.json, lora_rater.pt
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import fast_train as FT  # noqa: E402
import poc_profile as PF  # noqa: E402
import poc_rater as RT  # noqa: E402
import poc_whole as W  # noqa: E402

T0 = time.perf_counter()
NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")


def encode(tx, tok, fact, q):
    return tok.encode(tx.apply(tok, tx.msgs([fact], W.RATES["change"].format(q=q)), add_generation_prompt=True), add_special_tokens=False)


def scores(model, seqs, yes, no, torch, dev):
    """log-odds Yes vs No at each sequence's last token; right padding."""
    n = max(len(s) for s in seqs)
    ids = torch.zeros(len(seqs), n, dtype=torch.long, device=dev)
    mask = torch.zeros(len(seqs), n, dtype=torch.long, device=dev)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    h = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
    last = h[torch.arange(len(seqs), device=dev), mask.sum(1) - 1]
    w = model.lm_head.weight
    lg = last.float() @ w[yes + no].float().t()
    return torch.logsumexp(lg[:, :len(yes)], -1) - torch.logsumexp(lg[:, len(yes):], -1)


def evaluate(model, tok, tx, K, yes, no, torch, dev, label):
    model.eval()
    out = {}
    for src in ("rate_sweep.json", "rate_s192.json"):
        rows = json.loads((K.OUT / src).read_text())["rows"]
        rows = rows[::3] if src == "rate_s192.json" else rows  # HF scoring is slow: every third profile at 192 facts
        seqs = [encode(tx, tok, f, r["q"]) for r in rows for f in r["facts"]]
        order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
        sc = np.zeros(len(seqs))
        with torch.no_grad():
            for b in range(0, len(order), 96):
                idx = order[b:b + 96]
                sc[idx] = scores(model, [seqs[i] for i in idx], yes, no, torch, dev).cpu().numpy()
        n = len(rows[0]["facts"])
        rk = np.array([int((sc[i * n:(i + 1) * n] > sc[i * n + r["ci"]]).sum()) for i, r in enumerate(rows)])
        ty = np.array([r["type"] for r in rows])
        yes_share = float(np.mean(sc > 0))
        out[n] = {"top4": float(np.mean(rk < 4)), "top8": float(np.mean(rk < 8)), "median": float(np.median(rk)), "yes_share": yes_share,
                  "top8_by_type": {t: float(np.mean(rk[ty == t] < 8)) for t in sorted(set(ty))}, "scores": sc.round(3).tolist()}
        print(f"RLORA {label:7s} {n:3d} facts: clue in top 4 {np.mean(rk < 4):.2f}  top 8 {np.mean(rk < 8):.2f}  median {np.median(rk):.0f}  rated Yes {yes_share:.2f}"
              "  top 8 by type: " + "  ".join(f"{t} {v:.2f}" for t, v in out[n]["top8_by_type"].items()) + f"  (+{time.perf_counter() - T0:.0f}s)", flush=True)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--groups", type=int, default=4, help="requests per step; each = 1 known relevant + --neg random facts")
    ap.add_argument("--neg", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--opt", default="muon", choices=["muon", "adamw"])
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--skip-base", action="store_true")
    args = ap.parse_args()
    import torch
    import poc_agentmem as A
    from poc_clue import D, P
    K = PF.poc()
    runner = D.Runner([], K.DECODER)
    model, tok = runner.model, runner.tok
    dev = P.DEV
    tx = W.Text(tok)
    ids = lambda ws: sorted({tok.encode(w, add_special_tokens=False)[0] for w in ws})  # noqa: E731
    yes, no = ids([" Yes", "Yes", " yes"]), ids([" No", "No", " no"])
    su, qvec, fvec, pool, rel = RT.training_space(K)
    rng = random.Random(0)
    reqs = sorted(rel)
    val = set(rng.sample(reqs, 60))
    tr = [q for q in reqs if q not in val]
    res = {"args": vars(args)}
    if not args.skip_base:
        res["base"] = evaluate(model, tok, tx, K, yes, no, torch, dev, "base")
    lora = A.add_lora(model, 16, NAMES)
    lps = [t for m in lora for t in (m.A, m.B)]
    FT.fast_lora(lora)
    opt = FT.optimizer(lps, args.lr, args.opt)
    FT.checkpointing(model)
    scaler = torch.amp.GradScaler("cuda", enabled=next(model.parameters()).dtype == torch.float16)
    model.train()

    def group(q, r):
        negs = [f for f in r.sample(pool, args.neg + 5) if f not in rel[q]][: args.neg]
        return [encode(tx, tok, f, q) for f in [r.choice(sorted(rel[q]))] + negs]

    vr = random.Random(1)
    vgroups = [group(q, vr) for q in sorted(val)]

    def val_loss():
        model.eval()
        ls, top = [], []
        with torch.no_grad():
            for b in range(0, len(vgroups), 6):
                gs = vgroups[b:b + 6]
                s = scores(model, [x for g in gs for x in g], yes, no, torch, dev).view(len(gs), -1)
                ls.append(float(torch.nn.functional.cross_entropy(s / args.temp, torch.zeros(len(gs), dtype=torch.long, device=dev), reduction="sum")))
                top += (s.argmax(1) == 0).tolist()
        model.train()
        return sum(ls) / len(vgroups), float(np.mean(top))

    vl, vt = val_loss()
    print(f"RLORA step 0: val loss {vl:.3f}, known fact rated first among {args.neg + 1} {vt:.2f} (+{time.perf_counter() - T0:.0f}s)", flush=True)
    t0, u, run = time.perf_counter(), 0, []
    limit = args.minutes * 60
    while time.perf_counter() - t0 < limit:
        frac = (time.perf_counter() - t0) / limit
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (u + 1) / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        gs = [group(q, rng) for q in rng.sample(tr, args.groups)]
        s = scores(model, [x for g in gs for x in g], yes, no, torch, dev).view(len(gs), -1)
        loss = torch.nn.functional.cross_entropy(s / args.temp, torch.zeros(len(gs), dtype=torch.long, device=dev))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(lps, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        run.append(float(loss))
        u += 1
        if u % 40 == 0:
            vl, vt = val_loss()
            print(f"RLORA step {u}: train {np.mean(run[-40:]):.3f}  val loss {vl:.3f}, known fact first {vt:.2f} (+{time.perf_counter() - T0:.0f}s)", flush=True)
    res["trained"] = evaluate(model, tok, tx, K, yes, no, torch, dev, "trained")
    res["steps"] = u
    torch.save(A.lora_state(lora) if hasattr(A, "lora_state") else {f"{i}.{k}": v.detach().cpu() for i, m in enumerate(lora) for k, v in (("A", m.A), ("B", m.B))},
               K.OUT.parent / "agentmem" / f"lora_rater{args.tag}.pt")
    (K.OUT / f"rater_lora{args.tag}.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
