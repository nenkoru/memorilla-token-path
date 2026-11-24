"""Check 4 (Mac quick check): teach the decoder to rate facts from their 16-token blocks by copying the text rater.

Test 43 on the Mac (stride 3, 96 unseen-type questions): the plain decoder rating each fact from its block puts the clue in the
top 2 of 8 in 46% and the top 8 of 48 in 38%; the same decoder rating the fact as text gets 75% / 72%. The information is in
the block (POC probes), the decoder just does not judge relevance from it. Here a fresh LoRA (r16, q/k/v/o; core frozen)
learns, listwise, to match the text rater's scores (rater_train.json: 75,309 training-type pairs rated by the 8B from text,
test 36) while reading each fact only as its block. The target is the text rater's own judgement, not "known relevant"
labels (test 40 trained on those and learned the training types). Held-out types never appear in training.

    --minutes 45 -> runs/agentmem/lora_tokrater.pt; then evaluate with
    poc_tok.py rate --lora lora_tokrater.pt --raters tokL --wordings change --stride 3   (table tokL_change)
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
import poc_broad as B  # noqa: E402
import poc_profile as PF  # noqa: E402
import poc_tok as T  # noqa: E402

T0 = time.perf_counter()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=45)
    ap.add_argument("--groups", type=int, default=2, help="requests per step")
    ap.add_argument("--per", type=int, default=10, help="facts per request per step (its known relevant ones first)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--opt", default="adamw", choices=["muon", "adamw"])
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--val", type=int, default=30, help="held-out training-type requests for the agreement check")
    ap.add_argument("--out", default="lora_tokrater.pt")
    ap.add_argument("--data", default="rater_train.json", help="teacher scores: rater_train.json (8B) or rater_train27.json (27B, poc_rate27train.py)")
    args = ap.parse_args()

    import torch
    import poc_agentmem as A
    import poc_core as C
    from poc_agentmem import OUT, P
    from poc_clue import PERSONA_PROMPT
    K = PF.poc()
    su, qvec, fvec, _ = B.load_all()
    rows = [r for r in json.loads((K.OUT / args.data).read_text()) if r["q"] in qvec and r["f"] in fvec]
    by = {}
    for r in rows:
        by.setdefault(r["q"], []).append(r)
    rng = random.Random(0)
    reqs = sorted(q for q, v in by.items() if len(v) >= args.per)
    val = rng.sample(reqs, args.val)
    tr = [q for q in reqs if q not in set(val)]
    print(f"TOKRATER {len(rows)} rated pairs, {len(tr)} train / {len(val)} val requests", flush=True)

    st = A.Stack("lora", train=True)  # fresh LoRA, B = 0: starts as the plain decoder (the 46% / 38% baseline)
    FT.fast_lora(st.lora)  # no dropout, branch in the model's dtype
    core = st.core.float().eval()
    model, tok = st.r.model, st.r.tok
    dt = next(model.parameters()).dtype
    lps = [t for m in st.lora for t in (m.A, m.B)]
    opt = FT.optimizer(lps, args.lr, args.opt)
    FT.checkpointing(model)
    first = lambda ws: sorted({tok.encode(w, add_special_tokens=False)[0] for w in ws})  # noqa: E731
    yes, no = first([" Yes", "Yes", " yes"]), first([" No", "No", " no"])
    head = model.lm_head.weight[yes + no].detach().float()
    ids = {}

    def prompt(q):
        if q not in ids:
            ids[q] = T.rating_ids(A, tok, st.r.add_special, PERSONA_PROMPT, T.WORD["change"].format(q=q), 16)
        return ids[q]

    def scores(pairs):
        """Yes-vs-No log-odds for (request, fact) pairs, each fact as its 16-token block conditioned on its request."""
        with torch.no_grad():
            d = torch.from_numpy(np.stack([fvec[f] for _, f in pairs]).astype(np.float32))[:, None].to(P.DEV)
            q = torch.from_numpy(np.stack([qvec[x] for x, _ in pairs]).astype(np.float32)).to(P.DEV)
            mem = C.core_fwd(core, d, q).to(dt)
            embs = [P.input_embeds(model, tok, st.r.memory_id, torch.tensor([prompt(x)], device=P.DEV), mem[j:j + 1])[0] for j, (x, _) in enumerate(pairs)]
            t = max(e.shape[0] for e in embs)
            x = embs[0].new_zeros(len(embs), t, embs[0].shape[1])
            mask = torch.zeros(len(embs), t, dtype=torch.long, device=P.DEV)
            for j, e in enumerate(embs):  # left padding: the last position answers
                x[j, t - e.shape[0]:] = e
                mask[j, t - e.shape[0]:] = 1
        h = model.model(inputs_embeds=x, attention_mask=mask).last_hidden_state[:, -1]
        lg = h.float() @ head.t()
        return torch.logsumexp(lg[:, :len(yes)], -1) - torch.logsumexp(lg[:, len(yes):], -1)

    def group(q, r):
        v = by[q]
        known = [x for x in v if x["known"]]
        rest = r.sample([x for x in v if not x["known"]], args.per - min(len(known), 2))
        return known[:2] + rest

    vr = random.Random(1)
    vgroups = [group(q, vr) for q in val]

    def agreement():
        """Mean Spearman between student and text-rater scores within a request, and how often the text rater's top fact
        is in the student's top 3."""
        model.eval()
        rho, top = [], []
        with torch.no_grad():
            for g in vgroups:
                s = scores([(x["q"], x["f"]) for x in g]).cpu().numpy()
                t = np.array([x["s"] for x in g])
                rs, rt = np.argsort(np.argsort(s)), np.argsort(np.argsort(t))
                rho.append(np.corrcoef(rs, rt)[0, 1])
                top.append(int(np.argmax(t)) in set(np.argsort(-s)[:3].tolist()))
        model.train()
        return float(np.nanmean(rho)), float(np.mean(top))

    rho, top = agreement()
    print(f"TOKRATER step 0: val spearman {rho:.3f}, text rater's top fact in the student's top 3 {top:.2f} (+{time.perf_counter() - T0:.0f}s)", flush=True)
    model.train()
    t0, u, run = time.perf_counter(), 0, []
    limit = args.minutes * 60
    while time.perf_counter() - t0 < limit:
        frac = (time.perf_counter() - t0) / limit
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (u + 1) / 10) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        gs = [group(q, rng) for q in rng.sample(tr, args.groups)]
        s = scores([(x["q"], x["f"]) for g in gs for x in g]).view(len(gs), -1)
        t = torch.tensor([[x["s"] for x in g] for g in gs], device=s.device, dtype=torch.float32)
        loss = torch.nn.functional.kl_div(torch.log_softmax(s / args.temp, -1), torch.log_softmax(t / args.temp, -1), log_target=True, reduction="batchmean")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lps, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        run.append(float(loss))
        u += 1
        if u == 1 or u % 20 == 0:
            print(f"TOKRATER step {u}: train KL {np.mean(run[-20:]):.3f} (+{time.perf_counter() - T0:.0f}s)", flush=True)
        if u % 80 == 0:
            rho, top = agreement()
            print(f"TOKRATER step {u}: val spearman {rho:.3f}, top fact in top 3 {top:.2f}", flush=True)
    rho, top = agreement()
    print(f"TOKRATER done: {u} steps, {u * args.groups} requests; val spearman {rho:.3f}, top fact in top 3 {top:.2f} (+{(time.perf_counter() - T0) / 60:.1f} min)", flush=True)
    torch.save(st.state(), OUT / args.out)
    print(f"TOKRATER saved {OUT / args.out}", flush=True)


if __name__ == "__main__":
    main()
