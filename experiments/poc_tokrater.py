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
import os
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


import q8  # noqa: E402  (memory report; the decoder itself is quantized only when MPORT_Q8=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=45)
    ap.add_argument("--steps", type=int, default=0, help="if > 0: train this many steps (cosine over steps) instead of for --minutes")
    ap.add_argument("--groups", type=int, default=2, help="requests per step")
    ap.add_argument("--per", type=int, default=10, help="facts per request per step (its known relevant ones first)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--opt", default="adamw", choices=["muon", "adamw"])
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--val", type=int, default=30, help="held-out training-type requests for the agreement check")
    ap.add_argument("--out", default="lora_tokrater.pt")
    ap.add_argument("--layout", default="sys", choices=["sys", "qfirst"], help="memory in the system prompt (original) or after the request")
    ap.add_argument("--shared", type=int, default=1, help="1: each request's prefix computed once for its facts (T.shared_hidden)")
    ap.add_argument("--init", default="", help="start from this LoRA (in runs/agentmem) instead of a fresh one")
    ap.add_argument("--lora-from", type=int, default=0, help="LoRA only on layers >= this; backward then stops there (cheaper step)")
    ap.add_argument("--check", action="store_true", help="before training: shared-prefix scores vs the plain forward on one val group (layout sys)")
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
    if args.init:
        C.load_lora(st, args.init)
    FT.fast_lora(st.lora)  # no dropout, branch in the model's dtype
    core = st.core.float().eval()
    if q8.lean():  # the memory core runs on the CPU (no gradient, a few vectors per step): 0.8 GB less on the GPU
        core = core.cpu()
    cdev = next(core.parameters()).device
    model, tok = st.r.model, st.r.tok
    dt = next(model.parameters()).dtype
    for i, m in enumerate(st.lora):  # keys are layer * 4 + projection
        if i // 4 < args.lora_from:
            m.A.requires_grad_(False)
            m.B.requires_grad_(False)  # B starts at 0, so these layers stay the plain decoder
    lps = [t for m in st.lora for t in (m.A, m.B) if t.requires_grad]
    opt = FT.optimizer(lps, args.lr, args.opt)
    FT.checkpointing(model)
    first = lambda ws: sorted({tok.encode(w, add_special_tokens=False)[0] for w in ws})  # noqa: E731
    yes, no = first([" Yes", "Yes", " yes"]), first([" No", "No", " no"])
    head = model.lm_head.weight[yes + no].detach().float().to(P.DEV)
    q8.drop_lm_head(model)  # MPORT_LEAN=1 only: training reads model.model and these rows
    ids = {}
    timing = os.environ.get("MPORT_TIMING") == "1"
    clock = {"last": 0.0}
    spent = {}

    def tick(name):
        """MPORT_TIMING=1: wall time per phase of a step (synchronised, so it slows the run a little)."""
        if timing:
            torch.cuda.synchronize()
            now = time.perf_counter()
            if name:
                spent[name] = spent.get(name, 0.0) + now - clock["last"]
            clock["last"] = now

    assert args.shared or args.layout == "sys", "--layout qfirst needs --shared 1"

    def prompt(q, layout=args.layout):
        if (q, layout) not in ids:
            ids[q, layout] = T.prompt_ids(A, tok, st.r.add_special, st.r.memory_id, PERSONA_PROMPT, T.WORD["change"].format(q=q), layout)
        return ids[q, layout]

    def scores(pairs, shared=args.shared, layout=args.layout):
        """Yes-vs-No log-odds for (request, fact) pairs, each fact as its 16-token block conditioned on its request."""
        with torch.no_grad():
            d = torch.from_numpy(np.stack([fvec[f] for _, f in pairs]).astype(np.float32))[:, None].to(cdev)
            q = torch.from_numpy(np.stack([qvec[x] for x, _ in pairs]).astype(np.float32)).to(cdev)
            mem = C.core_fwd(core, d, q).to(P.DEV, dt)
            if shared:
                qs = list(dict.fromkeys(x for x, _ in pairs))
                pre = [P.input_embeds(model, tok, st.r.memory_id, torch.tensor([prompt(x, layout)[0]], device=P.DEV), None)[0] for x in qs]
                suf = [P.input_embeds(model, tok, st.r.memory_id, torch.tensor([prompt(x, layout)[1]], device=P.DEV), mem[j:j + 1])[0] for j, (x, _) in enumerate(pairs)]
        tick("prep (core, embeddings)")
        if shared:
            h = T.shared_hidden(model, pre, suf, [qs.index(x) for x, _ in pairs])
            tick("forward")
            lg = h.float() @ head.t()
            return torch.logsumexp(lg[:, :len(yes)], -1) - torch.logsumexp(lg[:, len(yes):], -1)
        with torch.no_grad():
            embs = [P.input_embeds(model, tok, st.r.memory_id, torch.tensor([sum(prompt(x, layout), [])], device=P.DEV), mem[j:j + 1])[0] for j, (x, _) in enumerate(pairs)]
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

    if args.check:
        model.eval()
        with torch.no_grad():
            pairs = [(x["q"], x["f"]) for x in vgroups[0]]
            a_, b_ = scores(pairs, shared=0, layout="sys"), scores(pairs, shared=1, layout="sys")
        print(f"TOKRATER check: shared-prefix vs plain forward on {len(pairs)} pairs: max |diff| {float((a_ - b_).abs().max()):.4f},"
              f" scores {[round(v, 2) for v in a_.tolist()[:4]]} vs {[round(v, 2) for v in b_.tolist()[:4]]}", flush=True)
        model.train()
    rho, top = agreement()
    print(f"TOKRATER step 0: val spearman {rho:.3f}, text rater's top fact in the student's top 3 {top:.2f} (+{time.perf_counter() - T0:.0f}s)", flush=True)
    model.train()
    t0, u, run = time.perf_counter(), 0, []
    prof = None
    if os.environ.get("MPORT_PROFILE") == "1":  # profile steps 3-4, print the top kernels, stop
        prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                      schedule=torch.profiler.schedule(wait=1, warmup=1, active=2))
        prof.start()
    limit = args.minutes * 60
    while (u < args.steps) if args.steps else (time.perf_counter() - t0 < limit):
        frac = u / args.steps if args.steps else (time.perf_counter() - t0) / limit
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (u + 1) / 10) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        gs = [group(q, rng) for q in rng.sample(tr, args.groups)]
        tick("")
        s = scores([(x["q"], x["f"]) for g in gs for x in g]).view(len(gs), -1)
        t = torch.tensor([[x["s"] for x in g] for g in gs], device=s.device, dtype=torch.float32)
        loss = torch.nn.functional.kl_div(torch.log_softmax(s / args.temp, -1), torch.log_softmax(t / args.temp, -1), log_target=True, reduction="batchmean")
        tick("loss")
        loss.backward()
        tick("backward")
        torch.nn.utils.clip_grad_norm_(lps, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        tick("optimizer")
        if prof is not None:
            prof.step()
            if u == 4:
                prof.stop()
                print(prof.key_averages().table(sort_by="self_device_time_total", row_limit=18), flush=True)
                print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=8), flush=True)
                return
        run.append(float(loss))
        u += 1
        if u == 1 or u % 20 == 0:
            print(f"TOKRATER step {u}: train KL {np.mean(run[-20:]):.3f} (+{time.perf_counter() - T0:.0f}s, {q8.mem_gib()})", flush=True)
            if timing:
                print("TOKRATER time per step: " + ", ".join(f"{k} {v / u:.2f}s" for k, v in spent.items()), flush=True)
        if u % 80 == 0:
            rho, top = agreement()
            print(f"TOKRATER step {u}: val spearman {rho:.3f}, top fact in top 3 {top:.2f}", flush=True)
    rho, top = agreement()
    print(f"TOKRATER done: {u} steps, {u * args.groups} requests; val spearman {rho:.3f}, top fact in top 3 {top:.2f} (+{(time.perf_counter() - T0) / 60:.1f} min, {q8.mem_gib()})", flush=True)
    torch.save(st.state(), OUT / args.out)
    print(f"TOKRATER saved {OUT / args.out}", flush=True)


if __name__ == "__main__":
    main()
