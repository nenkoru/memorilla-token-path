"""Batched versions of the two hot loops of the reader experiments.

Measured on the H15 run (poc_profile.py): training ran one example per forward/backward (about 1 example/s) and the
evaluation generated one answer at a time (612 answers in 55 min, as long as the training itself). These helpers pad
several examples to one left-padded batch so the decoder runs once per batch.

    loss_batch(st, items)          items = [(question, mem [1, n, 4096], answer)]; mean of per-example losses
    generate_batch(st, items, n)   items = [(question, mem)]; list of answers
    selftest                       GPU: checks both against the one-at-a-time versions and prints examples per second

New training and evaluation scripts should import these instead of looping over A.generate / one-example losses.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402


def pad_batch(st, items):
    """Left-pad the input embeddings of several (question, mem, answer or None) examples into one batch."""
    import torch
    from poc_agentmem import P
    embs, labs = [], []
    for question, mem, answer in items:
        ids, labels = st.ids(question, mem.shape[1], answer)
        embs.append(P.input_embeds(st.r.model, st.r.tok, st.r.memory_id, ids, mem)[0])
        labs.append(None if labels is None else labels[0])
    b, t = len(embs), max(e.shape[0] for e in embs)
    x = embs[0].new_zeros(b, t, embs[0].shape[1])
    mask = torch.zeros(b, t, dtype=torch.long, device=x.device)
    lab = torch.full((b, t), -100, dtype=torch.long, device=x.device)
    for i, e in enumerate(embs):
        x[i, t - e.shape[0]:] = e
        mask[i, t - e.shape[0]:] = 1
        if labs[i] is not None:
            lab[i, t - e.shape[0]:] = labs[i]
    return x, mask, lab


def loss_batch(st, items):
    """Mean over examples of the per-example mean token loss (same weighting as averaging one-example losses)."""
    import torch
    x, mask, lab = pad_batch(st, items)
    logits = st.r.model(inputs_embeds=x, attention_mask=mask, use_cache=False).logits[:, :-1].float()
    tgt = lab[:, 1:]
    ce = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), ignore_index=-100,
                                           reduction="none").view(tgt.shape)
    return (ce.sum(1) / (tgt != -100).sum(1).clamp(min=1)).mean()


def generate_batch(st, items, max_new):
    import torch
    x, mask, _ = pad_batch(st, [(q, m, None) for q, m in items])
    with torch.no_grad():
        out = st.r.model.generate(inputs_embeds=x, attention_mask=mask, max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=st.r.tok.pad_token_id)
    return [st.r.tok.decode(o, skip_special_tokens=True).strip() for o in out]


class Plateau:
    """Stop a time-boxed training early: call with a held-out loss every check; True once it has not improved by
    `tol` for `patience` checks in a row."""

    def __init__(self, patience=3, tol=0.01):
        self.best, self.bad, self.patience, self.tol = float("inf"), 0, patience, tol

    def __call__(self, value):
        if value < self.best - self.tol:
            self.best, self.bad = value, 0
        else:
            self.bad += 1
        return self.bad >= self.patience


def cmd_selftest(args):
    import torch
    import poc_agentmem as A
    import poc_core as C
    import poc_profile as PF
    from poc_agentmem import P
    K = PF.poc()
    *_, emap = PF.setup()
    gen = json.loads((K.OUT / "profile_gen.json").read_text())["rows"]
    rows = [r for r in gen if r["fact"] in emap and r["request"] in emap][:: 7][: args.n]
    st = A.Stack("lora", train=True)
    C.load_lora(st, args.lora)
    core = st.core.float().eval()
    dt = next(st.r.model.parameters()).dtype
    with torch.no_grad():
        mems = [PF.blocks(C, core, np.stack([emap[r["fact"]]]), emap[r["request"]], 4, P, torch).to(dt) for r in rows]
    items = [(r["request"] + PF.SUFFIX, m, r["answer"]) for r, m in zip(rows, mems)]
    lps = [t for m in st.lora for t in (m.A, m.B)]

    def sync():
        torch.mps.synchronize()  # mapped to CUDA by modal_entry.py

    st.r.model.train()
    with torch.no_grad():
        single = [float(st.r.model(inputs_embeds=P.input_embeds(st.r.model, st.r.tok, st.r.memory_id, ids, m),
                                   attention_mask=torch.ones_like(ids), labels=lab, use_cache=False).loss)
                  for (q, m, a) in items for ids, lab in [st.ids(q, m.shape[1], a)]]
        batched = float(loss_batch(st, items))
    print(f"FAST loss: one at a time {np.mean(single):.4f}  batched {batched:.4f}  diff {abs(np.mean(single) - batched):.4f}", flush=True)

    def time_train(bs):
        for p in lps:
            p.grad = None
        sync()
        t0 = time.perf_counter()
        for rep in range(args.reps):
            for b in range(0, len(items), bs):
                loss_batch(st, items[b:b + bs]).backward()
        sync()
        dt_s = time.perf_counter() - t0
        for p in lps:
            p.grad = None
        torch.mps.empty_cache()
        return args.reps * len(items) / dt_s

    if not args.skip_train:
        time_train(1)  # warm-up
        for bs in [int(x) for x in args.batches.split(",")]:
            print(f"FAST train batch {bs}: {time_train(bs):.2f} examples/s", flush=True)

    st.r.model.eval()
    gi = [(q, m) for q, m, _ in items]
    sync()
    t0 = time.perf_counter()
    one = [A.generate(st, q, m, args.max_new_tokens).strip() for q, m in gi]
    sync()
    t1 = time.perf_counter()
    many = generate_batch(st, gi, args.max_new_tokens)
    sync()
    t2 = time.perf_counter()
    same = sum(a == b for a, b in zip(one, many))
    print(f"FAST generate: one at a time {len(gi) / (t1 - t0):.2f} answers/s  batch of {len(gi)} {len(gi) / (t2 - t1):.2f} answers/s  "
          f"identical answers {same}/{len(gi)}", flush=True)
    for a, b in list(zip(one, many))[:2]:
        print("FAST sample", json.dumps({"single": a[:160], "batched": b[:160]}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("selftest")
    t.add_argument("--lora", default="lora_prof.pt")
    t.add_argument("--n", type=int, default=8)
    t.add_argument("--reps", type=int, default=2)
    t.add_argument("--skip-train", action="store_true")
    t.add_argument("--batches", default="1,2")
    t.add_argument("--max-new-tokens", type=int, default=70)
    args = ap.parse_args()
    {"selftest": cmd_selftest}[args.cmd](args)


if __name__ == "__main__":
    main()
