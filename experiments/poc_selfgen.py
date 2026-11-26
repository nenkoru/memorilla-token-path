"""Test 55, step 1: the reader samples its own answers from the memory blocks of training fact sets (poc_read27.py outputs),
for poc_broad.py train --mix27 after judge27.py --rows has kept the ones that take the relevant fact into account.

    poc_selfgen.py --lora lora_kd27a.pt --data read27.json+read27far.json --sets 600 --k 4 --min-facts 5 [--part 0/2] --out selfgen_0.json
Only sets with a relevant fact and at least --min-facts facts (the reader fails most when it is shown many). Facts in the
order item27 gives them in training (by the 27B's score; half the time the relevant fact moved to a random place), stored in
that order. --part i/n takes every n-th set from i, so two GPUs split one draw.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import poc_broad as B  # noqa: E402
import poc_profile as PF  # noqa: E402
import poc_tok as T  # noqa: E402
import q8  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default="lora_kd27a.pt")
    ap.add_argument("--data", default="read27.json+read27far.json")
    ap.add_argument("--sets", type=int, default=600)
    ap.add_argument("--k", type=int, default=4, help="samples per set")
    ap.add_argument("--min-facts", type=int, default=5)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="selfgen.json")
    args = ap.parse_args()
    from poc_agentmem import P
    import fast_gen as FG
    K = PF.poc()
    su, qvec, fvec, _ = B.load_all()
    rows = [r for f in args.data.split("+") for r in json.loads((K.OUT / f).read_text())["rows"]]
    rows = [r for r in rows if r["relevant"] is not None and len(r["facts"]) >= args.min_facts
            and r["request"] in qvec and all(f in fvec for f in r["facts"])]
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    i, n = (int(x) for x in args.part.split("/"))
    sets = rows[: args.sets][i::n]
    for r in sets:  # the order item27 trains with (--sort)
        order = sorted(range(len(r["facts"])), key=lambda j: -r["scores"][j])
        if rng.random() < 0.5:
            k = order.pop(next(j for j, x in enumerate(order) if r["facts"][x] == r["relevant"]))
            order.insert(rng.randint(0, len(order)), k)
        r["facts"], r["scores"] = [r["facts"][j] for j in order], [r["scores"][j] for j in order]
    print(f"SELFGEN {len(rows)} usable sets, {len(sets)} in this part ({args.part}), {args.k} samples each at temperature {args.temp}, "
          f"top-p {args.top_p}; mean facts {np.mean([len(r['facts']) for r in sets]):.1f}", flush=True)
    torch, A, C, st, core, dt = T.load_stack(args.lora)
    if q8.lean():  # as in poc_tok.py answer
        core = q8.place_core(core)
        q8.head_to_gpu(st.r.model)
    t0 = time.perf_counter()
    items = []
    with torch.no_grad():
        for r in sets:
            mem = PF.blocks(C, core, np.stack([fvec[f] for f in r["facts"]]), qvec[r["request"]], 16, P, torch).to(dt).cpu()
            items += [(r["request"], mem)] * args.k
    core.cpu()
    torch.cuda.empty_cache()
    print(f"SELFGEN blocks for {len(sets)} sets in {time.perf_counter() - t0:.0f}s", flush=True)
    ans = FG.generate(st, items, args.max_new_tokens, 0, log=lambda m: print(m, flush=True),
                      temp=args.temp, top_p=args.top_p, seed=args.seed * 100 + i)
    out = []
    for j, a in enumerate(ans):
        r = sets[j // args.k]
        note = ""
        if a.startswith("Relevant fact:"):
            note, _, a = a.partition("\n")
        out.append({"request": r["request"], "facts": r["facts"], "scores": r["scores"], "relevant": r["relevant"],
                    "sample": j % args.k, "note": note, "answer": a.strip()})
    (K.OUT / args.out).write_text(json.dumps({"args": vars(args), "rows": out}))
    print(f"SELFGEN done: {len(out)} answers, mean words {np.mean([len(o['answer'].split()) for o in out]):.0f}, "
          f"+{(time.perf_counter() - t0) / 60:.1f} min; sample: {out[0]['answer'][:300]!r}", flush=True)


if __name__ == "__main__":
    main()
