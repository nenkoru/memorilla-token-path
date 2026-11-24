"""H35c: does the wording of the per-fact rating question change where the clue ranks? Rank only, no answers.

For every unseen-type eval row, the plain decoder rates each of the 48 profile facts alone with each prompt below
(log P(Yes) - log P(No)); prints where the clue ranks, and the same for the rating plus embedding similarity.
  MPORT_VLLM_URL=... poc_rate_sweep.py   -> work/runs/poc/rate_sweep.json
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import poc_broad as B  # noqa: E402
import poc_profile as PF  # noqa: E402
import poc_whole as W  # noqa: E402

PROMPTS = {
    "base": W.RATE,
    "worse": 'The user asks: "{q}"\n\nWould a reply that ignores the fact above be worse for this user? Answer Yes or No.',
    "change": 'The user asks: "{q}"\n\nDoes the fact above change what a good reply should say or recommend? Answer Yes or No.',
    "wrong": 'The user asks: "{q}"\n\nCould a typical reply to this request be wrong, unsafe or unsuitable for this user because of the fact above? Answer Yes or No.',
    "mention": 'The user asks: "{q}"\n\nShould a thoughtful assistant mention or adapt to the fact above in its reply? Answer Yes or No.',
    # request first, fact last: the request and question form a prefix vLLM caches across a profile's facts
    "change_qfirst": 'The user asks: "{q}"\n\nDoes the following fact about the user change what a good reply should say or recommend? Answer Yes or No.\n\nFact: {f}',
    "mention_qfirst": 'The user asks: "{q}"\n\nShould a thoughtful assistant mention or adapt to the following fact about the user in its reply? Answer Yes or No.\n\nFact: {f}',
}


def main():
    import argparse
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma-separated prompt names")
    ap.add_argument("--out", default="rate_sweep.json")
    ap.add_argument("--size", type=int, default=48, help="above 48: pad each profile with random training-pool facts (other users')")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--from-file", action="store_true")
    ap.add_argument("--select-key", default="", help="also save scores to select_scores.json as <key>_<prompt>")
    args = ap.parse_args()
    prompts = {k: v for k, v in PROMPTS.items() if not args.only or k in args.only.split(",")}
    K = PF.poc()
    T0 = time.perf_counter()
    su, qvec, fvec, _ = B.load_all()
    import os
    tx = W.Text(AutoTokenizer.from_pretrained(os.environ.get("MPORT_RATER_TOK", K.DECODER)))  # H41: a bigger rater brings its own tokenizer
    ev, pmap = B.test_set(su, qvec, True)
    rows = []
    if args.size > 48:
        import random
        import poc_rater as RT
        pool = RT.training_space(K)[3]
    for e in ev:
        if e["group"] != "held-out types":
            continue
        lst = W.profile(e, 48, True, "rank")
        q = pmap[e["pair"]]["question"]
        if args.size > 48:
            rng = random.Random(e["pair"] + "pad")
            lst = lst + [f for f in rng.sample(pool, args.size) if f not in lst][: args.size - 48]
            rng.shuffle(lst)
        rows.append({"type": e["type"], "q": q, "facts": lst, "ci": lst.index(e["clue"]),
                     "cos": [float(B.unit(qvec[q]) @ B.unit(fvec[f])) if q in qvec and f in fvec else 0.0 for f in lst]})
    rows = rows[::args.every]  # every Nth profile (same subset as poc_rater_lora.py's 192-fact eval with --every 3)
    out = {}
    if args.from_file:  # no rating: reuse the rows saved in --out (to rewrite select_scores.json tables)
        rows = json.loads((K.OUT / args.out).read_text())["rows"]
        prompts = {k: v for k, v in prompts.items() if k in rows[0]}
    for name, pr in ({} if args.from_file else prompts).items():
        t1 = time.perf_counter()
        sc = W.yes_score(tx, [tx.msgs([], pr.format(q=r["q"], f=f)) if "{f}" in pr else tx.msgs([f], pr.format(q=r["q"])) for r in rows for f in r["facts"]])
        for i, r in enumerate(rows):
            r[name] = sc[i * len(r["facts"]):(i + 1) * len(r["facts"])]
        print(f"SWEEP {name}: {len(sc)} ratings in {time.perf_counter() - t1:.0f}s, at +{time.perf_counter() - T0:.0f}s", flush=True)

    def report(key, fn):
        rk = []
        for r in rows:
            s = np.array(fn(r))
            rk.append(int((s > s[r["ci"]]).sum()))
        rk = np.array(rk)
        ty = np.array([r["type"] for r in rows])
        o = {"top4": float(np.mean(rk < 4)), "top8": float(np.mean(rk < 8)), "top12": float(np.mean(rk < 12)), "top24": float(np.mean(rk < 24)), "median": float(np.median(rk)),
             "top8_by_type": {t: float(np.mean(rk[ty == t] < 8)) for t in sorted(set(ty))}}
        out[key] = o
        print(f"SWEEP {key:22s} clue in top 4 {o['top4']:.2f}  top 8 {o['top8']:.2f}  top 12 {o['top12']:.2f}  top 24 {o['top24']:.2f}  median {o['median']:.0f}  top 8 by type: "
              + "  ".join(f"{t} {v:.2f}" for t, v in o["top8_by_type"].items()), flush=True)

    z = lambda x: (np.array(x) - np.mean(x)) / (np.std(x) + 1e-6)  # noqa: E731
    report("cosine", lambda r: r["cos"])
    for name in prompts:
        report(name, lambda r, n=name: r[n])
    report("mean of all prompts", lambda r: sum(z(r[n]) for n in prompts))
    for name in prompts:
        report(f"{name} + cosine", lambda r, n=name: z(r[n]) + z(r["cos"]))
    (K.OUT / args.out).write_text(json.dumps({"results": out, "rows": rows}))
    if args.select_key and args.size == 48:  # per-fact scores for poc_whole.py's "saved <key>" conditions (any subset of these profiles)
        sp = K.OUT / "select_scores.json"
        allsel = json.loads(sp.read_text())
        for name in prompts:
            allsel[f"{args.select_key}_{name}"] = {W.fact_key(r["q"], f): float(v) for r in rows for f, v in zip(r["facts"], r[name])}
        tmp = sp.with_suffix(".tmp")
        tmp.write_text(json.dumps(allsel))
        tmp.replace(sp)


if __name__ == "__main__":
    main()
