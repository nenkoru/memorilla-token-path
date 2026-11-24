"""Check 4c teacher data: the 27B rates the same training pairs the 8B rated for check 4 (rater_train.json).

Test 43: the block rater (poc_tokrater.py) is capped by its teacher. The 8B text rater puts the clue first of 48 in 0.16,
the student 0.09; the 27B text rater ("wrong") 0.35. This re-rates a subset of rater_train.json's training-type pairs with
the 27B, same schema, so poc_tokrater.py --data rater_train27.json can learn from it. Held-out types never appear.

On the box, with the 27B served (v100.sh serve27):
  MPORT_VLLM_URL=http://127.0.0.1:8150 MPORT_VLLM_MODEL=qwen3-27b MPORT_RATER_TOK=<27B dir> poc_rate27train.py --requests 1000
  -> work/runs/poc/rater_train27.json
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import poc_profile as PF  # noqa: E402
import poc_whole as W  # noqa: E402
from poc_rate_sweep import PROMPTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=1000)
    ap.add_argument("--prompt", default="wrong", choices=sorted(PROMPTS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rater_train27.json")
    ap.add_argument("--exclude", default="", help="comma-separated earlier outputs: their requests are not drawn again")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    K = PF.poc()
    t0 = time.perf_counter()
    by = {}
    for r in json.loads((K.OUT / "rater_train.json").read_text()):
        by.setdefault(r["q"], []).append(r)
    done = {r["q"] for f in filter(None, args.exclude.split(",")) for r in json.loads((K.OUT / f).read_text())}
    pool = sorted(q for q in by if q not in done)
    reqs = random.Random(args.seed).sample(pool, min(args.requests, len(pool)))
    jobs = [r for q in reqs for r in by[q]]
    tx = W.Text(AutoTokenizer.from_pretrained(os.environ.get("MPORT_RATER_TOK", K.DECODER)))
    pr = PROMPTS[args.prompt]
    assert "{f}" not in pr, "fact-last wordings are not supported here"
    print(f"RATE27 {len(reqs)} requests ({len(done)} excluded), {len(jobs)} pairs, prompt {args.prompt}", flush=True)
    sc = W.yes_score(tx, [tx.msgs([r["f"]], pr.format(q=r["q"])) for r in jobs])
    out = [{"q": r["q"], "f": r["f"], "s": round(s, 3), "known": r["known"], "s8": r["s"]} for r, s in zip(jobs, sc)]
    (K.OUT / args.out).write_text(json.dumps(out))
    s, s8, kn = (np.array([o[k] for o in out]) for k in ("s", "s8", "known"))
    print(f"RATE27 rated {len(out)} pairs in {time.perf_counter() - t0:.0f}s; known relevant {s[kn].mean():.2f}, random {s[~kn].mean():.2f};"
          f" correlation with the 8B {np.corrcoef(s, s8)[0, 1]:.2f}", flush=True)


if __name__ == "__main__":
    main()
