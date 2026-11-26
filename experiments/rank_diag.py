"""Why does the block rater rank the clue below other facts? CPU only (no model).

For the held-out clue types it compares a rater's score with plain topical closeness (cosine between the
question vector and the fact vector), and compares the eval distractors with the rater's training negatives.

    rank_diag.py [--tables tokLr27w_full_change,tokLq4_full_change,r27_wrong] [--train rater_train.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import poc_broad as B  # noqa: E402
import poc_profile as PF  # noqa: E402
import poc_tok as T  # noqa: E402
import poc_whole as W  # noqa: E402


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 and np.std(ra) > 0 and np.std(rb) > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default="tokLr27w_full_change,tokLq4_full_change,r27_wrong,r8_change")
    ap.add_argument("--train", default="rater_train.json")
    args = ap.parse_args()
    K = PF.poc()
    su, qvec, fvec, _ = B.load_all()
    ev, pmap = B.test_set(su, qvec, True)
    held = [e for e in ev if e["group"] == "held-out types"]
    full = T.load_tables(K)
    print(f"DIAG {len(held)} held-out questions; tables present: {[t for t in args.tables.split(',') if t in full]}", flush=True)

    # 1. how topical is the clue: its rank by cosine alone, and the cosine of the clue against the distractors
    for size in T.SIZES:
        cr, cc, dc = [], [], []
        for e in held:
            q = pmap[e["pair"]]["question"]
            facts = T.profile_facts(e, size)
            cos = np.array([float(unit(qvec[q]) @ unit(fvec[f])) for f in facts])
            cr.append(int((cos > cos[-1]).sum()))  # the clue is the last fact
            cc.append(cos[-1])
            dc += cos[:-1].tolist()
        cr = np.array(cr)
        print(f"DIAG cosine alone, from {size:2d}: clue first {np.mean(cr == 0):.2f}  in top 2 {np.mean(cr < 2):.2f}  in top 4 {np.mean(cr < 4):.2f}"
              f"   mean cosine: clue {np.mean(cc):.3f}  distractors {np.mean(dc):.3f} (90th pct {np.percentile(dc, 90):.3f})", flush=True)

    # 2. per rater: clue rank, how much its scores follow cosine, and what the facts that beat the clue look like
    for name in args.tables.split(","):
        tab = full.get(name)
        if tab is None:
            continue
        for size in T.SIZES:
            first, rho, beat_pct, n_beat, miss = [], [], [], [], 0
            for e in held:
                q = pmap[e["pair"]]["question"]
                facts = T.profile_facts(e, size)
                sc = [tab.get(W.fact_key(q, f)) for f in facts]
                if any(s is None for s in sc):
                    miss += 1
                    continue
                sc = np.array(sc)
                cos = np.array([float(unit(qvec[q]) @ unit(fvec[f])) for f in facts])
                d_sc, d_cos = sc[:-1], cos[:-1]
                rho.append(spearman(d_sc, d_cos))  # among distractors only
                beat = d_sc > sc[-1]
                first.append(not beat.any())
                n_beat.append(int(beat.sum()))
                if beat.any():  # cosine percentile (within this profile's distractors) of the facts ranked above the clue
                    pct = np.argsort(np.argsort(d_cos)) / max(len(d_cos) - 1, 1)
                    beat_pct.append(float(pct[beat].mean()))
            print(f"DIAG {name:24s} from {size:2d}: clue first {np.mean(first):.2f}  facts above the clue {np.mean(n_beat):.1f}"
                  f"  score~cosine among distractors (Spearman) {np.nanmean(rho):+.2f}"
                  f"  cosine percentile of the facts above the clue {np.mean(beat_pct) if beat_pct else float('nan'):.2f} (0.5 = no topical bias)"
                  f"  skipped {miss}", flush=True)

    # 3. agreement between raters on the distractors (does the student follow its teacher where it matters?)
    names = [t for t in args.tables.split(",") if t in full]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rho = []
            for e in held:
                q = pmap[e["pair"]]["question"]
                facts = T.profile_facts(e, 48)
                sa = [full[a].get(W.fact_key(q, f)) for f in facts]
                sb = [full[b].get(W.fact_key(q, f)) for f in facts]
                if any(s is None for s in sa + sb):
                    continue
                rho.append(spearman(np.array(sa), np.array(sb)))
            if rho:
                print(f"DIAG agreement over 48 facts (Spearman) {a} ~ {b}: {np.nanmean(rho):+.2f}  (n {len(rho)})", flush=True)

    # 4. the rater's training candidates: are the negatives as close to the request as the eval distractors are?
    p = K.OUT / args.train
    if p.exists():
        d = json.loads(p.read_text())
        rows = d["rows"] if isinstance(d, dict) else d
        pos, neg = [], []
        for r in rows[:: max(1, len(rows) // 20000)]:
            if r["q"] in qvec and r["f"] in fvec:
                (pos if r.get("known") else neg).append(float(unit(qvec[r["q"]]) @ unit(fvec[r["f"]])))
        if pos and neg:
            print(f"DIAG training candidates ({args.train}): cosine request~fact  known-relevant {np.mean(pos):.3f}  other {np.mean(neg):.3f}"
                  f" (90th pct {np.percentile(neg, 90):.3f})  n {len(pos)} / {len(neg)}", flush=True)
        by = {}
        for r in rows:
            by.setdefault(r["q"], []).append(r)
        rk = []
        for q, rs in list(by.items())[:3000]:
            if q not in qvec or not any(r.get("known") for r in rs) or any(r["f"] not in fvec for r in rs):
                continue
            cos = np.array([float(unit(qvec[q]) @ unit(fvec[r["f"]])) for r in rs])
            k = [i for i, r in enumerate(rs) if r.get("known")][0]
            rk.append(int((cos > cos[k]).sum()))
        if rk:
            rk = np.array(rk)
            print(f"DIAG training requests: cosine alone puts a known-relevant fact first in {np.mean(rk == 0):.2f} of {len(rk)} requests"
                  f" (~{np.mean([len(v) for v in by.values()]):.0f} candidates each)", flush=True)
    print("DIAG done", flush=True)


if __name__ == "__main__":
    main()
