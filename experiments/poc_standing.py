"""H30: a per-fact score for "matters whatever the topic" (standing facts), on saved vectors only (no model).

The relevance map picks by topic and misses facts such as a health condition (see "Where the losses are" in
AGENTMEM_HYPOTHESES.md). Here each stored fact gets one score that does not depend on the request, learned from
the generated facts, and the picker keeps the best-scoring facts always on next to the map's picks.

Scores, each a ridge fit on the encoder vector of the fact:
  labels narrow / wide  the fact's generated category is in a hand-written list of standing categories
                        (narrow: body and dependents; wide: also money, time, fears, habits)
  gap                   1 - mean cosine between the fact and the requests it was generated for
  spread                1 - mean cosine among those requests

  MPORT_SCRATCH=$PWD/work work/venv/bin/python -I experiments/poc_standing.py [--map relmap_big.pt]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import poc_broad as B  # noqa: E402
import poc_profile as PF  # noqa: E402
import poc_relevance as R  # noqa: E402

NARROW = """food preferences and diets; elderly relatives they look after; allergies to things other than food;
hearing, eyesight and other physical limits; chronic conditions other than the common ones; mobility and stamina;
family structure and who lives with them; exercise routine and injuries; daily medication and appointments; fasting
and meal timing; water, heat and cold tolerance; accessibility needs when reading or typing; pets and animals at home"""
WIDE = NARROW + """; money and spending; savings goals and debts; job and working hours; time they have available and
when they are busy; sleep and daily routine; religion and beliefs; alcohol, smoking and caffeine habits; fears of
animals, heights, crowds or water; strong dislikes and things that scare them; things they never learned to do;
commute and getting around the city; vehicles other than their own that they rely on"""


def cats(text):
    return {" ".join(c.split()) for c in text.split(";")}


def ridge(X, y, lam):
    """-> score function for unit vectors. X [n, d] unit vectors, y [n]."""
    mu, ym = X.mean(0), y.mean()
    Xc = X - mu
    A = Xc.T @ Xc / len(X) + lam * np.eye(X.shape[1], dtype=np.float64)
    w = np.linalg.solve(A, Xc.T @ (y - ym) / len(X))
    return lambda V: (np.asarray(V, dtype=np.float64) - mu) @ w


def auc(score, y):
    order = np.argsort(score)
    rank = np.empty(len(score))
    rank[order] = np.arange(1, len(score) + 1)
    pos = y > 0.5
    return float((rank[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def fit_scores(K, qvec, fvec, lams):
    """-> {name: score function}, printed cross-validation lines."""
    rows = []
    for name in ("broad_gen",) + B.PAIR_FILES:
        rows += json.loads((K.OUT / f"{name}.json").read_text())["rows"]
    cat, reqs = {}, defaultdict(list)
    for r in rows:
        if r["fact"] in fvec and r["request"] in qvec:
            cat[r["fact"]] = " ".join(r["cat"].split())
            reqs[r["fact"]].append(r["request"])
    facts = sorted(cat)
    X = B.unit(np.stack([fvec[f] for f in facts])).astype(np.float64)
    allc = sorted(set(cat.values()))
    for name, text in (("narrow", NARROW), ("wide", WIDE)):
        missing = cats(text) - set(allc)
        assert not missing, missing
    print(f"STAND generated facts {len(facts)} in {len(allc)} categories; standing categories: narrow {len(cats(NARROW))}, wide {len(cats(WIDE))}")
    out = {}
    # label probes, cross-validated over whole categories: does "standing" carry over to categories never seen?
    fold_of = {c: i % 5 for i, c in enumerate(allc)}
    fold = np.array([fold_of[cat[f]] for f in facts])
    for name, text in (("labels narrow", NARROW), ("labels wide", WIDE)):
        y = np.array([float(cat[f] in cats(text)) for f in facts])
        best = None
        for lam in lams:
            sc = np.zeros(len(facts))
            for k in range(5):
                sc[fold == k] = ridge(X[fold != k], y[fold != k], lam)(X[fold == k])
            a = auc(sc, y)
            if best is None or a > best[0]:
                best = (a, lam)
        print(f"STAND {name:14s} standing facts {int(y.sum())} of {len(y)}; AUC on held-out categories {best[0]:.2f} (ridge {best[1]:g})")
        out[name] = ridge(X, y, best[1])
    # label-free targets from the generated requests
    gap = np.array([1 - float(np.mean([B.unit(qvec[q]) @ B.unit(fvec[f]) for q in reqs[f]])) for f in facts])
    many = np.array([len(reqs[f]) >= 2 for f in facts])
    spread = np.zeros(len(facts))
    for i, f in enumerate(facts):
        if many[i]:
            Q = B.unit(np.stack([qvec[q] for q in reqs[f]]))
            S = Q @ Q.T
            spread[i] = 1 - float((S.sum() - len(Q)) / (len(Q) * (len(Q) - 1)))
    rng = np.random.default_rng(0)
    rfold = rng.integers(0, 5, len(facts))
    for name, y, use in (("gap", gap, np.ones(len(facts), bool)), ("spread", spread, many)):
        best = None
        for lam in lams:
            sc = np.zeros(len(facts))
            for k in range(5):
                tr, te = use & (rfold != k), use & (rfold == k)
                sc[te] = ridge(X[tr], y[tr], lam)(X[te])
            c = float(np.corrcoef(sc[use], y[use])[0, 1])
            if best is None or c > best[0]:
                best = (c, lam)
        print(f"STAND {name:14s} target mean {y[use].mean():.2f} sd {y[use].std():.2f}; correlation on held-out facts {best[0]:.2f} (ridge {best[1]:g})")
        out[name] = ridge(X[use], y[use], best[1])
    return out


def z(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def evaluate(ev, qvec, fvec, mscore, sfun):
    """-> {(group, type, size): {metric: rate}}; the clue is index 0 of every profile."""
    acc = defaultdict(lambda: defaultdict(list))
    for e in ev:
        for size in R.SIZES:
            Fm = B.unit(np.stack([fvec[t] for t in [e["clue"]] + e["others"][: size - 1]]))
            m = np.asarray(mscore(B.unit(qvec[e["question"]]), Fm), dtype=np.float64)
            s = np.asarray(sfun(Fm), dtype=np.float64)
            mo, so = list(np.argsort(-m)), list(np.argsort(-s))
            got = {}
            got["map top 3"] = 0 in mo[:3]
            got["map top 5"] = 0 in mo[:5]
            got["standing top 1"] = so[0] == 0
            got["standing top 3"] = 0 in so[:3]
            for a, b in ((1, 2), (2, 1), (2, 3), (3, 3)):  # a always-on facts by standing score + b by the map among the rest
                on = so[:a]
                got[f"{a} standing + {b} map"] = 0 in on or 0 in [i for i in mo if i not in on][:b]
            got["map top 6"] = 0 in mo[:6]
            for lam in (0.5, 1.0, 2.0):
                got[f"map + {lam:g} x standing, top 3"] = 0 in list(np.argsort(-(z(m) + lam * z(s))))[:3]
            for key in ((e["group"], "all", size), (e["group"], e["type"], size)):
                for k, v in got.items():
                    acc[key][k].append(v)
    return {key: {k: float(np.mean(v)) for k, v in d.items()} for key, d in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="relmap_big.pt")
    ap.add_argument("--lams", type=lambda x: [float(v) for v in x.split(",")], default=[1e-5, 1e-4, 1e-3])
    args = ap.parse_args()
    import torch
    K = PF.poc()
    su, qvec, fvec, _ = B.load_all()
    ev, _ = B.test_set(su, qvec, True)
    m = torch.load(K.OUT / args.map, map_location="cpu")
    mscore = lambda q, Fm: B.map_score(m, q, Fm)  # noqa: E731
    scores = fit_scores(K, qvec, fvec, args.lams)
    rng = np.random.default_rng(1)
    scores["random (control)"] = lambda V: rng.random(len(V))
    held = [e for e in ev if e["group"] == "held-out types"]
    types = sorted({e["type"] for e in held})
    result = {}
    for name, sfun in scores.items():
        res = evaluate(ev, qvec, fvec, mscore, sfun)
        result[name] = {"|".join(map(str, k)): v for k, v in res.items()}
        print(f"\nSTAND score: {name}   (map {args.map}; clue kept among the facts handed over)")
        for g in dict.fromkeys(e["group"] for e in ev):
            for size in R.SIZES:
                r = res[(g, "all", size)]
                print(f"STAND   {g:25s} profile of {size:2d}: " + "  ".join(f"{k} {v:.2f}" for k, v in r.items()))
        for size in R.SIZES:
            for k in ("map top 3", "standing top 3", "1 standing + 2 map", "2 standing + 1 map", "map top 6", "3 standing + 3 map"):
                print(f"STAND   by type, profile of {size:2d}, {k:20s}: " + "  ".join(f"{t} {res[('held-out types', t, size)][k]:.2f}" for t in types))
    # a look at what the scores call standing: the test profiles' own facts
    pool = sorted({t for e in held for t in e["others"]})
    P = B.unit(np.stack([fvec[t] for t in pool]))
    clues = sorted({(e["type"], e["clue"]) for e in held})
    for name in ("labels narrow", "gap"):
        s = scores[name](P)
        print(f"\nSTAND {name}: highest among the {len(pool)} other test facts")
        for i in np.argsort(-s)[:6]:
            print(f"STAND   {s[i]:+.2f}  {pool[i][:110]}")
        print(f"STAND {name}: lowest")
        for i in np.argsort(s)[:3]:
            print(f"STAND   {s[i]:+.2f}  {pool[i][:110]}")
        print(f"STAND {name}: the held-out clues (share of the other test facts scoring lower)")
        for t, c in clues:
            v = float(scores[name](B.unit(np.stack([fvec[c]])))[0])
            print(f"STAND   {t:8s} {v:+.2f}  {float(np.mean(s < v)):.2f}  {c[:90]}")
    (K.OUT / "standing.json").write_text(json.dumps({"map": args.map, "scores": result}, indent=1))


if __name__ == "__main__":
    main()
