"""H36: distill the decoder's own per-fact rating into a fast scorer on the embeddings we already have.

Test 35b showed the plain decoder, asked about one fact at a time ("should a good reply to this request take this fact
into account?", yes/no log-probability), puts an unseen health clue in its top 4 of 8 in 94% (embedding map: 22-24%).
Rating every fact with the decoder costs one forward pass per fact; a memory layer needs something cheaper.
Here: the decoder rates random (training request, training fact) pairs (any fact with any request: a rating per pair is
well-posed, unlike a teacher answer), then a small scorer on the request and fact embeddings learns to rank facts the way
the rater does. Held-out types never appear in training.

  rate: MPORT_VLLM_URL=... --mode rate --requests 3000 --per 24   -> work/runs/poc/rater_train.json
  fit:  --mode fit                                                -> rater_fit.json: where the clue ranks among profile facts
        (8 / 16 / 48) on unseen types, for the scorer, raw embedding similarity, and the rater itself (from zero_rate.json)
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
import poc_whole as W  # noqa: E402

T0 = time.perf_counter()


def training_space(K):
    su, qvec, fvec, rows = B.load_all()
    pairs = B.known_pairs(su, rows, qvec, fvec)
    full = [(r["request"], r["fact"]) for r in json.loads((K.OUT / "broad_long.json").read_text())["rows"] if r["request"] in qvec and r["fact"] in fvec]
    facts = su[2]
    pool = sorted({f for _, f in pairs} | {f for _, f in full} | {f["text"] for f in facts if f["train"] and f["text"] in fvec})
    rel = {}
    for q, f in pairs + full:
        rel.setdefault(q, set()).add(f)
    return su, qvec, fvec, pool, rel


def rate(args, K):
    from transformers import AutoTokenizer
    su, qvec, fvec, pool, rel = training_space(K)
    tx = W.Text(AutoTokenizer.from_pretrained(K.DECODER))
    rng = random.Random(args.seed)
    reqs = rng.sample(sorted(rel), min(args.requests, len(rel)))
    jobs = []
    for q in reqs:
        fs = rng.sample(pool, args.per) + rng.sample(sorted(rel[q]), min(2, len(rel[q])))  # mostly random facts, plus a known relevant one or two
        jobs += [(q, f) for f in dict.fromkeys(fs)]
    print(f"RATER rate: {len(reqs)} requests, {len(jobs)} pairs, pool {len(pool)} facts", flush=True)
    sc = W.yes_score(tx, [tx.msgs([f], W.RATE.format(q=q)) for q, f in jobs])
    out = [{"q": q, "f": f, "s": round(s, 3), "known": f in rel[q]} for (q, f), s in zip(jobs, sc)]
    (K.OUT / "rater_train.json").write_text(json.dumps(out))
    s = np.array([o["s"] for o in out])
    kn = np.array([o["known"] for o in out])
    print(f"RATER rated {len(out)} pairs in {time.perf_counter() - T0:.0f}s; yes-no mean {s.mean():.2f}, known relevant {s[kn].mean():.2f},"
          f" random {s[~kn].mean():.2f}; share above 0: known {np.mean(s[kn] > 0):.2f}, random {np.mean(s[~kn] > 0):.2f}", flush=True)


def fit(args, K):
    import torch
    su, qvec, fvec, pool, rel = training_space(K)
    data = json.loads((K.OUT / "rater_train.json").read_text())
    reqs = sorted({o["q"] for o in data})
    rng = random.Random(1)
    val = set(rng.sample(reqs, len(reqs) // 10))
    groups = {}
    for o in data:
        groups.setdefault(o["q"], []).append(o)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    E = lambda m, keys: torch.tensor(B.unit(np.stack([m[k] for k in keys])), device=dev)  # noqa: E731

    def pack(qs):
        Q, Fs, S = [], [], []
        n = min(len(groups[q]) for q in qs)
        for q in qs:
            g = groups[q][:n]
            Q.append(qvec[q])
            Fs.append(np.stack([fvec[o["f"]] for o in g]))
            S.append([o["s"] for o in g])
        return (torch.tensor(B.unit(np.stack(Q)), device=dev), torch.tensor(B.unit(np.stack(Fs)), device=dev),
                torch.tensor(S, device=dev, dtype=torch.float32))

    tr = [q for q in reqs if q not in val]
    Qt, Ft, St = pack(tr)
    Qv, Fv, Sv = pack(sorted(val))
    d = Qt.shape[1]
    torch.manual_seed(0)
    # residual on top of scaled cosine: starts exactly at cosine ranking, learns a low-rank correction and a per-fact term
    ls = torch.nn.Parameter(torch.tensor(3.0, device=dev))
    A = torch.nn.Parameter(torch.randn(d, args.rank, device=dev) / d ** 0.5)
    Bm = torch.nn.Parameter(torch.zeros(d, args.rank, device=dev))
    c = torch.nn.Parameter(torch.zeros(d, device=dev))
    params = [ls, A, Bm, c] if args.rank else [ls, c]
    if not args.rank:
        A, Bm = torch.zeros(d, 1, device=dev), torch.zeros(d, 1, device=dev)

    def score(Q, F):  # Q (n, d), F (n, m, d) -> (n, m)
        return ls.exp() * torch.einsum("nd,nmd->nm", Q, F) + torch.einsum("nr,nmr->nm", Q @ A, F @ Bm) + F @ c

    def lossf(Q, F, S):  # listwise: the scorer's softmax over a request's facts matches the rater's (temperature args.temp)
        p = torch.softmax(S / args.temp, -1)
        return -(p * torch.log_softmax(score(Q, F), -1)).sum(-1).mean()

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    best = (float("inf"), None, -1)
    for step in range(args.steps):
        idx = torch.randint(0, len(Qt), (256,), device=dev)
        loss = lossf(Qt[idx], Ft[idx], St[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == args.steps - 1:
            with torch.no_grad():
                vl = lossf(Qv, Fv, Sv).item()
                top = (score(Qv, Fv).argmax(-1) == Sv.argmax(-1)).float().mean().item()
            if vl < best[0]:
                best = (vl, [p.detach().clone() for p in params], step)
            if step % 250 == 0:
                print(f"RATER fit step {step}: train {loss.item():.3f}  val {vl:.3f}  val: scorer's top fact = rater's top fact {top:.2f}", flush=True)
    with torch.no_grad():
        for p, b in zip(params, best[1]):
            p.copy_(b)
    print(f"RATER fit: best val {best[0]:.3f} at step {best[2]}, scale {ls.exp().item():.1f}", flush=True)

    ev, pmap = B.test_set(su, qvec, True)
    rated = {}
    zr = K.OUT / "zero_rate.json"
    if zr.exists():
        for r in json.loads(zr.read_text())["rows"]:
            if r["how"] == "rate" and "from" in r["condition"]:
                rated[(r["pair"], len(r["facts"]))] = int(r["clue_rank"])
    res = {}
    for size in (8, 16, 48):
        rows = []
        for e in ev:
            if e["group"] != "held-out types":
                continue
            lst = W.profile(e, size, True, "rank")
            q = pmap[e["pair"]]["question"]
            if q not in qvec or any(f not in fvec for f in lst):
                continue
            ci = lst.index(e["clue"])
            Q, F = E(qvec, [q]), E(fvec, lst)[None]
            with torch.no_grad():
                s_fit = score(Q, F)[0].cpu().numpy()
            s_cos = (F[0] @ Q[0]).cpu().numpy()
            rk = lambda s: int((s > s[ci]).sum())  # noqa: E731
            rows.append({"type": e["type"], "fit": rk(s_fit), "cos": rk(s_cos), "rater": rated.get((e["pair"], size), -1)})
        out = {}
        for key in ("fit", "cos", "rater"):
            x = [r for r in rows if r[key] >= 0]
            if not x:
                continue
            out[key] = {"n": len(x), "top1": float(np.mean([r[key] == 0 for r in x])), "top4": float(np.mean([r[key] < 4 for r in x])),
                        "top8": float(np.mean([r[key] < 8 for r in x])),
                        "top4_by_type": {t: float(np.mean([r[key] < 4 for r in x if r["type"] == t])) for t in sorted({r["type"] for r in x})}}
            o = out[key]
            print(f"RATER rank {size:2d} facts  {key:6s} n={o['n']:3d}  clue first {o['top1']:.2f}  top 4 {o['top4']:.2f}  top 8 {o['top8']:.2f}  top 4 by type: "
                  + "  ".join(f"{t} {v:.2f}" for t, v in o["top4_by_type"].items()), flush=True)
        res[size] = out
    (K.OUT / f"rater_fit{args.tag}.json").write_text(json.dumps({"args": vars(args), "results": res}, indent=1))
    torch.save({"ls": ls.detach().cpu(), "A": A.detach().cpu(), "B": Bm.detach().cpu(), "c": c.detach().cpu()}, K.OUT / f"rater_scorer{args.tag}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["rate", "fit"], required=True)
    ap.add_argument("--requests", type=int, default=3000)
    ap.add_argument("--per", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--temp", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--tag", default="")
    ap.add_argument("--steps", type=int, default=3000)
    args = ap.parse_args()
    K = PF.poc()
    return rate(args, K) if args.mode == "rate" else fit(args, K)


if __name__ == "__main__":
    main()
