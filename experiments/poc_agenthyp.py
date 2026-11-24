"""Hypothesis tests for agent-session memory with the core FROZEN: what is the missing part?

    embed   (encoder) extra vectors -> hyp_emb.npz: field-specific questions, anchored questions for test sessions,
            a one-line "session note" per session, an atomic one-fact chunk per test session, candidate name strings.
    probe   (core on CPU, no decoder) H1: are exact names linearly readable from the core's 16 tokens?
            Ridge from flattened core tokens -> encoder vector of the true name, fit on train repos, tested on
            held-out repos by nearest candidate. Variants: generic vs field-specific question vector; with a note.
    gen     (decoder) generation on test repos:
            H2 field-specific question (text + question vector) on the raw session chunks;
            H3 short name-dense documents: an atomic fact chunk / a one-line session note, alone, mixed into the raw
               chunks, and as "one note per session" memories of 8 / 32 / 150 sessions.
    rank    (decoder + LoRA) H4: can the decoder RECOGNISE the right repo from memory tokens? Log-likelihood of
            "Repository: <cand>" for every candidate repo; top-1 raw and column-centred (PMI-like).
Outputs in $MPORT_SCRATCH/runs/agentmem/: hyp_emb.npz, hyp_probe.json, hyp_gen_{variant}.json, hyp_rank.json.
"""

import argparse
import json
import random
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import poc_agentmem as A  # noqa: E402
from poc_agentmem import D, P, OUT, save  # noqa: E402

Q = {"repo": "Which repository was the coding session in your memory about? Answer with the repository name.",
     "files": "Which files were changed in the coding session in your memory? Answer with the file paths.",
     "funcs": "Which functions or classes were touched in the coding session in your memory? Answer with their names."}
QA = {"repo": "Your memory holds several coding sessions. One of them was about this issue: \"{title}\". "
              "Which repository was that session in? Answer with the repository name.",
      "files": "Your memory holds several coding sessions. One of them was about this issue: \"{title}\". "
               "Which files were changed in that session? Answer with the file paths."}
QKEYS = list(Q)
AKEYS = list(QA)


def note(s):
    return (f"Session note. Repository: {s['repo']}. Issue: {s['title']}. Files changed: {', '.join(s['files'])}. "
            f"Functions/classes: {', '.join(s['funcs']) if s['funcs'] else 'none'}.")


def atomic(s):
    return f"This coding session was in the repository {s['repo']}."


def names_of(s):
    return {"repo": s["repo"], "file": s["files"][0], "func": s["funcs"][0] if s["funcs"] else None}


def cmd_embed(args):
    ss, _, _ = A.load_emb()
    test = [s for s in ss if s["split"] == "test"]
    names = sorted({v for s in ss for v in names_of(s).values() if v})
    texts = ([Q[k] for k in QKEYS] + [QA[k].format(title=s["title"]) for s in test for k in AKEYS]
             + [note(s) for s in ss] + [atomic(s) for s in test] + names)
    enc = D.Encoder()
    t0, vecs = time.perf_counter(), []
    for i in range(0, len(texts), 256):
        vecs.append(enc(texts[i:i + 256], batch_size=16).astype(np.float16))
        print(f"{i + len(vecs[-1])}/{len(texts)}  {time.perf_counter() - t0:.0f}s", flush=True)
        torch.mps.empty_cache()
    np.savez(OUT / "hyp_emb.npz", vecs=np.concatenate(vecs), n_all=len(ss), test_ids=np.array([s["id"] for s in test]),
             names=np.array(names))
    print("done", flush=True)


class Extra:
    """Row lookups into hyp_emb.npz."""

    def __init__(self, ss):
        z = np.load(OUT / "hyp_emb.npz")
        self.v = z["vecs"]
        test_ids = [str(x) for x in z["test_ids"]]
        nt, na = len(test_ids), int(z["n_all"])
        o = 0
        self.q = {k: o + i for i, k in enumerate(QKEYS)}
        o += len(QKEYS)
        self.qa = {(sid, k): o + i * len(AKEYS) + j for i, sid in enumerate(test_ids) for j, k in enumerate(AKEYS)}
        o += nt * len(AKEYS)
        self.note = {s["id"]: o + i for i, s in enumerate(ss)}
        o += na
        self.atomic = {sid: o + i for i, sid in enumerate(test_ids)}
        o += nt
        self.name = {str(n): o + i for i, n in enumerate(z["names"])}
        assert o + len(self.name) == len(self.v)


# ---------------------------------------------------------------- probe (CPU)


@torch.no_grad()
def cmd_probe(args):
    ss, chunks, qv = A.load_emb()
    ex = Extra(ss)
    core = P.load_memory("native", A.DECODER).eval().float().cpu()
    f32 = lambda a: a.astype(np.float32)  # noqa: E731
    feats = {"encoder mean (ceiling for a linear probe)": [], "core, generic question": [],
             "core, 'which repository' question": [], "core, 'which files' question": [],
             "core, raw + session note, generic question": []}
    t0 = time.perf_counter()
    for i, s in enumerate(ss):
        raw = f32(chunks[s["rows"]])
        feats["encoder mean (ceiling for a linear probe)"].append(raw.mean(0))
        x = torch.from_numpy(raw)[None]
        xn = torch.from_numpy(np.concatenate([raw, f32(ex.v[[ex.note[s["id"]]]])]))[None]
        run = lambda d, q: core(d, None, torch.from_numpy(f32(q))[None])[0].flatten().numpy()  # noqa: E731
        feats["core, generic question"].append(run(x, qv[0]))
        feats["core, 'which repository' question"].append(run(x, ex.v[ex.q["repo"]]))
        feats["core, 'which files' question"].append(run(x, ex.v[ex.q["files"]]))
        feats["core, raw + session note, generic question"].append(run(xn, qv[0]))
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(ss)}  {time.perf_counter() - t0:.0f}s", flush=True)
    split = np.array([s["split"] for s in ss])
    tr, va, te = split == "train", split == "val", split == "test"
    out = {}
    for fname, X in feats.items():
        X = np.stack(X).astype(np.float64)
        X = X - X[tr].mean(0)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        K = X @ X.T
        out[fname] = {}
        for kind in ("repo", "file", "func"):
            have = np.array([names_of(s)[kind] is not None for s in ss])
            Y = np.stack([f32(ex.v[ex.name[names_of(s)[kind]]]) if h else np.zeros(ex.v.shape[1], np.float32)
                          for s, h in zip(ss, have)]).astype(np.float64)
            mu = Y[tr & have].mean(0)

            def top1(fit, ev, lam):
                alpha = np.linalg.solve(K[np.ix_(fit, fit)] + lam * np.eye(fit.sum()), Y[fit] - mu)
                pred = K[np.ix_(ev, fit)] @ alpha
                cands = sorted({names_of(s)[kind] for s, e in zip(ss, ev) if e})
                C = np.stack([f32(ex.v[ex.name[c]]) for c in cands]).astype(np.float64) - mu
                C /= np.linalg.norm(C, axis=1, keepdims=True)
                hit = [cands[j] == names_of(s)[kind] for j, s in zip((pred @ C.T).argmax(1), [s for s, e in zip(ss, ev) if e])]
                return float(np.mean(hit)), len(cands), int(ev.sum())

            lams = [1e-3, 1e-2, 1e-1, 1.0]
            lam = max(lams, key=lambda l: top1(tr & have, va & have, l)[0])
            acc, nc, n = top1((tr | va) & have, te & have, lam)
            out[fname][kind] = {"top1": acc, "candidates": nc, "n": n, "lambda": lam}
        print(fname, {k: round(v["top1"], 2) for k, v in out[fname].items()}, flush=True)
        save("hyp_probe.json", out)


# ---------------------------------------------------------------- gen


def build_conditions():
    """name -> (question kind, anchored?, docs builder(s, others) -> list of groups of vectors-spec)."""
    c = []
    for k in QKEYS:
        c.append((f"raw chunks, ask {k}", k, False, "raw", 0, 1))
    c.append(("atomic fact only, ask repo", "repo", False, "atomic", 0, 1))
    c.append(("atomic fact + raw chunks, ask repo", "repo", False, "atomic+raw", 0, 1))
    for k in ("repo", "files"):
        c.append((f"note only, ask {k}", k, False, "note", 0, 1))
        c.append((f"note + raw chunks, ask {k}", k, False, "note+raw", 0, 1))
    for n, g in ((8, 1), (8, 8), (32, 1), (32, 8), (150, 1), (150, 8)):
        for k in AKEYS:
            c.append((f"{n} session notes, {16 * g} tok, ask {k}", k, True, "notes", n - 1, g))
    return c


def docs_for(kind, s, others, chunks, ex, nrng):
    v = lambda rows: ex.v[rows].astype(np.float32)  # noqa: E731
    raw = chunks[s["rows"]].astype(np.float32)
    if kind == "raw":
        return raw
    if kind == "atomic":
        return v([ex.atomic[s["id"]]])
    if kind == "atomic+raw":
        d = np.concatenate([raw, v([ex.atomic[s["id"]]])])
    elif kind == "note":
        return v([ex.note[s["id"]]])
    elif kind == "note+raw":
        d = np.concatenate([raw, v([ex.note[s["id"]]])])
    else:  # notes of several sessions
        d = v([ex.note[o["id"]] for o in others + [s]])
    return d[nrng.permutation(len(d))]


@torch.no_grad()
def cmd_gen(args):
    ss, chunks, _ = A.load_emb()
    ex = Extra(ss)
    test = [s for s in ss if s["split"] == "test"]
    random.Random(3).shuffle(test)
    pick = test[: args.n]
    st = A.Stack(args.variant)
    conds = [c for c in build_conditions() if not args.only or any(o in c[0] for o in args.only)]
    path = f"hyp_gen_{args.variant}.json"
    prev = json.loads((OUT / path).read_text()) if (OUT / path).exists() and args.resume else {"rows": []}
    done = {(r["session"], r["condition"]) for r in prev["rows"]}
    rows, t0 = prev["rows"], time.perf_counter()
    for i, s in enumerate(pick):
        for name, k, anchored, kind, n_other, g in conds:
            if (s["id"], name) in done:
                continue
            rng = random.Random(zlib.crc32(f"{s['id']}|{n_other}".encode()))
            nrng = np.random.default_rng(zlib.crc32(s["id"].encode()))
            pool = [o for o in (test if n_other < 100 else ss) if o["repo"] != s["repo"]]
            others = rng.sample(pool, n_other) if n_other else []
            d = docs_for(kind, s, others, chunks, ex, nrng)
            question = (QA[k].format(title=s["title"]) if anchored else Q[k])
            qvec = ex.v[ex.qa[(s["id"], k)]] if anchored else ex.v[ex.q[k]]
            groups = [x for x in np.array_split(np.arange(len(d)), min(g, len(d))) if len(x)]
            mem = st.memory(d, groups, qvec)
            ans = A.generate(st, question, mem, args.max_new_tokens)
            sc = A.score(s, ans, True)
            rows.append({"session": s["id"], "condition": name, "n_tokens": int(mem.shape[1]), "answer": ans, **sc})
        torch.mps.empty_cache()
        if (i + 1) % 3 == 0 or i + 1 == len(pick):
            save(path, {"variant": args.variant, "rows": rows, "table": table(rows)})
            print(f"{i + 1}/{len(pick)} sessions  {(time.perf_counter() - t0) / 60:.0f} min", flush=True)
    for k, v in table(rows).items():
        print(f"{k:44s} " + "  ".join(f"{m} {x:.2f}" for m, x in v.items()))


def table(rows):
    out = {}
    for r in rows:
        out.setdefault(r["condition"], []).append(r)
    return {c: {"n": len(rs), **{m: float(np.mean([r[m] for r in rs if r.get(m) is not None] or [0]))
                                 for m in ("repo", "files_path", "files_name", "funcs")}} for c, rs in out.items()}


# ---------------------------------------------------------------- rank


@torch.no_grad()
def cmd_rank(args):
    ss, chunks, qv = A.load_emb()
    test = [s for s in ss if s["split"] == "test"]
    cands = sorted({s["repo"] for s in test})
    random.Random(3).shuffle(test)
    pick = test[: args.n]
    st = A.Stack("lora")
    tok, model = st.r.tok, st.r.model
    emb = model.get_input_embeddings()
    c_ids = [tok.encode(f"Repository: {c}\n", add_special_tokens=False) for c in cands]
    L = max(map(len, c_ids))
    pad = tok.pad_token_id
    ids_c = torch.tensor([x + [pad] * (L - len(x)) for x in c_ids], device=P.DEV)
    m_c = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in c_ids], device=P.DEV)
    S, t0 = [], time.perf_counter()
    for i, s in enumerate(pick):
        mem = st.memory(chunks, [s["rows"]], qv[0])
        p_ids, _ = st.ids(A.Q_SINGLE, mem.shape[1])
        pe = P.input_embeds(model, tok, st.r.memory_id, p_ids, mem)  # [1, Lp, H]
        row = []
        for b in range(0, len(cands), args.batch):
            ic, mc = ids_c[b:b + args.batch], m_c[b:b + args.batch]
            x = torch.cat([pe.expand(len(ic), -1, -1), emb(ic)], 1)
            am = torch.cat([torch.ones(len(ic), pe.shape[1], device=P.DEV, dtype=mc.dtype), mc], 1)
            logits = model(inputs_embeds=x, attention_mask=am, use_cache=False).logits[:, pe.shape[1] - 1:-1].float()
            lp = torch.log_softmax(logits, -1).gather(2, ic[..., None])[..., 0] * mc
            row += lp.sum(1).tolist()
        S.append(row)
        torch.mps.empty_cache()
        if (i + 1) % 5 == 0 or i + 1 == len(pick):
            M = np.array(S)
            true = np.array([cands.index(x["repo"]) for x in pick[: len(M)]])
            raw = float((M.argmax(1) == true).mean())
            Mc = M - M.mean(0)
            cen = float((Mc.argmax(1) == true).mean())
            rank_c = [int((Mc[j] > Mc[j, t]).sum()) for j, t in enumerate(true)]
            res = {"n": len(M), "candidates": len(cands), "top1_raw": raw, "top1_centred": cen,
                   "top5_centred": float(np.mean(np.array(rank_c) < 5)), "median_rank_centred": float(np.median(rank_c)),
                   "chance": 1 / len(cands)}
            save("hyp_rank.json", {"result": res, "scores": M.tolist(), "true": true.tolist(), "cands": cands})
            print(f"{i + 1}/{len(pick)}  {(time.perf_counter() - t0) / 60:.0f} min  {res}", flush=True)


# ---------------------------------------------------------------- contrastive decoding


@torch.no_grad()
def cmd_cad(args):
    """H6: generation fails because the name prior wins. Decode with lp(mem) + a * (lp(mem) - mean lp(other memories)),
    restricted to tokens the memory-conditioned model finds plausible (p >= 0.1 * max). a = 0 is plain greedy."""
    ss, chunks, qv = A.load_emb()
    ex = Extra(ss)
    test = [s for s in ss if s["split"] == "test"]
    random.Random(3).shuffle(test)
    pick = test[: args.n]
    st = A.Stack("lora")
    tok, model = st.r.tok, st.r.model
    emb = model.get_input_embeddings()
    eos = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    rows, t0 = [], time.perf_counter()
    for i, s in enumerate(pick):
        rng = random.Random(zlib.crc32(s["id"].encode()))
        refs = rng.sample([o for o in test if o["repo"] != s["repo"]], args.refs)
        for kind in ("raw", "note"):
            mems = []
            for o in [s] + refs:
                d = chunks[o["rows"]].astype(np.float32) if kind == "raw" else ex.v[[ex.note[o["id"]]]].astype(np.float32)
                mems.append(st.memory(d, [np.arange(len(d))], qv[0]))
            p_ids, _ = st.ids(A.Q_SINGLE, mems[0].shape[1])
            pe = torch.cat([P.input_embeds(model, tok, st.r.memory_id, p_ids, m) for m in mems])
            for a in args.alphas:
                out = model(inputs_embeds=pe, use_cache=True)
                ids = []
                for _ in range(args.max_new_tokens):
                    lp = torch.log_softmax(out.logits[:, -1].float(), -1)
                    sc = lp[0] + a * (lp[0] - lp[1:].mean(0))
                    sc[lp[0] < lp[0].max() + np.log(0.1)] = -1e9
                    t = int(sc.argmax())
                    if t in eos:
                        break
                    ids.append(t)
                    nxt = emb(torch.tensor([[t]] * len(mems), device=P.DEV))
                    out = model(inputs_embeds=nxt, past_key_values=out.past_key_values, use_cache=True)
                ans = tok.decode(ids, skip_special_tokens=True).strip()
                rows.append({"session": s["id"], "condition": f"{kind}, alpha {a}", "answer": ans, **A.score(s, ans, False)})
                del out
            torch.mps.empty_cache()
        if (i + 1) % 5 == 0 or i + 1 == len(pick):
            save("hyp_cad.json", {"rows": rows, "table": table(rows)})
            print(f"{i + 1}/{len(pick)}  {(time.perf_counter() - t0) / 60:.0f} min  " +
                  "  ".join(f"[{c}] repo {v['repo']:.2f} file {v['files_name']:.2f}" for c, v in table(rows).items()), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("embed")
    sub.add_parser("probe")
    g = sub.add_parser("gen")
    g.add_argument("--variant", default="lora")
    g.add_argument("--n", type=int, default=60)
    g.add_argument("--max_new_tokens", type=int, default=40)
    g.add_argument("--only", nargs="*")
    g.add_argument("--resume", action="store_true")
    r = sub.add_parser("rank")
    r.add_argument("--n", type=int, default=60)
    r.add_argument("--batch", type=int, default=41)
    c = sub.add_parser("cad")
    c.add_argument("--n", type=int, default=60)
    c.add_argument("--refs", type=int, default=3)
    c.add_argument("--alphas", type=float, nargs="*", default=[0.0, 1.0, 3.0])
    c.add_argument("--max_new_tokens", type=int, default=40)
    args = ap.parse_args()
    {"embed": cmd_embed, "probe": cmd_probe, "gen": cmd_gen, "rank": cmd_rank, "cad": cmd_cad}[args.cmd](args)


if __name__ == "__main__":
    main()
