"""H8: one fact per chunk. The spelling reader (lora_mix.pt) spells a name when the memory is ONE one-name sentence,
but not from a session note that packs ~6 names into one chunk vector. Does it work when each fact is its own chunk?

    embed   atoms_emb.npz: one sentence per fact (repo, each file, each function) for every test session
    eval    reader = lora_mix.pt, zero-shot. Per test session and per kind (repo / file / func):
              one atom            only the asked fact's chunk                         -> 16 tokens
              all atoms, pooled   all of the session's fact chunks in one core call   -> 16 tokens
              all atoms, 16 each  one core call per fact chunk                        -> 16 x facts tokens
              atoms + raw, pooled fact chunks mixed into the raw agent-step chunks    -> 16 tokens
            Score: the answer contains the repo / any of the session's files / any of its functions (exact string).
"""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import poc_agentmem as A  # noqa: E402
import poc_spell as S  # noqa: E402
from poc_agentmem import D, P, OUT, save  # noqa: E402


def facts(s):
    return [("repo", s["repo"])] + [("file", f) for f in s["files"]] + [("func", f) for f in s["funcs"]]


def test_sessions():
    ss, chunks, _ = A.load_emb()
    return [s for s in ss if s["split"] == "test"], chunks


def cmd_embed(args):
    test, _ = test_sessions()
    texts = [S.SENT[k].format(x=x) for s in test for k, x in facts(s)]
    enc = D.Encoder()
    v = np.concatenate([enc(texts[b:b + 512], batch_size=32).astype(np.float16) for b in range(0, len(texts), 512)])
    np.savez(OUT / "atoms_emb.npz", vecs=v)
    print("done", len(texts), flush=True)


@torch.no_grad()
def cmd_eval(args):
    test, chunks = test_sessions()
    v = np.load(OUT / "atoms_emb.npz")["vecs"]
    o = 0
    for s in test:
        s["facts"] = facts(s)
        s["arows"] = np.arange(o, o + len(s["facts"]))
        o += len(s["facts"])
    q = np.load(OUT / "spell_emb.npz")["q"]
    st = A.Stack("lora", train=True)
    sd = torch.load(OUT / args.lora, map_location="cpu")
    for i, m in enumerate(st.lora):
        m.A.data.copy_(sd[f"{i}.A"])
        m.B.data.copy_(sd[f"{i}.B"])
        m.drop.p = 0.0
    st.r.model.eval()
    random.Random(3).shuffle(test)
    rows, t0 = [], time.perf_counter()
    f32 = lambda a: a.astype(np.float32)  # noqa: E731
    for i, s in enumerate(test[: args.n]):
        nrng = np.random.default_rng(i)
        atoms, raw = f32(v[s["arows"]]), f32(chunks[s["rows"]])
        for kind in S.KINDS:
            truth = [x for k, x in s["facts"] if k == kind]
            if not truth:
                continue
            first = next(j for j, (k, _) in enumerate(s["facts"]) if k == kind)
            perm = nrng.permutation(len(atoms))
            both = np.concatenate([raw, atoms])
            both = both[nrng.permutation(len(both))]
            conds = {"one atom": (atoms[[first]], [np.array([0])]),
                     "all atoms, pooled 16 tok": (atoms[perm], [np.arange(len(atoms))]),
                     "all atoms, 16 tok each": (atoms[perm], [np.array([j]) for j in range(len(atoms))]),
                     "atoms + raw steps, pooled 16 tok": (both, [np.arange(len(both))])}
            for name, (d, groups) in conds.items():
                mem = st.memory(d, groups, q[S.KINDS.index(kind)])
                ans = A.generate(st, S.QS[kind], mem, 28).strip()
                rows.append({"session": s["id"], "kind": kind, "condition": name, "n_tokens": int(mem.shape[1]),
                             "truth": truth, "answer": ans, "hit": float(any(t.lower() in ans.lower() for t in truth)),
                             "exact": float(ans in truth)})
        torch.mps.empty_cache()
        if (i + 1) % 5 == 0 or i + 1 == min(args.n, len(test)):
            tab = {}
            for r in rows:
                tab.setdefault(r["condition"], {}).setdefault(r["kind"], []).append(r["hit"])
            tab = {c: {k: float(np.mean(x)) for k, x in ks.items()} for c, ks in tab.items()}
            save(args.out, {"lora": args.lora, "rows": rows, "table": tab})
            print(f"{i + 1}  {(time.perf_counter() - t0) / 60:.0f} min", flush=True)
    for c, ks in tab.items():
        print(f"ATOMS {c:34s} " + "  ".join(f"{k} {x:.2f}" for k, x in ks.items()), flush=True)


# ---------------------------------------------------------------- H9: train the reader on one-fact-per-chunk memories

import math  # noqa: E402
import os  # noqa: E402


def cmd_embedall(args):
    ss, _, _ = A.load_emb()
    texts = [S.SENT[k].format(x=x) for s in ss for k, x in facts(s)]
    enc = D.Encoder()
    t0, vs = time.perf_counter(), []
    for b in range(0, len(texts), 512):
        vs.append(enc(texts[b:b + 512], batch_size=32).astype(np.float16))
        print(f"{b + len(vs[-1])}/{len(texts)}  {time.perf_counter() - t0:.0f}s", flush=True)
        torch.mps.empty_cache()
    np.savez(OUT / "atoms_all.npz", vecs=np.concatenate(vs))
    print("done", flush=True)


def load_all():
    ss, chunks, _ = A.load_emb()
    v = np.load(OUT / "atoms_all.npz")["vecs"]
    o = 0
    for s in ss:
        s["facts"] = facts(s)
        s["arows"] = np.arange(o, o + len(s["facts"]))
        o += len(s["facts"])
    assert o == len(v)
    return ss, chunks, v


def layout(name, s, chunks, v, nrng, sel=None):
    """-> (doc vectors, groups). Layouts: each (16 tok per fact), pooled (16 tok), each+raw (16 per fact + 16 for raw)."""
    f32 = lambda a: a.astype(np.float32)  # noqa: E731
    rows = s["arows"] if sel is None else s["arows"][sel]
    atoms = f32(v[rows])[nrng.permutation(len(rows))]
    if name == "pooled":
        return atoms, [np.arange(len(atoms))]
    groups = [np.array([j]) for j in range(len(atoms))]
    if name == "each":
        return atoms, groups
    raw = f32(chunks[s["rows"]])
    d = np.concatenate([atoms, raw])
    groups.insert(int(nrng.integers(0, len(groups) + 1)), np.arange(len(atoms), len(d)))
    return d, groups


def cmd_train(args):
    torch.manual_seed(0)
    ss, chunks, v = load_all()
    train = [s for s in ss if s["split"] == "train"]
    items, svecs, q = S.load()
    for j, it in enumerate(items):
        it["row"] = j
    names = [i for i in items if i["split"] == "train"]
    st = A.Stack("lora", train=True)
    if args.init:
        sd = torch.load(OUT / args.init, map_location="cpu")
        for i, m in enumerate(st.lora):
            m.A.data.copy_(sd[f"{i}.A"])
            m.B.data.copy_(sd[f"{i}.B"])
    ps = [t for m in st.lora for t in (m.A, m.B)]
    opt = torch.optim.AdamW(ps, lr=args.lr, weight_decay=0.0)
    rng, nrng = random.Random(0), np.random.default_rng(0)

    def loss_one():
        mode = rng.choices(["spell", "each", "pooled", "each+raw"], [0.25, 0.40, 0.15, 0.20])[0]
        if mode == "spell":
            it = rng.choice(names)
            mem = st.memory(svecs, [np.array([it["row"]])], q[S.KINDS.index(it["kind"])])
            kind, answer = it["kind"], it["name"]
        else:
            s = rng.choice(train)
            # cap at 8 facts per example: longer memories ran out of GPU memory in backward
            sel = np.array(sorted(rng.sample(range(len(s["facts"])), min(8, len(s["facts"])))))
            fs = [s["facts"][j] for j in sel]
            kind = rng.choice([k for k in S.KINDS if any(fk == k for fk, _ in fs)])
            d, groups = layout(mode, s, chunks, v, nrng, sel)
            mem = st.memory(d, groups, q[S.KINDS.index(kind)])
            answer = ", ".join(x for k, x in fs if k == kind)
        ids, labels = st.ids(S.QS[kind], mem.shape[1], answer)
        emb = P.input_embeds(st.r.model, st.r.tok, st.r.memory_id, ids, mem)
        if os.environ.get("ATOMS_DEBUG"):
            print(f"dbg {mode} mem {tuple(mem.shape)} seq {ids.shape[1]} mps {torch.mps.current_allocated_memory() / 2**30:.1f}/{torch.mps.driver_allocated_memory() / 2**30:.1f} GB", flush=True)
        return st.r.model(inputs_embeds=emb, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False).loss

    st.r.model.train()
    t0, u, run = time.perf_counter(), 0, []
    budget, next_save = args.minutes * 60, 900
    while time.perf_counter() - t0 < budget:
        frac = min(1.0, (time.perf_counter() - t0) / budget)
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (u + 1) / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for _ in range(4):
            loss = loss_one() / 4
            loss.backward()
            run.append(loss.item() * 4)
        torch.nn.utils.clip_grad_norm_(ps, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        u += 1
        torch.mps.empty_cache()  # every update: variable-length memories fragment the MPS cache until it hits the cap
        if u % 25 == 0:
            print(f"update {u}  examples {u * 4}  loss {np.mean(run[-100:]):.3f}  {(time.perf_counter() - t0) / 60:.0f} min", flush=True)
        if time.perf_counter() - t0 > next_save:
            next_save += 900
            torch.save(st.state(), OUT / "lora_atoms.pt")
    torch.save(st.state(), OUT / "lora_atoms.pt")
    save("atoms_train.json", {"updates": u, "examples": u * 4, "loss": float(np.mean(run[-100:])), "minutes": args.minutes})
    print("saved", flush=True)


@torch.no_grad()
def cmd_eval2(args):
    ss, chunks, v = load_all()
    test = [s for s in ss if s["split"] == "test"]
    q = np.load(OUT / "spell_emb.npz")["q"]
    st = A.Stack("lora", train=True)
    sd = torch.load(OUT / args.lora, map_location="cpu")
    for i, m in enumerate(st.lora):
        m.A.data.copy_(sd[f"{i}.A"])
        m.B.data.copy_(sd[f"{i}.B"])
        m.drop.p = 0.0
    st.r.model.eval()
    random.Random(3).shuffle(test)
    rows, t0 = [], time.perf_counter()
    for i, s in enumerate(test[: args.n]):
        for kind in S.KINDS:
            truth = [x for k, x in s["facts"] if k == kind]
            if not truth:
                continue
            for name in ("each", "pooled", "each+raw"):
                d, groups = layout(name, s, chunks, v, np.random.default_rng(i))
                mem = st.memory(d, groups, q[S.KINDS.index(kind)])
                ans = A.generate(st, S.QS[kind], mem, 64).strip()
                got = [t.lower() in ans.lower() for t in truth]
                rows.append({"session": s["id"], "kind": kind, "condition": name, "n_tokens": int(mem.shape[1]),
                             "truth": truth, "answer": ans, "hit": float(any(got)), "recall": float(np.mean(got))})
        torch.mps.empty_cache()
        if (i + 1) % 5 == 0 or i + 1 == min(args.n, len(test)):
            tab = {}
            for r in rows:
                tab.setdefault(r["condition"], {}).setdefault(r["kind"], []).append((r["hit"], r["recall"], r["n_tokens"]))
            tab = {c: {k: {"any": float(np.mean([a for a, _, _ in x])), "recall": float(np.mean([b for _, b, _ in x])),
                           "tokens": float(np.mean([t for _, _, t in x]))} for k, x in ks.items()} for c, ks in tab.items()}
            save(args.out, {"lora": args.lora, "rows": rows, "table": tab})
            print(f"{i + 1}  {(time.perf_counter() - t0) / 60:.0f} min", flush=True)
    for c, ks in tab.items():
        print(f"ATOMS2 {c:10s} " + "  ".join(f"{k} any {x['any']:.2f} recall {x['recall']:.2f}" for k, x in ks.items())
              + f"  tokens {np.mean([x['tokens'] for x in ks.values()]):.0f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("embed")
    e = sub.add_parser("eval")
    e.add_argument("--n", type=int, default=60)
    e.add_argument("--lora", default="lora_mix.pt")
    e.add_argument("--out", default="atoms_eval.json")
    sub.add_parser("embedall")
    t = sub.add_parser("train")
    t.add_argument("--minutes", type=float, default=120)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--init", default="lora_mix.pt")
    e2 = sub.add_parser("eval2")
    e2.add_argument("--n", type=int, default=60)
    e2.add_argument("--lora", default="lora_atoms.pt")
    e2.add_argument("--out", default="atoms_eval2.json")
    args = ap.parse_args()
    {"embed": cmd_embed, "eval": cmd_eval, "embedall": cmd_embedall, "train": cmd_train, "eval2": cmd_eval2}[args.cmd](args)


if __name__ == "__main__":
    main()
