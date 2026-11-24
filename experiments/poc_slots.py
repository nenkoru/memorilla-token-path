"""H10: how few memory tokens per fact? The core turns each chunk into 16 tokens. With one fact per chunk and a reader
trained for it (H9), function names come back 81% of the time at 16 tokens per fact. A name as plain text costs ~5
tokens, so 16 per fact is no saving. Here the reader sees only the first k of the 16 slots per fact, or facts pooled in
pairs.

    train   init lora_atoms.pt; layouts k2 / k4 / k8 / k16 (first k slots of each fact), pair (2 facts -> 16 tokens),
            pooled (all facts -> 16 tokens), plus single-name spelling at a random k. Saves lora_slots.pt.
    eval    60 held-out sessions x 3 kinds x the six layouts -> slots_eval.json, lines starting SLOTS.
"""

import argparse
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import poc_agentmem as A  # noqa: E402
import poc_atoms as T  # noqa: E402
import poc_spell as S  # noqa: E402
from poc_agentmem import P, OUT, save  # noqa: E402

LAYOUTS = ["k2", "k4", "k8", "k16", "pair", "pooled"]


def memory(st, name, s, v, nrng, qv, sel=None):
    rows = s["arows"] if sel is None else s["arows"][sel]
    atoms = v[rows].astype(np.float32)[nrng.permutation(len(rows))]
    n = len(atoms)
    if name == "pooled":
        return st.memory(atoms, [np.arange(n)], qv)
    if name == "pair":
        return st.memory(atoms, [np.arange(j, min(j + 2, n)) for j in range(0, n, 2)], qv)
    k = int(name[1:])
    mem = st.memory(atoms, [np.array([j]) for j in range(n)], qv)
    return mem.reshape(1, n, 16, mem.shape[-1])[:, :, :k].reshape(1, n * k, mem.shape[-1])


def load_lora(st, name, eval_mode=False):
    sd = torch.load(OUT / name, map_location="cpu")
    for i, m in enumerate(st.lora):
        m.A.data.copy_(sd[f"{i}.A"])
        m.B.data.copy_(sd[f"{i}.B"])
        if eval_mode:
            m.drop.p = 0.0


def cmd_train(args):
    torch.manual_seed(0)
    ss, _, v = T.load_all()
    train = [s for s in ss if s["split"] == "train"]
    items, svecs, q = S.load()
    for j, it in enumerate(items):
        it["row"] = j
    names = [i for i in items if i["split"] == "train"]
    st = A.Stack("lora", train=True)
    load_lora(st, args.init)
    ps = [t for m in st.lora for t in (m.A, m.B)]
    opt = torch.optim.AdamW(ps, lr=args.lr, weight_decay=0.0)
    rng, nrng = random.Random(0), np.random.default_rng(0)

    def loss_one():
        if rng.random() < 0.15:
            it, k = rng.choice(names), rng.choice([2, 4, 8, 16])
            mem = st.memory(svecs, [np.array([it["row"]])], q[S.KINDS.index(it["kind"])])[:, :k]
            kind, answer = it["kind"], it["name"]
        else:
            s = rng.choice(train)
            sel = np.array(sorted(rng.sample(range(len(s["facts"])), min(8, len(s["facts"])))))
            fs = [s["facts"][j] for j in sel]
            kind = rng.choice([k for k in S.KINDS if any(fk == k for fk, _ in fs)])
            name = rng.choices(LAYOUTS, [0.2, 0.2, 0.2, 0.1, 0.2, 0.1])[0]
            mem = memory(st, name, s, v, nrng, q[S.KINDS.index(kind)], sel)
            answer = ", ".join(x for k, x in fs if k == kind)
        ids, labels = st.ids(S.QS[kind], mem.shape[1], answer)
        emb = P.input_embeds(st.r.model, st.r.tok, st.r.memory_id, ids, mem)
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
        torch.mps.empty_cache()
        if u % 25 == 0:
            print(f"update {u}  examples {u * 4}  loss {np.mean(run[-100:]):.3f}  {(time.perf_counter() - t0) / 60:.0f} min", flush=True)
        if time.perf_counter() - t0 > next_save:
            next_save += 900
            torch.save(st.state(), OUT / "lora_slots.pt")
    torch.save(st.state(), OUT / "lora_slots.pt")
    save("slots_train.json", {"updates": u, "examples": u * 4, "loss": float(np.mean(run[-100:])), "minutes": args.minutes})
    print("saved", flush=True)


@torch.no_grad()
def cmd_eval(args):
    ss, _, v = T.load_all()
    test = [s for s in ss if s["split"] == "test"]
    q = np.load(OUT / "spell_emb.npz")["q"]
    st = A.Stack("lora", train=True)
    load_lora(st, args.lora, eval_mode=True)
    st.r.model.eval()
    random.Random(3).shuffle(test)
    rows, t0, tab = [], time.perf_counter(), {}
    for i, s in enumerate(test[: args.n]):
        for kind in S.KINDS:
            truth = [x for k, x in s["facts"] if k == kind]
            if not truth:
                continue
            for name in LAYOUTS:
                mem = memory(st, name, s, v, np.random.default_rng(i), q[S.KINDS.index(kind)])
                ans = A.generate(st, S.QS[kind], mem, 64).strip()
                got = [t.lower() in ans.lower() for t in truth]
                rows.append({"session": s["id"], "kind": kind, "condition": name, "n_tokens": int(mem.shape[1]),
                             "n_facts": len(s["facts"]), "truth": truth, "answer": ans,
                             "hit": float(any(got)), "recall": float(np.mean(got))})
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
        print(f"SLOTS {c:7s} " + "  ".join(f"{k} any {x['any']:.2f} recall {x['recall']:.2f}" for k, x in ks.items())
              + f"  tokens {np.mean([x['tokens'] for x in ks.values()]):.0f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--minutes", type=float, default=90)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--init", default="lora_atoms.pt")
    e = sub.add_parser("eval")
    e.add_argument("--n", type=int, default=60)
    e.add_argument("--lora", default="lora_slots.pt")
    e.add_argument("--out", default="slots_eval.json")
    args = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
