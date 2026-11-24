"""H5: is SPELLING a name from memory tokens a reader skill that more distinct names can teach? (core frozen)

The smallest possible case: the memory holds ONE short sentence containing one name ("This coding session was in the
repository X." / "The file changed in this coding session was X." / "The function or class touched ... was X."),
the question asks for the name, the answer is the name. Names come from the local SWE-agent trajectories (repos,
patched file paths, def/class names), split by repository, so test names come from repos never trained on.
A LoRA on the decoder is trained on K distinct names; we measure exact-match on held-out names as K grows.

    build   spell.json (names, type, split)
    embed   spell_emb.npz (sentence vectors + 3 question vectors)
    run     --k K --tag T --minutes M : train a fresh LoRA on K names, then generate for test names -> spell_T.json
"""

import argparse
import glob
import json
import math
import random
import re
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import poc_agentmem as A  # noqa: E402
from poc_agentmem import D, P, OUT, save  # noqa: E402

SENT = {"repo": "This coding session was in the repository {x}.",
        "file": "The file changed in this coding session was {x}.",
        "func": "The function or class touched in this coding session was {x}."}
QS = {"repo": "Which repository was the coding session in your memory about? Answer with the repository name only.",
      "file": "Which file was changed in the coding session in your memory? Answer with the file path only.",
      "func": "Which function or class was touched in the coding session in your memory? Answer with its name only."}
KINDS = list(SENT)


def cmd_build(args):
    import pandas as pd

    sess_split = {s["repo"]: s["split"] for s in A.load_sessions()}
    by = {k: {} for k in KINDS}  # name -> repo
    for f in sorted(glob.glob(str(A.SRC / "*.parquet"))):
        df = pd.read_parquet(f, columns=["instance_id", "generated_patch"])
        for inst, p in zip(df.instance_id, df.generated_patch.fillna("").astype(str)):
            repo = inst.rsplit("-", 1)[0].replace("__", "/")
            by["repo"].setdefault(repo, repo)
            for m in re.finditer(r"(?m)^diff --git a/(\S+) b/(\S+)", p):
                if len(m.group(2)) <= 80:
                    by["file"].setdefault(m.group(2), repo)
            for m in re.finditer(r"\b(?:def|class)\s+([A-Za-z_]\w{5,40})", p):
                by["func"].setdefault(m.group(1), repo)

    def split_of(repo):
        if repo in sess_split:
            return sess_split[repo]
        return "test" if zlib.crc32(repo.encode()) % 100 < 15 else "train"

    rng = random.Random(0)
    items = []
    for k in KINDS:
        names = sorted(by[k].items())
        rng.shuffle(names)
        quota = {"train": args.train_per_kind, "test": 100, "val": 30}
        for name, repo in names:
            sp = split_of(repo)
            if quota[sp] > 0:
                quota[sp] -= 1
                items.append({"kind": k, "name": name, "repo": repo, "split": sp})
    rng.shuffle(items)
    save("spell.json", {"items": items})
    for sp in ("train", "val", "test"):
        print(sp, {k: sum(i["split"] == sp and i["kind"] == k for i in items) for k in KINDS})


def load():
    items = json.loads((OUT / "spell.json").read_text())["items"]
    z = np.load(OUT / "spell_emb.npz")
    return items, z["vecs"], z["q"]


def cmd_embed(args):
    items = json.loads((OUT / "spell.json").read_text())["items"]
    texts = [SENT[i["kind"]].format(x=i["name"]) for i in items]
    enc = D.Encoder()
    t0, vecs = time.perf_counter(), []
    for b in range(0, len(texts), 512):
        vecs.append(enc(texts[b:b + 512], batch_size=32).astype(np.float16))
        print(f"{b + len(vecs[-1])}/{len(texts)}  {time.perf_counter() - t0:.0f}s", flush=True)
        torch.mps.empty_cache()
    q = enc([QS[k] for k in KINDS], batch_size=4).astype(np.float16)
    np.savez(OUT / "spell_emb.npz", vecs=np.concatenate(vecs), q=q)
    print("done", flush=True)


def sim(a, b):
    import difflib

    return difflib.SequenceMatcher(None, a, b).ratio()


def cmd_run(args):
    torch.manual_seed(0)
    items, vecs, q = load()
    for j, it in enumerate(items):
        it["row"] = j
    train = [i for i in items if i["split"] == "train"]
    per = args.k // len(KINDS)
    train = [i for k in KINDS for i in [x for x in train if x["kind"] == k][:per]]
    test = [i for i in items if i["split"] == "test"]
    st = A.Stack("lora", train=True, r=args.rank)
    ps = [t for m in st.lora for t in (m.A, m.B)]
    opt = torch.optim.AdamW(ps, lr=args.lr, weight_decay=0.0)
    print(f"train names {len(train)}, test {len(test)}", flush=True)

    def mem_of(it):
        return st.memory(vecs, [np.array([it["row"]])], q[KINDS.index(it["kind"])])

    def loss_of(it):
        mem = mem_of(it)
        ids, labels = st.ids(QS[it["kind"]], mem.shape[1], it["name"])
        emb = P.input_embeds(st.r.model, st.r.tok, st.r.memory_id, ids, mem)
        return st.r.model(inputs_embeds=emb, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False).loss

    @torch.no_grad()
    def evaluate(pool, n):
        st.r.model.eval()
        rows = []
        for it in pool[:n]:
            ans = A.generate(st, QS[it["kind"]], mem_of(it), 28).strip()
            rows.append({"kind": it["kind"], "name": it["name"], "answer": ans, "exact": float(ans == it["name"]),
                         "contains": float(it["name"].lower() in ans.lower()), "sim": sim(ans, it["name"])})
        torch.mps.empty_cache()
        st.r.model.train()
        return rows

    def summ(rows):
        out = {"all": {m: float(np.mean([r[m] for r in rows])) for m in ("exact", "contains", "sim")}}
        for k in KINDS:
            rs = [r for r in rows if r["kind"] == k]
            if rs:
                out[k] = {m: float(np.mean([r[m] for r in rs])) for m in ("exact", "contains", "sim")}
        return out

    rng = random.Random(0)
    st.r.model.train()
    t0, u, run, log = time.perf_counter(), 0, [], []
    budget = args.minutes * 60
    next_eval = budget / 4
    order = []
    while time.perf_counter() - t0 < budget:
        lr = args.lr * min(1.0, (u + 1) / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, (time.perf_counter() - t0) / budget))))
        for g in opt.param_groups:
            g["lr"] = lr
        for _ in range(args.accum):
            if not order:
                order = train[:]
                rng.shuffle(order)
            loss = loss_of(order.pop()) / args.accum
            loss.backward()
            run.append(loss.item() * args.accum)
        torch.nn.utils.clip_grad_norm_(ps, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        u += 1
        if u % 25 == 0:
            torch.mps.empty_cache()
            print(f"update {u}  examples {u * args.accum}  loss {np.mean(run[-100:]):.3f}  {(time.perf_counter() - t0) / 60:.0f} min", flush=True)
        if time.perf_counter() - t0 > next_eval:
            next_eval += budget / 4
            s = summ(evaluate(test, 60))
            log.append({"update": u, "examples": u * args.accum, "train_loss": float(np.mean(run[-100:])), "test60": s["all"]})
            print("  quick test:", s["all"], flush=True)
    rows = evaluate(test, len(test))
    tr_rows = evaluate(train, 60)
    res = {"k": len(train), "updates": u, "examples": u * args.accum, "minutes": args.minutes, "log": log,
           "test": summ(rows), "train_sample": summ(tr_rows), "rows": rows, "train_rows": tr_rows}
    save(f"spell_{args.tag}.json", res)
    print("TEST", json.dumps(res["test"]), flush=True)
    print("TRAIN", json.dumps(res["train_sample"]), flush=True)


# ---------------------------------------------------------------- H7: does spelling transfer to real sessions?

MIX = [("spell", 0.40), ("note", 0.20), ("raw", 0.15), ("note+raw", 0.10), ("notes", 0.15)]


def mix_setup():
    import poc_agenthyp as H

    items, vecs, q = load()
    for j, it in enumerate(items):
        it["row"] = j
    ss, chunks, qv = A.load_emb()
    return items, vecs, q, ss, chunks, qv, H.Extra(ss)


def mix_memory(st, kind, s, others, chunks, qv, ex, nrng, anchored):
    f = lambda a: a.astype(np.float32)  # noqa: E731
    if kind == "raw":
        d = f(chunks[s["rows"]])
    elif kind == "note":
        d = f(ex.v[[ex.note[s["id"]]]])
    elif kind == "note+raw":
        d = np.concatenate([f(chunks[s["rows"]]), f(ex.v[[ex.note[s["id"]]]])])
    else:
        d = f(ex.v[[ex.note[o["id"]] for o in others + [s]]])
    d = d[nrng.permutation(len(d))]
    return st.memory(d, [np.arange(len(d))], qv[s["qi"]] if anchored else qv[0])


def cmd_mix(args):
    torch.manual_seed(0)
    items, vecs, q, ss, chunks, qv, ex = mix_setup()
    names = [i for i in items if i["split"] == "train"]
    train = [s for s in ss if s["split"] == "train"]
    st = A.Stack("lora", train=True, r=args.rank)
    ps = [t for m in st.lora for t in (m.A, m.B)]
    opt = torch.optim.AdamW(ps, lr=args.lr, weight_decay=0.0)
    rng, nrng = random.Random(0), np.random.default_rng(0)
    kinds, w = zip(*MIX)

    def loss_one():
        kind = rng.choices(kinds, w)[0]
        if kind == "spell":
            it = rng.choice(names)
            mem = st.memory(vecs, [np.array([it["row"]])], q[KINDS.index(it["kind"])])
            question, answer = QS[it["kind"]], it["name"]
        else:
            s = rng.choice(train)
            anchored = kind == "notes"
            others = rng.sample([o for o in train if o["repo"] != s["repo"]], rng.choice([3, 7, 15])) if anchored else []
            mem = mix_memory(st, kind, s, others, chunks, qv, ex, nrng, anchored)
            question = A.Q_ANCHOR.format(title=s["title"]) if anchored else A.Q_SINGLE
            answer = A.target(s, anchored)
        ids, labels = st.ids(question, mem.shape[1], answer)
        emb = P.input_embeds(st.r.model, st.r.tok, st.r.memory_id, ids, mem)
        return st.r.model(inputs_embeds=emb, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False).loss

    st.r.model.train()
    t0, u, run = time.perf_counter(), 0, []
    budget, next_save = args.minutes * 60, 900
    while time.perf_counter() - t0 < budget:
        frac = min(1.0, (time.perf_counter() - t0) / budget)
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (u + 1) / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for _ in range(args.accum):
            loss = loss_one() / args.accum
            loss.backward()
            run.append(loss.item() * args.accum)
        torch.nn.utils.clip_grad_norm_(ps, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        u += 1
        if u % 25 == 0:
            torch.mps.empty_cache()
            print(f"update {u}  examples {u * args.accum}  loss {np.mean(run[-100:]):.3f}  {(time.perf_counter() - t0) / 60:.0f} min", flush=True)
        if time.perf_counter() - t0 > next_save:
            next_save += 900
            torch.save(st.state(), OUT / "lora_mix.pt")
    torch.save(st.state(), OUT / "lora_mix.pt")
    save("mix_train.json", {"updates": u, "examples": u * args.accum, "loss": float(np.mean(run[-100:])), "minutes": args.minutes})
    print("saved", flush=True)


@torch.no_grad()
def cmd_mixeval(args):
    items, vecs, q, ss, chunks, qv, ex = mix_setup()
    st = A.Stack("lora", train=True)
    sd = torch.load(OUT / "lora_mix.pt", map_location="cpu")
    for i, m in enumerate(st.lora):
        m.A.data.copy_(sd[f"{i}.A"])
        m.B.data.copy_(sd[f"{i}.B"])
        m.drop.p = 0.0
    st.r.model.eval()
    test = [s for s in ss if s["split"] == "test"]
    random.Random(3).shuffle(test)
    pick = test[: args.n]
    conds = [("raw session, 16 tok", "raw", 0), ("note only", "note", 0), ("note + raw session", "note+raw", 0),
             ("8 session notes", "notes", 7), ("32 session notes", "notes", 31), ("150 session notes", "notes", 149)]
    rows, t0 = [], time.perf_counter()
    for i, s in enumerate(pick):
        for name, kind, k in conds:
            rng = random.Random(zlib.crc32(f"{s['id']}|{k}".encode()))
            nrng = np.random.default_rng(zlib.crc32(s["id"].encode()))
            pool = [o for o in (test if k < 100 else ss) if o["repo"] != s["repo"]]
            others = rng.sample(pool, k) if k else []
            mem = mix_memory(st, kind, s, others, chunks, qv, ex, nrng, bool(k))
            question = A.Q_ANCHOR.format(title=s["title"]) if k else A.Q_SINGLE
            ans = A.generate(st, question, mem, 72)
            rows.append({"session": s["id"], "condition": name, "answer": ans, **A.score(s, ans, bool(k))})
        torch.mps.empty_cache()
        if (i + 1) % 5 == 0 or i + 1 == len(pick):
            tab = {}
            for r in rows:
                tab.setdefault(r["condition"], []).append(r)
            tab = {c: {m: float(np.mean([r[m] for r in rs if r.get(m) is not None] or [0]))
                       for m in ("repo", "files_path", "files_name", "funcs")} for c, rs in tab.items()}
            save("mix_eval.json", {"rows": rows, "table": tab})
            print(f"{i + 1}/{len(pick)}  {(time.perf_counter() - t0) / 60:.0f} min", flush=True)
    for c, v in tab.items():
        print(f"MIX {c:24s} " + "  ".join(f"{m} {x:.2f}" for m, x in v.items()), flush=True)
    srows = []
    for it in [i for i in items if i["split"] == "test"]:
        ans = A.generate(st, QS[it["kind"]], st.memory(vecs, [np.array([it["row"]])], q[KINDS.index(it["kind"])]), 28).strip()
        srows.append({"kind": it["kind"], "name": it["name"], "answer": ans, "exact": float(ans == it["name"])})
    sp = {k: float(np.mean([r["exact"] for r in srows if r["kind"] == k])) for k in KINDS}
    save("mix_spell.json", {"exact": sp, "rows": srows})
    print("MIX spelling exact", sp, flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--train_per_kind", type=int, default=3000)
    sub.add_parser("embed")
    r = sub.add_parser("run")
    r.add_argument("--k", type=int, required=True)
    r.add_argument("--tag", required=True)
    r.add_argument("--minutes", type=float, default=35)
    r.add_argument("--rank", type=int, default=16)
    r.add_argument("--lr", type=float, default=2e-4)
    r.add_argument("--accum", type=int, default=4)
    m = sub.add_parser("mix")
    m.add_argument("--minutes", type=float, default=90)
    m.add_argument("--rank", type=int, default=16)
    m.add_argument("--lr", type=float, default=2e-4)
    m.add_argument("--accum", type=int, default=4)
    e = sub.add_parser("mixeval")
    e.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    {"build": cmd_build, "embed": cmd_embed, "run": cmd_run, "mix": cmd_mix, "mixeval": cmd_mixeval}[args.cmd](args)


if __name__ == "__main__":
    main()
