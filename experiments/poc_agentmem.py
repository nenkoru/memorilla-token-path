"""Agent-session memory: can Memorilla's frozen core + a LoRA'd Qwen3-8B recall WHAT happened in an earlier coding-agent
session (repo, issue, files changed, functions touched) from memory tokens alone, and how many tokens does it take?

Data (untrusted, read as data only): nebius/SWE-agent-trajectories, shards 0-1 (CC-BY-4.0), in
$MPORT_SCRATCH/ext/swe-traj/. One "session" = one SWE-agent trajectory. Targets are extracted mechanically:
    repository   from instance_id (owner__repo-N -> owner/repo)
    issue        first line of the issue text
    files        files the final patch modifies (new scratch files such as reproduce.py excluded)
    functions    def/class names from the patch's hunk headers and added lines
Documents = chunks of the trajectory (issue, then one chunk per agent step + its observation), each embedded once by
the frozen Qwen3-Embedding-4B. Splits are by repository, so test repos are never seen in training.

Token budget without retraining the core (fixed 16 slots): split a memory's chunks into G groups, run the core on each
group, concatenate -> G x 16 tokens (num_memories = 16G placeholders). "per-session" grouping = one group per session.

Subcommands (outputs in $MPORT_SCRATCH/runs/agentmem/):
    build   sessions.json (facts, chunk texts, split)
    embed   emb.npz: chunk vectors (fp16) + question vectors
    train   LoRA (r16 on q/k/v/o of all 36 layers); core, encoder and base decoder frozen. Mixed examples:
            single session with G in {1,2,4,8}, or an "anchored" question about one session among K distractors
            (pooled G groups or per-session groups). Early stop on val loss, wall-clock cap, best checkpoint kept.
    eval    --variant base|lora on test repos: recall of each fact under several token budgets / histories.

RAM: one model per process under poc_guard.sh; encoder and decoder never loaded together.
"""

import argparse
import json
import math
import random
import re
import sys
import time
import zlib
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import poc_clue as C  # noqa: E402  (env defaults, memorilla path)
from poc_clue import D, P, HEAD, DECODER  # noqa: E402
from poc_adapt import LoRALinear  # noqa: E402
from memorilla.data import apply_chat_template, build_messages  # noqa: E402

SRC = P.SCRATCH / "ext/swe-traj/data"
OUT = P.SCRATCH / "runs/agentmem"
SLOTS = 16
SYSTEM = ("You are a coding assistant with memory of the user's earlier coding sessions. "
          "Answer from that memory, briefly and exactly.")
Q_SINGLE = ("Summarize the coding session in your memory: the repository, the issue, the files changed and the "
            "functions or classes touched.")
Q_ANCHOR = ("Your memory holds several coding sessions. One of them was about this issue: \"{title}\". For that "
            "session, list the repository, the files changed and the functions or classes touched.")
MAX_CHUNKS = 24
CHUNK_CHARS = 800
OBS_CHARS = 300
EMB = OUT / __import__("os").environ.get("AGENTMEM_EMB", "emb.npz")


def save(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / (name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(OUT / name)


# ---------------------------------------------------------------- build


def patch_facts(patch):
    files, funcs = [], []
    for block in re.split(r"(?m)^diff --git ", patch or ""):
        m = re.match(r"a/(\S+) b/(\S+)", block)
        if not m or "new file mode" in block[:400] or "deleted file mode" in block[:400]:
            continue
        files.append(m.group(2))
        for line in block.splitlines():
            ctx = re.match(r"@@ [^@]* @@\s*(.*)", line)
            src = ctx.group(1) if ctx else (line[1:] if line.startswith("+") and not line.startswith("+++") else "")
            for name in re.findall(r"\b(?:def|class)\s+([A-Za-z_]\w*)", src):
                if name not in funcs:
                    funcs.append(name)
    return files[:4], funcs[:4]


def chunks_of(traj):
    msgs = [m for m in traj if m["role"] in ("user", "ai") and m["text"]]
    if not msgs:
        return [], ""
    issue = msgs[0]["text"]
    m = re.search(r"ISSUE:\s*\n(.*?)(?:\nINSTRUCTIONS:|$)", issue, re.S)
    issue_body = (m.group(1) if m else issue).strip()
    title = next((ln.strip() for ln in issue_body.splitlines() if ln.strip()), "")[:120]
    out = ["Issue:\n" + issue_body[:CHUNK_CHARS]]
    i = 1
    while i < len(msgs):
        step = msgs[i]["text"] if msgs[i]["role"] == "ai" else ""
        obs = ""
        if msgs[i]["role"] == "ai" and i + 1 < len(msgs) and msgs[i + 1]["role"] == "user":
            obs = msgs[i + 1]["text"]
            i += 1
        elif msgs[i]["role"] == "user":
            obs = msgs[i]["text"]
        i += 1
        text = ("Agent: " + step.strip())[: CHUNK_CHARS - OBS_CHARS] if step else ""
        if obs:
            text += "\nOutput: " + obs.strip()[:OBS_CHARS]
        out.append(text.strip())
    if len(out) > MAX_CHUNKS:  # keep the issue, the last 3 steps (submit shows the diff) and an even sample between
        mid = out[1:-3]
        keep = np.linspace(0, len(mid) - 1, MAX_CHUNKS - 4).round().astype(int)
        out = [out[0]] + [mid[k] for k in keep] + out[-3:]
    return out, title


def cmd_build(args):
    import pandas as pd

    rows = []
    for f in sorted(SRC.glob("*.parquet")):
        df = pd.read_parquet(f, columns=["instance_id", "model_name", "target", "trajectory", "generated_patch"])
        df["generated_patch"] = df["generated_patch"].fillna("").astype(str)
        rows += df.to_dict("records")
    by_inst = defaultdict(list)
    for r in rows:
        by_inst[r["instance_id"]].append(r)
    sessions = []
    for inst, rs in sorted(by_inst.items()):
        rs.sort(key=lambda r: (not r["target"], -len(r["generated_patch"] or "")))
        for r in rs[: args.per_instance]:
            files, funcs = patch_facts(r["generated_patch"])
            if not files:
                continue
            chunks, title = chunks_of(list(r["trajectory"]))
            if len(chunks) < 4 or not title:
                continue
            repo = inst.rsplit("-", 1)[0].replace("__", "/")
            text = "\n".join(chunks).lower()
            sessions.append({"id": f"{inst}#{len(sessions)}", "instance": inst, "repo": repo, "title": title,
                             "files": files, "funcs": funcs, "model": r["model_name"], "resolved": bool(r["target"]),
                             "chunks": chunks,
                             "in_text": {"repo": repo.split("/")[1].lower() in text,
                                         "files": float(np.mean([fl.split("/")[-1].lower() in text for fl in files])),
                                         "funcs": float(np.mean([fn.lower() in text for fn in funcs])) if funcs else None}})
    repos = sorted({s["repo"] for s in sessions})
    random.Random(0).shuffle(repos)
    n_test, n_val = round(len(repos) * 0.15), round(len(repos) * 0.03)
    split = {r: "test" for r in repos[:n_test]} | {r: "val" for r in repos[n_test:n_test + n_val]}
    seen_test_inst = set()
    for s in sessions:
        s["split"] = split.get(s["repo"], "train")
    # one trajectory per instance outside train, so test items are distinct tasks
    keep = []
    for s in sessions:
        if s["split"] != "train":
            if s["instance"] in seen_test_inst:
                continue
            seen_test_inst.add(s["instance"])
        keep.append(s)
    sessions = keep
    cnt = defaultdict(int)
    for s in sessions:
        cnt[s["split"]] += 1
    cov = {k: float(np.mean([s["in_text"][k] for s in sessions if s["in_text"][k] is not None])) for k in
           ("repo", "files", "funcs")}
    print(f"{len(rows)} trajectories, {len(by_inst)} instances, {len(repos)} repos -> sessions {dict(cnt)}; "
          f"chunks/session {np.mean([len(s['chunks']) for s in sessions]):.1f}; facts present in chunk text {cov}")
    save("sessions.json", {"sessions": sessions, "coverage": cov, "counts": dict(cnt)})


# ---------------------------------------------------------------- embed


def load_sessions():
    return json.loads((OUT / "sessions.json").read_text())["sessions"]


def cmd_embed(args):
    ss = load_sessions()
    if args.limit:  # smoke test: a few sessions from every split
        ss = ([s for s in ss if s["split"] == "train"][: args.limit] + [s for s in ss if s["split"] == "val"][:6]
              + [s for s in ss if s["split"] == "test"][:6])
    texts = [c for s in ss for c in s["chunks"]]
    questions = [Q_SINGLE] + [Q_ANCHOR.format(title=s["title"]) for s in ss]
    enc = D.Encoder()
    t0 = time.perf_counter()
    vecs = []
    for i in range(0, len(texts), 256):
        vecs.append(enc(texts[i:i + 256], batch_size=args.batch).astype(np.float16))
        done = i + len(vecs[-1])
        el = time.perf_counter() - t0
        print(f"{done}/{len(texts)} chunks  {el:.0f}s  {done / el:.1f}/s  eta {(len(texts) - done) / (done / el) / 60:.0f} min",
              flush=True)
        if (i // 256) % 8 == 0:
            torch.mps.empty_cache()
    qv = enc(questions, batch_size=args.batch).astype(np.float16)
    offs = np.cumsum([0] + [len(s["chunks"]) for s in ss])
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / ("emb.npz" if not args.limit else "emb_probe.npz"), chunks=np.concatenate(vecs), offsets=offs,
             questions=qv, ids=np.array([s["id"] for s in ss]), seconds=time.perf_counter() - t0)
    print(f"embedded {len(texts)} chunks + {len(questions)} questions in {time.perf_counter() - t0:.0f}s", flush=True)


def load_emb():
    z = np.load(EMB)
    by_id = {s["id"]: s for s in load_sessions()}
    ss = [by_id[str(x)] for x in z["ids"]]
    offs = z["offsets"]
    for i, s in enumerate(ss):
        s["rows"] = np.arange(offs[i], offs[i + 1])
        s["qi"] = i + 1
    return ss, z["chunks"], z["questions"]


# ---------------------------------------------------------------- model


def add_lora(model, r, names):
    mods = []
    for layer in model.model.layers:
        for name in names:
            lin = LoRALinear(getattr(layer.self_attn, name), r=r, alpha=2 * r)
            setattr(layer.self_attn, name, lin)
            mods.append(lin)
    return mods


class Stack:
    def __init__(self, variant, train=False, r=16, names=("q_proj", "k_proj", "v_proj", "o_proj")):
        self.r = D.Runner([HEAD], DECODER)
        self.core = self.r.heads[HEAD].eval().requires_grad_(False)
        self.lora = add_lora(self.r.model, r, names) if variant == "lora" else []
        if variant == "lora" and not train:
            st = torch.load(OUT / "lora.pt", map_location="cpu")
            for i, m in enumerate(self.lora):
                m.A.data.copy_(st[f"{i}.A"])
                m.B.data.copy_(st[f"{i}.B"])
        for m in self.lora:
            m.drop.p = m.drop.p if train else 0.0

    def state(self):
        return {f"{i}.{k}": v.detach().cpu().clone() for i, m in enumerate(self.lora) for k, v in (("A", m.A), ("B", m.B))}

    @torch.no_grad()
    def memory(self, chunks, groups, q_vec):
        """groups: list of row-index arrays -> [1, 16 * len(groups), 4096]."""
        n = max(len(g) for g in groups)
        d = torch.zeros(len(groups), n, chunks.shape[1])
        mask = torch.ones(len(groups), n, dtype=torch.bool)
        for i, g in enumerate(groups):
            d[i, : len(g)] = torch.from_numpy(chunks[g].astype(np.float32))
            mask[i, : len(g)] = False
        q = torch.from_numpy(q_vec.astype(np.float32))[None].expand(len(groups), -1)
        tok = self.core(d.to(P.DEV), mask.to(P.DEV), q.to(P.DEV))
        return tok.reshape(1, -1, tok.shape[-1])

    def ids(self, question, n_mem, answer=None):
        msgs = build_messages(question, SYSTEM, num_memories=n_mem, answer=answer)
        tok = self.r.tok
        prompt = apply_chat_template(tok, msgs[:2], add_generation_prompt=True)
        p_ids = tok.encode(prompt, add_special_tokens=self.r.add_special)
        if answer is None:
            return torch.tensor([p_ids], device=P.DEV), None
        f_ids = tok.encode(apply_chat_template(tok, msgs, add_generation_prompt=False),
                           add_special_tokens=self.r.add_special)
        assert f_ids[: len(p_ids)] == p_ids
        ids = torch.tensor([f_ids], device=P.DEV)
        labels = ids.clone()
        labels[:, : len(p_ids)] = -100
        return ids, labels


def target(s, anchored):
    lines = [f"Repository: {s['repo']}"]
    if not anchored:
        lines.append(f"Issue: {s['title']}")
    lines.append("Files changed: " + ", ".join(s["files"]))
    lines.append("Functions/classes: " + (", ".join(s["funcs"]) if s["funcs"] else "none"))
    return "\n".join(lines)


def split_groups(rows, g, rng=None):
    rows = np.asarray(rows)
    if rng is not None:
        rows = rows[rng.permutation(len(rows))]
    g = max(1, min(g, len(rows)))
    return [x for x in np.array_split(rows, g) if len(x)]


def make_example(s, pool, rng, nrng, mode=None, g=None, k=None, grouping=None):
    """Returns (question, q_index, groups, answer)."""
    mode = mode or rng.choice(["single", "anchored"])
    if mode == "single":
        g = g or rng.choice([1, 2, 4, 8])
        return Q_SINGLE, 0, split_groups(s["rows"], g), target(s, False)
    k = k if k is not None else rng.choice([1, 3, 7])
    others = rng.sample([o for o in pool if o["repo"] != s["repo"]], k)
    sess = others + [s]
    rng.shuffle(sess)
    grouping = grouping or rng.choice(["pooled", "per_session"])
    if grouping == "per_session":
        groups = [o["rows"] for o in sess]
    else:
        g = g or rng.choice([1, 2, 4, 8])
        groups = split_groups(np.concatenate([o["rows"] for o in sess]), g, nrng)
    return Q_ANCHOR.format(title=s["title"]), s["qi"], groups, target(s, True)


def loss_of(st, chunks, qv, ex):
    question, qi, groups, answer = ex
    mem = st.memory(chunks, groups, qv[qi])
    ids, labels = st.ids(question, mem.shape[1], answer)
    embeds = P.input_embeds(st.r.model, st.r.tok, st.r.memory_id, ids, mem)
    return st.r.model(inputs_embeds=embeds, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False).loss


def cmd_train(args):
    torch.manual_seed(args.seed)
    ss, chunks, qv = load_emb()
    train = [s for s in ss if s["split"] == "train"]
    val = [s for s in ss if s["split"] == "val"]
    st = Stack("lora", train=True, r=args.rank)
    ps = [t for m in st.lora for t in (m.A, m.B)]
    print(f"train sessions {len(train)}, val {len(val)}; LoRA {sum(p.numel() for p in ps) / 1e6:.2f}M params", flush=True)
    opt = torch.optim.AdamW(ps, lr=args.lr, weight_decay=0.0)
    total_updates = args.updates
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda u: min(1.0, (u + 1) / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(u, total_updates) / total_updates))))
    vr, vn = random.Random(7), np.random.default_rng(7)
    val_ex = [make_example(val[i % len(val)], val + train, vr, vn) for i in range(args.n_val)]

    def val_loss():
        st.r.model.eval()
        with torch.no_grad():
            ls = [loss_of(st, chunks, qv, e).item() for e in val_ex]
        torch.mps.empty_cache()
        return float(np.mean(ls))

    rng, nrng = random.Random(args.seed), np.random.default_rng(args.seed)
    log, t0 = [], time.perf_counter()
    best = val_loss()
    log.append({"update": 0, "val": best})
    print(f"update 0  val {best:.3f}", flush=True)
    bad, step, run = 0, 0, []
    torch.save(st.state(), OUT / "lora.pt")
    stop = "updates"
    while step < total_updates * args.accum:
        s = rng.choice(train)
        st.r.model.train()
        loss = loss_of(st, chunks, qv, make_example(s, train, rng, nrng)) / args.accum
        loss.backward()
        run.append(loss.item() * args.accum)
        step += 1
        if step % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(ps, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
        if step % 8 == 0:
            torch.mps.empty_cache()
        u = step // args.accum
        if step % args.accum == 0 and u % args.eval_every == 0:
            v = val_loss()
            tl = float(np.mean(run))
            run = []
            el = time.perf_counter() - t0
            log.append({"update": u, "val": v, "train": tl, "seconds": el})
            print(f"update {u}  {el / 60:.0f} min  train {tl:.3f}  val {v:.3f}", flush=True)
            if v < best - 1e-3:
                best, bad = v, 0
                torch.save(st.state(), OUT / "lora.pt")
            else:
                bad += 1
            save("train_log.json", {"args": vars(args), "best_val": best, "log": log})
            if bad >= args.patience:
                stop = "early stop"
                break
            if el > args.max_hours * 3600:
                stop = "time cap"
                break
    print(f"{stop}; best val {best:.3f}", flush=True)
    save("train_log.json", {"args": vars(args), "best_val": best, "log": log, "stop": stop,
                            "seconds": time.perf_counter() - t0})


# ---------------------------------------------------------------- eval


def norm(x):
    return re.sub(r"[^a-z0-9_./]+", " ", x.lower())


def score(s, answer, anchored):
    a = norm(answer)
    words = lambda t: [w for w in re.findall(r"[a-z0-9_]{4,}", t.lower())]  # noqa: E731
    tw = words(s["title"])
    out = {"repo": float(s["repo"].split("/")[1].lower() in a),
           "files_path": float(np.mean([f.lower() in a for f in s["files"]])),
           "files_name": float(np.mean([f.split("/")[-1].lower() in a for f in s["files"]])),
           "funcs": float(np.mean([f.lower() in a for f in s["funcs"]])) if s["funcs"] else None}
    if not anchored:
        out["issue"] = float(np.mean([w in a for w in tw])) if tw else None
    return out


CONDITIONS = [  # name, mode, k, grouping, g, counterfactual
    ("single 16 tok", "single", 0, "pooled", 1, False),
    ("single 32 tok", "single", 0, "pooled", 2, False),
    ("single 64 tok", "single", 0, "pooled", 4, False),
    ("single 128 tok", "single", 0, "pooled", 8, False),
    ("8 sessions, pooled 16 tok", "anchored", 7, "pooled", 1, False),
    ("8 sessions, pooled 128 tok", "anchored", 7, "pooled", 8, False),
    ("8 sessions, 16 tok each (128)", "anchored", 7, "per_session", None, False),
    ("32 sessions, pooled 16 tok", "anchored", 31, "pooled", 1, False),
    ("32 sessions, 16 tok each (512)", "anchored", 31, "per_session", None, False),
    ("8 sessions without it, 16 tok each", "anchored", 7, "per_session", None, True),
]


@torch.no_grad()
def generate(st, question, mem, max_new):
    ids, _ = st.ids(question, mem.shape[1])
    embeds = P.input_embeds(st.r.model, st.r.tok, st.r.memory_id, ids, mem)
    out = st.r.model.generate(inputs_embeds=embeds, attention_mask=torch.ones(ids.shape, device=P.DEV),
                              max_new_tokens=max_new, do_sample=False, pad_token_id=st.r.tok.pad_token_id)
    return st.r.tok.decode(out[0], skip_special_tokens=True).strip()


def cmd_eval(args):
    ss, chunks, qv = load_emb()
    test = [s for s in ss if s["split"] == "test"]
    random.Random(3).shuffle(test)
    test = test[: args.n]
    pool = [s for s in ss if s["split"] == "test"]
    pool = pool if len(pool) > 40 else ss  # smoke test on a small subset
    st = Stack(args.variant)
    conds = [c for c in CONDITIONS if not args.conditions or c[0] in args.conditions]
    path = f"eval_{args.variant}.json"
    prev = json.loads((OUT / path).read_text()) if (OUT / path).exists() and args.resume else {"rows": []}
    done = {(r["session"], r["condition"]) for r in prev["rows"]}
    rows, t0 = prev["rows"], time.perf_counter()
    for i, s in enumerate(test):
        for name, mode, k, grouping, g, cf in conds:
            if (s["id"], name) in done:
                continue
            rng, nrng = random.Random(zlib.crc32(f"{s['id']}|{k}".encode())), np.random.default_rng(i)
            if mode == "single":
                q, qi, groups, _ = make_example(s, pool, rng, nrng, "single", g=g)
            else:
                q, qi, groups, _ = make_example(s, pool, rng, nrng, "anchored", g=g, k=k, grouping=grouping)
                if cf:  # same distractors, target session's chunks removed
                    own = set(s["rows"].tolist())
                    groups = [x for x in ([r for r in grp if r not in own] for grp in groups) if len(x)]
                    groups = [np.asarray(x) for x in groups]
            ans = generate(st, q, st.memory(chunks, groups, qv[qi]), args.max_new_tokens)
            rows.append({"session": s["id"], "condition": name, "n_tokens": SLOTS * len(groups), "answer": ans,
                         **score(s, ans, mode == "anchored")})
        torch.mps.empty_cache()
        if (i + 1) % 5 == 0 or i + 1 == len(test):
            save(path, {"variant": args.variant, "rows": rows, "table": table(rows)})
            print(f"{i + 1}/{len(test)} sessions  {(time.perf_counter() - t0) / 60:.0f} min", flush=True)
    for k, v in table(rows).items():
        print(f"{k:40s} " + "  ".join(f"{m} {x:.2f}" for m, x in v.items() if m != "n") + f"  n={v['n']}")


def table(rows):
    out = {}
    for name, *_ in CONDITIONS:
        rs = [r for r in rows if r["condition"] == name]
        if not rs:
            continue
        t = {"n": len(rs)}
        for m in ("repo", "issue", "files_path", "files_name", "funcs"):
            vals = [r[m] for r in rs if r.get(m) is not None]
            if vals:
                t[m] = float(np.mean(vals))
        out[name] = t
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--per_instance", type=int, default=1)
    e = sub.add_parser("embed")
    e.add_argument("--batch", type=int, default=8)
    e.add_argument("--limit", type=int, default=0)
    t = sub.add_parser("train")
    t.add_argument("--rank", type=int, default=16)
    t.add_argument("--lr", type=float, default=2e-4)
    t.add_argument("--warmup", type=int, default=20)
    t.add_argument("--updates", type=int, default=1500)
    t.add_argument("--accum", type=int, default=4)
    t.add_argument("--eval_every", type=int, default=50)
    t.add_argument("--patience", type=int, default=4)
    t.add_argument("--n_val", type=int, default=48)
    t.add_argument("--max_hours", type=float, default=4.0)
    t.add_argument("--seed", type=int, default=0)
    v = sub.add_parser("eval")
    v.add_argument("--variant", choices=["base", "lora"], required=True)
    v.add_argument("--n", type=int, default=80)
    v.add_argument("--max_new_tokens", type=int, default=72)
    v.add_argument("--conditions", nargs="*")
    v.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    {"build": cmd_build, "embed": cmd_embed, "train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
