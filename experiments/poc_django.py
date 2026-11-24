"""H16: does the 16-token memory hold up at codebase scale? One fact per class / function / method of the Django 5.2
source tree (tests included), all behind one 16-token block. The source is read as DATA only (ast.parse, never
imported or run). Django is public and the base model knows a lot of it, so every score is paired with the same
question asked with the fact taken out of the store.

    facts   CPU: parse the tree -> django_facts.json (fact text, name, path, docstring line, split by file)
    embed   vectors for every fact and question -> django_emb.npz
    cos     CPU: does plain cosine rank the asked fact first among all of them?
    para    Qwen3-8B rewrites the test docstring lines as a developer's question (name not given)
    train   reader LoRA only, on facts of the training files (init lora_gen.pt)
    eval    held-out files: name and file-path recall with 1 / 1,000 / 10,000 / all facts in the store, and with
            the asked fact removed. Lines starting DJ.

At this size the key / value projections of the core's two attention layers are computed once for the whole store
(they do not depend on the question), which is also how a real store would keep them.
"""

import argparse
import ast
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

ROOT_NAME = "django-5.2"
Q_NAME = 'In the Django codebase, what is the name of the {kind} described as: "{doc}"? Answer with the name only.'
Q_PATH = 'In the Django codebase, which file defines the {kind} described as: "{doc}"? Answer with the file path only.'
Q_PARA = "In the Django codebase: {q} Answer with the name only."
PARA_SYSTEM = "You rewrite code documentation as questions a developer would ask."
PARA_USER = """Docstring of a {kind} in a Python web framework: "{doc}"

Write one short question a developer might ask when looking for this {kind} without knowing its name, for example "Which {kind} ...?" or "What do we call the {kind} that ...?". Use your own words, do not copy the docstring, and do not guess a name. Write only the question."""


def out_dir():
    import pv4_port as P
    d = P.SCRATCH / "runs/django"
    d.mkdir(parents=True, exist_ok=True)
    return d


def first_sentence(doc):
    doc = " ".join((doc or "").split())
    m = re.match(r"(.+?[.!?])(\s|$)", doc)
    return (m.group(1) if m else doc).strip()


def cmd_facts(args):
    import pv4_port as P
    root = P.SCRATCH / "ext/django" / ROOT_NAME
    facts, bad = [], 0
    for path in sorted(root.rglob("*.py")):
        if path.is_symlink() or not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if not (rel.startswith("django/") or rel.startswith("tests/")):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except (SyntaxError, ValueError):
            bad += 1
            continue
        split = "test" if int(hashlib.md5(rel.encode()).hexdigest(), 16) % 5 == 0 else "train"

        def visit(node, prefix, in_class):
            for ch in node.body:
                if isinstance(ch, ast.ClassDef):
                    add(ch, "class", prefix + ch.name, "")
                    visit(ch, prefix + ch.name + ".", True)
                elif isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    a = [x.arg for x in ch.args.posonlyargs + ch.args.args if x.arg not in ("self", "cls")][:4]
                    add(ch, "method" if in_class else "function", prefix + ch.name, "(" + ", ".join(a) + ")")

        def add(ch, kind, qual, sig):
            doc = first_sentence(ast.get_docstring(ch))[:220]
            text = f"{rel}: {kind} {qual}{sig}." + (f" {doc}" if doc else "")
            name = ch.name
            spoken = name.replace("_", " ").strip().lower()
            ask = (rel.startswith("django/") and 30 <= len(doc) <= 200 and len(name) >= 5 and not name.startswith("__")
                   and name.lower() not in doc.lower() and spoken not in doc.lower())
            facts.append({"path": rel, "kind": kind, "qual": qual, "name": name, "doc": doc, "text": text, "split": split, "ask": ask})

        visit(tree, "", False)
    for i, f in enumerate(facts):
        f["row"] = i
    (out_dir() / "django_facts.json").write_text(json.dumps(facts))
    n = len(facts)
    src = sum(f["path"].startswith("django/") for f in facts)
    ask = [f for f in facts if f["ask"]]
    print(f"facts {n} (django/ {src}, tests/ {n - src}), unparsed files {bad}")
    print(f"askable {len(ask)}: train files {sum(f['split'] == 'train' for f in ask)}, held-out files {sum(f['split'] == 'test' for f in ask)}")
    names = {}
    for f in facts:
        names.setdefault(f["name"], 0)
        names[f["name"]] += 1
    print(f"distinct names {len(names)}; most repeated: {sorted(names.items(), key=lambda x: -x[1])[:8]}")
    for f in random.Random(0).sample(ask, 4):
        print("SAMPLE", f["text"], "| Q:", Q_NAME.format(kind=f["kind"], doc=f["doc"]))


def load():
    facts = json.loads((out_dir() / "django_facts.json").read_text())
    emb = out_dir() / "django_emb.npz"
    if not emb.exists():
        return facts, None, None
    z = np.load(emb)
    qmap = {str(t): i for i, t in enumerate(z["q_text"])}
    return facts, z["facts"], (z["q"], qmap)


def test_items(facts, n):
    items = [f for f in facts if f["ask"] and f["split"] == "test"]
    random.Random(7).shuffle(items)
    return items[:n]


def questions(f):
    return Q_NAME.format(kind=f["kind"], doc=f["doc"]), Q_PATH.format(kind=f["kind"], doc=f["doc"])


def cmd_embed(args):
    import torch
    import demos_port as D
    facts, _, _ = load()
    train = [f for f in facts if f["ask"] and f["split"] == "train"]
    random.Random(1).shuffle(train)
    ask = train[: args.train_q] + test_items(facts, args.test_q)
    qs = sorted({q for f in ask for q in questions(f)})
    enc = D.Encoder()
    t0 = time.perf_counter()

    def run(texts, tag):
        order = np.argsort([len(t) for t in texts])
        out = np.zeros((len(texts), 2560), np.float16)
        for b in range(0, len(texts), 2048):
            idx = order[b:b + 2048]
            out[idx] = enc([texts[i] for i in idx], batch_size=64).astype(np.float16)
            torch.mps.empty_cache()
            print(f"{tag} {b + len(idx)}/{len(texts)}  {time.perf_counter() - t0:.0f}s", flush=True)
        return out

    fv = run([f["text"] for f in facts], "facts")
    qv = run(qs, "questions")
    np.savez(out_dir() / "django_emb.npz", facts=fv, q=qv, q_text=np.array(qs))
    print("done", flush=True)


def cmd_para(args):
    import torch
    import demos_port as D
    import poc_clue as K
    from poc_clue import P, apply_chat_template
    facts, _, _ = load()
    items = test_items(facts, args.n)
    runner = D.Runner([], K.DECODER)
    model, tok = runner.model, runner.tok
    tok.padding_side = "left"
    out = {}
    for b in range(0, len(items), 8):
        batch = items[b:b + 8]
        texts = [apply_chat_template(tok, [{"role": "system", "content": PARA_SYSTEM},
                                           {"role": "user", "content": PARA_USER.format(kind=f["kind"], doc=f["doc"])}],
                                     add_generation_prompt=True) for f in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(P.DEV)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=60, do_sample=False, pad_token_id=tok.pad_token_id)
        D.sync()
        for f, x in zip(batch, o):
            q = re.sub(r"<think>.*?</think>", "", tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True), flags=re.S)
            q = q.strip().splitlines()[0].strip().strip('"') if q.strip() else ""
            if q and f["name"].lower() not in q.lower():
                out[str(f["row"])] = q
        del enc, o
        torch.mps.empty_cache()
        print(f"{b + len(batch)}/{len(items)}", flush=True)
    (out_dir() / "django_para.json").write_text(json.dumps(out, indent=1))
    print(f"paraphrased {len(out)} of {len(items)}", flush=True)
    for r, q in list(out.items())[:4]:
        print("SAMPLE", facts[int(r)]["name"], "|", facts[int(r)]["doc"], "->", q, flush=True)


def cmd_paraemb(args):
    import demos_port as D
    out = json.loads((out_dir() / "django_para.json").read_text())
    qv = D.Encoder()([Q_PARA.format(q=q) for q in out.values()], batch_size=32).astype(np.float16)
    np.savez(out_dir() / "django_para_emb.npz", q=qv, row=np.array([int(r) for r in out]))
    print(f"embedded {len(out)}", flush=True)


def cmd_cos(args):
    facts, F, (Q, qmap) = load()
    Fn = F.astype(np.float32)
    Fn /= np.linalg.norm(Fn, axis=1, keepdims=True)
    items = test_items(facts, args.n)
    pz = out_dir() / "django_para_emb.npz"
    para = {int(r): v for r, v in zip(np.load(pz)["row"], np.load(pz)["q"])} if pz.exists() else {}
    for tag, pick in (("name question", 0), ("path question", 1), ("paraphrased question", None)):
        ranks, same = [], []
        for f in items:
            if pick is None and f["row"] not in para:
                continue
            q = (para[f["row"]] if pick is None else Q[qmap[questions(f)[pick]]]).astype(np.float32)
            sc = Fn @ (q / np.linalg.norm(q))
            ranks.append(int((sc > sc[f["row"]]).sum()) + 1)
            same.append(facts[int(sc.argmax())]["name"] == f["name"])
        r = np.array(ranks)
        print(f"DJCOS {tag:22s} n={len(r)} among {len(facts)} facts: asked fact ranked first {np.mean(r == 1):.2f}, top 5 {np.mean(r <= 5):.2f}, "
              f"top 50 {np.mean(r <= 50):.2f}, median rank {int(np.median(r))}; top fact has the right name {np.mean(same):.2f}", flush=True)


class Store:
    """Key / value projections of both cross-attention layers for every fact, computed once. forward() repeats
    poc_core.core_fwd + memory() (question conditioning, similarity bias beta * cos) on any subset of rows."""

    def __init__(self, core, F, beta, torch, P):
        self.core, self.torch, self.P, self.beta = core, torch, P, beta
        self.h, self.dim = core.config.num_heads, F.shape[1]
        assert not core.config.retrieval_init and core.config.num_self_attn_layers == 0
        self.cos, self.k, self.v = [], [[] for _ in core.cross_attn_layers], [[] for _ in core.cross_attn_layers]
        lin = torch.nn.functional.linear
        with torch.no_grad():
            for b in range(0, len(F), 8192):
                d = torch.from_numpy(F[b:b + 8192].astype(np.float32)).to(P.DEV)
                self.cos.append(torch.nn.functional.normalize(d, dim=-1).half())
                docs = core.doc_norm(d)
                for li, attn in enumerate(core.cross_attn_layers):
                    W, bb, dim = attn.in_proj_weight, attn.in_proj_bias, self.dim
                    self.k[li].append(lin(docs, W[dim:2 * dim], bb[dim:2 * dim]).half())
                    self.v[li].append(lin(docs, W[2 * dim:], bb[2 * dim:]).half())
                del d, docs
                torch.mps.empty_cache()
        self.cos = torch.cat(self.cos)
        self.k = [torch.cat(x) for x in self.k]
        self.v = [torch.cat(x) for x in self.v]

    def forward(self, rows, qv, want_attn=False):
        torch, core, h, dim = self.torch, self.core, self.h, self.dim
        idx = None if rows is None else torch.as_tensor(np.asarray(rows), device=self.P.DEV)
        sel = (lambda t: t) if idx is None else (lambda t: t[idx])
        q = torch.from_numpy(qv.astype(np.float32))[None].to(self.P.DEV)
        question = q.unsqueeze(1)
        slots = core.memory_queries.expand(1, -1, -1)
        cond, _ = core.query_cond_attn(slots, question, question)
        slots = core.query_cond_norm(slots + cond)
        cos = sel(self.cos).float() @ torch.nn.functional.normalize(q[0], dim=-1)
        attns = []
        for li, (norm, attn, fnorm, ffn) in enumerate(zip(core.cross_attn_norms, core.cross_attn_layers, core.ffn_norms, core.ffn_layers)):
            x = norm(slots)
            W, b = attn.in_proj_weight, attn.in_proj_bias
            qh = torch.nn.functional.linear(x, W[:dim], b[:dim]).view(16, h, dim // h).transpose(0, 1)  # [h, 16, dh]
            kh = sel(self.k[li]).float().view(-1, h, dim // h).transpose(0, 1)  # [h, n, dh]
            scores = qh @ kh.transpose(-1, -2) * (dim // h) ** -0.5 + self.beta[li][:, None, None] * cos[None, None, :]
            w = torch.softmax(scores, dim=-1)
            del kh, scores
            vh = sel(self.v[li]).float().view(-1, h, dim // h).transpose(0, 1)
            att = attn.out_proj((w @ vh).transpose(0, 1).reshape(1, 16, dim))
            del vh
            if want_attn:
                attns.append(w.mean(0).float().cpu())  # [16, n]
            slots = slots + att
            slots = slots + ffn(fnorm(slots))
        out = core.memory_projection(core.layer_norm(slots))
        return (out, attns) if want_attn else out


def hit_name(f, ans):
    return float(re.search(r"(?<![A-Za-z0-9_])" + re.escape(f["name"]) + r"(?![A-Za-z0-9_])", ans) is not None)


def cmd_selftest(args):
    """The precomputed store must give the same block as poc_core.memory()."""
    import torch
    import poc_agentmem as A
    import poc_core as C
    from poc_agentmem import P, OUT
    facts, F, (Q, qmap) = load()
    st = A.Stack("lora", train=True)
    core = st.core.float().eval()
    beta = torch.load(OUT / args.core, map_location="cpu")["beta"].float().to(P.DEV)
    sub = F[:3000]
    store = Store(core, sub, beta, torch, P)
    rows = np.random.default_rng(0).permutation(3000)[:500]
    with torch.no_grad():
        a = store.forward(rows, Q[0])
        b = C.memory(core, rows, sub, sub, Q[0], beta)
    diff, scale = float((a - b).abs().max()), float(b.abs().mean())
    print(f"SELFTEST max abs diff {diff:.4f}  mean abs {scale:.4f}", flush=True)
    assert diff < 0.05 * scale, "precomputed store disagrees with poc_core.memory"


def cmd_train(args):
    import torch
    import poc_agentmem as A
    import poc_core as C
    import poc_general as G
    from poc_agentmem import P, OUT
    import poc_fast as FB
    torch.manual_seed(0)
    facts, F, (Q, qmap) = load()
    train_rows = np.array([f["row"] for f in facts if f["split"] == "train"])
    ask = [f for f in facts if f["ask"] and f["split"] == "train" and questions(f)[0] in qmap]
    print(f"training questions: {len(ask)} facts x 2 (name, path); store rows for training {len(train_rows)}", flush=True)
    people, PF, PQ = G.load()
    ptrain = [p for p in people if p["split"] == "train"]
    st = A.Stack("lora", train=True)
    C.load_lora(st, args.init)
    core = st.core.float().eval()
    beta = torch.load(OUT / args.core, map_location="cpu")["beta"].float().to(P.DEV)
    store = Store(core, F, beta, torch, P)
    lps = [t for m in st.lora for t in (m.A, m.B)]
    opt = torch.optim.AdamW(lps, lr=args.lr, weight_decay=0.0)
    rng, nrng = random.Random(0), np.random.default_rng(0)
    dt = next(st.r.model.parameters()).dtype
    sizes, w = [1, 8, 64, 512, 4096, 20000], [0.1, 0.15, 0.2, 0.2, 0.2, 0.15]

    def item():
        with torch.no_grad():
            if rng.random() < 0.15:  # keep the people-fact skill
                p = rng.choice(ptrain)
                j = rng.randrange(len(p["facts"]))
                others = rng.sample([o for o in ptrain if o is not p], rng.choice([1, 4, 16]) - 1)
                rows = nrng.permutation(G.build_pile(p, others))
                mem = C.memory(core, rows, PF, PF, PQ[p["rows"][j]], beta)
                question, answer = p["facts"][j]["question"], p["facts"][j]["value"]
            else:
                f = rng.choice(ask)
                kind = rng.randrange(2)
                question = questions(f)[kind]
                answer = f["name"] if kind == 0 else f["path"]
                n = min(rng.choices(sizes, w)[0], len(train_rows))
                rows = nrng.choice(train_rows, n, replace=False)
                rows = np.unique(np.append(rows[rows != f["row"]][: n - 1], f["row"]))
                mem = store.forward(rows, Q[qmap[question]])
        return question, mem.to(dt), answer

    st.r.model.train()
    t0, u, run = time.perf_counter(), 0, []
    budget, next_save = args.minutes * 60, 1200
    while time.perf_counter() - t0 < budget:
        frac = min(1.0, (time.perf_counter() - t0) / budget)
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (u + 1) / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        items = [item() for _ in range(args.batch)]
        if args.single:  # old path: one forward/backward per example
            for it in items:
                loss = FB.loss_batch(st, [it]) / len(items)
                loss.backward()
                run.append(loss.item() * len(items))
        else:
            loss = FB.loss_batch(st, items)
            loss.backward()
            run.append(loss.item())
        torch.nn.utils.clip_grad_norm_(lps, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        u += 1
        torch.mps.empty_cache()
        if u % 25 == 0:
            print(f"update {u}  examples {u * args.batch}  loss {np.mean(run[-(100 if args.single else 25):]):.3f}  {(time.perf_counter() - t0) / 60:.0f} min", flush=True)
        if time.perf_counter() - t0 > next_save:
            next_save += 1200
            torch.save(st.state(), OUT / f"lora_{args.tag}.pt")
    torch.save(st.state(), OUT / f"lora_{args.tag}.pt")
    (out_dir() / f"django_train_{args.tag}.json").write_text(json.dumps(
        {"updates": u, "examples": u * args.batch, "batch": args.batch, "loss": float(np.mean(run[-25:])), "minutes": args.minutes,
         "questions": len(ask) * 2}))
    print("saved", flush=True)


def cmd_eval(args):
    import torch
    import poc_agentmem as A
    import poc_core as C
    import poc_fast as FB
    from poc_agentmem import P, OUT
    facts, F, (Q, qmap) = load()
    items = test_items(facts, args.n)
    pp = out_dir() / "django_para.json"
    para = json.loads(pp.read_text()) if pp.exists() else {}
    pz = np.load(out_dir() / "django_para_emb.npz") if para else None
    pvec = {int(r): v for r, v in zip(pz["row"], pz["q"])} if para else {}
    st = A.Stack("lora", train=True)
    C.load_lora(st, args.lora, eval_mode=True)
    core = st.core.float().eval()
    beta = torch.load(OUT / args.core, map_location="cpu")["beta"].float().to(P.DEV)
    st.r.model.eval()
    dt = next(st.r.model.parameters()).dtype
    store = Store(core, F, beta, torch, P)
    n_all = len(facts)
    sizes = [int(x) for x in args.sizes.split(",")]
    rows_out, t0 = [], time.perf_counter()

    pending = []

    def ask(f, cond, kind, question, qv, rows):
        """rows: None = whole store; else row indices. Queues one generation; flush() runs the queue."""
        mem, attns = store.forward(rows, qv, want_attn=True)
        pos = [f["row"]] if rows is None else np.where(rows == f["row"])[0].tolist()
        a = attns[1][:, pos].sum(-1).mean() if pos else 0.0
        pending.append((question, mem.to(dt), {"row": f["row"], "name": f["name"], "path": f["path"], "condition": cond,
                                               "question_kind": kind, "n_facts": n_all if rows is None else int(len(rows)),
                                               "attn_asked": float(a)}, f))

    def flush():
        for is_path, max_new in ((False, 16), (True, 28)):
            todo = [x for x in pending if (x[2]["question_kind"] == "path") == is_path]
            if not todo:
                continue
            if args.single:
                answers = [A.generate(st, q, m, max_new).strip() for q, m, _, _ in todo]
            else:
                answers = FB.generate_batch(st, [(q, m) for q, m, _, _ in todo], max_new)
            for (q, m, r, f), ans in zip(todo, answers):
                rows_out.append({**r, "answer": ans, "hit": hit_name(f, ans) if not is_path else float(f["path"] in ans)})
        pending.clear()

    with torch.no_grad():
        for i, f in enumerate(items):
            qn, qp = questions(f)
            vn, vp = Q[qmap[qn]], Q[qmap[qp]]
            r2 = np.random.default_rng(i)
            others = r2.permutation(np.delete(np.arange(n_all), f["row"]))
            for n in sizes:
                rows = np.sort(np.append(others[: n - 1], f["row"]))
                ask(f, f"{n} facts", "name", qn, vn, rows)
            ask(f, "all facts", "name", qn, vn, None)
            removed = np.delete(np.arange(n_all), f["row"])
            ask(f, "all facts, asked fact removed", "name", qn, vn, removed)
            ask(f, "all facts", "path", qp, vp, None)
            ask(f, "all facts, asked fact removed", "path", qp, vp, removed)
            if f["row"] in pvec:
                q = Q_PARA.format(q=para[str(f["row"])])
                ask(f, "all facts", "paraphrased", q, pvec[f["row"]], None)
                ask(f, "all facts, asked fact removed", "paraphrased", q, pvec[f["row"]], removed)
            flush()
            torch.mps.empty_cache()
            if (i + 1) % 10 == 0 or i + 1 == len(items):
                (out_dir() / args.out).write_text(json.dumps({"lora": args.lora, "n_facts": n_all, "rows": rows_out}, indent=1))
                print(f"{i + 1}/{len(items)}  {(time.perf_counter() - t0) / 60:.0f} min", flush=True)
    tab = {}
    for r in rows_out:
        tab.setdefault((r["question_kind"], r["condition"]), []).append(r)
    for (kind, cond), rs in tab.items():
        print(f"DJ {kind:11s} {cond:30s} n={len(rs):3d} facts {np.mean([r['n_facts'] for r in rs]):6.0f}  correct {np.mean([r['hit'] for r in rs]):.2f}  "
              f"attention on asked fact {np.mean([r['attn_asked'] for r in rs]):.2f}", flush=True)
    for r in [r for r in rows_out if r["condition"] == "all facts" and not r["hit"]][:12]:
        print("DJMISS", r["question_kind"], "| truth", r["name"] if r["question_kind"] != "path" else r["path"], "| answer", r["answer"][:80], flush=True)


def cmd_textrag(args):
    """Baseline: ordinary retrieval. The top-k facts by cosine are pasted as TEXT into the plain decoder (no memory
    tokens, no LoRA). k = 0 is the decoder's own knowledge of Django. Lines starting DJTEXT."""
    import torch
    import demos_port as D
    import poc_clue as K
    from poc_clue import P, apply_chat_template
    facts, F, (Q, qmap) = load()
    Fn = F.astype(np.float32)
    Fn /= np.linalg.norm(Fn, axis=1, keepdims=True)
    items = test_items(facts, args.n)
    para = json.loads((out_dir() / "django_para.json").read_text())
    pz = np.load(out_dir() / "django_para_emb.npz")
    pvec = {int(r): v for r, v in zip(pz["row"], pz["q"])}
    runner = D.Runner([], K.DECODER)
    model, tok = runner.model, runner.tok
    tok.padding_side = "left"
    jobs = []  # (fact, question kind, k, prompt, context tokens, asked fact among the pasted ones)
    for f in items:
        qn, qp = questions(f)
        todo = [("name", qn, Q[qmap[qn]], (0, 1, 3, 10)), ("path", qp, Q[qmap[qp]], (0, 3))]
        if f["row"] in pvec:
            todo.append(("paraphrased", Q_PARA.format(q=para[str(f["row"])]), pvec[f["row"]], (0, 1, 3, 10)))
        for kind, q, v, ks in todo:
            v = v.astype(np.float32)
            top = np.argsort(-(Fn @ (v / np.linalg.norm(v))))[:10]
            for k in ks:
                ctx = "Notes about the codebase:\n" + "\n".join("- " + facts[int(i)]["text"] for i in top[:k]) + "\n\n" if k else ""
                jobs.append((f, kind, k, ctx + q, len(tok.encode(ctx, add_special_tokens=False)) if k else 0, bool(f["row"] in top[:k])))
    order = sorted(range(len(jobs)), key=lambda i: len(jobs[i][3]))
    rows, t0 = [], time.perf_counter()
    for b in range(0, len(order), 8):
        batch = [jobs[i] for i in order[b:b + 8]]
        texts = [apply_chat_template(tok, [{"role": "user", "content": j[3]}], add_generation_prompt=True) for j in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(P.DEV)
        with torch.no_grad():
            o = model.generate(**enc, max_new_tokens=28, do_sample=False, pad_token_id=tok.pad_token_id)
        D.sync()
        for (f, kind, k, _, ntok, has), x in zip(batch, o):
            ans = re.sub(r"<think>.*?</think>", "", tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True), flags=re.S).strip()
            rows.append({"row": f["row"], "name": f["name"], "path": f["path"], "question_kind": kind, "k": k, "context_tokens": ntok,
                         "retrieved": has, "answer": ans, "hit": hit_name(f, ans) if kind != "path" else float(f["path"] in ans)})
        del enc, o
        torch.mps.empty_cache()
        if (b // 8) % 10 == 0:
            print(f"{b + len(batch)}/{len(jobs)}  {(time.perf_counter() - t0) / 60:.0f} min", flush=True)
    (out_dir() / "django_textrag.json").write_text(json.dumps({"rows": rows}, indent=1))
    tab = {}
    for r in rows:
        tab.setdefault((r["question_kind"], r["k"]), []).append(r)
    for (kind, k), rs in sorted(tab.items()):
        print(f"DJTEXT {kind:11s} top {k:2d} facts as text  n={len(rs):3d}  context tokens {np.mean([r['context_tokens'] for r in rs]):5.0f}  "
              f"correct {np.mean([r['hit'] for r in rs]):.2f}  asked fact among them {np.mean([r['retrieved'] for r in rs]):.2f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("facts")
    sub.add_parser("paraemb")
    x = sub.add_parser("textrag")
    x.add_argument("--n", type=int, default=120)
    e = sub.add_parser("embed")
    e.add_argument("--train-q", type=int, default=3000)
    e.add_argument("--test-q", type=int, default=300)
    p = sub.add_parser("para")
    p.add_argument("--n", type=int, default=120)
    c = sub.add_parser("cos")
    c.add_argument("--n", type=int, default=300)
    s = sub.add_parser("selftest")
    s.add_argument("--core", default="core_gen.pt")
    t = sub.add_parser("train")
    t.add_argument("--minutes", type=float, default=40)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--init", default="lora_gen.pt")
    t.add_argument("--core", default="core_gen.pt")
    t.add_argument("--tag", default="dj")
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--single", action="store_true")
    v = sub.add_parser("eval")
    v.add_argument("--n", type=int, default=120)
    v.add_argument("--lora", default="lora_gen.pt")
    v.add_argument("--core", default="core_gen.pt")
    v.add_argument("--sizes", default="1,1000,10000")
    v.add_argument("--out", default="django_eval.json")
    v.add_argument("--single", action="store_true")
    args = ap.parse_args()
    {"facts": cmd_facts, "embed": cmd_embed, "para": cmd_para, "paraemb": cmd_paraemb, "textrag": cmd_textrag, "cos": cmd_cos, "selftest": cmd_selftest,
     "train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
