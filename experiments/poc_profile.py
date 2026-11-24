"""H15: always-on profile. Picking a buried clue by similarity fails (H14), but a person's lasting facts are few. So keep
them in memory always, a few tokens per fact (H10 layout: the first k slots of each fact's block, no pooling), and
train the reader to ACT on them when an open request arrives ("What should I pack for a long day hike?").

Until now the reader acted on a clue only for clue types it was trained on (8 types). Here the training set is widened
with generated (fact, request, answer) triples for every stored fact that is not from a held-out clue type.

    gen     Qwen3-8B: for each training fact, 2 requests where the fact matters but is not mentioned; then a teacher
            answer (fact given as text) and a generic answer (no fact) per request -> profile_gen.json
    embed   vectors for the requests and for the clean clue facts from poc_config.json -> profile_emb.npz
    train   reader LoRA only (init lora_gen.pt); 80% profile examples, 20% people facts (H13) -> lora_prof.pt
    eval    held-out clue types and held-out chats of trained types: clue fact alone, in a profile of 4 / 8 / 16 facts,
            and a profile of 8 without it (false alarms). Lines starting PROF.

Clue fact = the note writer's stored fact when it holds the clue (H14), else the clean fact from the config.
"""

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

REQ_SYSTEM = "You write realistic short messages that a user sends to a personal assistant."
REQ_USER = """Something the assistant knows about this user: {fact}

Write 2 different short requests (one sentence each) that this user might send on later, unrelated days. A good answer to each request has to take the fact above into account, but the request itself must not mention or hint at the fact. Write one request per line and nothing else."""
SUFFIX = "\n\nAnswer in one or two sentences."
INV_USER = """Invent 10 different lasting facts about 10 different imaginary users, all in this category: {cat}.

Each fact is one short, concrete sentence that starts with "The user". It should be the kind of thing that changes how a personal assistant ought to answer later requests: a constraint, a need, a circumstance or a strong preference. Write one fact per line and nothing else."""
CATS = """food preferences and diets; religion and beliefs; pets and animals at home; job and working hours; housing and
living situation; money and spending; hobbies and sports; languages and nationality; the place and weather where they
live; partner and relationships; elderly relatives they look after; phones, computers and gadgets they use; sleep and
daily routine; strong dislikes and things that scare them; studies and exams; allergies to things other than food;
hearing, eyesight and other physical limits; big events coming up in their life; ethical and lifestyle choices; things
they never learned to do; what their neighbourhood lacks; time zone and people they coordinate with"""
# anything close to a held-out clue type (health, no car, small children, flying) is kept out of training
BANNED = re.compile(r"diabet|insulin|blood sugar|glucose|migraine|sensitiv|\bcars?\b|driv|licen|toddler|\bkids?\b|child|\bsons?\b|"
                    r"daughter|baby|babies|newborn|infant|fl[yi]|flight|plane|airport|climate|carbon|emission|pregnan|vehicle", re.I)


def poc():
    import poc_clue as K
    return K


def mentions(text, rubrics):
    t = text.lower()
    return all(any(w in t for w in alt) for alt in rubrics["mentions"]["all_of"])


def load_facts():
    """Stored facts from H14. -> list of {doc, j, text, row}; vectors [n, dim]."""
    K = poc()
    z = np.load(K.OUT / "keys_emb.npz")
    keep = np.where(z["kind"] == "fact")[0]
    facts = [{"doc": str(z["doc"][i]), "j": int(z["fact"][i]), "text": str(z["text"][i]), "row": n} for n, i in enumerate(keep)]
    return facts, z["v"][keep]


def setup():
    """-> synth, facts, fact vectors, doc info, clue_of(pair) -> (vector, text, source)."""
    K = poc()
    s = K.load_synth()
    cfg = K.cfg()
    facts, fv = load_facts()
    info = {d["doc"]: d for d in s["docs"]}
    test_types = set(cfg["test_types"])
    held = {s["docs"][p["clue"]]["doc"] for p in s["pairs"] if p["split"] != "train"}  # clue chats of val / test pairs
    for f in facts:
        d = info.get(f["doc"])
        f["type"] = d["type"] if d else None
        f["train"] = f["doc"] not in held and not (d and d["kind"] == "clue" and d["type"] in test_types)
    by_doc = {}
    for f in facts:
        by_doc.setdefault(f["doc"], []).append(f)
    emb = K.OUT / "profile_emb.npz"
    ez = np.load(emb) if emb.exists() else None
    ev_, et_ = (ez["v"], ez["text"]) if ez is not None else ([], [])  # read once, not once per row
    emap = {str(t): ev_[i] for i, t in enumerate(et_)}
    gp = K.OUT / "profile_gen.json"
    if gp.exists() and emap:  # invented facts join the store for training only
        inv = [x for x in json.loads(gp.read_text())["invented"] if x in emap]
        facts += [{"doc": f"inv:{i}", "j": 0, "text": x, "row": len(fv) + i, "type": None, "train": True} for i, x in enumerate(inv)]
        if inv:
            fv = np.concatenate([fv, np.stack([emap[x] for x in inv]).astype(fv.dtype)])

    def clue_of(p):
        doc = s["docs"][p["clue"]]["doc"]
        for f in by_doc.get(doc, []):
            if mentions(f["text"], p["rubrics"]):
                return fv[f["row"]], f["text"], "note writer"
        text = clean_fact(cfg, p)
        return emap.get(text), text, "config"

    return s, cfg, facts, fv, info, clue_of, emap


def clean_fact(cfg, p):
    spec = cfg["types"][p["type"]]["values"][p["value"]]
    return "The user said: " + spec["fact"]


def cmd_gen(args):
    import torch
    K = poc()
    from poc_clue import D, P, PERSONA_PROMPT, apply_chat_template
    s, cfg, facts, fv, info, clue_of, _ = setup()
    train = [f for f in facts if f["train"]]
    runner = D.Runner([], K.DECODER)
    model, tok = runner.model, runner.tok
    tok.padding_side = "left"

    def run(messages, max_new):
        out_all = []
        for b in range(0, len(messages), args.batch):
            texts = [apply_chat_template(tok, m, add_generation_prompt=True) for m in messages[b:b + args.batch]]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(P.DEV)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            D.sync()
            out_all += [re.sub(r"<think>.*?</think>", "", tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True), flags=re.S).strip() for o in out]
            del enc, out
            torch.mps.empty_cache()
            print(f"  {len(out_all)}/{len(messages)}  {time.perf_counter() - t0:.0f}s", flush=True)
        return out_all

    t0 = time.perf_counter()
    clean = lambda x: x.strip().lstrip("-*0123456789. ").strip().strip('"')  # noqa: E731
    cats = [" ".join(c.split()) for c in CATS.split(";")]
    raw = run([[{"role": "user", "content": INV_USER.format(cat=c)}] for c in cats], 260)
    lines = [clean(x) for o in raw for x in o.splitlines()]
    cand = [x for x in dict.fromkeys(lines) if x.startswith("The user") and 20 < len(x) < 200]
    invented = [x for x in cand if not BANNED.search(x)]
    print(f"invented facts: {len(invented)} kept, {len(cand) - len(invented)} dropped as close to a held-out type", flush=True)
    for x in invented[::25]:
        print("INVENTED", x, flush=True)
    todo = [f["text"] for f in train] + invented
    raw = run([[{"role": "system", "content": REQ_SYSTEM}, {"role": "user", "content": REQ_USER.format(fact=x)}] for x in todo], 90)
    rows = []
    for x, o in zip(todo, raw):
        reqs = [clean(r) for r in o.splitlines() if len(r.strip()) > 12][:2]
        rows += [{"fact": x, "request": r, "first": i == 0} for i, r in enumerate(reqs)]
    print(f"requests: {len(rows)} for {len(todo)} facts", flush=True)
    ans = run([[{"role": "system", "content": PERSONA_PROMPT},
                {"role": "user", "content": D.user_message([r["fact"]], r["request"]) + SUFFIX}] for r in rows], 80)
    first = [r for r in rows if r["first"]]  # a no-fact answer for one request per fact is enough for negatives
    gen = run([[{"role": "system", "content": PERSONA_PROMPT}, {"role": "user", "content": r["request"] + SUFFIX}] for r in first], 80)
    for r, a in zip(rows, ans):
        r["answer"] = a
    for r, g in zip(first, gen):
        r["generic"] = g
    (K.OUT / "profile_gen.json").write_text(json.dumps({"invented": invented, "rows": rows}, indent=1))
    for r in rows[:3] + rows[-2:]:
        print("SAMPLE", json.dumps(r)[:700], flush=True)


def cmd_embed(args):
    import torch
    K = poc()
    from poc_clue import D
    s, cfg, *_ = setup()
    g = json.loads((K.OUT / "profile_gen.json").read_text())
    texts = sorted({r["request"] for r in g["rows"]} | set(g["invented"]) | {clean_fact(cfg, p) for p in s["pairs"]})
    enc = D.Encoder()
    out = []
    for b in range(0, len(texts), 512):
        out.append(enc(texts[b:b + 512], batch_size=32).astype(np.float16))
        torch.mps.empty_cache()
    np.savez(K.OUT / "profile_emb.npz", v=np.concatenate(out), text=np.array(texts))
    print(f"embedded {len(texts)}", flush=True)


def blocks(C, core, vecs, qv, k, P, torch):
    """Each fact through the untouched core on its own; keep the first k of its 16 slots. -> [1, n * k, dim]"""
    cdev = next(core.parameters()).device  # the CPU under MPORT_LEAN=1
    d = torch.from_numpy(np.asarray(vecs, dtype=np.float32))[:, None].to(cdev)
    q = torch.from_numpy(qv.astype(np.float32))[None].expand(d.size(0), -1).to(cdev)
    out = C.core_fwd(core, d, q)
    return out[:, :k].reshape(1, -1, out.size(-1)).to(P.DEV)


def cmd_train(args):
    import torch
    import poc_agentmem as A
    import poc_core as C
    import poc_general as G
    from poc_agentmem import P, OUT
    K = poc()
    torch.manual_seed(0)
    s, cfg, facts, fv, info, clue_of, emap = setup()
    _, qmap = K.load_emb()
    pairs = {p["pair"]: p for p in s["pairs"]}
    teach = json.loads((K.OUT / "teach.json").read_text())["rows"]
    gen = json.loads((K.OUT / "profile_gen.json").read_text())["rows"]
    train_facts = [f for f in facts if f["train"]]
    vec_of = {f["text"]: fv[f["row"]] for f in train_facts}
    ex = []  # (question, question vector, relevant fact vector or None, answer, doc or fact text to avoid, type to avoid)
    for row in teach:
        if row["split"] != "train" or not row["kept"]:
            continue
        p = pairs[row["pair"]]
        vec, _, _ = clue_of(p)
        doc = s["docs"][p["clue"]]["doc"]
        ex += [(p["question"], qmap[p["question"]], vec, a, doc, p["type"]) for a in row["kept"]]
        if row["twin_ok"]:
            ex.append((p["question"], qmap[p["question"]], None, row["twin_answer"], doc, p["type"]))
    n_teach = len(ex)
    for r in gen:
        if r["fact"] not in vec_of or r["request"] not in emap:
            continue
        ex.append((r["request"], emap[r["request"]], vec_of[r["fact"]], r["answer"], r["fact"], None))
        if "generic" in r:
            ex.append((r["request"], emap[r["request"]], None, r["generic"], r["fact"], None))
    print(f"examples: {n_teach} from teacher answers on 8 clue types, {len(ex) - n_teach} generated", flush=True)
    people, F, Q = G.load()
    ptrain = [p for p in people if p["split"] == "train"]
    st = A.Stack("lora", train=True)
    C.load_lora(st, args.init)
    core = st.core.float().eval()
    beta = torch.load(OUT / args.core, map_location="cpu")["beta"].float().to(P.DEV)
    lps = [t for m in st.lora for t in (m.A, m.B)]
    opt = torch.optim.AdamW(lps, lr=args.lr, weight_decay=0.0)
    rng, nrng = random.Random(0), np.random.default_rng(0)
    dt = next(st.r.model.parameters()).dtype

    def loss_one():
        with torch.no_grad():
            if rng.random() < 0.2:
                p = rng.choice(ptrain)
                j = rng.randrange(len(p["facts"]))
                others = rng.sample([o for o in ptrain if o is not p], rng.choice([1, 4, 16]) - 1)
                rows = nrng.permutation(G.build_pile(p, others))
                mem = C.memory(core, rows, F, F, Q[p["rows"][j]], beta)
                question, answer = p["facts"][j]["question"], p["facts"][j]["value"]
            else:
                question, qv, vec, answer, doc, typ = rng.choice(ex)
                n = rng.choice([1, 2, 4, 8])
                pool = [f for f in train_facts if doc not in (f["doc"], f["text"]) and (typ is None or f["type"] != typ)]
                others = [fv[f["row"]] for f in rng.sample(pool, n - (vec is not None))]
                vecs = others + ([vec] if vec is not None else [])
                if not vecs:
                    vecs = [fv[rng.choice(pool)["row"]]]
                rng.shuffle(vecs)
                k = rng.choice([4, 16]) if len(vecs) == 1 else 4
                mem = blocks(C, core, np.stack(vecs), qv, k, P, torch)
                question = question + SUFFIX
        mem = mem.to(dt)
        ids, labels = st.ids(question, mem.shape[1], answer)
        emb = P.input_embeds(st.r.model, st.r.tok, st.r.memory_id, ids, mem)
        return st.r.model(inputs_embeds=emb, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False).loss

    st.r.model.train()
    t0, u, run = time.perf_counter(), 0, []
    budget, next_save = args.minutes * 60, 1500
    while time.perf_counter() - t0 < budget:
        frac = min(1.0, (time.perf_counter() - t0) / budget)
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (u + 1) / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for _ in range(4):
            loss = loss_one() / 4
            loss.backward()
            run.append(loss.item() * 4)
        torch.nn.utils.clip_grad_norm_(lps, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        u += 1
        torch.mps.empty_cache()
        if u % 25 == 0:
            print(f"update {u}  examples {u * 4}  loss {np.mean(run[-100:]):.3f}  {(time.perf_counter() - t0) / 60:.0f} min", flush=True)
        if time.perf_counter() - t0 > next_save:
            next_save += 1500
            torch.save(st.state(), OUT / f"lora_{args.tag}.pt")
    torch.save(st.state(), OUT / f"lora_{args.tag}.pt")
    (K.OUT / f"profile_train_{args.tag}.json").write_text(json.dumps(
        {"updates": u, "examples": u * 4, "loss": float(np.mean(run[-100:])), "minutes": args.minutes,
         "teacher_examples": n_teach, "generated_examples": len(ex) - n_teach}))
    print("saved", flush=True)


def cmd_eval(args):
    import torch
    import poc_agentmem as A
    import poc_core as C
    from poc_agentmem import P
    K = poc()
    s, cfg, facts, fv, info, clue_of, emap = setup()
    _, qmap = K.load_emb()
    st = A.Stack("lora", train=True)
    C.load_lora(st, args.lora, eval_mode=True)
    core = st.core.float().eval()
    st.r.model.eval()
    dt = next(st.r.model.parameters()).dtype
    groups = [("held-out types", [p for p in s["pairs"] if p["split"] == "test"][:: args.stride]),
              ("trained types, new chats", [p for p in s["pairs"] if p["split"] == "val"][: args.n_val])]
    conds = [("clue fact alone, 16 tok", 1, 16, True), ("clue fact alone, 4 tok", 1, 4, True), ("profile of 4, 16 tok", 4, 4, True),
             ("profile of 8, 32 tok", 8, 4, True), ("profile of 16, 64 tok", 16, 4, True), ("profile of 8 without the clue", 8, 4, False)]
    rows, t0 = [], time.perf_counter()
    with torch.no_grad():
        for gname, ps in groups:
            for n, p in enumerate(ps):
                vec, text, source = clue_of(p)
                doc = s["docs"][p["clue"]]["doc"]
                pool = [f for f in facts if f["doc"] != doc and f["type"] != p["type"] and not f["doc"].startswith("inv:")]
                rng = random.Random(p["pair"])
                others = rng.sample(pool, 16)
                for cname, size, k, inc in conds:
                    vecs = [fv[f["row"]] for f in others[: size - inc]] + ([vec] if inc else [])
                    random.Random(p["pair"] + cname).shuffle(vecs)
                    mem = blocks(C, core, np.stack(vecs), qmap[p["question"]], k, P, torch).to(dt)
                    ans = A.generate(st, p["question"] + SUFFIX, mem, args.max_new_tokens).strip()
                    rows.append({"group": gname, "pair": p["pair"], "type": p["type"], "value": p["value"], "condition": cname,
                                 "fact": text, "fact_source": source, "answer": ans, **K.strict_score(ans, p["rubrics"], p["value"])})
                torch.mps.empty_cache()
                if (n + 1) % 6 == 0 or n + 1 == len(ps):
                    (K.OUT / args.out).write_text(json.dumps({"lora": args.lora, "rows": rows}, indent=1))
                    print(f"{gname} {n + 1}/{len(ps)}  {(time.perf_counter() - t0) / 60:.0f} min", flush=True)
    for gname, _ in groups:
        for cname, *_ in conds:
            x = [r for r in rows if r["group"] == gname and r["condition"] == cname]
            if not x:
                continue
            nw = [r for r in x if r["fact_source"] == "note writer"]
            print(f"PROF {gname:25s} {cname:30s} n={len(x):3d}  mentions {np.mean([r['mentions_strict'] for r in x]):.2f}  "
                  f"acts {np.mean([r['acts_strict'] for r in x]):.2f}  | note-writer facts only n={len(nw)} mentions "
                  f"{np.mean([r['mentions_strict'] for r in nw]) if nw else float('nan'):.2f}", flush=True)
    by = {}
    for r in rows:
        if r["group"] == "held-out types" and r["condition"] == "profile of 8, 32 tok":
            by.setdefault(r["type"], []).append(r["mentions_strict"])
    print("PROF held-out types, profile of 8, mentions by type: " + "  ".join(f"{t} {np.mean(v):.2f}" for t, v in sorted(by.items())), flush=True)


def cmd_text(args):
    """Control for H15: the same profiles and questions, but the facts are pasted as plain text into the prompt of the
    plain decoder (no core, no LoRA). Separates "soft tokens are too lossy" from "the reader was not trained enough"."""
    import torch
    K = poc()
    from poc_clue import D, P, PERSONA_PROMPT, apply_chat_template
    s, cfg, facts, fv, info, clue_of, emap = setup()
    runner = D.Runner([], K.DECODER)
    model, tok = runner.model, runner.tok
    tok.padding_side = "left"
    groups = [("held-out types", [p for p in s["pairs"] if p["split"] == "test"][:: args.stride]),
              ("trained types, new chats", [p for p in s["pairs"] if p["split"] == "val"][: args.n_val])]
    conds = [("text: clue fact alone", 1, True), ("text: profile of 8", 8, True), ("text: profile of 16", 16, True),
             ("text: profile of 8 without the clue", 8, False), ("text: no profile", 0, False)]
    rows, msgs = [], []
    for gname, ps in groups:
        for p in ps:
            vec, text, source = clue_of(p)
            doc = s["docs"][p["clue"]]["doc"]
            pool = [f for f in facts if f["doc"] != doc and f["type"] != p["type"] and not f["doc"].startswith("inv:")]
            others = random.Random(p["pair"]).sample(pool, 16)  # same distractors as cmd_eval
            for cname, size, inc in conds:
                lst = [f["text"] for f in others[: max(size - inc, 0)]] + ([text] if inc else [])
                random.Random(p["pair"] + cname).shuffle(lst)
                user = (D.user_message(lst, p["question"]) if lst else p["question"]) + SUFFIX
                msgs.append([{"role": "system", "content": PERSONA_PROMPT}, {"role": "user", "content": user}])
                rows.append({"group": gname, "pair": p["pair"], "type": p["type"], "value": p["value"], "condition": cname,
                             "fact": text, "fact_source": source,
                             "fact_tokens": len(tok("\n".join(lst), add_special_tokens=False)["input_ids"]) if lst else 0})
    t0 = time.perf_counter()
    for b in range(0, len(msgs), args.batch):
        texts = [apply_chat_template(tok, m, add_generation_prompt=True) for m in msgs[b:b + args.batch]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(P.DEV)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        D.sync()
        for r, o in zip(rows[b:b + args.batch], out):
            ans = re.sub(r"<think>.*?</think>", "", tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True), flags=re.S).strip()
            p = next(x for x in s["pairs"] if x["pair"] == r["pair"])
            r.update(answer=ans, **K.strict_score(ans, p["rubrics"], p["value"]))
        del enc, out
        torch.mps.empty_cache()
        if (b // args.batch) % 8 == 0:
            (K.OUT / args.out).write_text(json.dumps({"rows": rows[: b + args.batch]}, indent=1))
            print(f"{min(b + args.batch, len(msgs))}/{len(msgs)}  {(time.perf_counter() - t0) / 60:.0f} min", flush=True)
    (K.OUT / args.out).write_text(json.dumps({"rows": rows}, indent=1))
    for gname, _ in groups:
        for cname, *_ in conds:
            x = [r for r in rows if r["group"] == gname and r["condition"] == cname]
            if x:
                print(f"PROFTEXT {gname:25s} {cname:36s} n={len(x):3d}  mentions {np.mean([r['mentions_strict'] for r in x]):.2f}  "
                      f"acts {np.mean([r['acts_strict'] for r in x]):.2f}  fact tokens {np.mean([r['fact_tokens'] for r in x]):.0f}", flush=True)
    by = {}
    for r in rows:
        if r["condition"] == "text: profile of 8":
            by.setdefault(r["type"], []).append(r["acts_strict"])
    print("PROFTEXT profile of 8, acts by type: " + "  ".join(f"{t} {np.mean(v):.2f}" for t, v in sorted(by.items())), flush=True)


SEL_USER = """Facts known about the user:
{facts}

The user's new request: "{request}"

Which of these facts, if any, should change how an assistant answers this request? Reply with the fact numbers separated by commas, or NONE. Reply with numbers only."""


def cmd_select(args):
    """Two-step control: the plain decoder first reads the whole profile as a numbered list and names the facts that
    matter for the request, then answers with only those facts as text. Tests whether the model's own judgement (not
    similarity, not a graph) can link "no car" to "weekend getaway", and how that holds as the profile grows."""
    import torch
    K = poc()
    from poc_clue import D, P, PERSONA_PROMPT, apply_chat_template
    s, cfg, facts, fv, info, clue_of, emap = setup()
    runner = D.Runner([], K.DECODER)
    model, tok = runner.model, runner.tok
    tok.padding_side = "left"
    pairs = {p["pair"]: p for p in s["pairs"]}

    def run(msgs, max_new, label):
        outs, t0 = [], time.perf_counter()
        for b in range(0, len(msgs), args.batch):
            texts = [apply_chat_template(tok, m, add_generation_prompt=True) for m in msgs[b:b + args.batch]]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(P.DEV)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            D.sync()
            outs += [re.sub(r"<think>.*?</think>", "", tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True), flags=re.S).strip() for o in out]
            del enc, out
            torch.mps.empty_cache()
            if (b // args.batch) % 8 == 0:
                print(f"{label} {len(outs)}/{len(msgs)}  {(time.perf_counter() - t0) / 60:.0f} min", flush=True)
        return outs

    groups = [("held-out types", [p for p in s["pairs"] if p["split"] == "test"][:: args.stride]),
              ("trained types, new chats", [p for p in s["pairs"] if p["split"] == "val"][: args.n_val])]
    sizes = [int(x) for x in args.sizes.split(",")]
    rows = []
    for gname, ps in groups:
        for p in ps:
            vec, text, source = clue_of(p)
            doc = s["docs"][p["clue"]]["doc"]
            pool = [f for f in facts if f["doc"] != doc and f["type"] != p["type"] and not f["doc"].startswith("inv:")]
            others = random.Random(p["pair"]).sample(pool, 16)  # same first 16 distractors as cmd_eval
            rest = [f for f in pool if f not in others]
            others += random.Random(p["pair"] + "more").sample(rest, min(len(rest), max(sizes) - 16)) if max(sizes) > 16 else []
            for size in sizes:
                for inc in (True, False):
                    lst = [f["text"] for f in others[: size - inc]] + ([text] if inc else [])
                    random.Random(f"{p['pair']} {size} {inc}").shuffle(lst)
                    rows.append({"group": gname, "pair": p["pair"], "type": p["type"], "value": p["value"], "size": len(lst),
                                 "clue_in": inc, "clue_at": lst.index(text) + 1 if inc else 0, "facts": lst, "fact": text})
    raw = run([[{"role": "user", "content": SEL_USER.format(facts="\n".join(f"{i + 1}. {x}" for i, x in enumerate(r["facts"])),
                                                             request=pairs[r["pair"]]["question"])}] for r in rows], 24, "select")
    for r, o in zip(rows, raw):
        r["select_raw"] = o
        r["picked"] = sorted({int(x) for x in re.findall(r"\d+", o) if 1 <= int(x) <= r["size"]})
        r["clue_picked"] = bool(r["clue_in"] and r["clue_at"] in r["picked"])
    msgs = []
    for r in rows:
        chosen = [r["facts"][i - 1] for i in r["picked"]]
        q = pairs[r["pair"]]["question"]
        msgs.append([{"role": "system", "content": PERSONA_PROMPT},
                     {"role": "user", "content": (D.user_message(chosen, q) if chosen else q) + SUFFIX}])
    ans = run(msgs, args.max_new_tokens, "answer")
    for r, a in zip(rows, ans):
        p = pairs[r["pair"]]
        r.update(answer=a, **K.strict_score(a, p["rubrics"], p["value"]))
    (K.OUT / args.out).write_text(json.dumps({"rows": rows}, indent=1))
    for gname, _ in groups:
        for size in sizes:
            for inc in (True, False):
                x = [r for r in rows if r["group"] == gname and r["size"] == size and r["clue_in"] == inc]
                if x:
                    print(f"PROFSEL {gname:25s} profile of {size:3d} {'with the clue' if inc else 'clue removed':13s} n={len(x):3d}  "
                          f"clue picked {np.mean([r['clue_picked'] for r in x]):.2f}  facts picked {np.mean([len(r['picked']) for r in x]):.1f}  "
                          f"mentions {np.mean([r['mentions_strict'] for r in x]):.2f}  acts {np.mean([r['acts_strict'] for r in x]):.2f}", flush=True)
    by = {}
    for r in rows:
        if r["clue_in"] and r["size"] == sizes[-1]:
            by.setdefault(r["type"], []).append(r["clue_picked"])
    print(f"PROFSEL profile of {sizes[-1]}, clue picked by type: " + "  ".join(f"{t} {np.mean(v):.2f}" for t, v in sorted(by.items())), flush=True)
    for r in [r for r in rows if r["clue_in"] and r["size"] == sizes[-1]][:: 17][:6]:
        print("PROFSEL sample", json.dumps({k: r[k] for k in ("type", "fact", "clue_at", "select_raw", "answer")})[:500], flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--batch", type=int, default=8)
    sub.add_parser("embed")
    t = sub.add_parser("train")
    t.add_argument("--minutes", type=float, default=60)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--init", default="lora_gen.pt")
    t.add_argument("--core", default="core_gen.pt")
    t.add_argument("--tag", default="prof")
    e = sub.add_parser("eval")
    e.add_argument("--lora", default="lora_prof.pt")
    e.add_argument("--n-val", type=int, default=30)
    e.add_argument("--stride", type=int, default=1)
    e.add_argument("--max-new-tokens", type=int, default=70)
    e.add_argument("--out", default="profile_eval.json")
    x = sub.add_parser("text")
    x.add_argument("--n-val", type=int, default=30)
    x.add_argument("--stride", type=int, default=1)
    x.add_argument("--batch", type=int, default=8)
    x.add_argument("--max-new-tokens", type=int, default=70)
    x.add_argument("--out", default="profile_text.json")
    c = sub.add_parser("select")
    c.add_argument("--n-val", type=int, default=30)
    c.add_argument("--stride", type=int, default=1)
    c.add_argument("--batch", type=int, default=8)
    c.add_argument("--sizes", default="8,16,48")
    c.add_argument("--max-new-tokens", type=int, default=70)
    c.add_argument("--out", default="profile_select.json")
    args = ap.parse_args()
    {"gen": cmd_gen, "embed": cmd_embed, "train": cmd_train, "eval": cmd_eval, "text": cmd_text, "select": cmd_select}[args.cmd](args)


if __name__ == "__main__":
    main()
