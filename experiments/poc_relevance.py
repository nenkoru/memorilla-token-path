"""H17: can the decoder's own judgement, one fact at a time, pick the fact that matters for a request, and can that
judgement be distilled into a small map over the frozen vectors (to sit inside the core's attention instead of cosine)?

H14: similarity ranks the clue first in 1% (fair pool). H15 controls: the plain decoder reading a list of 8 facts acts
on the right one in 32-43%; asked to name the relevant numbers it includes the clue in 33-58%.

    teach     GPU: yes/no logit of the plain decoder for (request, fact) pairs. Ranks the clue inside profiles of
              8 / 16 / 48, answers with the top facts as text, and labels training pairs for `distill`.
    distill   CPU: trains a low-rank map score(q, f) over the frozen 2560-d vectors on those labels (and, for
              comparison, on the known pairs alone); reports the rank of the clue on held-out fact types.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import poc_profile as PF  # noqa: E402

YN_USER = """A fact known about the user: {fact}

The user's new request: "{request}"

Should an assistant take this fact into account when answering this request? Answer yes or no."""
SIZES = (8, 16, 48)


def eval_sets(s, facts, clue_of, n_val, stride):
    """Same questions and the same first distractors as poc_profile.py eval / text / select."""
    groups = [("held-out types", [p for p in s["pairs"] if p["split"] == "test"][:: stride]),
              ("trained types, new chats", [p for p in s["pairs"] if p["split"] == "val"][: n_val])]
    out = []
    for gname, ps in groups:
        for p in ps:
            vec, text, source = clue_of(p)
            doc = s["docs"][p["clue"]]["doc"]
            pool = [f for f in facts if f["doc"] != doc and f["type"] != p["type"] and not f["doc"].startswith("inv:")]
            others = random.Random(p["pair"]).sample(pool, 16)
            rest = [f for f in pool if f not in others]
            others += random.Random(p["pair"] + "more").sample(rest, min(len(rest), max(SIZES) - 16))
            out.append({"group": gname, "pair": p["pair"], "type": p["type"], "value": p["value"], "question": p["question"],
                        "clue": text, "others": [f["text"] for f in others]})
    return out


def train_sets(s, facts, clue_of, n_neg):
    """Requests with a known relevant fact (generated H15 data and the training pairs), each with random other facts."""
    K = PF.poc()
    gen = json.loads((K.OUT / "profile_gen.json").read_text())["rows"]
    pool = sorted({f["text"] for f in facts if f["train"]})
    known = [(r["request"], r["fact"]) for r in gen] + [(p["question"], clue_of(p)[1]) for p in s["pairs"] if p["split"] == "train"]
    out, seen = [], set()
    for request, fact in known:
        if request in seen:
            continue
        seen.add(request)
        neg = [x for x in random.Random(request).sample(pool, n_neg + 1) if x != fact][:n_neg]
        out.append({"request": request, "facts": [fact] + neg})
    return out


def yn_ids(tok):
    def ids(words):
        got = {tok(w, add_special_tokens=False)["input_ids"][0] for w in words if len(tok(w, add_special_tokens=False)["input_ids"]) == 1}
        return sorted(got)
    return ids(["yes", "Yes", " yes", " Yes", "YES"]), ids(["no", "No", " no", " No", "NO"])


def score_pairs(model, tok, pairs, batch, label):
    """pairs: [(request, fact)] -> yes-minus-no logit of the next token, one forward pass per batch."""
    import torch
    from poc_clue import P, apply_chat_template
    yes, no = yn_ids(tok)
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
    out, t0 = np.zeros(len(pairs), dtype=np.float32), time.perf_counter()
    for b in range(0, len(order), batch):
        idx = order[b:b + batch]
        texts = [apply_chat_template(tok, [{"role": "user", "content": YN_USER.format(fact=pairs[i][1], request=pairs[i][0])}],
                                     add_generation_prompt=True) for i in idx]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(P.DEV)
        with torch.no_grad():
            h = model.model(**enc)[0][:, -1]
            lg = model.lm_head(h).float()
        if b == 0:
            top = torch.topk(lg[0], 5).indices.tolist()
            print(f"{label} next-token check:", json.dumps([tok.decode([t]) for t in top]), "yes ids", yes, "no ids", no, flush=True)
        out[idx] = (torch.logsumexp(lg[:, yes], 1) - torch.logsumexp(lg[:, no], 1)).cpu().numpy()
        del enc, h, lg
        if (b // batch) % 20 == 0:
            print(f"{label} {min(b + batch, len(order))}/{len(order)}  {time.perf_counter() - t0:.0f}s", flush=True)
    torch.mps.empty_cache()
    print(f"{label} {len(pairs)} pairs in {time.perf_counter() - t0:.0f}s ({len(pairs) / (time.perf_counter() - t0):.0f} pairs/s)", flush=True)
    return out


def gen_batches(tok, texts, order, batch, max_new):
    """Fixed batches of `batch`, or, with MPORT_GEN_TOKENS set, as many prompts as fit that many
    tokens of prompt + answer (the KV cache), capped by MPORT_GEN_MAX (default 256)."""
    budget = int(os.environ.get("MPORT_GEN_TOKENS", "0"))
    if not budget:
        return [order[b:b + batch] for b in range(0, len(order), batch)]
    cap, out, cur, longest = int(os.environ.get("MPORT_GEN_MAX", "256")), [], [], 0
    for i in order:  # shortest first, so the newest prompt is the longest in the batch
        n = len(tok(texts[i], add_special_tokens=False)["input_ids"]) + max_new
        if cur and (max(longest, n) * (len(cur) + 1) > budget or len(cur) >= cap):
            out.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, n)
    return out + [cur] if cur else out


def vllm_generate(texts, max_new):
    """All prompts in one request to a vLLM server (MPORT_VLLM_URL, e.g. http://127.0.0.1:8137); greedy, like generate()."""
    import urllib.request
    url = os.environ["MPORT_VLLM_URL"].rstrip("/")
    body = {"model": os.environ.get("MPORT_VLLM_MODEL", "qwen3-8b"), "prompt": texts, "max_tokens": max_new,
            "temperature": 0.0, "skip_special_tokens": True}
    if os.environ.get("MPORT_VLLM_LORA"):
        body["model"] = os.environ["MPORT_VLLM_LORA"]
    req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        res = json.load(r)
    out = [None] * len(texts)
    for c in res["choices"]:
        out[c["index"]] = c["text"]
    return out


def generate(model, tok, msgs, batch, max_new, label):
    from poc_clue import apply_chat_template
    texts = [apply_chat_template(tok, m, add_generation_prompt=True) for m in msgs]
    if os.environ.get("MPORT_VLLM_URL"):
        t0 = time.perf_counter()
        out = [re.sub(r"<think>.*?</think>", "", o, flags=re.S).strip() for o in vllm_generate(texts, max_new)]
        print(f"{label} {len(texts)} answers in {time.perf_counter() - t0:.0f}s (vLLM)", flush=True)
        return out
    import torch
    from poc_clue import P
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out, t0 = [None] * len(texts), time.perf_counter()
    for idx in gen_batches(tok, texts, order, batch, max_new):
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(P.DEV)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for i, o in zip(idx, gen):
            out[i] = re.sub(r"<think>.*?</think>", "", tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True), flags=re.S).strip()
        del enc, gen
        torch.mps.empty_cache()
    print(f"{label} {len(texts)} answers in {time.perf_counter() - t0:.0f}s", flush=True)
    return out


def rank_of(scores, i):
    """1-based rank of item i (ties count against it)."""
    return int(1 + sum(1 for j, x in enumerate(scores) if j != i and x >= scores[i]))


def cmd_teach(args):
    K = PF.poc()
    from poc_clue import D, PERSONA_PROMPT
    s, cfg, facts, fv, info, clue_of, emap = PF.setup()
    ev = eval_sets(s, facts, clue_of, args.n_val, args.stride)
    tr = train_sets(s, facts, clue_of, args.n_neg)
    if args.limit:
        ev, tr = ev[: args.limit], tr[: args.limit]
    runner = D.Runner([], K.DECODER)
    model, tok = runner.model, runner.tok
    tok.padding_side = "left"
    pairs = [(e["question"], f) for e in ev for f in [e["clue"]] + e["others"]]
    sc = score_pairs(model, tok, pairs, args.batch, "eval")
    n = 0
    for e in ev:
        m = 1 + len(e["others"])
        e["clue_score"], e["other_scores"] = float(sc[n]), [float(x) for x in sc[n + 1:n + m]]
        n += m
    out = K.OUT / args.out
    out.write_text(json.dumps({"eval": ev}, indent=1))
    for gname in dict.fromkeys(e["group"] for e in ev):
        x = [e for e in ev if e["group"] == gname]
        print(f"RELT {gname:25s} n={len(x):3d}  clue says yes {np.mean([e['clue_score'] > 0 for e in x]):.2f}  "
              f"other facts say yes {np.mean([v > 0 for e in x for v in e['other_scores']]):.2f}  "
              f"mean score clue {np.mean([e['clue_score'] for e in x]):+.2f}  others {np.mean([v for e in x for v in e['other_scores']]):+.2f}", flush=True)
        for size in SIZES:
            rk = [rank_of([e["clue_score"]] + e["other_scores"][: size - 1], 0) for e in x]
            print(f"RELT {gname:25s} profile of {size:2d}: clue ranked first {np.mean([r == 1 for r in rk]):.2f}  top 3 {np.mean([r <= 3 for r in rk]):.2f}  "
                  f"mean rank {np.mean(rk):.1f}", flush=True)
    by = {}
    for e in ev:
        by.setdefault(e["type"], []).append(rank_of([e["clue_score"]] + e["other_scores"][:47], 0) <= 3)
    print("RELT profile of 48, clue in top 3 by type: " + "  ".join(f"{t} {np.mean(v):.2f}" for t, v in sorted(by.items())), flush=True)

    # answers with the top facts as plain text
    rows, msgs = [], []
    for e in ev:
        for size in args.answer_sizes:
            for inc in (True, False):
                lst = ([(e["clue"], e["clue_score"])] if inc else []) + list(zip(e["others"], e["other_scores"]))[: size - inc]
                ranked = sorted(lst, key=lambda t: -t[1])
                for policy, chosen in (("top 3", ranked[:3]), ("yes only, up to 3", [t for t in ranked[:3] if t[1] > 0])):
                    texts = [t[0] for t in chosen]
                    random.Random(e["pair"] + policy).shuffle(texts)
                    user = (D.user_message(texts, e["question"]) if texts else e["question"]) + PF.SUFFIX
                    msgs.append([{"role": "system", "content": PERSONA_PROMPT}, {"role": "user", "content": user}])
                    rows.append({"group": e["group"], "pair": e["pair"], "type": e["type"], "size": size, "clue_in": inc, "policy": policy,
                                 "chosen": texts, "clue_chosen": e["clue"] in texts})
    ans = generate(model, tok, msgs, args.gen_batch, args.max_new_tokens, "answer")
    pmap = {p["pair"]: p for p in s["pairs"]}
    for r, a in zip(rows, ans):
        p = pmap[r["pair"]]
        r.update(answer=a, **K.strict_score(a, p["rubrics"], p["value"]))
    out.write_text(json.dumps({"eval": ev, "answers": rows}, indent=1))
    for gname in dict.fromkeys(r["group"] for r in rows):
        for size in args.answer_sizes:
            for policy in ("top 3", "yes only, up to 3"):
                a = [r for r in rows if r["group"] == gname and r["size"] == size and r["policy"] == policy and r["clue_in"]]
                b = [r for r in rows if r["group"] == gname and r["size"] == size and r["policy"] == policy and not r["clue_in"]]
                print(f"RELA {gname:25s} profile of {size:2d} {policy:18s} n={len(a):3d}  clue chosen {np.mean([r['clue_chosen'] for r in a]):.2f}  "
                      f"facts used {np.mean([len(r['chosen']) for r in a]):.1f}  mentions {np.mean([r['mentions_strict'] for r in a]):.2f}  "
                      f"acts {np.mean([r['acts_strict'] for r in a]):.2f}  acts with the clue removed {np.mean([r['acts_strict'] for r in b]):.2f}", flush=True)

    # labels for the distilled map
    tp = [(t["request"], f) for t in tr for f in t["facts"]]
    sc = score_pairs(model, tok, tp, args.batch, "train")
    n = 0
    for t in tr:
        t["scores"] = [float(x) for x in sc[n:n + len(t["facts"])]]
        n += len(t["facts"])
    out.write_text(json.dumps({"eval": ev, "answers": rows, "train": tr}, indent=1))
    print(f"RELT training labels: {len(tr)} requests, {len(tp)} pairs; known fact says yes {np.mean([t['scores'][0] > 0 for t in tr]):.2f}  "
          f"random fact says yes {np.mean([v > 0 for t in tr for v in t['scores'][1:]]):.2f}  "
          f"known fact ranked first {np.mean([rank_of(t['scores'], 0) == 1 for t in tr]):.2f}", flush=True)
    print(f"saved {out}", flush=True)


def cmd_fair(args):
    """Fair profiles. In H15 the other facts of a profile came from other users' chats and many are real constraints
    for the request (a wheelchair, a nut allergy for "plan a weekend getaway"), so "acts on the clue" undercounts.
    Here the other facts are ones the decoder itself judges irrelevant to the request (yes/no score below --thresh).
    --mode text: plain decoder, profile as text. --mode soft: reader LoRA, 4 soft tokens per fact through the core."""
    import torch
    K = PF.poc()
    from poc_clue import D, PERSONA_PROMPT
    s, cfg, facts, fv, info, clue_of, emap = PF.setup()
    _, qmap = K.load_emb()
    pmap = {p["pair"]: p for p in s["pairs"]}
    fvec = {f["text"]: fv[f["row"]] for f in facts}
    ev = json.loads((K.OUT / args.teach).read_text())["eval"]
    clue_text = {e["pair"]: e["clue"] for e in ev}
    rows = []
    for e in ev:
        irr = [t for t, sc in zip(e["others"], e["other_scores"]) if sc < args.thresh and t in fvec]
        for size in args.sizes:
            if len(irr) < size - 1 and size > 1:
                continue
            for inc in ((True,) if size == 1 else (True, False)):
                lst = irr[: size - inc] + ([e["clue"]] if inc else [])
                random.Random(f"{e['pair']} {size} {inc}").shuffle(lst)
                rows.append({"group": e["group"], "pair": e["pair"], "type": e["type"], "size": size, "clue_in": inc, "facts": lst})
    if args.limit:
        rows = rows[: args.limit]
    if args.mode == "text":
        runner = D.Runner([], K.DECODER)
        model, tok = runner.model, runner.tok
        tok.padding_side = "left"
        msgs = [[{"role": "system", "content": PERSONA_PROMPT},
                 {"role": "user", "content": D.user_message(r["facts"], pmap[r["pair"]]["question"]) + PF.SUFFIX}] for r in rows]
        ans = generate(model, tok, msgs, args.gen_batch, args.max_new_tokens, "fair text")
    else:
        import poc_agentmem as A
        import poc_core as C
        import poc_fast as FB
        from poc_agentmem import P
        st = A.Stack("lora", train=True)
        C.load_lora(st, args.lora, eval_mode=True)
        core = st.core.float().eval()
        st.r.model.eval()
        dt = next(st.r.model.parameters()).dtype
        ans, t0 = [], time.perf_counter()
        for b in range(0, len(rows), args.soft_batch):
            items = []
            with torch.no_grad():
                for r in rows[b:b + args.soft_batch]:
                    p = pmap[r["pair"]]
                    vecs = [clue_of(p)[0] if t == clue_text[r["pair"]] else fvec[t] for t in r["facts"]]
                    mem = PF.blocks(C, core, np.stack(vecs), qmap[p["question"]], args.k, P, torch).to(dt)
                    items.append((p["question"] + PF.SUFFIX, mem))
            ans += FB.generate_batch(st, items, args.max_new_tokens)
        print(f"fair soft {len(rows)} answers in {time.perf_counter() - t0:.0f}s", flush=True)
    for r, a in zip(rows, ans):
        p = pmap[r["pair"]]
        r.update(answer=a, **K.strict_score(a, p["rubrics"], p["value"]))
    (K.OUT / f"relevance_fair_{args.mode}.json").write_text(json.dumps({"rows": rows}, indent=1))
    name = "text" if args.mode == "text" else f"soft tokens ({args.lora})"
    for gname in dict.fromkeys(r["group"] for r in rows):
        for size in args.sizes:
            a = [r for r in rows if r["group"] == gname and r["size"] == size and r["clue_in"]]
            b = [r for r in rows if r["group"] == gname and r["size"] == size and not r["clue_in"]]
            if a:
                print(f"RELF {name:28s} {gname:25s} profile of {size:2d} (others irrelevant) n={len(a):3d}  mentions {np.mean([r['mentions_strict'] for r in a]):.2f}  "
                      f"acts {np.mean([r['acts_strict'] for r in a]):.2f}"
                      + (f"  acts with the clue removed {np.mean([r['acts_strict'] for r in b]):.2f}" if b else ""), flush=True)
    for r in [r for r in rows if r["clue_in"] and r["size"] == args.sizes[-1]][:: 23][:4]:
        print("RELF sample", json.dumps({k: r[k] for k in ("type", "answer", "acts_strict")})[:400], flush=True)


def cmd_distill(args):
    """Low-rank map over the frozen vectors. score(q, f) = scale * (norm(q) A) . (norm(f) B) + w * cos(q, f)."""
    import torch
    K = PF.poc()
    s, cfg, facts, fv, info, clue_of, emap = PF.setup()
    _, qmap = K.load_emb()
    data = json.loads((K.OUT / args.teach).read_text())
    fvec = {f["text"]: fv[f["row"]] for f in facts}
    for p in s["pairs"]:
        vec, text, _ = clue_of(p)
        if vec is not None:
            fvec.setdefault(text, vec)
    fvec.update({t: v for t, v in emap.items() if t not in fvec})
    qvec = {**emap, **qmap}

    def unit(x):
        x = np.asarray(x, dtype=np.float32)
        return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    tr = [t for t in data["train"] if t["request"] in qvec and all(f in fvec for f in t["facts"])]
    ev = [e for e in data["eval"] if e["question"] in qvec and e["clue"] in fvec and all(f in fvec for f in e["others"])]
    print(f"RELD usable: {len(tr)}/{len(data['train'])} training requests, {len(ev)}/{len(data['eval'])} test questions", flush=True)
    n_f = len(tr[0]["facts"])
    Q = torch.tensor(unit(np.stack([qvec[t["request"]] for t in tr])))
    F = torch.tensor(unit(np.stack([[fvec[f] for f in t["facts"]] for t in tr])))
    T = torch.tensor(np.array([t["scores"] for t in tr], dtype=np.float32))
    EQ = torch.tensor(unit(np.stack([qvec[e["question"]] for e in ev])))
    EF = torch.tensor(unit(np.stack([[fvec[f] for f in [e["clue"]] + e["others"]] for e in ev])))
    ET = np.array([[e["clue_score"]] + e["other_scores"] for e in ev], dtype=np.float32)

    def report(name, score):  # score [n_eval, 48]; column 0 is the clue
        res = {}
        for gname in dict.fromkeys(e["group"] for e in ev):
            ix = [i for i, e in enumerate(ev) if e["group"] == gname]
            for size in SIZES:
                rk = [rank_of(list(score[i][:size]), 0) for i in ix]
                res[(gname, size)] = (float(np.mean([r == 1 for r in rk])), float(np.mean([r <= 3 for r in rk])))
        return res

    def show(name, runs):
        for key in runs[0]:
            a = np.array([r[key] for r in runs])
            print(f"RELD {name:34s} {key[0]:25s} profile of {key[1]:2d}: clue first {a[:, 0].mean():.2f}  top 3 {a[:, 1].mean():.2f}"
                  + (f"  (range of first over {len(runs)} seeds {a[:, 0].min():.2f}-{a[:, 0].max():.2f})" if len(runs) > 1 else ""), flush=True)

    cos = torch.einsum("nd,nkd->nk", EQ, EF).numpy()
    show("cosine", [report("cosine", cos)])
    show("teacher (decoder yes/no)", [report("teacher", ET)])

    def train(mode, seed):
        torch.manual_seed(seed)
        d = Q.shape[1]
        A = torch.nn.Parameter(torch.randn(d, args.rank) / d ** 0.5)
        B = torch.nn.Parameter(torch.randn(d, args.rank) / d ** 0.5)
        scale = torch.nn.Parameter(torch.tensor(5.0))
        w = torch.nn.Parameter(torch.tensor(1.0))
        bias = torch.nn.Parameter(torch.tensor(0.0))
        opt = torch.optim.AdamW([A, B, scale, w, bias], lr=args.lr, weight_decay=args.wd)
        drop = torch.nn.Dropout(args.dropout)
        g = torch.Generator().manual_seed(seed)
        n_val = max(len(tr) // 10, 1)
        perm = torch.randperm(len(tr), generator=g)
        vi, ti = perm[:n_val], perm[n_val:]

        def score(q, f, training=False):
            qa, fb = q @ A, f @ B
            if training:
                qa, fb = drop(qa), drop(fb)
            qa, fb = torch.nn.functional.normalize(qa, dim=-1), torch.nn.functional.normalize(fb, dim=-1)
            return scale * torch.einsum("nr,nkr->nk", qa, fb) + w * torch.einsum("nd,nkd->nk", q, f) + bias

        def loss_of(ix, training):
            sc = score(Q[ix], F[ix], training)
            if mode == "known pairs":
                # the known fact against the random ones, plus every other request's known fact in the batch
                extra = scale * torch.nn.functional.normalize(Q[ix] @ A, dim=-1) @ torch.nn.functional.normalize(F[ix][:, 0] @ B, dim=-1).T \
                    + w * Q[ix] @ F[ix][:, 0].T + bias
                extra = extra.masked_fill(torch.eye(len(ix), dtype=torch.bool), float("-inf"))
                return torch.nn.functional.cross_entropy(torch.cat([sc, extra], 1), torch.zeros(len(ix), dtype=torch.long))
            t = T[ix]
            point = torch.nn.functional.binary_cross_entropy_with_logits(sc, torch.sigmoid(t))
            lst = torch.nn.functional.kl_div(torch.log_softmax(sc, 1), torch.softmax(t / args.temp, 1), reduction="batchmean")
            return point + lst

        best, best_state, bad = float("inf"), None, 0
        for epoch in range(args.epochs):
            p2 = ti[torch.randperm(len(ti), generator=g)]
            for b in range(0, len(p2), args.batch):
                opt.zero_grad()
                loss_of(p2[b:b + args.batch], True).backward()
                opt.step()
            with torch.no_grad():
                v = float(loss_of(vi, False))
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = [x.detach().clone() for x in (A, B, scale, w, bias)]
            else:
                bad += 1
                if bad >= args.patience:
                    break
        with torch.no_grad():
            for x, y in zip((A, B, scale, w, bias), best_state):
                x.copy_(y)
            sc = score(EQ, EF).numpy()
        return report(mode, sc), epoch + 1, best, {"A": A.detach(), "B": B.detach(), "scale": scale.detach(), "w": w.detach(), "bias": bias.detach()}

    results = {}
    for mode in ("known pairs", "teacher labels"):
        runs = [train(mode, seed) for seed in range(args.seeds)]
        show(f"map r{args.rank}, {mode}", [r[0] for r in runs])
        print(f"RELD   epochs {[r[1] for r in runs]}  held-out loss {[round(r[2], 3) for r in runs]}", flush=True)
        results[mode] = [{f"{k[0]} | {k[1]}": v for k, v in r[0].items()} for r in runs]
        if args.save:
            torch.save(runs[0][3], K.OUT / f"relmap_{mode.split()[0]}.pt")
    (K.OUT / args.out).write_text(json.dumps(results, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("teach")
    t.add_argument("--n-val", type=int, default=30)
    t.add_argument("--stride", type=int, default=1)
    t.add_argument("--n-neg", type=int, default=15)
    t.add_argument("--batch", type=int, default=96)
    t.add_argument("--gen-batch", type=int, default=32)
    t.add_argument("--answer-sizes", type=lambda x: [int(v) for v in x.split(",")], default=[8, 48])
    t.add_argument("--max-new-tokens", type=int, default=70)
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--out", default="relevance_teach.json")
    f = sub.add_parser("fair")
    f.add_argument("--mode", choices=["text", "soft"], default="text")
    f.add_argument("--teach", default="relevance_teach.json")
    f.add_argument("--thresh", type=float, default=0.0)
    f.add_argument("--sizes", type=lambda x: [int(v) for v in x.split(",")], default=[1, 8, 16])
    f.add_argument("--lora", default="lora_prof.pt")
    f.add_argument("--k", type=int, default=4)
    f.add_argument("--gen-batch", type=int, default=32)
    f.add_argument("--soft-batch", type=int, default=16)
    f.add_argument("--max-new-tokens", type=int, default=70)
    f.add_argument("--limit", type=int, default=0)
    d = sub.add_parser("distill")
    d.add_argument("--teach", default="relevance_teach.json")
    d.add_argument("--rank", type=int, default=64)
    d.add_argument("--lr", type=float, default=1e-3)
    d.add_argument("--wd", type=float, default=0.05)
    d.add_argument("--dropout", type=float, default=0.2)
    d.add_argument("--temp", type=float, default=2.0)
    d.add_argument("--epochs", type=int, default=60)
    d.add_argument("--patience", type=int, default=5)
    d.add_argument("--batch", type=int, default=64)
    d.add_argument("--seeds", type=int, default=3)
    d.add_argument("--save", action="store_true")
    d.add_argument("--out", default="relevance_distill.json")
    args = ap.parse_args()
    {"teach": cmd_teach, "distill": cmd_distill, "fair": cmd_fair}[args.cmd](args)


if __name__ == "__main__":
    main()
