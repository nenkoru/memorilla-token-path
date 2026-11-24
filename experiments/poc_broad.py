"""H18: broad data for the two learned parts around the frozen core.

H17 found: (a) the plain decoder given 8 facts as text tries to use all of them, even irrelevant ones (acts on the right
one 0.69 alone -> 0.20 among 7 irrelevant facts), so something has to select before the decoder; (b) a small low-rank
relevance map over the frozen vectors, trained on only 629 (request, relevant fact) pairs, already ranks the clue first
more often than cosine or the decoder's own yes/no judgement on held-out fact types (0.39 vs 0.22 / 0.26 in a profile
of 8). This script scales the data about 10x and trains both parts on it:

    gen     GPU: invent facts in ~100 categories, 2 requests per fact, answers with and without the fact
    embed   GPU: vectors of the new facts and requests
    map     relevance map (known pairs, in-batch negatives); rank of the clue on held-out types, sizes 8 / 16 / 48
    train   reader LoRA: 1-3 facts at 16 tokens each (the relevant one plus look-alikes picked by the map, or none
            relevant with the generic answer), batched
    eval    profile of 8 / 16 / 48 -> map picks 3 facts -> 48 soft tokens -> answer; controls
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import poc_profile as PF  # noqa: E402
import poc_relevance as R  # noqa: E402

CATS = PF.CATS + """; commute and getting around the city; cooking skills and kitchen equipment; clothing, sizes and
dress codes; music, films and books they like or avoid; social habits and how they like to spend evenings; chronic
conditions other than the common ones; mobility and stamina; family structure and who lives with them; their work
tools and software; their employer's rules and policies; the team they manage or belong to; clients and customers they
serve; deadlines and recurring obligations; side projects and long-term goals; savings goals and debts; subscriptions
and memberships they have; things they own and things they deliberately do not own; garden, balcony and plants;
home repairs and renovation in progress; noise, neighbours and landlord rules; shopping habits and stores near them;
travel documents, visas and residency; places they refuse to go or cannot enter; holidays and traditions they observe;
volunteering and community roles; political or moral lines they will not cross; privacy and what they do not share
online; how they like information presented; reading level, attention span and learning style; writing and
communication style they prefer; the tone they dislike in answers; units, formats and conventions they use; their
coding stack and tooling preferences; their company's products and brand rules; their company's budget and approval
process; their company's legal and compliance limits; their business hours, suppliers and logistics; school or
university rules they must follow; exercise routine and injuries; things they are training or saving up for; daily
medication and appointments; sensory preferences such as smells, textures and sounds; alcohol, smoking and caffeine
habits; fasting and meal timing; water, heat and cold tolerance; fears of animals, heights, crowds or water; internet
access and device limits; accessibility needs when reading or typing; time they have available and when they are
busy; people they must not contact or places they avoid for personal reasons; past bad experiences that shape their
choices; gifts, celebrations and people they buy for; vehicles other than their own that they rely on; where they
were raised and cultural norms they keep; their artistic or craft practice; competitions and events they take part in;
collections and things they are a serious fan of; the kind of help they explicitly do not want"""
INV2 = """Invent 10 different lasting facts about 10 different imaginary users, all in this category: {cat}.

Make them unusual and specific rather than typical. Each fact is one short, concrete sentence that starts with "The user". It should be the kind of thing that changes how a personal assistant ought to answer later requests: a constraint, a need, a circumstance or a strong preference. Write one fact per line and nothing else."""


def clean(x):
    return x.strip().lstrip("-*0123456789. ").strip().strip('"')


def cmd_gen(args):
    K = PF.poc()
    from poc_clue import D, PERSONA_PROMPT
    cats = [" ".join(c.split()) for c in CATS.split(";")][: args.limit or None]
    runner = D.Runner([], K.DECODER)
    model, tok = runner.model, runner.tok
    tok.padding_side = "left"
    old = set(json.loads((K.OUT / "profile_gen.json").read_text())["invented"])
    msgs = [[{"role": "user", "content": t.format(cat=c)}] for c in cats for t in (PF.INV_USER, INV2)]
    raw = R.generate(model, tok, msgs, args.batch, 320, "invent")
    facts, seen = [], set(old)
    for (c, _), o in zip([(c, t) for c in cats for t in (0, 1)], raw):
        for x in o.splitlines():
            x = clean(x)
            if x.startswith("The user") and 20 < len(x) < 200 and x not in seen and not PF.BANNED.search(x):
                seen.add(x)
                facts.append({"fact": x, "cat": c})
    print(f"BROAD invented facts: {len(facts)} in {len(cats)} categories", flush=True)
    raw = R.generate(model, tok, [[{"role": "system", "content": PF.REQ_SYSTEM}, {"role": "user", "content": PF.REQ_USER.format(fact=f["fact"])}]
                                  for f in facts], args.batch, 90, "requests")
    rows = []
    for f, o in zip(facts, raw):
        reqs = [clean(r) for r in o.splitlines() if len(r.strip()) > 12][:2]
        rows += [{"fact": f["fact"], "cat": f["cat"], "request": r, "first": i == 0} for i, r in enumerate(reqs) if not PF.BANNED.search(r)]
    ans = R.generate(model, tok, [[{"role": "system", "content": PERSONA_PROMPT},
                                   {"role": "user", "content": D.user_message([r["fact"]], r["request"]) + PF.SUFFIX}] for r in rows],
                     args.batch, 80, "answers")
    first = [r for r in rows if r["first"]]
    gen = R.generate(model, tok, [[{"role": "system", "content": PERSONA_PROMPT}, {"role": "user", "content": r["request"] + PF.SUFFIX}]
                                  for r in first], args.batch, 80, "generic")
    for r, a in zip(rows, ans):
        r["answer"] = a
    for r, g in zip(first, gen):
        r["generic"] = g
    (K.OUT / "broad_gen.json").write_text(json.dumps({"rows": rows}, indent=1))
    print(f"BROAD rows: {len(rows)} requests for {len(facts)} facts; {len(first)} generic answers", flush=True)
    for r in rows[:: max(len(rows) // 5, 1)][:5]:
        print("BROAD sample", json.dumps(r)[:600], flush=True)


WHO = ["students", "retired people", "owners of a small business", "people who live in a big city",
       "people who live in a remote rural area", "freelancers who work from home", "people who recently moved to another country",
       "managers at a large company", "shift workers", "people on a tight budget"]
WHO += ["parents of teenagers", "people who live alone", "people who share a flat with several others", "people who work outdoors",
        "people who travel for work most weeks", "people who care for an elderly relative", "amateur athletes", "teachers",
        "nurses and doctors", "software engineers", "artists and musicians", "farmers", "people with a long commute",
        "people recovering from an illness or surgery", "people who run an online shop", "people learning a new language",
        "people who are strictly religious", "people who live on a boat or in a van", "restaurant and shop staff",
        "people planning a big life change this year"]
# H25: more questions for the held-out clue types (same clues, same rubrics), so close results can be told apart
XQ = {
    "health": ["Plan a beach day for me and a couple of friends.",
               "I'm going to an all-day music festival on Saturday - how should I prepare?",
               "Suggest a good routine for a long study day at the library.",
               "Help me plan a full day of skiing.",
               "Plan a long bike ride for me this Sunday.",
               "What's a good way to spend a sunny afternoon outdoors?",
               "I have a ten-hour day of back-to-back meetings tomorrow - how do I get through it?",
               "Give me a plan for a day of sightseeing on foot in Lisbon.",
               "I'm driving eight hours to visit a friend tomorrow - how should I plan the drive?"],
    "no_car": ["I need to pick up a sofa I bought second-hand on the other side of the city - how should I do it?",
               "Plan a day trip to a national park for me.",
               "I just got a job in a suburb 15 miles from my place - how should I plan my commute?",
               "Help me plan a camping weekend.",
               "I want to visit my grandmother who lives in a small town two hours away - plan the visit.",
               "How do I get a Christmas tree home?",
               "I'm hosting a barbecue for 20 people - how do I get all the supplies?",
               "Plan a trip to a furniture store to furnish my new bedroom.",
               "Plan a day at a beach about an hour from the city."],
    "toddler": ["Plan a dinner out for my partner's birthday.",
                "Help me plan a week-long vacation.",
                "What's a good workout routine I can stick to?",
                "I want to host a game night at my place - how should I set it up?",
                "Plan a day trip to a nearby city.",
                "Suggest a hobby I could pick up this year.",
                "How should I organize my mornings to be more productive?",
                "What should I look for in a new apartment?",
                "I want to start working from home two days a week - how should I set things up?"],
    "flying": ["I need to get from Boston to Washington DC for a conference - plan the travel.",
               "Plan a two-week trip around Europe for me, starting in London.",
               "My friend is getting married in Seattle and I live in San Francisco - how should I get there?",
               "Help me plan a honeymoon.",
               "I've got a job interview in Toronto next week and I live in Montreal - plan the trip.",
               "Plan a family reunion trip from Chicago to Denver.",
               "I want to go somewhere for New Year's - suggest a trip from Los Angeles.",
               "Plan a long weekend trip from Paris to Barcelona.",
               "I want to see the Grand Canyon this spring and I live in Dallas - plan it."],
}
PAIR_FILES = ("broad_pairs", "broad_pairs2")
INV3 = """Invent 10 different lasting facts about 10 different imaginary users, all in this category: {cat}.

All 10 users are {who}. Each fact is one short, concrete sentence that starts with "The user". It should be the kind of thing that changes how a personal assistant ought to answer later requests: a constraint, a need, a circumstance or a strong preference. Write one fact per line and nothing else."""


def cmd_pairs(args):
    """H20: many more (request, relevant fact) pairs for the map; no answers, so it is cheap."""
    K = PF.poc()
    from poc_clue import D
    cats = [" ".join(c.split()) for c in CATS.split(";")][: args.limit or None]
    runner = D.Runner([], K.DECODER)
    model, tok = runner.model, runner.tok
    tok.padding_side = "left"
    seen = set(json.loads((K.OUT / "profile_gen.json").read_text())["invented"])
    seen |= {r["fact"] for r in json.loads((K.OUT / "broad_gen.json").read_text())["rows"]}
    for name in PAIR_FILES:
        if (K.OUT / f"{name}.json").exists() and name != args.out:
            seen |= {r["fact"] for r in json.loads((K.OUT / f"{name}.json").read_text())["rows"]}
    jobs = [(c, w) for c in cats for w in WHO[args.who_from: args.who]]
    raw = R.generate(model, tok, [[{"role": "user", "content": INV3.format(cat=c, who=w)}] for c, w in jobs], args.batch, 320, "invent")
    facts = []
    for (c, w), o in zip(jobs, raw):
        for x in o.splitlines():
            x = clean(x)
            if x.startswith("The user") and 20 < len(x) < 200 and x not in seen and not PF.BANNED.search(x):
                seen.add(x)
                facts.append({"fact": x, "cat": c})
    print(f"BROADPAIRS invented facts: {len(facts)} from {len(jobs)} prompts", flush=True)
    req = PF.REQ_USER.replace("Write 2 different", "Write 3 different")
    raw = R.generate(model, tok, [[{"role": "system", "content": PF.REQ_SYSTEM}, {"role": "user", "content": req.format(fact=f["fact"])}]
                                  for f in facts], args.batch, 120, "requests")
    rows = []
    for f, o in zip(facts, raw):
        reqs = [clean(r) for r in o.splitlines() if len(r.strip()) > 12][:3]
        rows += [{"fact": f["fact"], "cat": f["cat"], "request": r} for r in reqs if not PF.BANNED.search(r)]
    (K.OUT / f"{args.out}.json").write_text(json.dumps({"rows": rows}, indent=1))
    print(f"BROADPAIRS rows: {len(rows)} requests for {len(facts)} facts", flush=True)
    for r in rows[:: max(len(rows) // 5, 1)][:5]:
        print("BROADPAIRS sample", json.dumps(r)[:400], flush=True)


def cmd_longans(args):
    """H21: teacher answers with no length limit, for a reader that answers at natural length."""
    K = PF.poc()
    from poc_clue import D, PERSONA_PROMPT
    rows = json.loads((K.OUT / "broad_gen.json").read_text())["rows"][: args.limit or None]
    runner = D.Runner([], K.DECODER)
    model, tok = runner.model, runner.tok
    tok.padding_side = "left"
    ans = R.generate(model, tok, [[{"role": "system", "content": PERSONA_PROMPT},
                                   {"role": "user", "content": D.user_message([r["fact"]], r["request"])}] for r in rows],
                     args.batch, args.max_new_tokens, "long answers")
    first = [r for r in rows if r.get("generic")]
    gen = R.generate(model, tok, [[{"role": "system", "content": PERSONA_PROMPT}, {"role": "user", "content": r["request"]}]
                                  for r in first], args.batch, args.max_new_tokens, "long generic")
    out = [{"request": r["request"], "fact": r["fact"], "answer": a} for r, a in zip(rows, ans)]
    for r, g in zip(first, gen):
        next(o for o in out if o["request"] == r["request"] and o["fact"] == r["fact"])["generic"] = g
    (K.OUT / "broad_long.json").write_text(json.dumps({"rows": out}, indent=1))
    print(f"BROADLONG {len(out)} answers, {len(first)} generic; mean words {np.mean([len(o['answer'].split()) for o in out]):.0f}", flush=True)
    print("BROADLONG sample", json.dumps(out[0])[:900], flush=True)


def cmd_embed(args):
    import torch
    K = PF.poc()
    from poc_clue import D
    name = args.name if args.pairs else "broad_gen"
    if args.test:
        name, texts = "broad_test", sorted({q for qs in XQ.values() for q in qs})
    else:
        rows = json.loads((K.OUT / f"{name}.json").read_text())["rows"]
        texts = sorted({r["request"] for r in rows} | {r["fact"] for r in rows})
    enc = D.Encoder()
    out = []
    for b in range(0, len(texts), 512):
        out.append(enc(texts[b:b + 512], batch_size=64).astype(np.float16))
    np.savez(K.OUT / ("broad_emb.npz" if name == "broad_gen" else f"{name}_emb.npz"), v=np.concatenate(out), text=np.array(texts))
    print(f"BROAD embedded {len(texts)}", flush=True)


EXTRA = []


def unit(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def load_all():
    """-> setup tuple, qvec (request / question text -> vector), fvec (fact text -> vector), broad rows."""
    K = PF.poc()
    su = PF.setup()
    s, cfg, facts, fv, info, clue_of, emap = su
    _, qmap = K.load_emb()
    z = np.load(K.OUT / "broad_emb.npz")
    zv, zt = z["v"], z["text"]  # read once: every z["v"] access re-reads the whole array from the file
    bmap = {str(t): zv[i] for i, t in enumerate(zt)}
    fvec = {f["text"]: fv[f["row"]] for f in facts}
    for p in s["pairs"]:
        vec, text, _ = clue_of(p)
        if vec is not None:
            fvec.setdefault(text, vec)
    for m in (emap, bmap):
        for t, v in m.items():
            fvec.setdefault(t, v)
    rows = [r for r in json.loads((K.OUT / "broad_gen.json").read_text())["rows"] if r["request"] in bmap and r["fact"] in bmap]
    EXTRA.clear()
    for name in PAIR_FILES:  # H20: pairs without answers, for the map only
        if not (K.OUT / f"{name}_emb.npz").exists():
            continue
        z = np.load(K.OUT / f"{name}_emb.npz")
        zv, zt = z["v"], z["text"]
        xmap = {str(t): zv[i] for i, t in enumerate(zt)}
        EXTRA.extend((r["request"], r["fact"]) for r in json.loads((K.OUT / f"{name}.json").read_text())["rows"]
                     if r["request"] in xmap and r["fact"] in xmap)
        for t, v in xmap.items():
            fvec.setdefault(t, v)
        bmap = {**xmap, **bmap}
    if (K.OUT / "broad_test_emb.npz").exists():  # H25: extra test questions
        z = np.load(K.OUT / "broad_test_emb.npz")
        zv, zt = z["v"], z["text"]
        bmap = {**{str(t): zv[i] for i, t in enumerate(zt)}, **bmap}
    return su, {**emap, **bmap, **qmap}, fvec, rows


def test_set(su, qvec, big):
    """-> eval entries and pair id -> pair. big: every held-out clue also gets the XQ questions of its type."""
    s, cfg, facts, fv, info, clue_of, emap = su
    ev = R.eval_sets(s, facts, clue_of, 30, 1)
    pmap = {p["pair"]: p for p in s["pairs"]}
    if not big:
        return ev, pmap
    by = {e["pair"]: e for e in ev}
    out = []
    for e in ev:
        out.append(e)
        v, j, qi = e["pair"].split(":")
        if e["group"] != "held-out types" or qi != "q0":
            continue
        for k, q in enumerate(XQ[e["type"]]):
            base = by[f"{v}:{j}:q{k % 3}"]  # same clue; the distractors of one of its three questions, rotated
            o, sh = base["others"], (7 * (k + 1)) % len(base["others"])
            pid = f"{v}:{j}:x{k}"
            assert q in qvec, "run: poc_broad.py embed --test"
            out.append({**base, "pair": pid, "question": q, "others": o[sh:] + o[:sh]})
            pmap[pid] = {**pmap[base["pair"]], "pair": pid, "question": q}
    return out, pmap


def known_pairs(su, rows, qvec, fvec):
    """(request, relevant fact) pairs for training: broad rows, the H15 generated rows, the training clue pairs."""
    s, cfg, facts, fv, info, clue_of, emap = su
    old = [(t["request"], t["facts"][0]) for t in R.train_sets(s, facts, clue_of, 1)]
    out = [(r["request"], r["fact"]) for r in rows] + old + list(EXTRA)
    return [(q, f) for q, f in dict.fromkeys(out) if q in qvec and f in fvec]


def fit_map(Q, F, fid, seed, args):
    """Q, F: unit vectors of each request and its relevant fact; fid: fact id per row (same fact = not a negative)."""
    import torch
    torch.manual_seed(seed)
    Q, F, fid = torch.tensor(Q), torch.tensor(F), torch.tensor(fid)
    d = Q.shape[1]
    A = torch.nn.Parameter(torch.randn(d, args.rank) / d ** 0.5)
    B = torch.nn.Parameter(torch.randn(d, args.rank) / d ** 0.5)
    scale, w = torch.nn.Parameter(torch.tensor(10.0)), torch.nn.Parameter(torch.tensor(5.0))
    params = [A, B, scale, w]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    drop = torch.nn.Dropout(args.dropout)
    g = torch.Generator().manual_seed(seed)
    ids = torch.unique(fid)
    held = set(ids[torch.randperm(len(ids), generator=g)[: max(len(ids) // 10, 1)]].tolist())
    is_val = torch.tensor([int(x) in held for x in fid])
    vi, ti = torch.where(is_val)[0], torch.where(~is_val)[0]
    nrm = torch.nn.functional.normalize

    def loss_of(ix, training):
        qa, fb = Q[ix] @ A, F[ix] @ B
        if training:
            qa, fb = drop(qa), drop(fb)
        lg = scale * nrm(qa, dim=-1) @ nrm(fb, dim=-1).T + w * Q[ix] @ F[ix].T
        same = ((fid[ix][:, None] == fid[ix][None, :]) | (F[ix] @ F[ix].T > args.dup)) & ~torch.eye(len(ix), dtype=torch.bool)
        return torch.nn.functional.cross_entropy(lg.masked_fill(same, float("-inf")), torch.arange(len(ix)))

    best, state, bad, epoch = float("inf"), None, 0, 0
    for epoch in range(args.epochs):
        p = ti[torch.randperm(len(ti), generator=g)]
        for b in range(0, len(p), args.batch):
            opt.zero_grad()
            loss_of(p[b:b + args.batch], True).backward()
            opt.step()
        with torch.no_grad():
            v = float(np.mean([float(loss_of(vi[b:b + args.batch], False)) for b in range(0, len(vi), args.batch)]))
        if v < best - 1e-4:
            best, bad, state = v, 0, [x.detach().clone() for x in params]
        else:
            bad += 1
            if bad >= args.patience:
                break
    return {"A": state[0], "B": state[1], "scale": state[2], "w": state[3], "epochs": epoch + 1, "val": best}


def map_score(m, q, Fm):
    """q [d], Fm [k, d] unit vectors -> [k] relevance scores."""
    A, B = m["A"].numpy(), m["B"].numpy()
    return float(m["scale"]) * unit(Fm @ B) @ unit(q @ A) + float(m["w"]) * Fm @ q


def ranks(ev, qvec, fvec, score):
    res = {}
    for gname in dict.fromkeys(e["group"] for e in ev):
        for size in R.SIZES:
            rk = []
            for e in [e for e in ev if e["group"] == gname]:
                Fm = unit(np.stack([fvec[t] for t in [e["clue"]] + e["others"][: size - 1]]))
                rk.append(R.rank_of(list(score(unit(qvec[e["question"]]), Fm)), 0))
            res[(gname, size)] = (float(np.mean([r == 1 for r in rk])), float(np.mean([r <= 3 for r in rk])))
    return res


def cmd_map(args):
    import torch
    K = PF.poc()
    su, qvec, fvec, rows = load_all()
    s, cfg, facts, fv, info, clue_of, emap = su
    ev, _ = test_set(su, qvec, args.big)
    pairs = known_pairs(su, rows, qvec, fvec)
    fids = {f: i for i, f in enumerate(dict.fromkeys(f for _, f in pairs))}
    Q, F = unit(np.stack([qvec[q] for q, _ in pairs])), unit(np.stack([fvec[f] for _, f in pairs]))
    fid = np.array([fids[f] for _, f in pairs])
    print(f"BROADMAP training pairs {len(pairs)}, distinct facts {len(fids)}", flush=True)

    def show(name, runs):
        for key in runs[0]:
            a = np.array([r[key] for r in runs])
            print(f"BROADMAP {name:22s} {key[0]:25s} profile of {key[1]:2d}: clue first {a[:, 0].mean():.2f}  top 3 {a[:, 1].mean():.2f}"
                  + (f"  (first, range over {len(runs)} seeds {a[:, 0].min():.2f}-{a[:, 0].max():.2f})" if len(runs) > 1 else ""), flush=True)

    show("cosine", [ranks(ev, qvec, fvec, lambda q, Fm: Fm @ q)])
    for frac in args.fractions:
        n = max(int(len(pairs) * frac), 64)
        maps = []
        for seed in range(args.seeds):
            ix = np.random.default_rng(seed).permutation(len(pairs))[:n]
            maps.append(fit_map(Q[ix], F[ix], fid[ix], seed, args))
        show(f"map, {n} pairs", [ranks(ev, qvec, fvec, lambda q, Fm, m=m: map_score(m, q, Fm)) for m in maps])
        print(f"BROADMAP   epochs {[m['epochs'] for m in maps]}  held-out-fact loss {[round(m['val'], 3) for m in maps]}", flush=True)
    torch.save(maps[0], K.OUT / args.save)  # full data, seed 0
    print(f"saved {args.save}", flush=True)


def cmd_combine(args):
    """H22 (CPU): does the decoder's yes/no add anything on top of the map? Uses the stored yes/no scores of test 17."""
    import torch
    K = PF.poc()
    su, qvec, fvec, rows = load_all()
    m = torch.load(K.OUT / args.map)
    ev = json.loads((K.OUT / "relevance_teach.json").read_text())["eval"]
    z = lambda x: (np.asarray(x, dtype=np.float32) - np.mean(x)) / (np.std(x) + 1e-6)
    for gname in dict.fromkeys(e["group"] for e in ev):
        for size in R.SIZES:
            res = {}
            for e in [e for e in ev if e["group"] == gname]:
                lst = [e["clue"]] + e["others"][: size - 1]
                yn = np.array([e["clue_score"]] + e["other_scores"][: size - 1], dtype=np.float32)
                mp = map_score(m, unit(qvec[e["question"]]), unit(np.stack([fvec[t] for t in lst])))
                for lam in (0.0, 0.25, 0.5, 1.0, 2.0):
                    res.setdefault(f"map + {lam} yes/no", []).append(R.rank_of(list(z(mp) + lam * z(yn)), 0) == 1)
                top = np.argsort(-mp)[:3]
                res.setdefault("map top 3, then yes/no", []).append(0 in top and R.rank_of(list(yn[top]), list(top).index(0)) == 1)
                top = np.argsort(-yn)[:3]
                res.setdefault("yes/no top 3, then map", []).append(0 in top and R.rank_of(list(mp[top]), list(top).index(0)) == 1)
                res.setdefault("yes/no only", []).append(R.rank_of(list(yn), 0) == 1)
            print(f"BROADCOMB {gname:25s} profile of {size:2d}: " + "  ".join(f"{k} {np.mean(v):.2f}" for k, v in res.items()), flush=True)


def make_delta(torch, rank, d_in=2560, d_out=4096, k=16):
    """H23: a small add-on to the core's output: 16 extra-token offsets per fact from its vector. The core is untouched."""
    class Delta(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = torch.nn.Linear(d_in, rank), torch.nn.Linear(rank, k * d_out)
            torch.nn.init.zeros_(self.b.weight)
            torch.nn.init.zeros_(self.b.bias)

        def forward(self, vecs, keep=k):  # [n, d_in] -> [1, n * keep, d_out]
            x = torch.nn.functional.normalize(vecs.float(), dim=-1) * 50.0
            return self.b(torch.nn.functional.gelu(self.a(x))).view(-1, k, d_out)[:, :keep].reshape(1, -1, d_out)
    return Delta()


def attn_memory(C, core, vecs, qv, score, beta, P, torch):
    """H25: one core call over all facts; the map's score for each fact is added to the attention logits. -> [1, 16, dim]"""
    d = torch.from_numpy(np.asarray(vecs, dtype=np.float32))[None].to(P.DEV)
    q = torch.from_numpy(qv.astype(np.float32))[None].to(P.DEV)
    sc = torch.from_numpy(np.asarray(score, dtype=np.float32)).to(P.DEV)
    return C.core_fwd(core, d, q, bias=beta[:, :, None] * sc[None, None, :])


def cmd_train(args):
    import torch
    import fast_train as FT
    import q8
    import poc_agentmem as A
    import poc_core as C
    import poc_fast as FB
    from poc_agentmem import P, OUT
    K = PF.poc()
    torch.manual_seed(0)
    su, qvec, fvec, rows = load_all()
    s, cfg, facts, fv, info, clue_of, emap = su
    m = torch.load(K.OUT / args.map)
    gen = json.loads((K.OUT / "profile_gen.json").read_text())["rows"]
    ex = [(r["request"], r["fact"], r["answer"], r.get("generic")) for r in rows + gen if r["request"] in qvec and r["fact"] in fvec]
    pairs = {p["pair"]: p for p in s["pairs"]}
    for row in json.loads((K.OUT / "teach.json").read_text())["rows"]:  # teacher answers on the trained clue types
        if row["split"] == "train" and row["kept"]:
            p = pairs[row["pair"]]
            text = clue_of(p)[1]
            if text in fvec and p["question"] in qvec:
                ex += [(p["question"], text, a, row["twin_answer"] if row["twin_ok"] else None) for a in row["kept"]]
    suffix = PF.SUFFIX
    if args.long:  # H21: natural-length answers only (broad rows), no length instruction
        ex = [(r["request"], r["fact"], r["answer"], r.get("generic")) for r in json.loads((K.OUT / "broad_long.json").read_text())["rows"]
              if r["request"] in qvec and r["fact"] in fvec]
        suffix = ""
    if args.restate:  # H23: only name the fact that matters; every known pair is usable, no teacher answers needed
        ex = [(q, f, None, None) for q, f in known_pairs(su, rows, qvec, fvec)]
    pool = sorted({f for _, f, _, _ in ex} | {f["text"] for f in facts if f["train"] and f["text"] in fvec})
    RAW = np.stack([fvec[t] for t in pool]).astype(np.float32)
    PV = unit(RAW)
    pidx = {t: i for i, t in enumerate(pool)}
    # look-alikes: for each request the pool facts the map scores highest (the kind that will sit next to the right one)
    PB = unit(PV @ m["B"].numpy())
    hard, qs = {}, list(dict.fromkeys(e[0] for e in ex))
    for b in range(0, len(qs), 1024):  # in chunks: one matrix product per 1,024 requests
        qu = unit(np.stack([qvec[q] for q in qs[b:b + 1024]]))
        sc = float(m["scale"]) * unit(qu @ m["A"].numpy()) @ PB.T + float(m["w"]) * qu @ PV.T
        top = np.argpartition(-sc, 12, axis=1)[:, :12]
        for q, row in zip(qs[b:b + 1024], top):
            hard[q] = row

    def unlike(i, fact):  # not a near-copy of the right fact, in plain or in map space (those are not real negatives)
        j = pidx[fact]
        return pool[i] != fact and float(PV[i] @ PV[j]) < args.dup and float(PB[i] @ PB[j]) < args.dup

    kept = [sum(unlike(i, f) for i in hard[q]) for q, f, _, _ in random.Random(1).sample(ex, min(300, len(ex)))]
    print(f"BROADTRAIN examples {len(ex)}  fact pool {len(pool)}  look-alikes kept after dropping near-copies: {np.mean(kept):.1f} of 12", flush=True)
    st = A.Stack("lora", train=True)
    C.load_lora(st, args.init)
    core = st.core.float().eval()
    loss_fn = FB.loss_batch
    if q8.lean():  # 4-bit lean box run: core on the CPU, output layer on the GPU, packed forward + chunked loss
        core = core.cpu()
        q8.head_to_gpu(st.r.model)
        FT.fast_lora(st.lora)  # branch in fp16, no dropout
        loss_fn = FB.lean_loss_batch
    lps = [t for mod in st.lora for t in (mod.A, mod.B)]
    groups = [{"params": lps, "lr": args.lr, "base": args.lr}]
    delta = None
    if args.delta:
        delta = make_delta(torch, args.delta).to(P.DEV)
        groups.append({"params": list(delta.parameters()), "lr": args.delta_lr, "base": args.delta_lr})
        lps = lps + list(delta.parameters())
    beta = None
    if args.attn:
        beta = torch.nn.Parameter(torch.ones(len(core.cross_attn_layers), core.config.num_heads, device=P.DEV))
        groups.append({"params": [beta], "lr": args.beta_lr, "base": args.beta_lr})
        lps = lps + [beta]
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    rng = random.Random(0)
    dt = next(st.r.model.parameters()).dtype
    mA, mB, msc, mw = m["A"].numpy(), m["B"].numpy(), float(m["scale"]), float(m["w"])

    def item():
        request, fact, answer, generic = rng.choice(ex)
        with_fact = (generic is None and not args.restate) or rng.random() > args.p_none
        n = rng.choice([1, 2, 3, 3]) if args.max_facts <= 3 else rng.randint(1, args.max_facts)
        cand = [i for i in (hard[request] if rng.random() < 0.5 else rng.sample(range(len(pool)), 12)) if unlike(i, fact)]
        if len(cand) < n:
            cand = [i for i in rng.sample(range(len(pool)), 4 * n + 12) if unlike(i, fact)]
        idx = rng.sample(list(cand), n - with_fact) + ([pidx[fact]] if with_fact else [])
        if not idx:
            idx = [rng.choice(list(cand))]
        rng.shuffle(idx)
        qu = unit(qvec[request])
        score = msc * unit(PV[idx] @ mB) @ unit(qu @ mA) + mw * PV[idx] @ qu
        if args.sort:  # H25: the block the map likes best comes first
            idx = [idx[i] for i in np.argsort(-score)]
        if beta is not None:
            mem = attn_memory(C, core, RAW[idx], qvec[request], score, beta, P, torch).to(dt)
        else:
            with torch.no_grad():
                mem = PF.blocks(C, core, RAW[idx], qvec[request], 16, P, torch).to(dt)
        if delta is not None:
            mem = mem + delta(torch.from_numpy(RAW[idx]).to(P.DEV)).to(dt)
        ans = answer if with_fact else generic
        if args.restate:
            ans = f"Relevant fact: {fact if with_fact else 'none'}"
        elif args.recite:  # H19: first restate the fact that matters (or none), then answer
            ans = f"Relevant fact: {fact if with_fact else 'none'}\n{ans}"
        return request + suffix, mem, ans

    st.r.model.train()
    t0, u, run = time.perf_counter(), 0, []
    budget = args.minutes * 60
    while time.perf_counter() - t0 < budget:
        frac = min(1.0, (time.perf_counter() - t0) / budget)
        for g in opt.param_groups:
            g["lr"] = g["base"] * min(1.0, (u + 1) / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        loss = loss_fn(st, [item() for _ in range(args.batch)])
        loss.backward()
        run.append(loss.item())
        torch.nn.utils.clip_grad_norm_(lps, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        u += 1
        if u % 50 == 0 or u in (1, 5):
            print(f"update {u}  examples {u * args.batch}  loss {np.mean(run[-50:]):.3f}  {(time.perf_counter() - t0) / 60:.1f} min  peak {torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0:.2f} GiB", flush=True)
    torch.save(st.state(), OUT / f"lora_{args.tag}.pt")
    if delta is not None:
        torch.save({"rank": args.delta, "state": {k: v.cpu() for k, v in delta.state_dict().items()}}, OUT / f"lora_{args.tag}.delta.pt")
        print(f"BROADTRAIN delta rank {args.delta}  |b| {float(delta.b.weight.norm()):.3f}", flush=True)
    if beta is not None:
        torch.save({"beta": beta.detach().cpu()}, OUT / f"lora_{args.tag}.beta.pt")
        print(f"BROADTRAIN beta {[[round(float(x), 2) for x in row] for row in beta.detach().cpu()]}", flush=True)
    print(f"BROADTRAIN saved lora_{args.tag}.pt  updates {u}  examples {u * args.batch}  loss {np.mean(run[-50:]):.3f}", flush=True)
    if args.restate or args.recite:  # check on training-style examples: how often does it write "none" when the fact is there?
        st.r.model.eval()
        p_none, args.p_none = args.p_none, 0.0
        with torch.no_grad():
            chk = [item() for _ in range(64)]
            outs = FB.generate_batch(st, [(a, b) for a, b, _ in chk], 40)
        args.p_none = p_none
        none = np.mean([o.strip().lower().startswith("relevant fact: none") for o in outs])
        same = np.mean([o.strip().splitlines()[0].strip() == t.splitlines()[0].strip() if o.strip() else False for o, (_, _, t) in zip(outs, chk)])
        print(f"BROADTRAIN check, 64 training-style examples with the fact present: writes none {none:.2f}, restates it word for word {same:.2f}", flush=True)


def cmd_eval(args):
    import torch
    import poc_agentmem as A
    import poc_core as C
    import poc_fast as FB
    from poc_agentmem import P
    K = PF.poc()
    su, qvec, fvec, rows = load_all()
    s, cfg, facts, fv, info, clue_of, emap = su
    m = torch.load(K.OUT / args.map)
    ev, pmap = test_set(su, qvec, args.big)
    ev = ev[:: args.stride]
    todo = []
    for e in ev:
        qu = unit(qvec[e["question"]])

        def pick(size, inc, how, n=3):
            lst = e["others"][: size - inc] + ([e["clue"]] if inc else [])
            Fm = unit(np.stack([fvec[t] for t in lst]))
            sc = map_score(m, qu, Fm) if how == "map" else Fm @ qu
            return [lst[i] for i in np.argsort(-sc)[:n]]

        conds = [("clue alone, 16 tok", [e["clue"]], 16, True)]
        for size in R.SIZES:
            conds.append((f"map picks 3 of {size}", pick(size, True, "map"), 16, True))
            conds.append((f"map picks 3 of {size}, clue removed", pick(size, False, "map"), 16, False))
        conds += [(f"similarity picks 3 of {size}", pick(size, True, "cos"), 16, True) for size in (8, 48)]
        conds.append(("all 8 facts, 4 tok each", e["others"][:7] + [e["clue"]], 4, True))
        for size in R.SIZES:  # H20: hand over only the top pick
            conds.append((f"map picks 1 of {size}", pick(size, True, "map", 1), 16, True))
        conds.append(("map picks 1 of 8, clue removed", pick(8, False, "map", 1), 16, False))
        for size in (8, 16):  # H24: no picker, the reader sees every fact at full size
            conds.append((f"whole profile of {size}, 16 tok each", e["others"][: size - 1] + [e["clue"]], 16, True))
        conds.append(("whole profile of 8, 16 tok each, clue removed", e["others"][:8], 16, False))
        if args.attn:  # H25: the core reads the whole profile into one block, the map inside its attention
            conds = [conds[0]] + [(f"core reads profile of {size}, one block", e["others"][: size - 1] + [e["clue"]], 16, True) for size in R.SIZES]
            conds += [(f"core reads profile of {size}, one block, clue removed", e["others"][:size], 16, False) for size in (8, 48)]
            conds.append(("map picks 1 of 8", pick(8, True, "map", 1), 16, True))
        if args.conds:
            conds = [c for c in conds if any(x in c[0] for x in args.conds.replace("_", " ").split(","))]
        for name, lst, k, inc in conds:
            lst = list(lst)
            random.Random(e["pair"] + name).shuffle(lst)
            if args.sort:
                lst = [lst[i] for i in np.argsort(-map_score(m, qu, unit(np.stack([fvec[t] for t in lst]))))]
            todo.append({"group": e["group"], "pair": e["pair"], "type": e["type"], "condition": name, "facts": lst, "k": k,
                         "clue_in": inc, "clue_used": e["clue"] in lst})
    ans, t0 = [], time.perf_counter()
    if args.text:  # control: the same picked facts pasted as text for the plain decoder
        from poc_clue import D, PERSONA_PROMPT
        if args.notes:  # second pass of H23: the plain decoder answers with only the fact the reader restated
            notes = {(r["pair"], r["condition"]): r["note"] for r in json.loads((K.OUT / args.notes).read_text())["rows"]}
            todo = [r for r in todo if (r["pair"], r["condition"]) in notes]
            for r in todo:
                n = notes[(r["pair"], r["condition"])].replace("Relevant fact:", "").strip()
                r["facts"] = [] if n.lower().strip(". ") in ("", "none") else [n]
        runner = D.Runner([], K.DECODER)
        runner.tok.padding_side = "left"
        msgs = [[{"role": "system", "content": PERSONA_PROMPT},
                 {"role": "user", "content": (D.user_message(r["facts"], pmap[r["pair"]]["question"]) if r["facts"] else pmap[r["pair"]]["question"])
                  + ("" if args.long else PF.SUFFIX)}] for r in todo]
        ans = R.generate(runner.model, runner.tok, msgs, 32, args.max_new_tokens, "broad text")
        args.lora = "plain decoder, text" + (", long answers" if args.long else "")
    else:
        st = A.Stack("lora", train=True)
        C.load_lora(st, args.lora, eval_mode=True)
        core = st.core.float().eval()
        st.r.model.eval()
        dt = next(st.r.model.parameters()).dtype
        delta = None
        from poc_agentmem import OUT as AOUT
        dpath = AOUT / args.lora.replace(".pt", ".delta.pt")
        if dpath.exists():
            dd = torch.load(dpath, map_location="cpu")
            delta = make_delta(torch, dd["rank"])
            delta.load_state_dict(dd["state"])
            delta = delta.to(P.DEV).eval()
            print(f"BROAD using {dpath.name}", flush=True)
        beta = None
        if args.attn:
            beta = torch.load(AOUT / args.lora.replace(".pt", ".beta.pt"), map_location="cpu")["beta"].to(P.DEV)
    for b in range(0, 0 if args.text else len(todo), args.batch):
        items = []
        with torch.no_grad():
            for r in todo[b:b + args.batch]:
                q = pmap[r["pair"]]["question"]
                vs = np.stack([fvec[t] for t in r["facts"]])
                if beta is not None:
                    mem = attn_memory(C, core, vs, qvec[q], map_score(m, unit(qvec[q]), unit(vs)), beta, P, torch).to(dt)
                else:
                    mem = PF.blocks(C, core, vs, qvec[q], r["k"], P, torch).to(dt)
                if delta is not None:
                    mem = mem + delta(torch.from_numpy(vs.astype(np.float32)).to(P.DEV), r["k"]).to(dt)
                items.append((q + ("" if args.long else PF.SUFFIX), mem))
        ans += FB.generate_batch(st, items, args.max_new_tokens)
    print(f"BROAD eval {len(todo)} answers in {time.perf_counter() - t0:.0f}s", flush=True)
    for r, a in zip(todo, ans):
        p = pmap[r["pair"]]
        note = ""
        if args.restate:
            note, a = (a.splitlines() or [""])[0], ""
        elif args.recite and a.startswith("Relevant fact:"):
            note, _, a = a.partition("\n")
        r.update(answer=a.strip(), note=note, recites_clue=bool(note) and PF.mentions(note, p["rubrics"]),
                 **K.strict_score(a, p["rubrics"], p["value"]))
    (K.OUT / args.out).write_text(json.dumps({"lora": args.lora, "rows": todo}, indent=1))
    for gname in dict.fromkeys(r["group"] for r in todo):
        for name in dict.fromkeys(r["condition"] for r in todo):
            x = [r for r in todo if r["group"] == gname and r["condition"] == name]
            print(f"BROAD {args.lora:16s} {gname:25s} {name:36s} n={len(x):3d}  clue among the picked {np.mean([r['clue_used'] for r in x]):.2f}  "
                  f"mentions {np.mean([r['mentions_strict'] for r in x]):.2f}  acts {np.mean([r['acts_strict'] for r in x]):.2f}"
                  + (f"  restates the clue first {np.mean([r['recites_clue'] for r in x]):.2f}  writes none {np.mean(['none' in r['note'].lower()[-8:] for r in x]):.2f}"
                     if args.recite or args.restate else ""), flush=True)
    by = {}
    for r in todo:
        if r["condition"] == "map picks 3 of 16":
            by.setdefault(r["type"], []).append(r["acts_strict"])
    print("BROAD map picks 3 of 16, acts by type: " + "  ".join(f"{t} {np.mean(v):.2f}" for t, v in sorted(by.items())), flush=True)
    for r in [r for r in todo if r["condition"] == "map picks 3 of 16"][:: 19][:5]:
        print("BROAD sample", json.dumps({k: r[k] for k in ("type", "facts", "note", "answer", "acts_strict")})[:600], flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--batch", type=int, default=48)
    g.add_argument("--limit", type=int, default=0)
    em = sub.add_parser("embed")
    em.add_argument("--pairs", action="store_true")
    em.add_argument("--test", action="store_true")
    em.add_argument("--name", default="broad_pairs")
    sub.add_parser("combine").add_argument("--map", default="relmap_broad.pt")
    la = sub.add_parser("longans")
    la.add_argument("--batch", type=int, default=64)
    la.add_argument("--limit", type=int, default=0)
    la.add_argument("--max-new-tokens", type=int, default=220)
    pr = sub.add_parser("pairs")
    pr.add_argument("--batch", type=int, default=96)
    pr.add_argument("--limit", type=int, default=0)
    pr.add_argument("--who", type=int, default=10)
    pr.add_argument("--who-from", type=int, default=0)
    pr.add_argument("--out", default="broad_pairs")
    mp = sub.add_parser("map")
    mp.add_argument("--rank", type=int, default=64)
    mp.add_argument("--lr", type=float, default=1e-3)
    mp.add_argument("--wd", type=float, default=0.05)
    mp.add_argument("--dropout", type=float, default=0.2)
    mp.add_argument("--epochs", type=int, default=80)
    mp.add_argument("--patience", type=int, default=6)
    mp.add_argument("--batch", type=int, default=128)
    mp.add_argument("--seeds", type=int, default=3)
    mp.add_argument("--save", default="relmap_broad.pt")
    mp.add_argument("--big", action="store_true")
    mp.add_argument("--dup", type=float, default=2.0)
    mp.add_argument("--fractions", type=lambda x: [float(v) for v in x.split(",")], default=[0.1, 0.3, 1.0])
    t = sub.add_parser("train")
    t.add_argument("--init", default="lora_prof.pt")
    t.add_argument("--tag", default="broad")
    t.add_argument("--minutes", type=float, default=18)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--p-none", type=float, default=0.25)
    t.add_argument("--recite", action="store_true")
    t.add_argument("--long", action="store_true")
    t.add_argument("--restate", action="store_true")
    t.add_argument("--max-facts", type=int, default=3)
    t.add_argument("--delta", type=int, default=0)
    t.add_argument("--delta-lr", type=float, default=3e-4)
    t.add_argument("--map", default="relmap_broad.pt")
    t.add_argument("--dup", type=float, default=0.6)
    t.add_argument("--sort", action="store_true")
    t.add_argument("--attn", action="store_true")
    t.add_argument("--beta-lr", type=float, default=3e-3)
    e = sub.add_parser("eval")
    e.add_argument("--lora", default="lora_broad.pt")
    e.add_argument("--out", default="broad_eval.json")
    e.add_argument("--stride", type=int, default=1)
    e.add_argument("--batch", type=int, default=16)
    e.add_argument("--max-new-tokens", type=int, default=70)
    e.add_argument("--recite", action="store_true")
    e.add_argument("--text", action="store_true")
    e.add_argument("--long", action="store_true")
    e.add_argument("--map", default="relmap_broad.pt")
    e.add_argument("--conds", default="")
    e.add_argument("--restate", action="store_true")
    e.add_argument("--notes", default="")
    e.add_argument("--big", action="store_true")
    e.add_argument("--sort", action="store_true")
    e.add_argument("--attn", action="store_true")
    args = ap.parse_args()
    {"gen": cmd_gen, "pairs": cmd_pairs, "longans": cmd_longans, "combine": cmd_combine, "embed": cmd_embed, "map": cmd_map, "train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
