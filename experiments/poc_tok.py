"""Test 43: the overnight recipe on Memorilla's 16-token memory blocks. No fact is ever given as text.

Tests 35b-41 found that rating each fact alone with a yes/no and keeping the top few beats putting the whole profile in
the prompt, but every one of them gave the facts as text. Here both steps read the fact as its 16 memory tokens (frozen
core, Qwen3-8B, native head):

    rate     each fact of the 48-fact profile alone: its 16-token block in memory, the request and a yes/no question in the
             user turn, score = logit(Yes) - logit(No). Two raters: the plain decoder (table tok_<wording>) and the reader
             LoRA (tokL_<wording>). Tables go to select_scores.json keyed by W.fact_key, next to the text raters' tables
             (r8_change, r27_wrong), and the clue's rank is printed for all of them.
    answer   the reader LoRA (lora_lsort.pt, trained on best-first blocks and "Relevant fact:" first) answers from the top k
             blocks of a table. Controls: the clue alone, the whole profile of 8 as blocks, the clue removed.
    dry      no model: builds the rows, prints one rating prompt and checks the ranking and selection code (runs on the Mac).

Run on the V100 box (one GPU each; rate first, it writes the tables the answer runs read):
    experiments/v100.sh run tokrate 0 poc_tok.py rate
    experiments/v100.sh run toka 0 poc_tok.py answer --tables tok_change --out tok_a.json
    experiments/v100.sh run tokb 1 poc_tok.py answer --tables r27_wrong --no-controls --out tok_b.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import poc_broad as B  # noqa: E402
import poc_profile as PF  # noqa: E402
import poc_whole as W  # noqa: E402
import q8  # noqa: E402

SIZES = (8, 16, 48)
WORD = {  # the text wordings of tests 35c and 41 with "the fact above" -> "the fact in your memory"
    "change": 'The user asks: "{q}"\n\nDoes the fact in your memory change what a good reply should say or recommend? Answer Yes or No.',
    "wrong": 'The user asks: "{q}"\n\nCould a typical reply to this request be wrong, unsafe or unsuitable for this user because of the fact in your memory? Answer Yes or No.',
}
RATERS = {"tok": False, "tokL": True}  # table prefix -> reader LoRA on
TEXT_TABLES = ("r8_change", "r27_wrong")  # text raters, printed for comparison when present
T0 = time.perf_counter()


def profile_facts(e, size, inc=True):
    """The facts of a profile of `size` (order does not matter here: every fact is rated alone)."""
    return e["others"][: size - inc] + ([e["clue"]] if inc else [])


def ranked(tab, q, facts):
    """Facts best first by a per-fact table; a fact without a score goes last. -> (facts, number missing)"""
    sc = [tab.get(W.fact_key(q, f)) for f in facts]
    order = np.argsort([-(-1e9 if s is None else s) for s in sc], kind="stable")
    return [facts[i] for i in order], sum(s is None for s in sc)


def rank_report(name, tab, ev, pmap):
    """Where the clue ranks among the facts of a profile of 8 / 16 / 48, unseen types."""
    held = [e for e in ev if e["group"] == "held-out types"]
    for size in SIZES:
        pos, miss = [], 0
        for e in held:
            lst, m = ranked(tab, pmap[e["pair"]]["question"], profile_facts(e, size))
            miss += m
            pos.append(lst.index(e["clue"]))
        pos = np.array(pos)
        by = {}
        for e, p in zip(held, pos):
            by.setdefault(e["type"], []).append(p < 2)
        print(f"TOK rank {name:12s} from {size:2d}: clue first {np.mean(pos < 1):.2f}  top 2 {np.mean(pos < 2):.2f}  top 4 {np.mean(pos < 4):.2f}  "
              f"top 8 {np.mean(pos < 8):.2f}  median rank {int(np.median(pos))}  missing {miss}; top 2 by type: "
              + "  ".join(f"{t} {np.mean(v):.2f}" for t, v in sorted(by.items())), flush=True)


def load_tables(K):
    p = K.OUT / "select_scores.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_tables(K, new):
    """Merge into select_scores.json (other tables are kept); write through a temp file."""
    p = K.OUT / "select_scores.json"
    full = load_tables(K)
    full.update(new)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(full))
    tmp.replace(p)


def rating_ids(A, tok, add_special, system, user, n_mem):
    """Token ids of the rating prompt: system prompt with the memory placeholder, the yes/no question, thinking off."""
    msgs = A.build_messages(user, system, num_memories=n_mem)
    return tok.encode(A.apply_chat_template(tok, msgs[:2], add_generation_prompt=True), add_special_tokens=add_special)


def qfirst_ids(A, tok, add_special, system, user, n_mem, memory_id):
    """Rating prompt with the request first and the memory block last (layout "qfirst"): plain system prompt, then the
    user turn = yes/no question + the 16 memory tokens. Everything before the first memory token depends only on the
    request, so all facts of a request share it (shared_hidden)."""
    m = tok.convert_ids_to_tokens(memory_id)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": f"{user}\n\nMemory{m * n_mem}Memory"}]
    return tok.encode(A.apply_chat_template(tok, msgs, add_generation_prompt=True), add_special_tokens=add_special)


def prompt_ids(A, tok, add_special, memory_id, system, user, layout):
    """(prefix, suffix) token ids, split at the first memory token. layout "sys": memory in the system prompt (the
    original); "qfirst": request first, memory last."""
    ids = rating_ids(A, tok, add_special, system, user, 16) if layout == "sys" else qfirst_ids(A, tok, add_special, system, user, 16, memory_id)
    k = ids.index(memory_id)
    return ids[:k], ids[k:]


def shared_hidden(model, pre, suf, grp, every=False):
    """Final hidden state at the last position of each suffix, every suffix reading after its group's prefix.

    pre: G prefix embeddings [Lp_g, dim]; suf: M suffix embeddings [Ls_m, dim]; grp[m]: the prefix of suffix m.
    Each prefix is computed once and its keys/values are shared by its suffixes (a training-time prefix cache).
    Tokens stay one flat matrix for every projection, so each 4-bit weight is unpacked once per pass for all of them;
    only attention is split. Right padding, positions counted from each row's first real token. Same maths as
    the Qwen3 decoder layer (pre-norm, q/k norm, RoPE, GQA); each layer is checkpointed when training.
    every=True: the normed hidden state at every suffix position, [M, Ls, dim] (padding rows included)."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    mm = model.model
    dev, dim = pre[0].device, pre[0].shape[-1]
    G, M = len(pre), len(suf)
    lp = torch.tensor([x.shape[0] for x in pre], device=dev)
    ls = torch.tensor([x.shape[0] for x in suf], device=dev)
    Lp, Ls = int(lp.max()), int(ls.max())
    g = torch.tensor(grp, device=dev)
    hp, hs = pre[0].new_zeros(G, Lp, dim), suf[0].new_zeros(M, Ls, dim)
    for i, x in enumerate(pre):
        hp[i, :x.shape[0]] = x
    for i, x in enumerate(suf):
        hs[i, :x.shape[0]] = x
    ar_p, ar_s = torch.arange(Lp, device=dev), torch.arange(Ls, device=dev)
    pos = torch.cat([ar_p.expand(G, -1).reshape(-1), (lp[g][:, None] + ar_s[None]).reshape(-1)])[None]
    cos, sin = (x[0] for x in mm.rotary_emb(hp, pos))  # [T, head_dim]
    valid = ar_p[None] < lp[:, None]  # [G, Lp] real prefix keys
    mask_p = (torch.tril(torch.ones(Lp, Lp, dtype=torch.bool, device=dev))[None] & valid[:, None, :])[:, None]
    mask_s = torch.cat([valid[g][:, None, :].expand(M, Ls, Lp), torch.tril(torch.ones(Ls, Ls, dtype=torch.bool, device=dev))[None].expand(M, -1, -1)], -1)[:, None]
    n_p = G * Lp
    h = torch.cat([hp.reshape(-1, dim), hs.reshape(-1, dim)])

    def layer_fn(layer, h):
        a = layer.self_attn
        hd = a.head_dim
        x = layer.input_layernorm(h)
        q = a.q_norm(a.q_proj(x).view(h.shape[0], -1, hd))
        k = a.k_norm(a.k_proj(x).view(h.shape[0], -1, hd))
        v = a.v_proj(x).view(h.shape[0], -1, hd)
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1)
        heads = lambda t, n, L: t.view(n, L, t.shape[1], hd).transpose(1, 2)  # noqa: E731
        qp, kp, vp = (heads(t[:n_p], G, Lp) for t in (q, k, v))
        qs, ks, vs = (heads(t[n_p:], M, Ls) for t in (q, k, v))
        op = F.scaled_dot_product_attention(qp, kp, vp, attn_mask=mask_p, scale=a.scaling, enable_gqa=True)
        os_ = F.scaled_dot_product_attention(qs, torch.cat([kp[g], ks], 2), torch.cat([vp[g], vs], 2), attn_mask=mask_s, scale=a.scaling, enable_gqa=True)
        o = torch.cat([op.transpose(1, 2).reshape(n_p, -1), os_.transpose(1, 2).reshape(M * Ls, -1)])
        h = h + a.o_proj(o)
        return h + layer.mlp(layer.post_attention_layernorm(h))

    ckpt = model.training and torch.is_grad_enabled()
    for layer in mm.layers:
        h = checkpoint(layer_fn, layer, h, use_reentrant=False) if ckpt else layer_fn(layer, h)
    if every:
        return mm.norm(h[n_p:].view(M, Ls, dim))
    last = h[n_p:].view(M, Ls, dim)[torch.arange(M, device=dev), ls - 1]
    return mm.norm(last)


def load_stack(lora):
    import torch
    import poc_agentmem as A
    import poc_core as C
    st = A.Stack("lora", train=True)  # train=True: do not auto-load lora.pt
    C.load_lora(st, lora, eval_mode=True)
    st.r.model.eval()
    core = st.core.float().eval()
    return torch, A, C, st, core, next(st.r.model.parameters()).dtype


def set_lora(st, on):
    """Reader LoRA on or off (off = the plain decoder reading the memory tokens)."""
    if not hasattr(st, "lora_scale"):
        st.lora_scale = [m.scale for m in st.lora]
    for m, s in zip(st.lora, st.lora_scale):
        m.scale = s if on else 0.0


def cmd_rate(args):
    from poc_agentmem import P
    from poc_clue import PERSONA_PROMPT
    K = PF.poc()
    su, qvec, fvec, _ = B.load_all()
    ev, pmap = B.test_set(su, qvec, True)
    ev = ev[:: args.stride]
    jobs = list(dict.fromkeys((pmap[e["pair"]]["question"], f) for e in ev for f in profile_facts(e, 48)))
    print(f"TOK rate: {len(ev)} questions, {len(jobs)} (request, fact) pairs, wordings {args.wordings}, raters {args.raters}", flush=True)
    torch, A, C, st, core, dt = load_stack(args.lora)
    model, tok = st.r.model, st.r.tok
    first = lambda ws: sorted({tok.encode(w, add_special_tokens=False)[0] for w in ws})  # noqa: E731
    yes, no = first([" Yes", "Yes", " yes"]), first([" No", "No", " no"])
    head = model.lm_head.weight[yes + no].float().to(P.DEV)
    q8.drop_lm_head(model)  # MPORT_LEAN=1 only: rating reads model.model and these rows
    if q8.lean():  # the memory core runs on the CPU: a few hundred vectors per batch, and 0.8 GB less on the GPU
        core = q8.place_core(core)
    cdev = next(core.parameters()).device
    bq = dict(qvec)  # which question vector the core gets for a block (test 49: the reader does not need the right one)
    if args.block_q == "mean":  # one fixed vector: the blocks could then be built once, when the fact is stored
        qs_all = sorted({q for q, _ in jobs})
        m = np.mean([qvec[q] for q in qs_all], axis=0)
        m = (m * (np.mean([np.linalg.norm(qvec[q]) for q in qs_all]) / np.linalg.norm(m))).astype(qvec[qs_all[0]].dtype)
        bq = {q: m for q in qs_all}
        print(f"TOK rate: blocks built with the mean question vector ({len(qs_all)} distinct questions)", flush=True)
    for wd in args.wordings.split(","):
        if args.shared:
            sp = {q: prompt_ids(A, tok, st.r.add_special, st.r.memory_id, PERSONA_PROMPT, WORD[wd].format(q=q), args.layout) for q in dict.fromkeys(q for q, _ in jobs)}
        else:
            assert args.layout == "sys", "--layout qfirst needs --shared 1"
            ids = [rating_ids(A, tok, st.r.add_special, PERSONA_PROMPT, WORD[wd].format(q=q), 16) for q, _ in jobs]
            order = sorted(range(len(jobs)), key=lambda i: len(ids[i]))
        for rater in args.raters.split(","):
            set_lora(st, RATERS[rater])
            sc, t0 = [0.0] * len(jobs), time.perf_counter()
            if args.shared:
                byq = {}
                for i, (q, _) in enumerate(jobs):
                    byq.setdefault(q, []).append(i)
                parts = [(q, ix[k:k + args.batch]) for q, ix in byq.items() for k in range(0, len(ix), args.batch)]
                batches, cur = [], []
                for part in parts:  # whole requests packed up to --batch facts; bigger ones split
                    if cur and sum(len(p[1]) for p in cur) + len(part[1]) > args.batch:
                        batches.append(cur)
                        cur = []
                    cur.append(part)
                batches.append(cur)
                for bn, chunk in enumerate(batches):
                    qs = [q for q, _ in chunk]
                    idx = [i for _, ix in chunk for i in ix]
                    with torch.no_grad():
                        d = torch.from_numpy(np.stack([fvec[jobs[i][1]] for i in idx]).astype(np.float32))[:, None].to(cdev)
                        qv = torch.from_numpy(np.stack([bq[jobs[i][0]] for i in idx]).astype(np.float32)).to(cdev)
                        mem = C.core_fwd(core, d, qv).to(P.DEV, dt)
                        pre = [P.input_embeds(model, tok, st.r.memory_id, torch.tensor([sp[q][0]], device=P.DEV), None)[0] for q in qs]
                        suf = [P.input_embeds(model, tok, st.r.memory_id, torch.tensor([sp[jobs[i][0]][1]], device=P.DEV), mem[j:j + 1])[0] for j, i in enumerate(idx)]
                        h = shared_hidden(model, pre, suf, [qs.index(jobs[i][0]) for i in idx])
                        lg = h.float() @ head.t()
                        out = torch.logsumexp(lg[:, :len(yes)], -1) - torch.logsumexp(lg[:, len(yes):], -1)
                    for i, v in zip(idx, out.tolist()):
                        sc[i] = v
                    if bn == 0:
                        print(f"TOK rate first batch: {len(qs)} requests, {len(idx)} facts in {time.perf_counter() - t0:.1f}s (shared prefix, layout {args.layout})", flush=True)
            for b in range(0, len(order), args.batch) if not args.shared else ():
                idx = order[b:b + args.batch]
                with torch.no_grad():
                    d = torch.from_numpy(np.stack([fvec[jobs[i][1]] for i in idx]).astype(np.float32))[:, None].to(cdev)
                    q = torch.from_numpy(np.stack([bq[jobs[i][0]] for i in idx]).astype(np.float32)).to(cdev)
                    mem = C.core_fwd(core, d, q).to(P.DEV, dt)  # [n, 16, dim]: each fact alone, conditioned on its request
                    embs = [P.input_embeds(model, tok, st.r.memory_id, torch.tensor([ids[i]], device=P.DEV), mem[j:j + 1])[0] for j, i in enumerate(idx)]
                    t = max(x.shape[0] for x in embs)
                    x = embs[0].new_zeros(len(embs), t, embs[0].shape[1])
                    mask = torch.zeros(len(embs), t, dtype=torch.long, device=P.DEV)
                    for j, e in enumerate(embs):  # left padding: the last position is the one that answers
                        x[j, t - e.shape[0]:] = e
                        mask[j, t - e.shape[0]:] = 1
                    h = model.model(inputs_embeds=x, attention_mask=mask).last_hidden_state[:, -1]
                    lg = h.float() @ head.t()
                    out = torch.logsumexp(lg[:, :len(yes)], -1) - torch.logsumexp(lg[:, len(yes):], -1)
                for i, v in zip(idx, out.tolist()):
                    sc[i] = v
                if b == 0:
                    print(f"TOK rate first batch: {len(idx)} prompts of {t} positions in {time.perf_counter() - t0:.1f}s", flush=True)
            name = f"{rater}{args.tag}_{wd}"
            tab = {W.fact_key(q, f): s for (q, f), s in zip(jobs, sc)}
            save_tables(K, {name: tab})
            print(f"TOK rate {name}: {len(jobs)} ratings in {time.perf_counter() - t0:.0f}s, says Yes to {np.mean(np.array(sc) > 0):.2f}", flush=True)
            rank_report(name, tab, ev, pmap)
    set_lora(st, True)
    full = load_tables(K)
    for name in TEXT_TABLES:
        if name in full:
            rank_report(name + " (text)", full[name], ev, pmap)
    print(f"TOK rate done at +{(time.perf_counter() - T0) / 60:.1f} min", flush=True)


def conditions(tables, controls, extra=""):
    """(name, size, clue included, table or None, k). Blocks go best first, the order the reader was trained on."""
    out = []
    if controls:
        out += [("clue alone", 1, True, None, 1), ("whole profile of 8, 16 tok each", 8, True, None, 8)]
    for t in tables:
        out += [(f"{t} top {k} from {size}", size, True, t, k) for k in (1, 2) for size in SIZES]
        out += [(f"{t} top 4 from {size}", size, True, t, 4) for size in (8, 48)]
        out.append((f"{t} top 2 from 8, clue removed", 8, False, t, 2))
        for x in filter(None, extra.split(",")):  # e.g. "8:48" -> top 8 from 48; "8:48:x" the same with the clue removed
            k, size, *rm = x.split(":")
            out.append((f"{t} top {k} from {size}" + (", clue removed" if rm else ""), int(size), not rm, t, int(k)))
    return out


def build_rows(ev, pmap, full, conds):
    todo = []
    for name, size, inc, table, k in conds:
        miss = n = 0
        for e in ev:
            q = pmap[e["pair"]]["question"]
            facts = profile_facts(e, size, inc)
            if table is None:
                shown = W.profile(e, size, inc, name)[:k]
            else:
                lst, m = ranked(full[table], q, facts)
                shown = lst[:k]
                held = e["group"] == "held-out types"
                miss += m * held
                n += len(facts) * held
            todo.append({"group": e["group"], "pair": e["pair"], "type": e["type"], "condition": name, "facts": shown,
                         "clue_in": inc, "clue_shown": e["clue"] in shown, "clue_first": shown[0] == e["clue"]})
        assert miss <= 0.01 * max(n, 1), f"{name}: saved scores missing for {miss} of {n} unseen-type facts (run: poc_tok.py rate)"
    return todo


def cmd_answer(args):
    import poc_fast as FB
    from poc_agentmem import P
    K = PF.poc()
    su, qvec, fvec, _ = B.load_all()
    ev, pmap = B.test_set(su, qvec, True)
    ev = ev[:: args.stride]
    full = load_tables(K)
    conds = conditions(args.tables.split(","), not args.no_controls, args.extra)
    if args.conds:
        conds = [c for c in conds if any(x in c[0] for x in args.conds.replace("+", " ").split(","))]
    print(f"TOK answer: {len(ev)} questions, reader {args.lora}, conditions: " + "; ".join(c[0] for c in conds), flush=True)
    todo = build_rows(ev, pmap, full, conds)
    if args.text_url:  # another model on a server answers: nothing is loaded here
        args.text = True
    else:
        torch, A, C, st, core, dt = load_stack(args.lora)
        if q8.lean():  # 4-bit lean box run: core on the CPU, the whole output layer on the GPU for generation
            core = q8.place_core(core)
            q8.head_to_gpu(st.r.model)
    ans, t0 = [], time.perf_counter()
    # control: which question vector the core gets when it builds a block (the decoder always reads the right question as text)
    qs = sorted({pmap[r["pair"]]["question"] for r in todo})
    if args.block_q == "wrong":  # another test question's vector, half the list away
        block_q = {q: qvec[qs[(i + len(qs) // 2) % len(qs)]] for i, q in enumerate(qs)}
    elif args.block_q == "mean":  # the same averaged vector for every question, at the usual length
        m = np.mean([qvec[q] for q in qs], axis=0)
        m = m * (np.mean([np.linalg.norm(qvec[q]) for q in qs]) / np.linalg.norm(m))
        block_q = {q: m.astype(qvec[qs[0]].dtype) for q in qs}
    else:
        block_q = {q: qvec[q] for q in qs}
    if args.block_q != "right":
        print(f"TOK answer: blocks built with the {args.block_q} question vector ({len(qs)} distinct questions)", flush=True)

    def text_batch(rows):  # control: the plain decoder (LoRA off) reads the same facts, in the same order, as text
        from poc_clue import D, PERSONA_PROMPT
        tok = st.r.tok
        prompts = [A.apply_chat_template(tok, [{"role": "system", "content": PERSONA_PROMPT},
                                               {"role": "user", "content": D.user_message(list(r["facts"]), pmap[r["pair"]]["question"].strip())}],
                                         add_generation_prompt=True) for r in rows]
        side, tok.padding_side = tok.padding_side, "left"
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=st.r.add_special).to(P.DEV)
        tok.padding_side = side
        with torch.no_grad(), FB.lora_off(st):
            out = st.r.model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        return [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]

    def server_batch(rows):  # control: another model on a vLLM server reads the same facts as text (thinking off), one request each
        import urllib.request
        from concurrent.futures import ThreadPoolExecutor
        from poc_clue import D, PERSONA_PROMPT
        url = args.text_url.rstrip("/")

        def one(r):
            body = {"model": args.text_model, "max_tokens": args.max_new_tokens, "temperature": 0,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "messages": [{"role": "system", "content": PERSONA_PROMPT},
                                 {"role": "user", "content": D.user_message(list(r["facts"]), pmap[r["pair"]]["question"].strip())}]}
            req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
            return (json.loads(urllib.request.urlopen(req, timeout=1800).read())["choices"][0]["message"]["content"] or "").strip()

        while W._waiting(url) > 0:  # a shared server: someone else is queued, let them go first
            time.sleep(2)
        with ThreadPoolExecutor(len(rows)) as ex:
            return list(ex.map(one, rows))

    if not args.text and os.environ.get("MPORT_FASTGEN") == "1":  # fast_gen.py: one pass over all rows, batch from free memory
        import fast_gen as FG
        items = []
        with torch.no_grad():
            for r in todo:
                q = pmap[r["pair"]]["question"]
                items.append((q, PF.blocks(C, core, np.stack([fvec[t] for t in r["facts"]]), block_q[q], 16, P, torch).to(dt).cpu()))
        print(f"TOK answer: {len(items)} blocks sets built in {time.perf_counter() - t0:.0f}s", flush=True)
        core.cpu()  # every block is built: the core's 0.8 GB on the GPU becomes room for ~14 more sequences
        torch.cuda.empty_cache()
        ans = FG.generate(st, items, args.max_new_tokens, 0, log=lambda m: print(m, flush=True))
        print(f"TOK answer first row: {ans[0][:300]!r}", flush=True)
    for b in range(0, 0 if ans else len(todo), args.batch):
        if args.text:
            ans += (server_batch if args.text_url else text_batch)(todo[b:b + args.batch])
            if b == 0:
                print(f"TOK answer (text control) first batch in {time.perf_counter() - t0:.0f}s: {ans[0][:300]!r}", flush=True)
            if (b // args.batch) % 10 == 9:
                print(f"TOK answer {len(ans)}/{len(todo)} at +{(time.perf_counter() - T0) / 60:.1f} min", flush=True)
            continue
        items = []
        with torch.no_grad():
            for r in todo[b:b + args.batch]:
                q = pmap[r["pair"]]["question"]
                vs = np.stack([fvec[t] for t in r["facts"]])
                items.append((q, PF.blocks(C, core, vs, block_q[q], 16, P, torch).to(dt)))
        ans += FB.generate_batch(st, items, args.max_new_tokens)
        if b == 0:
            print(f"TOK answer first batch in {time.perf_counter() - t0:.0f}s: {ans[0][:300]!r}", flush=True)
        if (b // args.batch) % 10 == 9:
            print(f"TOK answer {len(ans)}/{len(todo)} at +{(time.perf_counter() - T0) / 60:.1f} min", flush=True)
    for r, a in zip(todo, ans):
        p = pmap[r["pair"]]
        note = ""
        if a.startswith("Relevant fact:"):
            note, _, a = a.partition("\n")
        r.update(answer=a.strip(), note=note, recites_clue=bool(note) and PF.mentions(note, p["rubrics"]), **K.strict_score(a, p["rubrics"], p["value"]))
    (K.OUT / args.out).write_text(json.dumps({"lora": args.lora, "args": vars(args), "rows": todo}, indent=1))
    report(todo)
    print(f"TOK answer done: {len(todo)} answers, +{(time.perf_counter() - T0) / 60:.1f} min", flush=True)


def report(todo):
    for g in dict.fromkeys(r["group"] for r in todo):
        for name in dict.fromkeys(r["condition"] for r in todo):
            x = [r for r in todo if r["group"] == g and r["condition"] == name]
            print(f"TOK {g:25s} {name:36s} n={len(x):3d}  acts {np.mean([r['acts_strict'] for r in x]):.2f}  mentions {np.mean([r['mentions_strict'] for r in x]):.2f}  "
                  f"clue shown {np.mean([r['clue_shown'] for r in x]):.2f}  restates the clue first {np.mean([r['recites_clue'] for r in x]):.2f}  "
                  f"writes none {np.mean(['none' in r['note'].lower()[-8:] for r in x]):.2f}", flush=True)
            if g == "held-out types":
                by = {}
                for r in x:
                    by.setdefault(r["type"], []).append(r["acts_strict"])
                print(f"TOK {name} acts by type: " + "  ".join(f"{t} {np.mean(v):.2f}" for t, v in sorted(by.items())), flush=True)


def cmd_dry(args):
    """No model: rows, one rendered rating prompt, ranking and selection with made-up scores."""
    import poc_agentmem as A
    from poc_agentmem import P
    from poc_clue import PERSONA_PROMPT
    from transformers import AutoTokenizer
    K = PF.poc()
    su, qvec, fvec, _ = B.load_all()
    ev, pmap = B.test_set(su, qvec, True)
    jobs = list(dict.fromkeys((pmap[e["pair"]]["question"], f) for e in ev for f in profile_facts(e, 48)))
    print(f"TOK dry: {len(ev)} questions, {len(jobs)} pairs, vectors missing: facts {sum(f not in fvec for _, f in jobs)}, requests {sum(q not in qvec for q, _ in jobs)}")
    tok = AutoTokenizer.from_pretrained(K.DECODER)
    P.prepare_tokenizer(tok)
    ids = rating_ids(A, tok, False, PERSONA_PROMPT, WORD["change"].format(q=jobs[0][0]), 16)
    print(f"TOK dry prompt ({len(ids)} tokens):\n" + tok.decode(ids))
    first = lambda ws: sorted({tok.encode(w, add_special_tokens=False)[0] for w in ws})  # noqa: E731
    print("TOK dry yes/no ids", first([" Yes", "Yes", " yes"]), first([" No", "No", " no"]))
    rng = np.random.default_rng(0)
    clues = {(pmap[e["pair"]]["question"], e["clue"]) for e in ev}
    tab = {W.fact_key(q, f): float(rng.normal()) + 10.0 * ((q, f) in clues) for q, f in jobs}  # the clue always wins
    rank_report("made-up", tab, ev, pmap)
    conds = conditions(["made-up"], True)
    todo = build_rows(ev, pmap, {"made-up": tab}, conds)
    for name in dict.fromkeys(r["condition"] for r in todo):
        x = [r for r in todo if r["condition"] == name and r["group"] == "held-out types"]
        print(f"TOK dry {name:36s} n={len(x)}  blocks {np.mean([len(r['facts']) for r in x]):.1f}  clue shown {np.mean([r['clue_shown'] for r in x]):.2f}  clue first {np.mean([r['clue_first'] for r in x]):.2f}")
    full = load_tables(K)
    print("TOK dry tables on this machine:", sorted(full))
    for name in TEXT_TABLES:
        if name in full:
            rank_report(name + " (text)", full[name], ev, pmap)


def cmd_ensemble(args):
    """No model: average several per-fact tables (each z-scored over its own values) into one, save it, and print the
    rank report of every input and of the average. Separate training runs of the block rater disagree by about 0.1 on
    "clue first"; their mean should be steadier than any one of them."""
    K = PF.poc()
    su, qvec, fvec, _ = B.load_all()
    ev, pmap = B.test_set(su, qvec, True)
    ev = ev[:: args.stride]
    full = load_tables(K)
    names = args.tables.split(",")
    keys = set.intersection(*(set(full[n]) for n in names))
    out = {k: 0.0 for k in keys}
    for n in names:
        v = np.array([full[n][k] for k in keys])
        mu, sd = float(v.mean()), float(v.std()) or 1.0
        for k in keys:
            out[k] += (full[n][k] - mu) / sd / len(names)
        rank_report(n, full[n], ev, pmap)
    print(f"TOK ensemble of {len(names)} tables over {len(keys)} pairs -> {args.name}", flush=True)
    rank_report(args.name, out, ev, pmap)
    save_tables(K, {args.name: out})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("rate")
    r.add_argument("--wordings", default="change,wrong")
    r.add_argument("--raters", default="tok,tokL")
    r.add_argument("--batch", type=int, default=96)
    r.add_argument("--tag", default="", help="suffix after the rater name in the table name, e.g. 90 -> tokL90_change")
    r.add_argument("--layout", default="sys", choices=["sys", "qfirst"], help="memory in the system prompt (original) or after the request")
    r.add_argument("--block-q", default="right", choices=["right", "mean"], help="question vector given to the core for each block")
    r.add_argument("--shared", type=int, default=1, help="1: compute each request's prefix once for all its facts (shared_hidden)")
    a = sub.add_parser("answer")
    a.add_argument("--tables", default="tok_change")
    a.add_argument("--no-controls", action="store_true")
    a.add_argument("--text", action="store_true", help="control: the plain decoder (LoRA off) reads the same picked facts as text")
    a.add_argument("--text-url", default="", help="text control answered by a vLLM server (e.g. http://127.0.0.1:8001); use a small --batch")
    a.add_argument("--text-model", default="", help="model name on that server")
    a.add_argument("--block-q", default="right", choices=["right", "wrong", "mean"],
                   help="question vector given to the core for each block: the right one, another question's, or the average")
    a.add_argument("--extra", default="", help="more picks as k:size pairs, e.g. 8:48,4:16 (top 8 from 48, top 4 from 16)")
    a.add_argument("--conds", default="", help="comma-separated substrings of condition names, + for a space")
    a.add_argument("--batch", type=int, default=32)
    a.add_argument("--max-new-tokens", type=int, default=300)
    a.add_argument("--out", default="tok_answer.json")
    d = sub.add_parser("dry")
    en = sub.add_parser("ensemble")
    en.add_argument("--tables", required=True, help="comma-separated table names in select_scores.json")
    en.add_argument("--name", required=True, help="name of the averaged table, e.g. tokLens_change")
    for p in (r, a, d, en):
        p.add_argument("--lora", default="lora_lsort.pt")
        p.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()
    {"rate": cmd_rate, "answer": cmd_answer, "dry": cmd_dry, "ensemble": cmd_ensemble}[args.cmd](args)


if __name__ == "__main__":
    main()
