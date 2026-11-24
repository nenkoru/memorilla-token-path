"""H31: no picker. The whole profile is in the prompt as plain text; the decoder's LoRA is trained to first restate the
fact that matters for the request and then answer acting on it. Text is the upper bound for memory tokens.

One process, time-boxed, results written after every step (the run may be cut off by the spending limit):
  probe      one worst-case training batch (memory) and two answers through the scoring path
  untrained  the plain decoder (the LoRA starts at zero), whole profile of 8 and 48, every second question
  train      --minutes; restate-only examples from all known pairs, restate + long answer from broad_long.json
  trained    whole profile of 8, 48, 8 with the clue removed, 16; a step is skipped once --deadline is reached

  experiments/modal.sh run whole poc_whole.py --minutes 11 --deadline 25

--mode zero (H32, no training): the plain decoder with the whole profile as text
  plain      answer directly
  quote      one pass: first quote the facts that should change the answer, then answer
  groups     two passes: quote such facts from every group of 8, then answer with only the quoted facts
  --conds picks conditions by substring (commas; _ for a space), --out names the result file.
  MPORT_SCRATCH=$PWD/work work/venv/bin/python -I experiments/poc_whole.py --dry     (no model: data and token counts)
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

import poc_broad as B  # noqa: E402
import poc_profile as PF  # noqa: E402
import poc_relevance as R  # noqa: E402
import fast_train as FT  # noqa: E402

LINE = "Relevant fact: "
QUOTE = '\n\nFirst write one line that starts with "Relevant facts:" and quotes the facts above that should change your answer, or says "none". Then answer.'
JUDGE = '\n\nDo not answer yet. Write one line that starts with "Relevant facts:" and quotes the facts above that should change the answer to this request, or says "none".'
BESTOF = ("Do not write a reply of your own. Using everything you know about this user, decide which reply serves them best: one that"
          " takes into account whatever about them matters for this request. Answer with the reply's letter only.")
RATE = 'The user asks: "{q}"\n\nShould a good reply to this request take the fact above into account? Answer Yes or No.'
RATES = {"change": 'The user asks: "{q}"\n\nDoes the fact above change what a good reply should say or recommend? Answer Yes or No.'}  # test 35c
STOP = {"user", "that", "with", "have", "their", "they", "this", "from", "when", "been", "does", "will"}
TRAIN_SIZES = [4, 8, 8, 12, 16, 16, 24, 32, 48, 48]
T0 = time.perf_counter()


def minutes():
    return (time.perf_counter() - T0) / 60


def norm(t):
    return " ".join(t.lower().split()).strip(". ")


def words(t):
    return [w for w in re.findall(r"[a-z']+", t.lower()) if len(w) > 3 and w not in STOP]


def quoted(text, facts):
    """Indices of the facts a note quotes: the whole fact, or at least 0.6 of its content words."""
    t, have = norm(text), set(words(text))
    out = []
    for i, f in enumerate(facts):
        w = words(f)
        if norm(f) in t or (w and sum(x in have for x in w) / len(w) >= 0.6):
            out.append(i)
    return out


def split_note(a):
    """-> (the leading "Relevant fact(s):" note, the answer after it)."""
    if not a.lstrip().lower().startswith("relevant fact"):
        return "", a
    lines = a.lstrip().split("\n")
    end = 1
    if len(lines) > 1 and re.match(r"\s*([-*\"\u201c]|\d+[.)])", lines[1]):  # quotes as a list: the note runs to the first blank line
        while end < len(lines) and lines[end].strip():
            end += 1
    return "\n".join(lines[:end]), "\n".join(lines[end:])


class Sampler:
    """Training examples: a request, a profile of n facts as text (the right fact among them, or not), the target."""

    def __init__(self, args, K, torch, dev):
        self.args, self.torch, self.rng = args, torch, random.Random(0)
        su, self.qvec, fvec, rows = B.load_all()
        self.su = su
        facts = su[2]
        self.full = [(r["request"], r["fact"], r["answer"], r.get("generic")) for r in json.loads((K.OUT / "broad_long.json").read_text())["rows"]
                     if r["request"] in self.qvec and r["fact"] in fvec]
        self.pairs = B.known_pairs(su, rows, self.qvec, fvec)
        self.pool = sorted({f for _, f in self.pairs} | {f for _, f, _, _ in self.full} | {f["text"] for f in facts if f["train"] and f["text"] in fvec})
        self.pidx = {t: i for i, t in enumerate(self.pool)}
        self.PV = torch.from_numpy(B.unit(np.stack([fvec[t] for t in self.pool]))).to(dev)
        self.dev, self.fvec = dev, fvec
        self.kinds = {"full": 0, "pair": 0, "none": 0}

    def item(self, n=None):
        """-> (facts, request, target text, whether the target ends the turn)"""
        rng, a, torch = self.rng, self.args, self.torch
        if rng.random() < a.p_full:
            request, fact, answer, generic = rng.choice(self.full)
            kind = "full"
        else:
            (request, fact), answer, generic, kind = rng.choice(self.pairs), None, None, "pair"
        n = n or rng.choice(TRAIN_SIZES)
        none = rng.random() < a.p_none and (kind == "pair" or bool(generic))
        fi = self.pidx[fact]
        copy = (self.PV @ self.PV[fi]) >= a.dup  # near-copies of the right fact are not real distractors
        idx = []
        if not none:  # a few facts that are close to the request, as a real profile may hold
            q = torch.from_numpy(B.unit(self.qvec[request])).to(self.dev)
            near = [i for i in (self.PV @ q).topk(16).indices.tolist() if not bool(copy[i])]
            idx = rng.sample(near, min(len(near), rng.choice([0, 0, 1, 2]), n - 1))
        want = n - (0 if none else 1)
        while len(idx) < want:
            cand = rng.sample(range(len(self.pool)), 2 * (want - len(idx)) + 4)
            ok = (~copy[torch.tensor(cand, device=self.dev)]).tolist()
            idx += [c for c, good in zip(cand, ok) if good and c not in idx][: want - len(idx)]
        lst = [self.pool[i] for i in idx] + ([] if none else [fact])
        rng.shuffle(lst)
        self.kinds["none" if none else kind] += 1
        line = LINE + ("none" if none else fact) + "\n"
        if kind == "pair":
            return lst, request, line, False
        return lst, request, line + (generic if none else answer), True


class Text:
    """Prompt and target token ids, the same prompt format as the earlier plain-text controls."""

    def __init__(self, tok):
        from poc_clue import D, PERSONA_PROMPT, apply_chat_template
        self.tok, self.D, self.sys, self.apply = tok, D, PERSONA_PROMPT, apply_chat_template

    def msgs(self, facts, question):
        return [{"role": "system", "content": self.sys}, {"role": "user", "content": self.D.user_message(list(facts), question)}]

    def encode(self, facts, question, target, ends):
        p = self.tok.encode(self.apply(self.tok, self.msgs(facts, question), add_generation_prompt=True), add_special_tokens=False)
        t = self.tok.encode(target + ("<|im_end|>" if ends else ""), add_special_tokens=False)
        return p, t


def next_batch(sm, tx, budget, state, max_bs=8):
    """Examples of one profile size until batch size x longest example would pass the token budget."""
    out, longest = [], 0
    n = sm.rng.choice(TRAIN_SIZES)
    while len(out) < max_bs:
        p, t = state.pop("carry", None) or tx.encode(*sm.item(n))
        size = len(p) + len(t)
        if out and (len(out) + 1) * max(longest, size) > budget:
            state["carry"] = (p, t)
            break
        out.append((p, t))
        longest = max(longest, size)
    return out


def tensors(batch, pad, torch, dev):
    width = max(len(p) + len(t) for p, t in batch)
    ids = torch.full((len(batch), width), pad, dtype=torch.long)
    lab = torch.full((len(batch), width), -100, dtype=torch.long)
    mask = torch.zeros((len(batch), width), dtype=torch.long)
    for i, (p, t) in enumerate(batch):
        ids[i, : len(p) + len(t)] = torch.tensor(p + t)
        lab[i, len(p): len(p) + len(t)] = torch.tensor(t)
        mask[i, : len(p) + len(t)] = 1
    return ids.to(dev), mask.to(dev), lab.to(dev)


def loss_fn(model, ids, mask, lab, torch):
    """Mean over examples of the mean target-token loss; the vocabulary layer runs on target positions only."""
    h = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, :-1]
    tgt = lab[:, 1:]
    sel = tgt != -100
    ce = torch.nn.functional.cross_entropy(model.lm_head(h[sel]).float(), tgt[sel], reduction="none")
    per = torch.zeros(ids.shape[0], device=ce.device).index_add_(0, sel.nonzero()[:, 0], ce)
    return (per / sel.sum(1).clamp(min=1)).mean()


def yes_score(tx, msgs):
    """log P(Yes) - log P(No) for the first reply token, from the vLLM server at MPORT_VLLM_URL (H35b)."""
    import os
    import urllib.request
    import time
    texts = [tx.apply(tx.tok, m, add_generation_prompt=True) for m in msgs]
    out = []
    n = int(os.environ.get("MPORT_VLLM_CHUNK", "2000"))  # small chunks + MPORT_VLLM_POLITE=1 on a shared (production) server
    t0 = time.perf_counter()
    for b in range(0, len(texts), n):
        while os.environ.get("MPORT_VLLM_POLITE") == "1" and _waiting(os.environ["MPORT_VLLM_URL"]) > 0:
            time.sleep(2)  # someone else is queued: let them go first
        if b and b // n % 100 == 0:
            print(f"yes_score {b}/{len(texts)} (+{time.perf_counter() - t0:.0f}s)", flush=True)
        body = json.dumps({"model": os.environ.get("MPORT_VLLM_MODEL", "qwen3-8b"), "prompt": texts[b:b + n], "max_tokens": 1, "temperature": 0, "logprobs": 20}).encode()
        req = urllib.request.Request(os.environ["MPORT_VLLM_URL"] + "/v1/completions", body, {"Content-Type": "application/json"})
        res = sorted(json.loads(urllib.request.urlopen(req, timeout=3600).read())["choices"], key=lambda c: c["index"])
        for c in res:
            top = c["logprobs"]["top_logprobs"][0]
            get = lambda w: max([v for k, v in top.items() if k.strip().lower() == w] or [-30.0])  # noqa: E731
            out.append(get("yes") - get("no"))
    return out


def _waiting(url):
    """Requests waiting in a vLLM server's queue (its /metrics), 0 if unknown."""
    import urllib.request
    try:
        m = urllib.request.urlopen(url + "/metrics", timeout=5).read().decode()
    except OSError:
        return 0
    return sum(float(x.split()[-1]) for x in m.splitlines() if x.startswith("vllm:num_requests_waiting{"))


def row_key(q, clue, facts):
    """select_scores.json key for one eval row's table {fact: score}. Request + clue alone is not unique: 84 of the 288
    unseen-type rows share it with a row that has other facts (bug found 2026-11-23, earlier "saved" results were inflated)."""
    import hashlib
    return q + "\n" + clue + "\n" + hashlib.sha1("\n".join(sorted(facts)).encode()).hexdigest()[:16]


def fact_key(q, fact):
    """key for a per-fact score that does not depend on the rest of the profile (a fact rated alone)."""
    return q + "\x1f" + fact


def profile(e, size, inc, name):
    lst = e["others"][: size - inc] + ([e["clue"]] if inc else [])
    random.Random(e["pair"] + name).shuffle(lst)
    return lst


def run_cond(model, tx, K, ev, pmap, name, size, inc, stage, max_new, batch, limit=0, how="plain"):
    todo = []
    for e in ev[:limit] if limit else ev:
        lst = profile(e, size, inc, name)
        todo.append({"stage": stage, "group": e["group"], "pair": e["pair"], "type": e["type"], "condition": name, "facts": lst,
                     "clue_in": bool(inc), "clue_pos": lst.index(e["clue"]) if inc else -1, "clue": e["clue"]})
    t0 = time.perf_counter()
    ask = lambda r: pmap[r["pair"]]["question"]  # noqa: E731
    if how == "groups":  # pass 1: every group of 8 facts is judged on its own; pass 2 sees only the quoted facts
        jobs = [(r, r["facts"][g:g + 8]) for r in todo for g in range(0, len(r["facts"]), 8)]
        notes = R.generate(model, tx.tok, [tx.msgs(grp, ask(r) + JUDGE) for r, grp in jobs], batch, 80, f"WHOLE {stage} {name}, pass 1")
        for r in todo:
            r["shown"], r["note"] = [], ""
        for (r, grp), n in zip(jobs, notes):
            r["shown"] += [grp[i] for i in quoted(n, grp)]
            r["note"] += n.strip() + " | "
        ans = R.generate(model, tx.tok, [tx.msgs(r["shown"], ask(r)) for r in todo], batch, max_new, f"WHOLE {stage} {name}, pass 2")
    elif how == "bestof":  # H35: one answer per single fact (and one with none); the decoder, seeing the whole profile, picks one
        jobs = [(r, f) for r in todo for f in r["facts"] + [None]]
        cands = R.generate(model, tx.tok, [tx.msgs([f] if f else [], ask(r)) for r, f in jobs], batch, max_new, f"WHOLE {stage} {name}, candidates")
        for r in todo:
            r["cands"] = []
        for (r, f), a in zip(jobs, cands):
            r["cands"].append((f, a.strip()))
        prompts = []
        for r in todo:
            order = list(range(len(r["cands"])))
            random.Random(r["pair"] + "judge").shuffle(order)
            r["order"] = order
            text = "".join(f"\n\nReply {chr(65 + k)}:\n{r['cands'][i][1][:500]}" for k, i in enumerate(order))
            prompts.append(tx.msgs(r["facts"], f"The user asked: \"{ask(r)}\"\n\nHere are {len(order)} possible replies.{text}\n\n" + BESTOF))
        picks = R.generate(model, tx.tok, prompts, batch, 6, f"WHOLE {stage} {name}, judge")
        ans = []
        for r, j in zip(todo, picks):
            m = re.search(r"[A-Z]", j.strip())
            k = ord(m.group(0)) - 65 if m else -1
            k = k if 0 <= k < len(r["order"]) else 0
            f, a = r["cands"][r["order"][k]]
            r["shown"], r["note"] = ([f] if f else []), j.strip()
            p = pmap[r["pair"]]
            r["oracle"] = any(K.strict_score(c, p["rubrics"], p["value"])["acts_strict"] for _, c in r["cands"])
            r["clue_alone_acts"] = bool(inc) and K.strict_score(r["cands"][r["clue_pos"]][1], p["rubrics"], p["value"])["acts_strict"]
            r.pop("cands")
            ans.append(a)
    elif how == "rate":  # H35b: each fact rated on its own (yes/no log-probability); the answer sees the top 4
        k = int(re.search(r"top (\d+)", name).group(1))
        jobs = [(r, i) for r in todo for i in range(len(r["facts"]))]
        saved = re.search(r"saved (\S+)", name)
        if saved:  # H37b: scores computed elsewhere -> select_scores.json: a table per eval row (row_key) or per fact (fact_key)
            tab = json.loads((K.OUT / "select_scores.json").read_text())[saved.group(1)]
            sc, miss, n_held = [], 0, 0
            for r, i in jobs:
                row = tab.get(row_key(ask(r), r["clue"], r["facts"]))
                v = row.get(r["facts"][i]) if row is not None else tab.get(fact_key(ask(r), r["facts"][i]))
                held = r["group"] == "held-out types"  # the saved tables cover the unseen-type rows only
                n_held += held
                miss += held and v is None
                sc.append(0.0 if v is None else v)
            print(f"WHOLE {name}: saved scores missing for {miss} of {n_held} unseen-type facts", flush=True)
            assert miss < 0.01 * n_held, "saved scores do not cover these profiles"
        else:
            wd = re.search(r"wording (\w+)", name)
            sc = yes_score(tx, [tx.msgs([r["facts"][i]], (RATES[wd.group(1)] if wd else RATE).format(q=ask(r))) for r, i in jobs])
        for r in todo:
            r["scores"] = []
        for (r, i), v in zip(jobs, sc):
            r["scores"].append(v)
        if "normalized" in name:  # minus the fact's mean score over 8 fixed unrelated requests: facts that sound important everywhere sink
            base_q = random.Random(0).sample(sorted({v["question"] for v in pmap.values()}), 8)
            uf = sorted({f for r in todo for f in r["facts"]})
            bs = yes_score(tx, [tx.msgs([f], RATE.format(q=q)) for f in uf for q in base_q])
            base = {f: float(np.mean(bs[j * 8:(j + 1) * 8])) for j, f in enumerate(uf)}
            for r in todo:
                r["scores"] = [v - base[f] for f, v in zip(r["facts"], r["scores"])]
        for r in todo:
            top = sorted(range(len(r["facts"])), key=lambda i: -r["scores"][i])
            r["shown"] = [r["facts"][i] for i in (top[:k][::-1] if "best last" in name else sorted(top[:k]))]  # best last: next to the request
            r["clue_rank"] = top.index(r["clue_pos"]) if inc else -1
            r["note"] = ""
        ans = R.generate(model, tx.tok, [tx.msgs(r["shown"], ask(r)) for r in todo], batch, max_new, f"WHOLE {stage} {name}")
    else:
        ans = R.generate(model, tx.tok, [tx.msgs(r["facts"], ask(r) + (QUOTE if how == "quote" else "")) for r in todo], batch, max_new, f"WHOLE {stage} {name}")
    for r, a in zip(todo, ans):
        p = pmap[r["pair"]]
        if how in ("groups", "bestof", "rate"):
            note, q = r["note"], [r["facts"].index(f) for f in r["shown"]]
        else:
            note, a = split_note(a)
            q = quoted(note, r["facts"]) if how == "quote" else []
        said = norm(note.replace(LINE.strip(), ""))
        picked = next((i for i, f in enumerate(r["facts"]) if said and norm(f) == said), -1)
        r.update(answer=a.strip(), note=note, restates_clue=bool(inc) and picked == r["clue_pos"], restates_other=picked >= 0 and picked != r["clue_pos"],
                 writes_none=said == "none", recites_clue=bool(note) and PF.mentions(note, p["rubrics"]), how=how,
                 quotes_clue=bool(inc) and r["clue_pos"] in q, n_quoted=len(q), **K.strict_score(a, p["rubrics"], p["value"]))
    for g in dict.fromkeys(r["group"] for r in todo):
        x = [r for r in todo if r["group"] == g]
        m = lambda k: float(np.mean([r[k] for r in x]))  # noqa: E731
        print(f"WHOLE {stage:9s} {g:25s} {name:36s} n={len(x):3d}  acts {m('acts_strict'):.2f}  mentions {m('mentions_strict'):.2f}  "
              + (f"restates the clue {m('restates_clue'):.2f}  another fact {m('restates_other'):.2f}  none {m('writes_none'):.2f}" if how == "plain" else
                 f"quotes the clue {m('quotes_clue'):.2f}  facts quoted {m('n_quoted'):.1f}  quotes nothing {float(np.mean([r['n_quoted'] == 0 for r in x])):.2f}"), flush=True)
    held = [r for r in todo if r["group"] == "held-out types"]
    if how == "rate" and inc:
        rk = np.array([r["clue_rank"] for r in held])
        k = int(re.search(r"top (\d+)", name).group(1))
        print(f"WHOLE {stage:9s} {name:36s} held-out: clue rated first {np.mean(rk == 0):.2f}, top 4 {np.mean(rk < 4):.2f}, top 8 {np.mean(rk < 8):.2f},"
              f" top 12 {np.mean(rk < 12):.2f}, median rank {np.median(rk):.0f}; by type top {k}: "
              + "  ".join(f"{t} {np.mean([r['clue_rank'] < k for r in held if r['type'] == t]):.2f}" for t in sorted({r['type'] for r in held})), flush=True)
    if how == "bestof":
        print(f"WHOLE {stage:9s} {name:36s} held-out: picks the clue's answer {np.mean([r['quotes_clue'] for r in held]):.2f}, picks no-fact answer"
              f" {np.mean([r['n_quoted'] == 0 for r in held]):.2f}, clue's answer acts {np.mean([r['clue_alone_acts'] for r in held]):.2f},"
              f" some answer acts (oracle) {np.mean([r['oracle'] for r in held]):.2f}", flush=True)
    for key, label in (("acts_strict", "acts"), ("restates_clue", "restates the clue") if how == "plain" else ("quotes_clue", "quotes the clue")):
        by = {}
        for r in held:
            by.setdefault(r["type"], []).append(r[key])
        print(f"WHOLE {stage:9s} {name:36s} {label} by type: " + "  ".join(f"{t} {np.mean(v):.2f}" for t, v in sorted(by.items())), flush=True)
    print(f"WHOLE {stage} {name}: {len(todo)} answers, {(time.perf_counter() - t0) / 60:.1f} min, at +{minutes():.1f} min", flush=True)
    return todo


ZERO = [(f"whole profile of {n}, text", n, True, "plain") for n in R.SIZES] + \
       [(f"whole profile of {n}, text, clue removed", n, False, "plain") for n in (8, 48)] + \
       [("whole profile of 8, text, quote first", 8, True, "quote"), ("whole profile of 48, text, quote first", 48, True, "quote"),
        ("whole profile of 48, text, quote first, clue removed", 48, False, "quote"),
        ("groups of 8 from 16, two passes", 16, True, "groups"), ("groups of 8 from 48, two passes", 48, True, "groups"),
        ("groups of 8 from 48, two passes, clue removed", 48, False, "groups"),
        ("best of single-fact answers, 8, model judges", 8, True, "bestof"), ("best of single-fact answers, 16, model judges", 16, True, "bestof"),
        ("each fact rated alone, top 4, from 8", 8, True, "rate"), ("each fact rated alone, top 4, from 16", 16, True, "rate"),
        ("each fact rated alone, top 4, from 48", 48, True, "rate"),
        ("rated alone, normalized, top 4, from 8", 8, True, "rate"), ("rated alone, normalized, top 4, from 16", 16, True, "rate"),
        ("rated alone, normalized, top 8, from 48", 48, True, "rate"), ("each fact rated alone, top 8, from 48", 48, True, "rate"),
        ("rated alone, normalized, top 12, from 48", 48, True, "rate"),
        ("saved change+attn top 4, from 48", 48, True, "rate"), ("saved change+attn top 8, from 48", 48, True, "rate"),
        ("saved all+attn top 8, from 48", 48, True, "rate"), ("saved change top 8, from 48", 48, True, "rate"),
        ("saved attn top 8, from 48", 48, True, "rate"), ("saved all+attn top 12, from 48", 48, True, "rate"),
        ("saved change top 8 best last, from 48", 48, True, "rate"), ("saved change top 4 best last, from 48", 48, True, "rate"),
        ("saved change+attn top 8 best last, from 48", 48, True, "rate"), ("saved change top 2 best last, from 48", 48, True, "rate"),
        ("rated, wording change, top 4, from 8", 8, True, "rate"), ("rated, wording change, top 4, from 16", 16, True, "rate"),
        ("rated, wording change, top 8, from 16", 16, True, "rate"),
        ("rated, wording change, top 4 best last, from 8", 8, True, "rate"), ("rated, wording change, top 4 best last, from 16", 16, True, "rate"),
        ("rated, wording change, top 2 best last, from 8", 8, True, "rate"),
        ("saved all+attn top 12 best last, from 48", 48, True, "rate"), ("saved change+attn top 6 best last, from 48", 48, True, "rate"),
        ("saved all+attn top 8 best last, from 48", 48, True, "rate"),
        ("saved pick8+s1 top 4 best last, from 48", 48, True, "rate"), ("saved pick8+s1 top 8 best last, from 48", 48, True, "rate"),
        ("saved think12+s1 top 4 best last, from 48", 48, True, "rate"), ("saved think12+s1 top 8 best last, from 48", 48, True, "rate"),
        # H41: the 27B rates ("wrong" wording, poc_rate_sweep.py --select-key r27), the 8B answers
        ("saved r27_wrong top 2 best last, from 48", 48, True, "rate"), ("saved r27_wrong top 4 best last, from 48", 48, True, "rate"),
        ("saved r27_wrong top 8 best last, from 48", 48, True, "rate"), ("saved r27_change top 4 best last, from 48", 48, True, "rate"),
        # smaller profiles are subsets of the 48 (same facts, same request), so the saved ratings cover them
        ("saved r27_wrong top 4 best last, from 8", 8, True, "rate"), ("saved r27_wrong top 4 best last, from 16", 16, True, "rate"),
        ("saved r27_wrong top 2 best last, from 8", 8, True, "rate"), ("saved r27_wrong top 2 best last, from 16", 16, True, "rate"),
        # H42: the same selections answered by the 27B (MPORT_VLLM_MODEL / MPORT_TOK); the 8B's own picks as per-fact tables (r8)
        ("clue alone, text", 1, True, "plain"),
        ("saved r8_change top 4 best last, from 8", 8, True, "rate"), ("saved r8_change top 4 best last, from 16", 16, True, "rate")]


def zero(args, K):
    """H32: the plain decoder, no training."""
    su, qvec, _, _ = B.load_all()
    ev, pmap = B.test_set(su, qvec, True)
    conds = [c for c in ZERO if not args.conds or any(x in c[0] for x in args.conds.replace("_", " ").split(","))]
    conds = [c for c in conds if not args.exclude or not any(x in c[0] for x in args.exclude.replace("_", " ").split(","))]
    print(f"WHOLE zero: {len(ev)} questions, conditions: " + "; ".join(c[0] for c in conds), flush=True)
    if args.dry:
        from transformers import AutoTokenizer
        tx = Text(AutoTokenizer.from_pretrained(K.DECODER))
        e = ev[0]
        lst = profile(e, 8, True, "x")
        for extra in (QUOTE, JUDGE):
            print("WHOLE dry\n" + tx.apply(tx.tok, tx.msgs(lst[:2], e["question"] + extra), add_generation_prompt=True))
        note = 'Relevant facts: "' + lst[3] + '"\n\nHere is my answer.'
        print("WHOLE dry split", split_note(note), "quoted", quoted(split_note(note)[0], lst), "expected [3]")
        print("WHOLE dry split list", split_note("Relevant facts:\n- " + lst[1] + "\n- " + lst[2].lower() + "\n\nAnswer text."))
        print("WHOLE dry none", split_note('Relevant facts: none\nAnswer.'), quoted("Relevant facts: none", lst))
        return
    import os
    if os.environ.get("MPORT_VLLM_URL"):  # generation goes to a vLLM server; only the tokenizer is needed here
        from transformers import AutoTokenizer
        model, tok = None, AutoTokenizer.from_pretrained(os.environ.get("MPORT_TOK", K.DECODER))  # H42: another answerer brings its own template
    else:
        from poc_clue import D
        runner = D.Runner([], K.DECODER)
        model, tok = runner.model, runner.tok
    tok.padding_side = "left"
    tx = Text(tok)
    out_path = K.OUT / args.out
    result = {"lora": os.environ.get("MPORT_VLLM_LORA") or "plain decoder", "args": vars(args), "rows": []}
    run_cond(model, tx, K, ev, pmap, "probe", 16, True, "probe", 12, 2, limit=2, how="groups")
    for name, size, inc, how in conds:
        result["rows"] += run_cond(model, tx, K, ev[:: args.stride], pmap, name, size, inc, "untrained", args.max_new_tokens + (60 if how == "quote" else 0), args.gen_batch, how=how)
        out_path.write_text(json.dumps(result, indent=1))
    print(f"WHOLE done at +{minutes():.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="train", choices=["train", "zero"])
    ap.add_argument("--conds", default="")
    ap.add_argument("--out", default="zero_eval.json")
    ap.add_argument("--exclude", default="", help="drop conditions whose name has one of these substrings")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--minutes", type=float, default=11)
    ap.add_argument("--deadline", type=float, default=25, help="minutes from the start; later evaluation steps are skipped")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--budget", type=int, default=2400, help="tokens per training batch (batch size x longest example)")
    ap.add_argument("--p-full", type=float, default=0.4)
    ap.add_argument("--p-none", type=float, default=0.1)
    ap.add_argument("--dup", type=float, default=0.6)
    ap.add_argument("--gen-batch", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=260)
    ap.add_argument("--tag", default="whole")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--opt", default="muon", choices=["muon", "adamw"])
    ap.add_argument("--ckpt", action="store_true", help="gradient checkpointing (bigger token budgets fit)")
    ap.add_argument("--bench", action="store_true", help="train for --minutes and report speed only: no evaluation, nothing saved")
    args = ap.parse_args()
    import torch
    from poc_clue import P
    K = PF.poc()
    torch.manual_seed(0)
    if args.mode == "zero":
        return zero(args, K)
    nw, rank = (1, 0) if args.dry else FT.setup()
    dev = "cpu" if args.dry else P.DEV
    sm = Sampler(args, K, torch, dev)
    sm.rng = random.Random(rank)  # each process draws its own examples
    ev, pmap = B.test_set(sm.su, sm.qvec, True)
    print(f"WHOLE examples with answers {len(sm.full)}  pairs {len(sm.pairs)}  fact pool {len(sm.pool)}  test questions {len(ev)}", flush=True)
    state = {}
    if args.dry:
        from transformers import AutoTokenizer
        tx = Text(AutoTokenizer.from_pretrained(K.DECODER))
        by = {}
        for _ in range(150):
            lst, request, target, ends = sm.item()
            p, t = tx.encode(lst, request, target, ends)
            by.setdefault(len(lst), []).append((len(p), len(t)))
        for n, v in sorted(by.items()):
            print(f"WHOLE dry profile of {n:2d}: prompt tokens {np.mean([a for a, _ in v]):.0f}  target tokens mean {np.mean([b for _, b in v]):.0f} max {max(b for _, b in v)}  n={len(v)}")
        sizes = [len(next_batch(sm, tx, args.budget, state)) for _ in range(60)]
        print(f"WHOLE dry kinds {sm.kinds}  batch sizes under {args.budget} tokens: mean {np.mean(sizes):.1f} min {min(sizes)} max {max(sizes)}")
        lst, request, target, ends = sm.item(4)
        print("WHOLE dry sample\n" + tx.apply(tx.tok, tx.msgs(lst, request), add_generation_prompt=True) + target + ("<|im_end|>" if ends else ""))
        for size in R.SIZES:
            L = [len(tx.encode(profile(e, size, True, "x"), e["question"], "", False)[0]) for e in ev[::20]]
            print(f"WHOLE dry test profile of {size}: prompt tokens mean {np.mean(L):.0f} max {max(L)}; others available min {min(len(e['others']) for e in ev)}")
        return
    import poc_agentmem as A
    from poc_clue import D
    runner = D.Runner([], K.DECODER)
    model, tok = runner.model, runner.tok
    tok.padding_side = "left"
    tx = Text(tok)
    lora = A.add_lora(model, 16, ("q_proj", "k_proj", "v_proj", "o_proj"))
    lps = [t for m in lora for t in (m.A, m.B)]
    opt = FT.optimizer(lps, args.lr, args.opt)
    if args.ckpt:
        FT.checkpointing(model)
    out_path = K.OUT / f"{args.tag}_eval.json"
    result = {"lora": f"lora_{args.tag}.pt", "args": vars(args), "train": {}, "rows": []}

    half = next(model.parameters()).dtype == torch.float16  # V100: scale the loss so float16 gradients do not underflow
    scaler = torch.amp.GradScaler("cuda", enabled=half)

    def step(batch):
        loss = loss_fn(model, *tensors(batch, tok.pad_token_id, torch, P.DEV), torch)
        scaler.scale(loss).backward()
        return float(loss)

    def save(rows):
        result["rows"] += rows
        out_path.write_text(json.dumps(result, indent=1))

    # probe: the largest training batch must fit; two answers go through generation and scoring
    model.train()
    budget = args.budget
    while True:
        worst = [tx.encode(*sm.item(48)) for _ in range(8)]
        longest = max(len(p) + len(t) for p, t in worst)
        worst = worst[: max(1, budget // longest)]
        try:
            torch.cuda.reset_peak_memory_stats()
            value = step(worst)
            opt.zero_grad(set_to_none=True)
            print(f"WHOLE probe: batch of {len(worst)} x {longest} tokens, loss {value:.3f}, peak memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
            break
        except torch.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            budget = int(budget * 0.75)
            print(f"WHOLE probe: out of memory, token budget lowered to {budget}", flush=True)
        torch.cuda.empty_cache()
    del worst
    torch.cuda.empty_cache()
    budget = int(min(FT.total(budget) / nw, budget))  # the same budget on every process
    if not args.bench and rank == 0:
        model.eval()
        run_cond(model, tx, K, ev, pmap, "probe", 8, True, "probe", 12, 2, limit=2)
        # the plain decoder (LoRA output is zero before training): whole profile, every second question
        for name, size in (("whole profile of 8, text", 8), ("whole profile of 48, text", 48)):
            save(run_cond(model, tx, K, ev[::2], pmap, name, size, True, "untrained", args.max_new_tokens, args.gen_batch))
    FT.agree(False)  # wait for rank 0's baselines

    model.train()
    t0, u, seen, toks, run = time.perf_counter(), 0, 0, 0, []
    limit = args.minutes * 60
    while True:
        frac = FT.total((time.perf_counter() - t0) / limit) / nw  # averaged: the same learning rate and stop on every process
        if frac >= 1:
            break
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (u + 1) / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        batch = next_batch(sm, tx, budget, state)
        oom = False
        try:
            run.append(step(batch))
        except torch.OutOfMemoryError:
            oom = True
        if FT.agree(oom):
            if not oom:
                run.pop()
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            budget = int(budget * 0.8)
            print(f"WHOLE out of memory in training, token budget lowered to {budget}", flush=True)
            continue
        if FT.agree(not math.isfinite(run[-1])):
            run.pop()
            opt.zero_grad(set_to_none=True)
            scaler.update()
            print("WHOLE non-finite loss, batch skipped", flush=True)
            continue
        FT.average_grads(lps)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(lps, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        u, seen, toks = u + 1, seen + len(batch), toks + sum(len(p) + len(t) for p, t in batch)
        if u % 25 == 0:
            el = time.perf_counter() - t0
            all_seen, all_toks, mem = FT.total(seen), FT.total(toks), torch.cuda.max_memory_allocated() / 2**30
            if rank == 0:
                print(f"update {u}  examples {all_seen:.0f}  loss {np.mean(run[-25:]):.3f}  {all_seen / el:.1f} examples/s  {all_toks / el:.0f} tokens/s"
                      f"  ({nw} GPU, budget {budget}, peak {mem:.1f} GB)  {el / 60:.1f} min", flush=True)
    seen, toks = FT.total(seen), FT.total(toks)
    FT.done()
    if rank != 0:
        return
    if args.bench:
        el = time.perf_counter() - t0
        print(f"WHOLE bench: {args.opt}, {nw} GPU, checkpointing {args.ckpt}, budget {budget}: {u} updates, {seen / el:.1f} examples/s, {toks / el:.0f} tokens/s,"
              f" peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
        return
    state_dict = {f"{i}.{k}": v.detach().cpu().clone() for i, m in enumerate(lora) for k, v in (("A", m.A), ("B", m.B))}
    torch.save(state_dict, A.OUT / f"lora_{args.tag}.pt")
    result["train"] = {"updates": u, "examples": seen, "tokens": toks, "loss": float(np.mean(run[-50:])), "kinds": sm.kinds, "minutes": (time.perf_counter() - t0) / 60}
    print(f"WHOLE saved lora_{args.tag}.pt  {json.dumps(result['train'])}", flush=True)
    del opt
    torch.cuda.empty_cache()
    model.eval()
    spent = None
    for name, size, inc in (("whole profile of 8, text", 8, True), ("whole profile of 48, text", 48, True),
                            ("whole profile of 8, text, clue removed", 8, False), ("whole profile of 16, text", 16, True),
                            ("whole profile of 48, text, clue removed", 48, False)):
        if spent is not None and minutes() + spent > args.deadline:
            print(f"WHOLE skipped (deadline {args.deadline:g} min): {name}", flush=True)
            continue
        t1 = time.perf_counter()
        save(run_cond(model, tx, K, ev, pmap, name, size, inc, "trained", args.max_new_tokens + 30, args.gen_batch))
        spent = (time.perf_counter() - t1) / 60 * (1.5 if size < 48 else 1.0)  # a guess for the next step, with room for a larger profile
    print(f"WHOLE done at +{minutes():.1f} min", flush=True)


if __name__ == "__main__":
    main()
