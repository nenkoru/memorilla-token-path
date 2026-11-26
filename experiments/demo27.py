"""The video author's Buried Clue test (memorilla-demos, Sam's 150 chats, 3 buried clues + 1 control) with the current
pipeline: facts extracted per chat, one 16-token block per fact, the block rater picks the top k, the reader answers.

    demo27.py extract --url http://127.0.0.1:8001 --model <name>     production's 27B writes the facts (box, no GPU)
    demo27.py embed                                                  Qwen3-Embedding-4B: facts and questions (Modal)
    demo27.py rate --lora lora_tokrater_r27w.pt                      every fact's block rated for every question (box)
    demo27.py answer --lora lora_kd27a.pt --k 1,3,4,8 [--url ... --model ...]   answers, rubric and 27B judge (box)
Files under work/runs/poc: demo_facts.json, demo_emb.npz, demo_scores.json, demo_answers.json.
The chats are read as data only and passed to the model as text; its output is parsed as plain text.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import poc_profile as PF  # noqa: E402


def questions():
    import demos_port as D
    _, qs = D.load_clue()
    for q in qs.values():
        q["clue_chat"] = int(q["clue_chat"])
    return qs


def cmd_extract(args):
    import urllib.request
    import demos_port as D
    import poc_whole as W
    K = PF.poc()
    chats, _ = D.load_clue()
    url = args.url.rstrip("/")

    def one(chat):
        body = {"model": args.model, "max_tokens": 400, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False},
                "messages": [{"role": "system", "content": D.EXTRACT_SYSTEM}, {"role": "user", "content": D.EXTRACT_USER.format(chat=chat)}]}
        req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        return (json.loads(urllib.request.urlopen(req, timeout=1800).read())["choices"][0]["message"]["content"] or "").strip()

    out, t0 = [], time.perf_counter()
    for b in range(0, len(chats), 8):
        while W._waiting(url) > 0:  # a shared server: someone else is queued, let them go first
            time.sleep(2)
        with ThreadPoolExecutor(8) as ex:
            for i, raw in enumerate(ex.map(one, chats[b:b + 8]), b + 1):
                out.append({"chat": i, "facts": D.parse_facts(raw), "raw": raw})
    (K.OUT / "demo_facts.json").write_text(json.dumps({"extractor": args.model, "chats": out}, indent=1))
    n = sum(len(c["facts"]) for c in out)
    print(f"DEMO extract: {n} facts from {len(out)} chats in {time.perf_counter() - t0:.0f}s", flush=True)
    for qid, q in questions().items():
        print(f"DEMO {qid} clue chat {q['clue_chat']}: {out[q['clue_chat'] - 1]['facts']}", flush=True)


def load_facts(K):
    spec = json.loads((K.OUT / "demo_facts.json").read_text())["chats"]
    facts = [f for c in spec for f in c["facts"]]
    chat = [c["chat"] for c in spec for _ in c["facts"]]
    return facts, chat


def cmd_embed(args):
    import demos_port as D
    K = PF.poc()
    facts, _ = load_facts(K)
    qs = questions()
    enc = D.Encoder()
    fv = enc(facts, batch_size=64)
    qv = enc([q["question"] for q in qs.values()])
    np.savez(K.OUT / "demo_emb.npz", facts=fv, questions=qv, qids=np.array(list(qs)))
    print(f"DEMO embed: {len(facts)} facts, {len(qs)} questions", flush=True)


def load_emb(K):
    z = np.load(K.OUT / "demo_emb.npz")
    return z["facts"].astype(np.float32), {q: v.astype(np.float32) for q, v in zip(z["qids"].tolist(), z["questions"])}


def cmd_rate(args):
    import poc_tok as T
    import q8
    from poc_agentmem import P
    from poc_clue import PERSONA_PROMPT
    K = PF.poc()
    facts, chat = load_facts(K)
    fv, qv = load_emb(K)
    qs = questions()
    torch, A, C, st, core, dt = T.load_stack(args.lora)
    model, tok = st.r.model, st.r.tok
    first = lambda ws: sorted({tok.encode(w, add_special_tokens=False)[0] for w in ws})  # noqa: E731
    yes, no = first([" Yes", "Yes", " yes"]), first([" No", "No", " no"])
    head = model.lm_head.weight[yes + no].float().to(P.DEV)
    q8.drop_lm_head(model)
    if q8.lean():
        core = q8.place_core(core)
    cdev = next(core.parameters()).device
    out, t0 = {}, time.perf_counter()
    for qid, q in qs.items():
        ids = T.rating_ids(A, tok, st.r.add_special, PERSONA_PROMPT, T.WORD["change"].format(q=q["question"]), 16)
        sc = []
        for b in range(0, len(facts), args.batch):
            n = min(args.batch, len(facts) - b)
            with torch.no_grad():
                d = torch.from_numpy(fv[b:b + n])[:, None].to(cdev)
                mem = C.core_fwd(core, d, torch.from_numpy(np.repeat(qv[qid][None], n, 0)).to(cdev)).to(P.DEV, dt)
                x = torch.cat([P.input_embeds(model, tok, st.r.memory_id, torch.tensor([ids], device=P.DEV), mem[j:j + 1]) for j in range(n)])
                h = model.model(inputs_embeds=x).last_hidden_state[:, -1]
                lg = h.float() @ head.t()
                sc += (torch.logsumexp(lg[:, :len(yes)], -1) - torch.logsumexp(lg[:, len(yes):], -1)).tolist()
        out[qid] = sc
        order = np.argsort(-np.array(sc), kind="stable")
        rank = {int(i): r for r, i in enumerate(order, 1)}
        clue = [i for i, c in enumerate(chat) if c == q["clue_chat"]]
        cos = fv @ qv[qid]
        crank = {int(i): r for r, i in enumerate(np.argsort(-cos, kind="stable"), 1)}
        print(f"DEMO rate {qid}: clue chat's facts ranked " + ", ".join(f"{rank[i]} (cosine {crank[i]})" for i in clue)
              + f" of {len(facts)}; top 4: {[facts[i][:70] for i in order[:4]]}", flush=True)
    (K.OUT / "demo_scores.json").write_text(json.dumps({"lora": args.lora, "scores": out}))
    print(f"DEMO rate done: {len(facts) * len(qs)} ratings in {time.perf_counter() - t0:.0f}s", flush=True)


def cmd_answer(args):
    import demos_port as D
    import fast_gen as FG
    import judge27 as J
    import poc_tok as T
    import q8
    from poc_agentmem import P
    K = PF.poc()
    facts, chat = load_facts(K)
    fv, qv = load_emb(K)
    qs = questions()
    scores = json.loads((K.OUT / "demo_scores.json").read_text())["scores"]
    torch, A, C, st, core, dt = T.load_stack(args.lora)
    if q8.lean():
        core = q8.place_core(core)
        q8.head_to_gpu(st.r.model)
    todo = []
    for qid, q in qs.items():
        order = [int(i) for i in np.argsort(-np.array(scores[qid]), kind="stable")]
        clue = [i for i, c in enumerate(chat) if c == q["clue_chat"]]
        conds = [(f"top {k} of {len(facts)}", order[:k]) for k in args.k] + [("clue chat's facts only", clue)]
        conds.append((f"top {max(args.k)} of {len(facts)}, clue chat removed", [i for i in order if chat[i] != q["clue_chat"]][:max(args.k)]))
        for name, shown in conds:
            todo.append({"question": qid, "condition": name, "facts": [facts[i] for i in shown],
                         "clue_shown": sum(chat[i] == q["clue_chat"] for i in shown)})
    items = []
    with torch.no_grad():
        for r in todo:
            vs = np.stack([fv[facts.index(f)] for f in r["facts"]])
            items.append((qs[r["question"]]["question"], PF.blocks(C, core, vs, qv[r["question"]], 16, P, torch).to(dt).cpu()))
    core.cpu()
    torch.cuda.empty_cache()
    ans = FG.generate(st, items, args.max_new_tokens, 0, log=lambda m: print(m, flush=True))
    url = args.url.rstrip("/")
    for r, a in zip(todo, ans):
        note = ""
        if a.startswith("Relevant fact:"):
            note, _, a = a.partition("\n")
        q = qs[r["question"]]
        r.update(note=note, answer=a.strip(), rubric=D.rubric_pass(a, q["rubric"]))
    if url:
        with ThreadPoolExecutor(8) as ex:
            for r, p in zip(todo, ex.map(lambda r: J.ask(url, args.model, qs[r["question"]]["question"], qs[r["question"]]["clue_summary"], r["answer"]), todo)):
                r["judge"] = round(p, 3)
    (K.OUT / "demo_answers.json").write_text(json.dumps({"lora": args.lora, "rows": todo}, indent=1))
    for r in todo:
        print(f"DEMO [{r['question']:14s}] {r['condition']:34s} clue facts shown {r['clue_shown']}  rubric {'PASS' if r['rubric'] else 'fail'}"
              + (f"  judge {r['judge']:.2f}" if "judge" in r else "") + f"  {r['note'][:90]!r} {r['answer'][:160]!r}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract")
    sub.add_parser("embed")
    r = sub.add_parser("rate")
    r.add_argument("--batch", type=int, default=64)
    a = sub.add_parser("answer")
    a.add_argument("--k", type=lambda x: [int(v) for v in x.split(",")], default=[1, 3, 4, 8])
    a.add_argument("--max-new-tokens", type=int, default=300)
    for p in (e, a):
        p.add_argument("--url", default="" if p is a else None, required=p is e)
        p.add_argument("--model", default="<qwen-27b-served>")
    for p in (r, a):
        p.add_argument("--lora", required=True)
    args = ap.parse_args()
    {"extract": cmd_extract, "embed": cmd_embed, "rate": cmd_rate, "answer": cmd_answer}[args.cmd](args)


if __name__ == "__main__":
    main()
