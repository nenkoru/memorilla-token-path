"""H14: buried clues. The similarity bias (H12, H13) picks a fact when the question resembles it. A buried-clue question
does not ("What should I pack for a long day hike?" needs "the user has type 1 diabetes"); cosine on the chat ranked
the clue 12th to 103rd of 150. Idea: at ingest time the note writer also writes, for each fact, the kinds of future
requests it matters for. Those lines become the fact's KEY vectors; the question is matched against keys, not text.
No question is seen at ingest and nothing is retrieved as text at question time.

    gen     Qwen3-8B reads every candidate chat of the buried-clue set (371 chats + Sam's 4 clue chats) and writes up
            to 3 "Fact: / Matters when:" pairs -> keys_raw.jsonl (resumable)
    embed   vectors for every fact and every matters-when item -> keys_emb.npz
    rank    CPU: where does the clue chat rank among its 150 candidates, by chat cosine, fact cosine, key cosine?

Chat texts from memorilla-demos are passed to the model as data only; its output is parsed as plain text.
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

SYSTEM = ("You keep a long-term memory about a user for a personal assistant. You read one past chat and write down what "
          "is worth remembering about the user as a person.")
USER = """Past chat:
---
{text}
---

Write up to 3 lasting facts about the user that this chat reveals: health, diet, family, pets, possessions, work, schedule, budget, beliefs, plans, habits or strong preferences. Do not record the one-off topic of the chat unless it says something lasting about the user. If the chat reveals nothing lasting, write only: NONE

For each fact, add a line listing 4 different kinds of future requests in which an assistant should take the fact into account, even if the request does not mention it. Use exactly this format:

Fact: <one sentence about the user>
Matters when: <kind of request>; <kind of request>; <kind of request>; <kind of request>"""


def docs_all():
    import poc_clue as K
    s = K.load_synth()
    out = [{"id": d["doc"], "text": d["text"]} for d in s["docs"]]
    sam = K.sam_chats()
    out += [{"id": f"sam:{c['id']}", "text": c["text"]} for c in sam if c["id"] in K.SAM_CLUES]
    return out


def parse(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).replace("**", "")
    facts, cur = [], None
    for line in text.splitlines():
        line = line.strip().lstrip("-*0123456789. ").strip()
        if line.lower().startswith("fact:"):
            cur = {"fact": line[5:].strip(), "matters": []}
            facts.append(cur)
        elif line.lower().startswith("matters when:") and cur is not None:
            cur["matters"] = [x.strip(" .") for x in line[13:].split(";") if x.strip(" .")][:6]
    return [f for f in facts if f["fact"]][:3]


def raw_path():
    import poc_clue as K
    return K.OUT / "keys_raw.jsonl"


def load_raw():
    p, done = raw_path(), {}
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            done[r["id"]] = r
    return done


def cmd_gen(args):
    import torch
    import poc_clue as K
    from poc_clue import D, P, apply_chat_template
    docs, done = docs_all(), load_raw()
    pending = [d for d in docs if d["id"] not in done][: args.limit or None]
    pending.sort(key=lambda d: len(d["text"]))
    runner = D.Runner([], K.DECODER)
    model, tok = runner.model, runner.tok
    tok.padding_side = "left"
    t0 = time.perf_counter()
    for b in range(0, len(pending), args.batch):
        batch = pending[b: b + args.batch]
        texts = [apply_chat_template(tok, [{"role": "system", "content": SYSTEM},
                                           {"role": "user", "content": USER.format(text=d["text"][:4000])}],
                                     add_generation_prompt=True) for d in batch]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(P.DEV)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        D.sync()
        outs = [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]
        del enc, out
        torch.mps.empty_cache()
        with raw_path().open("a") as f:
            for d, o in zip(batch, outs):
                f.write(json.dumps({"id": d["id"], "raw": o, "facts": parse(o)}) + "\n")
        print(f"{b + len(batch)}/{len(pending)}  {time.perf_counter() - t0:.0f}s", flush=True)
        if b == 0:
            print("SAMPLE", json.dumps(outs[0])[:900], flush=True)
    done = load_raw()
    n = [len(r["facts"]) for r in done.values()]
    print(f"done: {len(done)} chats, {sum(n)} facts, {sum(1 for x in n if x == 0)} chats with none", flush=True)


def cmd_embed(args):
    import torch
    import poc_clue as K
    from poc_clue import D
    done = load_raw()
    items = []  # (doc id, fact index, "fact" | "matter", text)
    for d in docs_all():
        for j, f in enumerate(done[d["id"]]["facts"]):
            items.append((d["id"], j, "fact", f["fact"]))
            items += [(d["id"], j, "matter", m) for m in f["matters"]]
    enc = D.Encoder()
    t0, out = time.perf_counter(), []
    for b in range(0, len(items), 512):
        out.append(enc([x[3] for x in items[b:b + 512]], batch_size=32).astype(np.float16))
        print(f"{b + len(out[-1])}/{len(items)}  {time.perf_counter() - t0:.0f}s", flush=True)
        torch.mps.empty_cache()
    np.savez(K.OUT / "keys_emb.npz", v=np.concatenate(out), doc=np.array([x[0] for x in items]),
             fact=np.array([x[1] for x in items]), kind=np.array([x[2] for x in items]), text=np.array([x[3] for x in items]))
    print("done", flush=True)


def cmd_rank(args):
    import poc_clue as K
    s = K.load_synth()
    dv, qmap = K.load_emb()
    z = np.load(K.OUT / "keys_emb.npz")
    norm = lambda x: x / np.linalg.norm(x, axis=-1, keepdims=True)  # noqa: E731
    kv, dv = norm(z["v"].astype(np.float32)), norm(dv.astype(np.float32))
    by_doc = {}
    for i, (d, kind) in enumerate(zip(z["doc"], z["kind"])):
        by_doc.setdefault(str(d), {"fact": [], "matter": []})[str(kind)].append(i)
    ids = [d["doc"] for d in s["docs"]]

    def doc_scores(q, cand, mode):
        if mode == "chat":
            return dv[cand] @ q
        sc = np.full(len(cand), -1.0)
        for n, c in enumerate(cand):
            e = by_doc.get(ids[c], {"fact": [], "matter": []})
            rows = e["fact"] if mode == "fact" else e["matter"] if mode == "matter" else e["fact"] + e["matter"]
            if rows:
                sc[n] = float((kv[rows] @ q).max())
        return sc

    rows = []
    for split in ("test", "val", "train", "all"):
        ps = [p for p in s["pairs"] if split == "all" or p["split"] == split]
        line = f"RANK {split:5s} n={len(ps):3d}"
        for mode in ("chat", "fact", "matter", "fact+matter"):
            rk, mass = [], []
            for p in ps:
                q = norm(qmap[p["question"]].astype(np.float32))
                cand = [p["clue"]] + p["fillers"]
                sc = doc_scores(q, cand, mode)
                rk.append(int((sc > sc[0]).sum()) + 1)
                e = np.exp(args.beta * (sc - sc.max()))
                mass.append(float(e[0] / e.sum()))
                if split == "all":
                    rows.append({"pair": p["pair"], "split": p["split"], "type": p["type"], "mode": mode, "rank": rk[-1], "mass": mass[-1]})
            rk = np.array(rk)
            line += f" | {mode}: R@1 {np.mean(rk <= 1):.2f} R@5 {np.mean(rk <= 5):.2f} R@10 {np.mean(rk <= 10):.2f} median {int(np.median(rk))} mass {np.mean(mass):.2f}"
        print(line, flush=True)
    by_type = {}
    for r in rows:
        if r["mode"] == "fact+matter":
            by_type.setdefault(r["type"], []).append(r["rank"])
    print("RANK by clue type (fact+matter, R@5): " + "  ".join(f"{t} {np.mean(np.array(v) <= 5):.2f}" for t, v in sorted(by_type.items())), flush=True)
    (K.OUT / "keys_rank.json").write_text(json.dumps(rows))
    # Sam's demo questions: rank of each clue chat among all 150 of Sam's chats
    sam = K.sam_chats()
    qs = json.loads((K.D.DEMOS / "data/sam/questions.json").read_text())["questions"]
    sam_dv = norm(np.load(K.D.OUT / "emb_sam.npz")["v"].astype(np.float32))
    for qd in qs:
        q = norm(qmap[qd["question"]].astype(np.float32))
        best = {}
        for mode in ("chat", "fact+matter"):
            sc = []
            for c in sam:
                e = by_doc.get(f"sam:{c['id']}", {"fact": [], "matter": []})
                r = e["fact"] + e["matter"]
                sc.append(float(sam_dv[c["id"] - 1] @ q) if mode == "chat" else (float((kv[r] @ q).max()) if r else -1.0))
            order = np.argsort(-np.array(sc))
            best[mode] = [(sam[i]["id"], round(sc[i], 2)) for i in order[:5]]
            for cid in sorted(K.SAM_CLUES):
                pos = int(np.where(np.array([sam[i]["id"] for i in order]) == cid)[0][0]) + 1
                best[mode].append((f"clue chat {cid} rank", pos))
        print("SAM", json.dumps({"question": qd["question"][:70], "keys": [k for k in qd if k != "question"][:6], **best}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--batch", type=int, default=8)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--max-new-tokens", type=int, default=230)
    sub.add_parser("embed")
    r = sub.add_parser("rank")
    r.add_argument("--beta", type=float, default=100.0)
    args = ap.parse_args()
    {"gen": cmd_gen, "embed": cmd_embed, "rank": cmd_rank}[args.cmd](args)


if __name__ == "__main__":
    main()
