"""The video author's Memorilla tests (andrisgauracs/memorilla-demos), re-run on LFM3-1.7B with the frozen pv4 core
and our adapter-only head. Their data and rubrics are read as JSON; none of their code is imported.

Subcommands:
    embedcheck  local Qwen3-Embedding-4B (sentence-transformers) vs the vectors Memorilla was trained on
    clue        The Buried Clue: 4 questions x {none, rag5, full, memorilla heads, wrong-docs control}
    diagnose    their diagnostic: Memorilla from only the clue chat, the clue + 4 others, all 150
    book        The Hound of the Baskervilles: 8 questions x {none, rag5, full, memorilla}
    speed       answer latency for 1..512 passages of 250 tokens: full context vs Memorilla

Differences from the original (all on purpose, all reported):
    decoder LFM3-1.7B instead of Qwen3-8B; core pv4 for every test (they used pmv2 for the clue, stage3 for the book
    and speed); HF transformers on MPS instead of vLLM on an H100; LFM3's 262K window, so full context never truncates.
"""

import argparse
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import pv4_port as P  # noqa: E402  (also puts memorilla and mport on sys.path)
from memorilla.benchmarks import BRIEF_ANSWER_PROMPT, PERSONA_PROMPT  # noqa: E402
from memorilla.data import apply_chat_template, build_messages  # noqa: E402

DEMOS = Path(os.environ.get("MPORT_DEMOS", P.SCRATCH / "ext/memorilla-demos"))
OUT = P.SCRATCH / "runs/demos"
DECODER = "LiquidAI/LFM3-1.7B"
HEADS = {"cal_ridge": P.SCRATCH / "runs/cal_ridge/head.pt", "cal_random": P.SCRATCH / "runs/cal_random/head.pt"}
SEPARATOR = "\n\n"  # evaluate_baselines.DOCUMENT_SEPARATOR


def sync():
    if P.DEV.type == "mps":
        torch.mps.synchronize()


# ---------------------------------------------------------------- scoring (same rule as their demos/rubric.py)


def _norm(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‑", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text)


def rubric_pass(answer: str, rubric: dict) -> bool:
    text = _norm(answer)
    groups_ok = all(any(_norm(p) in text for p in group) for group in rubric.get("all_of", []))
    return groups_ok and not any(_norm(p) in text for p in rubric.get("none_of", []))


# ---------------------------------------------------------------- models


class Encoder:
    def __init__(self):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(str(P.snapshot(P.ENCODER)), device=str(P.DEV),
                                         model_kwargs={"dtype": torch.float16})

    def __call__(self, texts: list[str], batch_size: int = 8) -> np.ndarray:
        out = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=True)
        sync()
        return out.astype(np.float32)


class Runner:
    def __init__(self, heads: list[str], decoder: str = DECODER):
        self.model, self.tok, self.memory_id = P.load_decoder(decoder)
        self.heads = {name: P.load_memory(name if name == "native" else str(HEADS[name]), decoder) for name in heads}
        probe = apply_chat_template(self.tok, [{"role": "user", "content": "x"}], add_generation_prompt=True)
        self.add_special = self.tok.bos_token is None or not probe.startswith(self.tok.bos_token)

    def ids(self, messages) -> torch.Tensor:
        text = apply_chat_template(self.tok, messages, add_generation_prompt=True)
        return torch.tensor([self.tok.encode(text, add_special_tokens=self.add_special)], device=P.DEV)

    @torch.no_grad()
    def _generate(self, ids, memory_tokens, max_new_tokens):
        sync()
        start = time.perf_counter()
        kwargs = dict(attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens, do_sample=False,
                      pad_token_id=self.tok.pad_token_id)
        if memory_tokens is None:
            out = self.model.generate(input_ids=ids, prefill_chunk_size=4096, **kwargs)[0, ids.shape[1]:]
        else:
            embeds = P.input_embeds(self.model, self.tok, self.memory_id, ids, memory_tokens)
            out = self.model.generate(inputs_embeds=embeds, **kwargs)[0]
        sync()
        return {"answer": self.tok.decode(out, skip_special_tokens=True).strip(),
                "prompt_tokens": int(ids.shape[1]), "new_tokens": int(len(out)),
                "generate_s": round(time.perf_counter() - start, 3)}

    def text(self, system: str, content: str, max_new_tokens: int):
        return self._generate(self.ids([{"role": "system", "content": system}, {"role": "user", "content": content}]),
                              None, max_new_tokens)

    @torch.no_grad()
    def memorilla(self, head: str, system: str, question: str, doc_vecs: np.ndarray, q_vec: np.ndarray,
                  max_new_tokens: int):
        memory = self.heads[head]
        sync()
        start = time.perf_counter()
        docs = torch.from_numpy(doc_vecs)[None].to(P.DEV)
        mask = torch.zeros(1, len(doc_vecs), dtype=torch.bool, device=P.DEV)  # True = padding; none here
        tokens = memory(docs, mask, torch.from_numpy(q_vec)[None].to(P.DEV))
        result = self._generate(self.ids(build_messages(question, system, num_memories=memory.num_memories)),
                                tokens, max_new_tokens)
        sync()
        result["answer_s"] = round(time.perf_counter() - start, 3)
        return result


def user_message(documents: list[str], question: str) -> str:
    return SEPARATOR.join(documents) + SEPARATOR + question if documents else question


def top_k(doc_vecs: np.ndarray, q_vec: np.ndarray, k: int) -> list[int]:
    return [int(i) for i in np.argsort(-(doc_vecs @ q_vec), kind="stable")[:k]]


def cached_vectors(encoder: Encoder, name: str, texts: list[str]) -> tuple[np.ndarray, float]:
    """Embed once per corpus (the timing of that one pass is kept with the vectors)."""
    path = OUT / f"emb_{name}.npz"
    if path.exists():
        z = np.load(path)
        return z["v"], float(z["seconds"])
    start = time.perf_counter()
    vecs = encoder(texts)
    seconds = time.perf_counter() - start
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(path, v=vecs, seconds=seconds)
    return vecs, seconds


def save(name: str, data: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(data, indent=2))
    print(f"wrote {path}")


# ---------------------------------------------------------------- subcommands


def cmd_embedcheck(args):
    import pyarrow.parquet as pq

    table = pq.read_table(P.DATA_ROOT / "data/pv4/test-00000-of-00001.parquet", columns=["collection_id", "documents"])
    row = table.slice(0, 1).to_pylist()[0]
    _, store = P.load_split("test")
    stored = store.documents(row["collection_id"]).float().numpy()[: args.n]
    local = Encoder()(row["documents"][: args.n])
    cos = (local * stored).sum(1) / np.linalg.norm(stored, axis=1)
    print(json.dumps({"stored_norm": float(np.linalg.norm(stored, axis=1).mean()),
                      "cos_local_vs_stored": [round(float(c), 4) for c in cos]}))


def load_clue():
    chats = [c["text"] for c in json.loads((DEMOS / "data/sam/chats.json").read_text())["chats"]]
    questions = {q["id"]: q for q in json.loads((DEMOS / "data/sam/questions.json").read_text())["questions"]}
    return chats, questions


def load_hound():
    return [p["text"] for p in json.loads((DEMOS / "data/hound/passages.json").read_text())["passages"]]


def cmd_clue(args):
    chats, questions = load_clue()
    encoder = Encoder()
    vecs, embed_s = cached_vectors(encoder, "sam", chats)
    wrong, _ = cached_vectors(encoder, "hound", load_hound())
    wrong = wrong[: len(chats)]
    runner = Runner(list(HEADS))
    max_new = args.max_new_tokens
    runner.text(PERSONA_PROMPT, "Any tips for keeping a houseplant alive?", 8)  # warm-up, not scored
    rows = []
    for qid, q in questions.items():
        question = q["question"]
        q_vec = encoder([question])[0]
        retrieved = top_k(vecs, q_vec, 5)
        methods = {
            "no memory": lambda: runner.text(PERSONA_PROMPT, question, max_new),
            "RAG top-5": lambda: runner.text(PERSONA_PROMPT, user_message([chats[i] for i in retrieved], question),
                                             max_new) | {"retrieved": [i + 1 for i in retrieved]},
            "full context": lambda: runner.text(PERSONA_PROMPT, user_message(chats, question), max_new),
            "Memorilla (cal_ridge)": lambda: runner.memorilla("cal_ridge", PERSONA_PROMPT, question, vecs, q_vec,
                                                              max_new),
            "Memorilla (cal_random)": lambda: runner.memorilla("cal_random", PERSONA_PROMPT, question, vecs, q_vec,
                                                               max_new),
            "Memorilla, Hound docs": lambda: runner.memorilla("cal_ridge", PERSONA_PROMPT, question, wrong, q_vec,
                                                              max_new),
        }
        for method, fn in methods.items():
            result = fn()
            result |= {"question": qid, "method": method, "pass": rubric_pass(result["answer"], q["rubric"])}
            rows.append(result)
            print(f"[{qid:14s}] {method:24s} {'PASS' if result['pass'] else 'fail'}  {result['generate_s']:6.2f}s  "
                  f"{result['answer'][:110]!r}", flush=True)
    save("clue", {"embed_all_chats_s": round(embed_s, 2), "max_new_tokens": max_new, "rows": rows})


def cmd_diagnose(args):
    chats, questions = load_clue()
    conditions = {  # copied from their demos/diagnose_clue.py (1-based chat numbers); None = all 150
        "control": {"only the clue chat": [57], "clue + 4 others": [57, 117, 119, 80, 76], "all 150": None},
        "parents_dinner": {"only the clue chat": [88], "clue + 4 others": [88, 76, 30, 140, 119], "all 150": None},
    }
    encoder = Encoder()
    vecs, _ = cached_vectors(encoder, "sam", chats)
    runner = Runner(["cal_ridge"])
    rows = []
    for qid, conds in conditions.items():
        q = questions[qid]
        q_vec = encoder([q["question"]])[0]
        for label, ids in conds.items():
            sub = vecs if ids is None else vecs[[i - 1 for i in ids]]
            result = runner.memorilla("cal_ridge", PERSONA_PROMPT, q["question"], sub, q_vec, args.max_new_tokens)
            result |= {"question": qid, "condition": label, "num_docs": len(sub),
                       "pass": rubric_pass(result["answer"], q["rubric"])}
            rows.append(result)
            print(f"[{qid}] {label:20s} {'PASS' if result['pass'] else 'fail'}  {result['answer'][:160]!r}",
                  flush=True)
    save("diagnose", {"rows": rows})


def cmd_book(args):
    passages = load_hound()
    questions = json.loads((DEMOS / "data/hound/questions.json").read_text())["questions"]
    encoder = Encoder()
    vecs, embed_s = cached_vectors(encoder, "hound", passages)
    runner = Runner(["cal_ridge"])
    runner.text(BRIEF_ANSWER_PROMPT, "Who is Sherlock Holmes's friend and chronicler?", 8)  # warm-up
    rows = []
    for q in questions:
        question = q["question"]
        q_vec = encoder([question])[0]
        retrieved = top_k(vecs, q_vec, 5)
        methods = {
            "no memory": lambda: runner.text(BRIEF_ANSWER_PROMPT, question, 64),
            "RAG top-5": lambda: runner.text(BRIEF_ANSWER_PROMPT,
                                             user_message([passages[i] for i in retrieved], question), 64)
            | {"retrieved": [i + 1 for i in retrieved]},
            "full context": lambda: runner.text(BRIEF_ANSWER_PROMPT, user_message(passages, question), 64),
            "Memorilla (cal_ridge)": lambda: runner.memorilla("cal_ridge", PERSONA_PROMPT, question, vecs, q_vec, 64),
        }
        for method, fn in methods.items():
            result = fn() | {"question": q["id"], "method": method}
            result["pass"] = rubric_pass(result["answer"], q["rubric"])
            rows.append(result)
            print(f"[{q['id']:18s}] {method:22s} {'PASS' if result['pass'] else 'fail'}  {result['generate_s']:6.2f}s  "
                  f"{result['answer'][:110]!r}", flush=True)
    save("book", {"embed_book_s": round(embed_s, 2), "rows": rows})


def cmd_speed(args):
    spec = json.loads((DEMOS / "data/speed/passages.json").read_text())
    passages = [p["text"] for p in spec["passages"]]
    question = "Why does Dr. Mortimer come to see Sherlock Holmes, and what is he afraid of?"
    encoder = Encoder()
    runner = Runner(["cal_ridge"])
    runner.text(PERSONA_PROMPT, question, 4)
    encoder(passages[:4])
    rows = []
    for n in args.sizes:
        docs = passages[:n]
        start = time.perf_counter()  # the write path, timed once per size (deterministic, the slow part on a laptop)
        doc_vecs = encoder(docs)
        q_vec = encoder([question])[0]
        embed_s = time.perf_counter() - start
        mem = [runner.memorilla("cal_ridge", PERSONA_PROMPT, question, doc_vecs, q_vec, 32)
               for _ in range(args.timed_runs + 1)][1:]
        full_runs = args.timed_runs if n <= args.full_repeat_max else 1
        full = [runner.text(PERSONA_PROMPT, user_message(docs, question), 32) for _ in range(full_runs)]
        row = {"passages": n, "prompt_tokens_full": full[0]["prompt_tokens"],
               "full_context_s": statistics.median(r["generate_s"] for r in full), "full_runs": full_runs,
               "full_new_tokens": full[0]["new_tokens"],
               "memorilla_answer_s": statistics.median(r["answer_s"] for r in mem), "embed_s": round(embed_s, 3),
               "memorilla_new_tokens": mem[0]["new_tokens"]}
        row["memorilla_end_to_end_s"] = round(row["memorilla_answer_s"] + row["embed_s"], 3)
        rows.append(row)
        print(json.dumps(row), flush=True)
        save("speed", {"question": question, "rows": rows})


# ---------------------------------------------------------------- finer documents: one fact per document

EXTRACT_SYSTEM = "You extract facts about the user from a conversation between the user and an assistant."
EXTRACT_USER = (
    "Conversation:\n{chat}\n\n"
    "List every fact this conversation reveals about the user: preferences, habits, diet, pets, family, job, "
    "schedule, health, possessions, plans. Include facts mentioned only in passing. One fact per line, each a short "
    "standalone sentence starting with \"The user\". If there are none, write NONE."
)


def parse_facts(text: str) -> list[str]:
    """One fact per line. The extractor sometimes writes a "The user:" header and then bare bullets
    ("Has a bakery shift..."), so bare lines get the subject put back instead of being dropped."""
    meta = re.compile(r"\b(assistant|we need|we can|list (every|of)|facts?\b|conversation|no explicit|none)\b", re.I)
    facts, under_header = [], False
    for raw in text.splitlines():
        bullet = bool(re.match(r"^\s*(?:[-*•]|\d+[.)])\s+", raw))
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw).strip().strip("*").strip()
        if re.fullmatch(r"the user\s*:?", line, re.I):
            under_header = True  # "The user:" followed by bare bullets
            continue
        if not bullet:
            under_header = False
        if len(line) < 12 or meta.search(line):
            continue
        if line.lower().startswith("the user"):
            facts.append(line)
        elif bullet and under_header:
            facts.append(f"The user {line[0].lower()}{line[1:]}")
    return facts


def cmd_extract(args):
    """Write path: LFM3 turns each of Sam's chats into short standalone facts (facts.json)."""
    chats, _ = load_clue()
    runner = Runner([])
    out, t0 = [], time.perf_counter()
    for i, chat in enumerate(chats, 1):
        result = runner.text(EXTRACT_SYSTEM, EXTRACT_USER.format(chat=chat), args.max_new_tokens)
        out.append({"chat": i, "facts": parse_facts(result["answer"]), "raw": result["answer"]})
        if i % 10 == 0:
            print(f"{i}/{len(chats)}  {time.perf_counter() - t0:.0f}s", flush=True)
    save("facts", {"extractor": DECODER, "seconds": round(time.perf_counter() - t0, 1), "chats": out})
    for qid, chat_no in (("getaway", 23), ("parents_dinner", 88), ("gym_time", 131), ("control", 57)):
        print(qid, chat_no, out[chat_no - 1]["facts"])


DECODERS = {"lfm3": (DECODER, "cal_ridge"), "qwen8b": ("Qwen/Qwen3-8B", "native")}


def cmd_facts(args):
    """The Buried Clue with chats vs extracted facts as Memorilla's documents, for one decoder."""
    chats, questions = load_clue()
    facts_spec = json.loads((OUT / "facts.json").read_text())["chats"]
    for c in facts_spec:  # re-parse so parser fixes apply without re-running the extractor
        c["facts"] = parse_facts(c["raw"])
    facts = [f for c in facts_spec for f in c["facts"]]
    fact_chat = [c["chat"] for c in facts_spec for _ in c["facts"]]
    encoder = Encoder()
    chat_vecs, _ = cached_vectors(encoder, "sam", chats)
    fact_vecs, fact_embed_s = cached_vectors(encoder, "sam_facts", facts)
    hound_vecs, _ = cached_vectors(encoder, "hound", load_hound())
    q_vecs = {qid: encoder([q["question"]])[0] for qid, q in questions.items()}
    del encoder  # the 8B decoder and the 4B encoder do not need to share memory
    if P.DEV.type == "mps":
        torch.mps.empty_cache()

    model_id, head = DECODERS[args.decoder]
    runner = Runner([head], model_id)
    runner.text(PERSONA_PROMPT, "Any tips for keeping a houseplant alive?", 8)  # warm-up

    rows, max_new = [], args.max_new_tokens
    for qid, q in questions.items():
        question, q_vec = q["question"], q_vecs[qid]
        clue_facts = [i for i, c in enumerate(fact_chat) if c == q["clue_chat"]]
        rag = top_k(fact_vecs, q_vec, 5)
        mem = lambda vecs: runner.memorilla(head, PERSONA_PROMPT, question, vecs, q_vec, max_new)  # noqa: E731
        methods = {
            "no memory": lambda: runner.text(PERSONA_PROMPT, question, max_new),
            "RAG top-5 facts (text)": lambda: runner.text(PERSONA_PROMPT, user_message([facts[i] for i in rag],
                                                          question), max_new)
            | {"retrieved_from_chats": [fact_chat[i] for i in rag]},
            "Memorilla, 150 chats": lambda: mem(chat_vecs),
            f"Memorilla, {len(facts)} facts": lambda: mem(fact_vecs),
            "Memorilla, clue chat's facts only": lambda: mem(fact_vecs[clue_facts]) if clue_facts
            else {"answer": "(no facts extracted from the clue chat)", "generate_s": 0.0},
            "Memorilla, Hound passages (wrong docs)": lambda: mem(hound_vecs[: len(facts)]),
        }
        for method, fn in methods.items():
            result = fn() | {"question": qid, "method": method}
            result["pass"] = rubric_pass(result["answer"], q["rubric"])
            rows.append(result)
            print(f"[{qid:14s}] {method:40s} {'PASS' if result['pass'] else 'fail'}  {result['generate_s']:6.2f}s  "
                  f"{result['answer'][:120]!r}", flush=True)
    save(f"facts_{args.decoder}", {"decoder": model_id, "head": head, "num_facts": len(facts),
                                   "embed_facts_s": round(fact_embed_s, 2), "rows": rows})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p = sub.add_parser("facts")
    p.add_argument("--decoder", choices=list(DECODERS), required=True)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p = sub.add_parser("embedcheck")
    p.add_argument("--n", type=int, default=6)
    for name in ("clue", "diagnose"):
        p = sub.add_parser(name)
        p.add_argument("--max_new_tokens", type=int, default=256)
    sub.add_parser("book")
    p = sub.add_parser("speed")
    p.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    p.add_argument("--timed_runs", type=int, default=3)
    p.add_argument("--full_repeat_max", type=int, default=128)
    args = parser.parse_args()
    {"embedcheck": cmd_embedcheck, "clue": cmd_clue, "diagnose": cmd_diagnose, "book": cmd_book,
     "speed": cmd_speed, "extract": cmd_extract, "facts": cmd_facts}[args.cmd](args)


if __name__ == "__main__":
    main()
