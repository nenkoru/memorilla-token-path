"""Proof of concept (POC_PLAN.md): with everything frozen, could retraining Memorilla's core fix buried-clue recall?

Subcommands (outputs in $MPORT_SCRATCH/runs/poc/):
    synth    step 0: Qwen3-8B writes clue chats, topic-matched twins without the clue, and neutral fillers;
             then builds synthetic users (1 clue + 149 fillers) and their counterfactuals (no clue)
    embed    Qwen3-Embedding-4B vectors for every synthetic chat and question (cached)
    probe    step 1: can a small read over the frozen vectors find the clue? cosine / bilinear / MLP / doc-only MLP
    oracle   step 2: frozen core + frozen Qwen3-8B with the clue + k random fillers; text ceiling; no memory
    select   step 3: top-m documents by cosine or by the learned scorer -> frozen core; RAG baseline

Nothing here trains the core, the encoder or the decoder. Only the step-1 probes (< 5M params) are trained.
Data from memorilla-demos is read as JSON only.
"""

import argparse
import json
import os
from pathlib import Path
import random
import re
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import demos_port as D  # noqa: E402
import pv4_port as P  # noqa: E402
from memorilla.benchmarks import PERSONA_PROMPT  # noqa: E402
from memorilla.data import apply_chat_template  # noqa: E402

OUT = P.SCRATCH / "runs/poc"
CONFIG = Path(__file__).resolve().parent / "poc_config.json"
DECODER, HEAD = D.DECODERS["qwen8b"]
SAM_CLUES = {23, 88, 131, 57}  # 1-based; Sam's own clue and control chats are never fillers
N_FILLERS = 149


def cfg():
    return json.loads(CONFIG.read_text())


def save(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(data, indent=1))
    print(f"wrote {OUT / name}", flush=True)


def values(c):
    """(type, value_id, spec) in config order."""
    return [(t, v, s) for t, ts in c["types"].items() for v, s in ts["values"].items()]


# ---------------------------------------------------------------- step 0: synthetic chats
# Two phases. "base": Qwen3-8B writes every chat with no personal constraint (these are the twins and the neutral
# fillers). "clue": Qwen3-8B rewrites only the first user message of a twin so it mentions the constraint in passing.
# So a clue chat and its twin differ by one clause: same topic, same style, same assistant turns.

GEN_SYSTEM = "You write realistic sample conversations between a person and an AI assistant, for a test dataset."
GEN_USER = (
    "Write a short conversation in which a person asks an AI assistant for help with: {topic}.\n\n"
    "Rules:\n"
    "- Format: lines starting with \"User:\" and \"Assistant:\". Two exchanges (User, Assistant, User, "
    "Assistant).\n"
    "- 140 to 190 words in total.\n"
    "- The user writes casually, like a real chat message, and gives a bit of everyday context for why they are "
    "asking. The assistant is helpful and concrete; a short bulleted list is fine.\n"
    "- Output only the conversation, nothing else."
)
REWRITE_SYSTEM = "You edit chat messages for a test dataset. You output only the edited message."
REWRITE_USER = (
    "Here is a message a person sent to an AI assistant:\n\n{message}\n\n"
    "Rewrite it so that the person also mentions, in passing and in the first person, that {fact}. Weave it into "
    "the context of the message in a few words (for example as a reason, a circumstance or an aside in the middle of "
    "a sentence); do not add it as a separate sentence at the very end. The detail must not change what they are "
    "asking about. Keep everything else the same: the question, the tone, the casual style. Output only the "
    "rewritten message, without a \"User:\" prefix."
)


def gen_jobs(c):
    topics = list(c["topics"])
    random.Random(0).shuffle(topics)
    jobs, k = [], 0
    for t, v, s in values(c):
        for j in range(3):
            topic = topics[k]
            k += 1
            jobs.append({"id": f"twin:{v}:{j}", "kind": "twin", "type": t, "value": v, "topic": topic})
            jobs.append({"id": f"clue:{v}:{j}", "kind": "clue", "type": t, "value": v, "topic": topic,
                         "fact": s["fact"], "base": f"twin:{v}:{j}"})
    for topic in topics[k:]:
        jobs.append({"id": f"neutral:{topic}", "kind": "neutral", "type": None, "value": None, "topic": topic})
    return jobs


def clean_chat(text):
    text = text.replace("**", "").strip()
    text = re.sub(r"^```\w*\s*|\s*```$", "", text).strip()
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    start = text.find("User:")
    return text[start:].strip() if start >= 0 else text


def split_first_user(chat):
    """(first user message, rest starting at the first 'Assistant:')."""
    i = chat.find("Assistant:")
    return chat[len("User:"):i].strip(), chat[i:]


def check_chat(job, text, c):
    """Return None if the chat is usable, else the reason."""
    words = len(text.split())
    if not text.startswith("User:") or "Assistant:" not in text:
        return "format"
    if not 80 <= words <= 280:
        return f"length {words}"
    if job["kind"] == "clue":
        spec = c["types"][job["type"]]["values"][job["value"]]
        first, _ = split_first_user(text)
        if not D.rubric_pass(first, spec["mentions"]):
            return "clue missing"
    return None


@torch.no_grad()
def batch_generate(model, tok, prompts, max_new, seed):
    texts = [apply_chat_template(tok, m, add_generation_prompt=True) for m in prompts]
    tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(P.DEV)
    torch.manual_seed(seed)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=0.9,
                         pad_token_id=tok.pad_token_id)
    D.sync()
    texts = [tok.decode(o[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for o in out]
    del enc, out
    torch.mps.empty_cache()
    return texts


def load_raw():
    done = {}
    raw = OUT / "synth_raw.jsonl"
    if raw.exists():
        for line in raw.read_text().splitlines():
            r = json.loads(line)
            if r["id"] not in done or done[r["id"]].get("problem"):
                done[r["id"]] = r  # a good attempt is never replaced
    return done


def cmd_gen(args):
    c = cfg()
    jobs = gen_jobs(c)
    if args.only:
        jobs = [j for j in jobs if re.search(args.only, j["id"])]
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "synth_raw.jsonl"
    done = load_raw()
    runner = D.Runner([], DECODER)
    model, tok = runner.model, runner.tok
    t0 = time.perf_counter()
    for phase in ("base", "clue"):
        for attempt in range(args.retries + 1):
            ok = lambda j: j["id"] in done and not done[j["id"]].get("problem")  # noqa: E731
            pending = [j for j in jobs if (j["kind"] == "clue") == (phase == "clue") and not ok(j)
                       and (phase == "base" or ok({"id": j["base"]}))]
            if args.limit:
                pending = pending[: args.limit]
            if not pending:
                break
            for b in range(0, len(pending), args.batch):
                batch = pending[b: b + args.batch]
                if phase == "base":
                    prompts = [[{"role": "system", "content": GEN_SYSTEM},
                                {"role": "user", "content": GEN_USER.format(topic=j["topic"])}] for j in batch]
                else:
                    prompts = [[{"role": "system", "content": REWRITE_SYSTEM},
                                {"role": "user", "content": REWRITE_USER.format(
                                    message=split_first_user(done[j["base"]]["text"])[0], fact=j["fact"])}]
                               for j in batch]
                seed = 1000 * attempt + b + (500 if phase == "clue" else 0)
                outs = batch_generate(model, tok, prompts, args.max_new_tokens if phase == "base" else 160, seed)
                with raw.open("a") as f:
                    for j, o in zip(batch, outs):
                        if phase == "base":
                            text = clean_chat(o)
                        else:
                            msg = re.sub(r"^(User:)?\s*", "", o.replace("**", "").strip()).strip().strip('"')
                            text = f"User: {msg}\n{split_first_user(done[j['base']]['text'])[1]}"
                        rec = {k: j[k] for k in ("id", "kind", "type", "value", "topic")}
                        rec |= {"text": text, "problem": check_chat(j, text, c), "attempt": attempt, "seed": seed}
                        if j["id"] not in done or done[j["id"]].get("problem"):
                            done[j["id"]] = rec
                        f.write(json.dumps(rec) + "\n")
                n_ok = sum(1 for r in done.values() if not r.get("problem"))
                print(f"{phase} attempt {attempt} batch {b // args.batch + 1}/{-(-len(pending) // args.batch)}  "
                      f"ok {n_ok}  {time.perf_counter() - t0:.0f}s", flush=True)


def sam_chats():
    return json.loads((D.DEMOS / "data/sam/chats.json").read_text())["chats"]


def cmd_build(args):
    """Freeze the generated chats into synth.json and build the synthetic users."""
    c = cfg()
    latest = load_raw()
    jobs = gen_jobs(c)
    chats = [latest[j["id"]] for j in jobs if j["id"] in latest and not latest[j["id"]].get("problem")]
    bad = [j["id"] for j in jobs if j["id"] not in latest or latest[j["id"]].get("problem")]
    good_ids = {ch["id"] for ch in chats}
    # a clue needs its twin and vice versa
    chats = [ch for ch in chats if ch["kind"] == "neutral"
             or ({f"clue:{ch['value']}:{ch['id'].split(':')[2]}", f"twin:{ch['value']}:{ch['id'].split(':')[2]}"}
                 <= good_ids)]
    conflict = {t: re.compile(ts["conflict"], re.I) for t, ts in c["types"].items()}
    test_types = set(c["test_types"])

    sam = [s for s in sam_chats() if s["id"] not in SAM_CLUES]
    rng = random.Random(0)
    rng.shuffle(sam)
    half = len(sam) // 2
    for i, s in enumerate(sam):
        s["split"] = "train" if i < half else "test"

    docs = []  # every candidate document: synthetic chats then Sam fillers
    for ch in chats:
        split = "test" if (ch["type"] in test_types) else "train"
        if ch["kind"] == "neutral":
            split = "test"  # neutrals pad the test users (train users have enough Sam + twin fillers)
        docs.append({"doc": f"syn:{ch['id']}", "kind": ch["kind"], "type": ch["type"], "value": ch["value"],
                     "split": split, "text": ch["text"]})
    for s in sam:
        docs.append({"doc": f"sam:{s['id']}", "kind": "sam", "type": None, "value": None, "split": s["split"],
                     "text": s["text"]})
    for d in docs:
        d["conflicts"] = [t for t, rx in conflict.items() if d["kind"] != "clue" and rx.search(d["text"])]

    pairs = []
    for i, d in enumerate(docs):
        if d["kind"] != "clue":
            continue
        t, v, j = d["type"], d["value"], d["doc"].split(":")[-1]
        split = "test" if t in test_types else ("val" if j == "2" else "train")
        pool = "test" if split == "test" else "train"
        twin = next(k for k, e in enumerate(docs) if e["doc"] == f"syn:twin:{v}:{j}")
        fill = [k for k, e in enumerate(docs) if e["kind"] in ("twin", "neutral", "sam") and e["split"] == pool
                and t not in e["conflicts"] and k != twin]
        for qi, question in enumerate(c["types"][t]["questions"]):
            prng = random.Random(f"{d['doc']}|{qi}")
            others = prng.sample(fill, min(N_FILLERS - 1, len(fill)))
            fillers = [twin] + others  # the clue's own topic twin is always a distractor
            extra = next((k for k in fill if k not in others), None)  # counterfactual: one more filler, no clue
            spec = c["types"][t]["values"][v]
            acts = spec.get("acts_q", {}).get(str(qi), spec["acts"])
            pairs.append({"pair": f"{v}:{j}:q{qi}", "split": split, "type": t, "value": v, "q_index": qi,
                          "question": question, "clue": i, "fillers": fillers,
                          "counterfactual": fillers + ([extra] if extra is not None else []),
                          "rubrics": {"acts": acts, "mentions": spec["mentions"]}})
    save("synth.json", {"docs": docs, "pairs": pairs, "failed_jobs": bad})
    from collections import Counter
    print("docs", Counter((d["kind"], d["split"]) for d in docs))
    print("pairs", Counter(p["split"] for p in pairs), "fillers/pair",
          Counter(len(p["fillers"]) for p in pairs), "failed jobs", len(bad))


# ---------------------------------------------------------------- embeddings


def load_synth():
    return json.loads((OUT / "synth.json").read_text())


def cmd_embed(args):
    """Vectors for every candidate document and question. Sam's chats reuse the earlier run's cache (same encoder)."""
    s = load_synth()
    questions = sorted({p["question"] for p in s["pairs"]})
    sam_q = [q["question"] for q in json.loads((D.DEMOS / "data/sam/questions.json").read_text())["questions"]]
    sam_cache = D.OUT / "emb_sam.npz"
    sam_vecs = np.load(sam_cache)["v"] if sam_cache.exists() else None
    encoder = D.Encoder()
    if sam_vecs is None:  # cache lost with /private/tmp: embed Sam's chats from the same texts
        t_sam = time.perf_counter()
        sam_vecs = encoder([c["text"] for c in sam_chats()])
        sam_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(sam_cache, v=sam_vecs, seconds=time.perf_counter() - t_sam)
    syn = [d for d in s["docs"] if d["doc"].startswith("syn:")]
    t0 = time.perf_counter()
    syn_vecs = encoder([d["text"] for d in syn])
    qv = encoder(questions + sam_q)
    vecs = np.zeros((len(s["docs"]), syn_vecs.shape[1]), np.float32)
    k = 0
    for i, d in enumerate(s["docs"]):
        if d["doc"].startswith("syn:"):
            vecs[i] = syn_vecs[k]
            k += 1
        else:
            vecs[i] = sam_vecs[int(d["doc"].split(":")[1]) - 1]
    np.savez(OUT / "emb_poc.npz", docs=vecs, questions=qv, question_text=np.array(questions + sam_q),
             seconds=time.perf_counter() - t0)
    print(f"embedded {len(syn)} chats + {len(qv)} questions in {time.perf_counter() - t0:.0f}s")


def load_emb():
    z = np.load(OUT / "emb_poc.npz")
    qmap = {str(t): z["questions"][i] for i, t in enumerate(z["question_text"])}
    return z["docs"], qmap


# ---------------------------------------------------------------- step 1: probes over frozen vectors


class Scorer(torch.nn.Module):
    """score(q, d) for every candidate. kinds: cosine | bilinear | mlp | doc_mlp | cos+doc_mlp."""

    def __init__(self, kind, dim=2560, rank=48, hidden=256, dropout=0.1):
        super().__init__()
        self.kind = kind
        self.scale = dim ** 0.5  # unit-norm vectors -> roughly unit-variance coordinates
        self.log_temp = torch.nn.Parameter(torch.tensor(np.log(20.0), dtype=torch.float32))
        self.drop = torch.nn.Dropout(dropout)
        if kind == "bilinear":
            self.U = torch.nn.Parameter(torch.randn(dim, rank) / dim ** 0.5 * 0.1)
            self.V = torch.nn.Parameter(torch.randn(dim, rank) / dim ** 0.5 * 0.1)
        elif kind == "mlp":
            self.net = torch.nn.Sequential(torch.nn.Linear(3 * dim, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout),
                                           torch.nn.Linear(hidden, 1))
        elif kind in ("doc_mlp", "cos+doc_mlp"):
            self.net = torch.nn.Sequential(torch.nn.Linear(dim, hidden), torch.nn.GELU(), torch.nn.Dropout(dropout),
                                           torch.nn.Linear(hidden, 1))

    def forward(self, q, d):
        """q [B, D], d [B, N, D] -> [B, N]."""
        cos = torch.einsum("bd,bnd->bn", q, d)
        base = cos * self.log_temp.exp()
        if self.kind == "cosine":
            return cos
        qs, ds = self.drop(q * self.scale), self.drop(d * self.scale)
        if self.kind == "bilinear":
            return base + torch.einsum("br,bnr->bn", qs @ self.U, ds @ self.V)
        if self.kind == "mlp":
            qe = qs[:, None].expand_as(ds)
            return base + self.net(torch.cat([qe, ds, qe * ds], -1)).squeeze(-1)
        if self.kind == "doc_mlp":
            return self.net(ds).squeeze(-1)
        return base + self.net(ds).squeeze(-1)  # cos+doc_mlp: question relevance plus a learned "constraint" prior


def pair_tensors(pairs, docs, qmap, key="fillers"):
    """Candidates are [clue] + fillers (target index 0). Ragged sets are padded with -inf masks."""
    n = max(1 + len(p[key]) if key == "fillers" else len(p[key]) for p in pairs)
    dim = docs.shape[1]
    d = np.zeros((len(pairs), n, dim), np.float32)
    mask = np.ones((len(pairs), n), bool)  # True = padding
    for i, p in enumerate(pairs):
        idx = ([p["clue"]] + p[key]) if key == "fillers" else p[key]
        d[i, : len(idx)] = docs[idx]
        mask[i, : len(idx)] = False
    q = np.stack([qmap[p["question"]] for p in pairs])
    return torch.from_numpy(q), torch.from_numpy(d), torch.from_numpy(mask)


def ranks(scores, mask):
    """1-based rank of candidate 0 (ties count against it)."""
    s = scores.masked_fill(mask, float("-inf"))
    return (s >= s[:, :1]).sum(1)


def metrics(r, n=None):
    r = np.asarray(r)
    out = {"n": int(len(r)), "R@1": round(float((r <= 1).mean()), 3), "R@5": round(float((r <= 5).mean()), 3),
           "R@10": round(float((r <= 10).mean()), 3), "mean_rank": round(float(r.mean()), 1),
           "median_rank": float(np.median(r))}
    return out


def train_scorer(kind, train, val, seed, epochs=400, lr=1e-3, wd=1e-2, patience=40):
    torch.manual_seed(seed)
    model = Scorer(kind)
    if kind == "cosine":
        return model, {}
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    qt, dt, mt = train
    best, best_state, bad, history = -1.0, None, 0, []
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(qt))
        for b in range(0, len(perm), 32):
            ix = perm[b: b + 32]
            logits = model(qt[ix], dt[ix]).masked_fill(mt[ix], float("-inf"))
            loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(ix), dtype=torch.long))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            r = ranks(model(val[0], val[1]), val[2]).numpy()
        score = float((r <= 5).mean()) - 1e-4 * float(r.mean())  # val R@5, ties broken by mean rank
        history.append(round(score, 4))
        if score > best:
            best, bad = score, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"epochs": epoch + 1, "best_val": round(best, 4)}


def sam_real(model, docs_sam, qmap):
    """Rank of Sam's real clue chats among his 150 chats (note: their clue values are in the train types)."""
    qs = json.loads((D.DEMOS / "data/sam/questions.json").read_text())["questions"]
    out = {}
    for q in qs:
        d = torch.from_numpy(docs_sam)[None]
        with torch.no_grad():
            s = model(torch.from_numpy(qmap[q["question"]])[None], d)[0]
        out[q["id"]] = int((s >= s[q["clue_chat"] - 1]).sum())
    return out


def cmd_probe(args):
    s = load_synth()
    docs, qmap = load_emb()
    sam_vecs = np.load(D.OUT / "emb_sam.npz")["v"]
    by = {k: [p for p in s["pairs"] if p["split"] == k] for k in ("train", "val", "test")}
    T = {k: pair_tensors(v, docs, qmap) for k, v in by.items()}
    # twin rank: is the clue scored above its own topic twin (fillers[0])?
    kinds = ["cosine", "bilinear", "mlp", "doc_mlp", "cos+doc_mlp"]
    results, per_pair = {}, {}
    for kind in kinds:
        runs = []
        for seed in range(args.seeds if kind != "cosine" else 1):
            model, info = train_scorer(kind, T["train"], T["val"], seed)
            with torch.no_grad():
                sc = {k: model(T[k][0], T[k][1]) for k in ("val", "test")}
            r = {k: ranks(sc[k], T[k][2]).numpy() for k in sc}
            twin = (sc["test"][:, 0] > sc["test"][:, 1]).float().mean().item()
            runs.append({"seed": seed, **info, "val": metrics(r["val"]), "test": metrics(r["test"]),
                         "test_beats_twin": round(twin, 3), "sam_real_rank": sam_real(model, sam_vecs, qmap),
                         "test_ranks": r["test"].tolist()})
            if seed == 0:
                torch.save(model.state_dict(), OUT / f"scorer_{kind}.pt")
                per_pair[kind] = scorer_topm(model, by["test"], docs, qmap)
        agg = {split: {m: round(float(np.mean([x[split][m] for x in runs])), 3) for m in ("R@1", "R@5", "R@10",
                                                                                          "mean_rank")}
               for split in ("val", "test")}
        agg["test_beats_twin"] = round(float(np.mean([x["test_beats_twin"] for x in runs])), 3)
        agg["test_R@5_per_seed"] = [x["test"]["R@5"] for x in runs]
        results[kind] = {"mean_over_seeds": agg, "runs": runs}
        print(f"{kind:12s} val R@5 {agg['val']['R@5']:.3f}  test R@1 {agg['test']['R@1']:.3f}  R@5 "
              f"{agg['test']['R@5']:.3f}  R@10 {agg['test']['R@10']:.3f}  mean rank {agg['test']['mean_rank']:6.1f}  "
              f"beats twin {agg['test_beats_twin']:.2f}  seeds {agg['test_R@5_per_seed']}  "
              f"sam {runs[0]['sam_real_rank']}", flush=True)
    # per test type, for every scorer (seed 0)
    by_type = {}
    for kind in kinds:
        r = np.array(results[kind]["runs"][0]["test_ranks"])
        by_type[kind] = {t: metrics(r[[i for i, p in enumerate(by["test"]) if p["type"] == t]])
                         for t in sorted({p["type"] for p in by["test"]})}
    save("probe.json", {"n_pairs": {k: len(v) for k, v in by.items()},
                        "candidates_per_test_pair": int(T["test"][1].shape[1]), "results": results,
                        "by_type_seed0": by_type, "topm": per_pair})


def scorer_topm(model, pairs, docs, qmap, m=16):
    """Top-m document indices (into synth docs) for each pair's real set and its counterfactual set."""
    out = {}
    for key in ("fillers", "counterfactual"):
        q, d, mask = pair_tensors(pairs, docs, qmap, key)
        with torch.no_grad():
            sc = model(q, d).masked_fill(mask, float("-inf"))
        order = sc.argsort(1, descending=True)[:, :m].numpy()
        for i, p in enumerate(pairs):
            idx = ([p["clue"]] + p["fillers"]) if key == "fillers" else p["counterfactual"]
            out.setdefault(p["pair"], {})[key] = [int(idx[j]) for j in order[i]]
    return out


def cmd_curve(args):
    """Step-1 extra: held-out-type R@5 as a function of how many train types the scorer saw (no new data)."""
    s = load_synth()
    docs, qmap = load_emb()
    train_types = sorted({p["type"] for p in s["pairs"] if p["split"] != "test"})
    test = [p for p in s["pairs"] if p["split"] == "test"]
    T_test = pair_tensors(test, docs, qmap)
    rows = []
    for n in args.sizes:
        for rep in range(args.reps if n < len(train_types) else 1):
            types = sorted(random.Random(f"curve|{n}|{rep}").sample(train_types, n))
            tr = pair_tensors([p for p in s["pairs"] if p["split"] == "train" and p["type"] in types], docs, qmap)
            va = pair_tensors([p for p in s["pairs"] if p["split"] == "val" and p["type"] in types], docs, qmap)
            for kind in args.kinds:
                model, info = train_scorer(kind, tr, va, seed=rep)
                with torch.no_grad():
                    r = ranks(model(T_test[0], T_test[1]), T_test[2]).numpy()
                row = {"n_types": n, "rep": rep, "types": types, "kind": kind, **metrics(r)}
                rows.append(row)
                print(json.dumps({k: row[k] for k in ("n_types", "rep", "kind", "R@5", "R@10", "mean_rank")}),
                      flush=True)
    summary = {}
    for kind in args.kinds:
        for n in args.sizes:
            rs = [r for r in rows if r["kind"] == kind and r["n_types"] == n]
            summary[f"{kind}|{n}"] = {"R@5": round(float(np.mean([r["R@5"] for r in rs])), 3),
                                      "R@10": round(float(np.mean([r["R@10"] for r in rs])), 3),
                                      "mean_rank": round(float(np.mean([r["mean_rank"] for r in rs])), 1),
                                      "reps": len(rs)}
    save("curve.json", {"summary": summary, "rows": rows})
    for k, v in summary.items():
        print(k, v)


# ---------------------------------------------------------------- steps 2 and 3: frozen core + frozen Qwen3-8B


def eval_pairs(s, n_per_value):
    """A fixed, stratified subset of the held-out-type pairs: n per value."""
    test = [p for p in s["pairs"] if p["split"] == "test"]
    out = []
    for v in sorted({p["value"] for p in test}):
        group = [p for p in test if p["value"] == v]
        random.Random(f"eval|{v}").shuffle(group)
        out += group[:n_per_value]
    return out


class Gen:
    """Batched greedy generation on the frozen decoder, with or without the frozen core's memory tokens."""

    def __init__(self, max_new, text_max=None):
        self.text_max = text_max or max_new
        self.runner = D.Runner([HEAD], DECODER)
        self.memory = self.runner.heads[HEAD]
        self.max_new = max_new
        self.no_memory = {}
        self.runner.text(PERSONA_PROMPT, "Any tips for keeping a houseplant alive?", 8)  # warm-up

    def _decode(self, out, start):
        tok = self.runner.tok
        return [tok.decode(o[start:], skip_special_tokens=True).strip() for o in out]

    @torch.no_grad()
    def memorilla(self, question, q_vec, doc_sets, docs):
        """One answer per document set; all share the question, so the prompts have equal length (no padding)."""
        n = max(len(x) for x in doc_sets)
        d = torch.zeros(len(doc_sets), n, docs.shape[1])
        mask = torch.ones(len(doc_sets), n, dtype=torch.bool)
        for i, idx in enumerate(doc_sets):
            d[i, : len(idx)] = torch.from_numpy(docs[idx])
            mask[i, : len(idx)] = False
        d, mask = d.to(P.DEV), mask.to(P.DEV)
        q = torch.from_numpy(q_vec)[None].expand(len(doc_sets), -1).to(P.DEV)
        tokens = self.memory(d, mask, q)
        r = self.runner
        ids = r.ids(D.build_messages(question, PERSONA_PROMPT, num_memories=self.memory.num_memories))
        ids = ids.expand(len(doc_sets), -1)
        embeds = P.input_embeds(r.model, r.tok, r.memory_id, ids, tokens)
        out = r.model.generate(inputs_embeds=embeds, attention_mask=torch.ones(ids.shape, device=P.DEV),
                               max_new_tokens=self.max_new, do_sample=False, pad_token_id=r.tok.pad_token_id)
        D.sync()
        return self._decode(out, 0), self.attention(d, mask, q)

    @torch.no_grad()
    def attention(self, d, mask, q):
        """Share of the core's cross-attention (averaged over slots and heads) on each document, per layer.
        Re-implements MemoryModule.forward up to the cross-attention weights (pv4: learned slots, no self-attn)."""
        m = self.memory
        slots = m._initial_slots(d, mask, q)
        cond, _ = m.query_cond_attn(slots, q[:, None], q[:, None])
        slots = m.query_cond_norm(slots + cond)
        docs = m.doc_norm(d)
        weights = []
        for norm, attn, fnorm, ffn in zip(m.cross_attn_norms, m.cross_attn_layers, m.ffn_norms, m.ffn_layers):
            att, w = attn(norm(slots), docs, docs, key_padding_mask=mask, need_weights=True,
                          average_attn_weights=True)
            weights.append(w.mean(1).float().cpu())  # [B, N]
            slots = slots + att
            slots = slots + ffn(fnorm(slots))
        return torch.stack(weights, 1).numpy()  # [B, layers, N]

    @torch.no_grad()
    def text(self, contents):
        """Plain-text answers (left-padded batch)."""
        r = self.runner
        texts = [apply_chat_template(r.tok, [{"role": "system", "content": PERSONA_PROMPT},
                                             {"role": "user", "content": c}], add_generation_prompt=True)
                 for c in contents]
        r.tok.padding_side = "left"
        enc = r.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(P.DEV)
        out = r.model.generate(**enc, max_new_tokens=self.text_max, do_sample=False, pad_token_id=r.tok.pad_token_id)
        D.sync()
        return self._decode(out, enc["input_ids"].shape[1])

    def none(self, question):
        if question not in self.no_memory:
            self.no_memory[question] = self.text([question])[0]
        return self.no_memory[question]


def score(answer, rubrics):
    return {name: D.rubric_pass(answer, rb) for name, rb in rubrics.items()}


def summarize(rows, conditions):
    table = {}
    for c in conditions:
        rs = [r for r in rows if r["condition"] == c]
        if rs:
            table[c] = {"n": len(rs), "acts": round(float(np.mean([r["acts"] for r in rs])), 3),
                        "mentions": round(float(np.mean([r["mentions"] for r in rs])), 3)}
            att = [r["clue_attention"] for r in rs if r.get("clue_attention") is not None]
            if att:
                table[c]["clue_attention_x_uniform"] = round(float(np.mean(att)), 2)
    return table


def log_row(row):
    print(f"[{row['pair']:22s}] {row['condition']:26s} acts {int(row['acts'])} mentions {int(row['mentions'])}  "
          f"{row['answer'][:120]!r}", flush=True)


def cmd_oracle(args):
    s = load_synth()
    docs, qmap = load_emb()
    pairs = eval_pairs(s, args.per_value)
    gen = Gen(args.max_new_tokens)
    ks = [0, 4, 16, 64, 149]
    rows, t0 = [], time.perf_counter()
    for n, p in enumerate(pairs):
        rng = random.Random(f"oracle|{p['pair']}")
        fillers = list(p["fillers"])
        rng.shuffle(fillers)
        sets = [[p["clue"]] + fillers[:k] for k in ks] + [p["counterfactual"]]
        names = [f"clue + {k} fillers" for k in ks] + ["counterfactual (no clue), 150"]
        answers, att = gen.memorilla(p["question"], qmap[p["question"]], sets, docs)
        for i, (name, a) in enumerate(zip(names, answers)):
            row = {"pair": p["pair"], "value": p["value"], "type": p["type"], "condition": name,
                   "num_docs": len(sets[i]), "answer": a, **score(a, p["rubrics"])}
            if i < len(ks):  # attention on the clue (index 0 of the set), as a multiple of uniform 1/N
                row["clue_attention"] = float(att[i, :, 0].mean() * len(sets[i]))
                row["clue_attention_by_layer"] = [round(float(x * len(sets[i])), 2) for x in att[i, :, 0]]
                if ks[i] > 0:
                    twin_pos = sets[i].index(p["fillers"][0]) if p["fillers"][0] in sets[i] else None
                    if twin_pos is not None:
                        row["twin_attention"] = float(att[i, :, twin_pos].mean() * len(sets[i]))
            rows.append(row)
            log_row(row)
        clue_text = s["docs"][p["clue"]]["text"]
        a_text = gen.text([D.user_message([clue_text], p["question"])])[0]
        for name, a in (("text ceiling: clue chat as text", a_text), ("no memory", gen.none(p["question"]))):
            row = {"pair": p["pair"], "value": p["value"], "type": p["type"], "condition": name, "answer": a,
                   **score(a, p["rubrics"])}
            rows.append(row)
            log_row(row)
        print(f"--- {n + 1}/{len(pairs)}  {time.perf_counter() - t0:.0f}s", flush=True)
        torch.mps.empty_cache()
        conditions = names + ["text ceiling: clue chat as text", "no memory"]
        save("oracle.json", {"max_new_tokens": args.max_new_tokens, "pairs": [p["pair"] for p in pairs],
                             "table": summarize(rows, conditions), "rows": rows})
    for c, v in summarize(rows, conditions).items():
        print(f"{c:34s} {v}")


def cmd_select(args):
    s = load_synth()
    docs, qmap = load_emb()
    probe = json.loads((OUT / "probe.json").read_text())
    kind = args.scorer or max((k for k in probe["results"] if k != "cosine"),
                              key=lambda k: (probe["results"][k]["mean_over_seeds"]["val"]["R@5"],
                                             -probe["results"][k]["mean_over_seeds"]["val"]["mean_rank"]))
    learned, cosine = probe["topm"][kind], probe["topm"]["cosine"]
    pairs = eval_pairs(s, args.per_value)
    gen = Gen(args.max_new_tokens, args.text_tokens)
    rows, t0 = [], time.perf_counter()
    for n, p in enumerate(pairs):
        L, C = learned[p["pair"]], cosine[p["pair"]]
        sets = {f"cosine top-4 -> core": C["fillers"][:4], "cosine top-8 -> core": C["fillers"][:8],
                f"learned top-4 -> core": L["fillers"][:4], f"learned top-8 -> core": L["fillers"][:8],
                "learned top-4, counterfactual -> core": L["counterfactual"][:4],
                "learned top-8, counterfactual -> core": L["counterfactual"][:8]}
        answers, att = gen.memorilla(p["question"], qmap[p["question"]], list(sets.values()), docs)
        for i, ((name, idx), a) in enumerate(zip(sets.items(), answers)):
            row = {"pair": p["pair"], "value": p["value"], "type": p["type"], "condition": name, "docs": idx,
                   "clue_selected": p["clue"] in idx, "answer": a, **score(a, p["rubrics"])}
            if p["clue"] in idx:
                row["clue_attention"] = float(att[i, :, idx.index(p["clue"])].mean() * len(idx))
            rows.append(row)
            log_row(row)
        texts = {"RAG: learned top-4 as text": L["fillers"][:4], "RAG: cosine top-4 as text": C["fillers"][:4]}
        answers = gen.text([D.user_message([s["docs"][i]["text"] for i in idx], p["question"])
                            for idx in texts.values()])
        for (name, idx), a in zip(texts.items(), answers):
            row = {"pair": p["pair"], "value": p["value"], "type": p["type"], "condition": name, "docs": idx,
                   "clue_selected": p["clue"] in idx, "answer": a, **score(a, p["rubrics"])}
            rows.append(row)
            log_row(row)
        print(f"--- {n + 1}/{len(pairs)}  {time.perf_counter() - t0:.0f}s", flush=True)
        torch.mps.empty_cache()
        conditions = list(sets) + list(texts)
        table = summarize(rows, conditions)
        for c in table:
            sel = [r["clue_selected"] for r in rows if r["condition"] == c]
            table[c]["clue_selected"] = round(float(np.mean(sel)), 3)
        save("select.json", {"scorer": kind, "max_new_tokens": args.max_new_tokens, "text_tokens": args.text_tokens,
                             "pairs": [p["pair"] for p in pairs], "table": table, "rows": rows})
    for c, v in table.items():
        print(f"{c:40s} {v}")


# Strict rubric scoring (fixed after reading the first 3 oracle pairs; reported next to the plan's substring rule).
# A term must start at a word boundary and may carry at most 4 more letters ("diabet" -> "diabetes", but "son" no
# longer hits "reason" and "rest" no longer hits "restaurant"). none_of lists gain the phrasings the substring lists
# missed ("Fly from New York ...", "flying into Miami", "rent a car").
NO_FLY = ["fly from", "flying from", "fly in", "flying in", "flying into", "fly out", "flying out", "fly back",
          "take a flight", "catch a flight", "book a round", "your flight", "flight from"]
NO_DRIVE = ["drive from", "driving from", "drive through", "drive along", "drive down", "drive out", "drive over",
            "scenic route", "rent a car", "rental car", "renting a car", "your vehicle", "pack the car", "pack your car"]
STRICT_EXTRA_NONE = {"fear_flying": NO_FLY, "climate_flying": NO_FLY, "no_car": NO_DRIVE,
                     "no_license": NO_DRIVE + ["rent a truck", "rent a van", "rental truck"]}


def _hit(text, term):
    return re.search(r"\b" + re.escape(term.lower()) + r"[a-z]{0,4}\b", text) is not None


def strict_pass(answer, rubric, value):
    t = answer.lower()
    if not all(any(_hit(t, term) for term in group) for group in rubric.get("all_of", [])):
        return False
    none = list(rubric.get("none_of", [])) + (STRICT_EXTRA_NONE.get(value, []) if "none_of" in rubric or
                                               value in STRICT_EXTRA_NONE else [])
    return not any(_hit(t, term) for term in none)


def strict_score(answer, rubrics, value):
    """acts gets the extra none_of list; mentions only the stricter matching."""
    return {"acts_strict": strict_pass(answer, rubrics["acts"], value),
            "mentions_strict": strict_pass(answer, {"all_of": rubrics["mentions"]["all_of"]}, None)}


def table_both(rows, conditions):
    out = {}
    for c in conditions:
        rs = [r for r in rows if r["condition"] == c]
        if not rs:
            continue
        e = {"n": len(rs)}
        for k in ("acts", "mentions", "acts_strict", "mentions_strict"):
            e[k] = round(float(np.mean([r[k] for r in rs])), 3)
        att = [r["clue_attention"] for r in rs if r.get("clue_attention") is not None]
        if att:
            e["clue_attention_x_uniform"] = round(float(np.mean(att)), 2)
        if "clue_selected" in rs[0]:
            e["clue_selected"] = round(float(np.mean([r["clue_selected"] for r in rs])), 3)
        out[c] = e
    return out


def paired_lift(rows, cond, control, key="acts_strict"):
    """Per pair: answer acts with the clue but not on the matched no-clue set (and the reverse)."""
    a = {r["pair"]: r[key] for r in rows if r["condition"] == cond}
    b = {r["pair"]: r[key] for r in rows if r["condition"] == control}
    common = sorted(set(a) & set(b))
    if not common:
        return None
    return {"n": len(common), "clue_only": round(float(np.mean([a[p] and not b[p] for p in common])), 3),
            "control_only": round(float(np.mean([b[p] and not a[p] for p in common])), 3),
            "diff": round(float(np.mean([a[p] for p in common]) - np.mean([b[p] for p in common])), 3)}


def cmd_rescore(args):
    s = load_synth()
    rub = {p["pair"]: p for p in s["pairs"]}
    for name in ("oracle.json", "ceiling.json", "select.json"):
        f = OUT / name
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        for r in d["rows"]:
            p = rub[r["pair"]]
            r.update(strict_score(r["answer"], p["rubrics"], p["value"]))
        conds = list(dict.fromkeys(r["condition"] for r in d["rows"]))
        d["table_both"] = table_both(d["rows"], conds)
        by_type = {}
        for t in sorted({r["type"] for r in d["rows"]}):
            by_type[t] = table_both([r for r in d["rows"] if r["type"] == t], conds)
        d["table_both_by_type"] = by_type
        if name == "oracle.json":
            d["paired_vs_counterfactual"] = {c: paired_lift(d["rows"], c, "counterfactual (no clue), 150")
                                             for c in conds if c.startswith("clue +")}
        if name == "select.json":
            d["paired_vs_counterfactual"] = {
                f"learned top-{m}": paired_lift(d["rows"], f"learned top-{m} -> core",
                                                f"learned top-{m}, counterfactual -> core") for m in (4, 8)}
        save(name, d)
        print(f"== {name}")
        for c, v in d["table_both"].items():
            print(f"  {c:42s} {v}")
        for c, v in d.get("paired_vs_counterfactual", {}).items():
            print(f"  paired {c:35s} {v}")


def cmd_ceiling(args):
    """Text ceiling and no-memory baseline again with a longer budget (plain Qwen3-8B answers open with a preamble)."""
    s = load_synth()
    pairs = eval_pairs(s, args.per_value)
    gen = Gen(args.max_new_tokens)
    rows, t0 = [], time.perf_counter()
    questions = sorted({p["question"] for p in pairs})
    none = dict(zip(questions, gen.text(questions)))
    for i in range(0, len(pairs), args.batch):
        chunk = pairs[i: i + args.batch]
        answers = gen.text([D.user_message([s["docs"][p["clue"]]["text"]], p["question"]) for p in chunk])
        for p, a in zip(chunk, answers):
            for name, ans in (("text ceiling: clue chat as text", a), ("no memory", none[p["question"]])):
                row = {"pair": p["pair"], "value": p["value"], "type": p["type"], "condition": name,
                       "answer": ans, **score(ans, p["rubrics"])}
                rows.append(row)
                log_row(row)
        print(f"--- {i + len(chunk)}/{len(pairs)}  {time.perf_counter() - t0:.0f}s", flush=True)
    save("ceiling.json", {"max_new_tokens": args.max_new_tokens, "rows": rows})


def cmd_coreprobe(args):
    """Does the clue survive the frozen core? Linear clue-vs-twin probe on (a) the encoder vector and (b) the core's
    memory tokens for that single document, trained on train types, tested on held-out types. CPU only."""
    s = load_synth()
    docs, qmap = load_emb()
    mem = P.load_memory(HEAD, DECODER).cpu().eval()
    # every (clue chat, question) pair; the twin is always fillers[0]
    items = [("test" if p["split"] == "test" else "train", p["type"], p["question"], p["clue"], p["fillers"][0])
             for p in s["pairs"]]
    feats = {"encoder": [], "core_mean": [], "core_flat": []}
    labels, splits, groups = [], [], []
    with torch.no_grad():
        for n, (split, t, q, ci, ti) in enumerate(items):
            for idx, y in ((ci, 1), (ti, 0)):
                x = torch.from_numpy(docs[[idx]]).float()[None]
                tok = mem(x, torch.zeros(1, 1, dtype=torch.bool), torch.from_numpy(qmap[q]).float()[None])[0]
                feats["encoder"].append(docs[idx])
                feats["core_mean"].append(tok.mean(0).numpy())
                feats["core_flat"].append(tok.flatten().numpy())
                labels.append(y); splits.append(split); groups.append(n)
    labels, splits, groups = np.array(labels), np.array(splits), np.array(groups)
    out = {}
    for name, X in feats.items():
        X = np.stack(X).astype(np.float32)
        tr, te = splits == "train", splits == "test"
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xt = torch.from_numpy((X - mu) / sd)
        res = []
        for seed in range(3):
            torch.manual_seed(seed)
            w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
            opt = torch.optim.Adam([w, b], lr=1e-3)
            y = torch.from_numpy(labels).float()
            for _ in range(300):
                opt.zero_grad()
                logit = Xt[tr] @ w + b
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y[tr]) + args.l2 * (w ** 2).sum()
                loss.backward(); opt.step()
            with torch.no_grad():
                sc = (Xt @ w + b).numpy()
            acc = float(((sc[te] > 0) == labels[te]).mean())
            g = groups[te]; scte = sc[te]; lte = labels[te]
            pair = [scte[(g == k) & (lte == 1)][0] > scte[(g == k) & (lte == 0)][0] for k in np.unique(g)]
            res.append({"test_acc": round(acc, 3), "test_pairwise_clue_above_twin": round(float(np.mean(pair)), 3)})
        out[name] = {"dim": int(X.shape[1]), "seeds": res,
                     "pairwise_mean": round(float(np.mean([r["test_pairwise_clue_above_twin"] for r in res])), 3)}
        # how far does the clue move the representation, relative to how far topics move it?
        Z = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        cos_ct = [float(Z[(groups == k) & (labels == 1)][0] @ Z[(groups == k) & (labels == 0)][0])
                  for k in np.unique(groups)]
        rnd = np.random.default_rng(0).permutation(len(Z))
        cos_rand = float(np.mean(np.sum(Z * Z[rnd], 1)))
        out[name]["cos_clue_vs_own_twin"] = round(float(np.mean(cos_ct)), 4)
        out[name]["cos_random_pair"] = round(cos_rand, 4)
        print(name, json.dumps(out[name]), flush=True)
    save("coreprobe.json", {"n_items": len(items), "l2": args.l2, "results": out})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("gen")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=400)
    p.add_argument("--retries", type=int, default=2)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--only", default="")
    sub.add_parser("build")
    sub.add_parser("embed")
    p = sub.add_parser("probe")
    p.add_argument("--seeds", type=int, default=3)
    p = sub.add_parser("curve")
    p.add_argument("--sizes", type=int, nargs="+", default=[2, 4, 6, 8])
    p.add_argument("--reps", type=int, default=4)
    p.add_argument("--kinds", nargs="+", default=["doc_mlp", "bilinear"])
    for name in ("oracle", "select"):
        p = sub.add_parser(name)
        p.add_argument("--per_value", type=int, default=5)
        p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--scorer", default="")
    p.add_argument("--text_tokens", type=int, default=320)
    sub.add_parser("rescore")
    p = sub.add_parser("coreprobe")
    p.add_argument("--l2", type=float, default=1e-3)
    p = sub.add_parser("ceiling")
    p.add_argument("--per_value", type=int, default=5)
    p.add_argument("--max_new_tokens", type=int, default=320)
    p.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    {"gen": cmd_gen, "build": cmd_build, "embed": cmd_embed, "probe": cmd_probe, "curve": cmd_curve,
     "oracle": cmd_oracle, "select": cmd_select,
     "rescore": cmd_rescore, "coreprobe": cmd_coreprobe, "ceiling": cmd_ceiling}[args.cmd](args)


if __name__ == "__main__":
    main()
