"""Diagnostics for poc_agentmem.py: where do exact identifiers (repo, file, function names) get lost?

    retrieve   encoder only: do a session's frozen chunk vectors pick out its own repo / file / function name string
               among those of all test sessions? (score = max cosine over the session's chunks; rank of the true one)
    text       decoder only (no memory, no LoRA): Qwen3-8B reads the same chunks AS TEXT and answers Q_SINGLE.
               Same scorer as the memory eval -> the ceiling for what the chunks contain.
Outputs in $MPORT_SCRATCH/runs/agentmem/: probe_retrieve.json, eval_text.json.
"""

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import poc_agentmem as A  # noqa: E402
from poc_agentmem import D, P, save  # noqa: E402
from memorilla.data import apply_chat_template  # noqa: E402


def test_sessions(ss, n=None):
    test = [s for s in ss if s["split"] == "test"]
    if n:
        random.Random(3).shuffle(test)
        test = test[:n]
    return test


def cmd_retrieve(args):
    ss, chunks, _ = A.load_emb()
    test = test_sessions(ss)
    enc = D.Encoder()
    out = {}
    kinds = {"repo": lambda s: [s["repo"]], "file": lambda s: s["files"][:1],
             "func": lambda s: s["funcs"][:1]}
    for kind, get in kinds.items():
        items = [(i, get(s)[0]) for i, s in enumerate(test) if get(s)]
        cands = sorted({c for _, c in items})
        cv = enc(cands, batch_size=16)
        idx = {c: j for j, c in enumerate(cands)}
        ranks = []
        for i, c in items:
            sims = chunks[test[i]["rows"]].astype(np.float32) @ cv.T  # [chunks, cands]
            score = sims.max(0)
            ranks.append(int((score > score[idx[c]]).sum()))
        ranks = np.array(ranks)
        out[kind] = {"n": len(items), "candidates": len(cands), "top1": float((ranks == 0).mean()),
                     "top5": float((ranks < 5).mean()), "chance_top1": 1 / len(cands),
                     "median_rank": float(np.median(ranks))}
        print(kind, out[kind], flush=True)
    save("probe_retrieve.json", out)


def cmd_text(args):
    ss, _, _ = A.load_emb()
    test = test_sessions(ss, args.n)
    r = D.Runner([], A.DECODER)
    rows, t0 = [], time.perf_counter()
    for i, s in enumerate(test):
        content = "Earlier coding session (excerpts):\n\n" + "\n\n".join(s["chunks"]) + "\n\n" + A.Q_SINGLE
        text = apply_chat_template(r.tok, [{"role": "system", "content": A.SYSTEM}, {"role": "user", "content": content}],
                                   add_generation_prompt=True)
        ids = torch.tensor([r.tok.encode(text, add_special_tokens=r.add_special)], device=P.DEV)
        with torch.no_grad():
            out = r.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.max_new_tokens,
                                   do_sample=False, pad_token_id=r.tok.pad_token_id)
        ans = r.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()
        rows.append({"session": s["id"], "condition": "text", "prompt_tokens": int(ids.shape[1]), "answer": ans,
                     **A.score(s, ans, False)})
        del out, ids
        torch.mps.empty_cache()
        if (i + 1) % 5 == 0 or i + 1 == len(test):
            tab = {m: float(np.mean([x[m] for x in rows if x.get(m) is not None]))
                   for m in ("repo", "issue", "files_path", "files_name", "funcs")}
            save("eval_text.json", {"rows": rows, "table": tab})
            print(f"{i + 1}/{len(test)}  {(time.perf_counter() - t0) / 60:.0f} min  {tab}", flush=True)


@torch.no_grad()
def cmd_coreid(args):
    """Does the frozen core keep session identity? Split each test session's chunks into two halves (odd/even), build
    memory from each half, and check whether half A's representation is nearest to half B of the SAME session among
    all test sessions. Compared for: mean chunk vector (encoder output) and the core's 16 tokens (flattened)."""
    ss, chunks, qv = A.load_emb()
    test = test_sessions(ss)
    core = P.load_memory("native", A.DECODER).eval().float().cpu()
    q = torch.from_numpy(qv[0].astype(np.float32))[None]
    reps = {"encoder mean": ([], []), "core 16 tokens": ([], [])}
    for s in test:
        rows = s["rows"]
        for h, part in enumerate((rows[0::2], rows[1::2])):
            x = torch.from_numpy(chunks[part].astype(np.float32))[None]
            reps["encoder mean"][h].append(x[0].mean(0).numpy())
            reps["core 16 tokens"][h].append(core(x, None, q)[0].flatten().numpy())
    out = {}
    for name, (a, b) in reps.items():
        a, b = np.stack(a), np.stack(b)
        a = (a - a.mean(0)) / (np.linalg.norm(a - a.mean(0), axis=1, keepdims=True) + 1e-8)
        b = (b - b.mean(0)) / (np.linalg.norm(b - b.mean(0), axis=1, keepdims=True) + 1e-8)
        sims = a @ b.T
        out[name] = {"n": len(a), "top1": float((sims.argmax(1) == np.arange(len(a))).mean()), "chance": 1 / len(a)}
        print(name, out[name], flush=True)
    save("probe_coreid.json", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("retrieve")
    sub.add_parser("coreid")
    t = sub.add_parser("text")
    t.add_argument("--n", type=int, default=60)
    t.add_argument("--max_new_tokens", type=int, default=96)
    args = ap.parse_args()
    {"retrieve": cmd_retrieve, "text": cmd_text, "coreid": cmd_coreid}[args.cmd](args)


if __name__ == "__main__":
    main()
