"""H39: two-stage rating. The one-token yes/no rating (test 35c, "change" wording) keeps the top N facts; then the decoder
thinks about each kept fact alone (Qwen3 thinking mode, up to --think tokens) before the yes/no, read as log P(Yes) - log P(No).
No training. Reads the saved first-stage scores (rate_sweep.json for 48 facts, rate_s96/s192.json for padded profiles).
  MPORT_VLLM_URL=... poc_think.py --src rate_sweep.json --top 12   -> prints where the clue ranks; select_scores.json "think<N>"
"""
import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import poc_profile as PF  # noqa: E402
import poc_whole as W  # noqa: E402

T0 = time.perf_counter()


def post(body):
    req = urllib.request.Request(os.environ["MPORT_VLLM_URL"] + "/v1/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return sorted(json.loads(urllib.request.urlopen(req, timeout=7200).read())["choices"], key=lambda c: c["index"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="rate_sweep.json")
    ap.add_argument("--stage1", default="change")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--think", type=int, default=512)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    K = PF.poc()
    tok = AutoTokenizer.from_pretrained(K.DECODER)
    tx = W.Text(tok)
    rows = json.loads((K.OUT / args.src).read_text())["rows"]
    jobs = []
    for i, r in enumerate(rows):
        s1 = np.array(r[args.stage1])
        r["cand"] = [int(j) for j in np.argsort(-s1, kind="stable")[: args.top]]
        jobs += [(i, j) for j in r["cand"]]
    texts = [tok.apply_chat_template(tx.msgs([rows[i]["facts"][j]], W.RATES["change"].format(q=rows[i]["q"])), tokenize=False,
                                     add_generation_prompt=True, enable_thinking=True) for i, j in jobs]
    thoughts, ntok = [], []
    for b in range(0, len(texts), 1000):
        res = post({"model": "qwen3-8b", "prompt": texts[b:b + 1000], "max_tokens": args.think, "temperature": 0, "stop": ["</think>"]})
        thoughts += [c["text"] for c in res]
        print(f"THINK generated {len(thoughts)} / {len(texts)} at +{time.perf_counter() - T0:.0f}s", flush=True)
    ntok = [len(tok.encode(t, add_special_tokens=False)) for t in thoughts]
    sc = []
    full = [t + th.rstrip() + "\n</think>\n\n" for t, th in zip(texts, thoughts)]
    for b in range(0, len(full), 2000):
        for c in post({"model": "qwen3-8b", "prompt": full[b:b + 2000], "max_tokens": 1, "temperature": 0, "logprobs": 20}):
            top = c["logprobs"]["top_logprobs"][0]
            get = lambda w: max([v for k, v in top.items() if k.strip().lower() == w] or [-30.0])  # noqa: E731
            sc.append(get("yes") - get("no"))
    print(f"THINK {len(sc)} ratings; thought length median {np.median(ntok):.0f} tokens, cut at the limit {np.mean(np.array(ntok) >= args.think - 1):.2f};"
          f" at +{time.perf_counter() - T0:.0f}s", flush=True)
    for r in rows:
        r["think"] = {}
    for (i, j), s, th in zip(jobs, sc, thoughts):
        rows[i]["think"][j] = s
        rows[i].setdefault("thoughts", {})[j] = th[-600:]
    z = lambda x: (np.asarray(x) - np.mean(x)) / (np.std(x) + 1e-6)  # noqa: E731
    ty = np.array([r["type"] for r in rows])
    out = {}

    def final(r, mode):
        s1 = z(r[args.stage1])
        f = s1 - 1000.0
        for j, s in r["think"].items():
            f[j] = (s if mode == "think" else s + 3 * s1[j] if mode == "think+stage1" else s1[j]) + (0.01 * s1[j] if mode == "think" else 0)
        return f

    for mode in ("stage1", "think", "think+stage1"):
        rk = np.array([int((final(r, mode) > final(r, mode)[r["ci"]]).sum()) for r in rows])
        out[mode] = {"top2": float(np.mean(rk < 2)), "top4": float(np.mean(rk < 4)), "top8": float(np.mean(rk < 8))}
        print(f"THINK {mode:14s} {len(rows[0]['facts'])} facts: clue first {np.mean(rk == 0):.2f}  top 2 {np.mean(rk < 2):.2f}  top 4 {np.mean(rk < 4):.2f}"
              f"  top 8 {np.mean(rk < 8):.2f}  (in the {args.top} kept {np.mean(rk < args.top):.2f})  top 4 by type: "
              + "  ".join(f"{t} {np.mean(rk[ty == t] < 4):.2f}" for t in sorted(set(ty))), flush=True)
    clue_kept = [r for r in rows if r["ci"] in r["think"]]
    cs = np.array([r["think"][r["ci"]] for r in clue_kept])
    other = np.array([s for r in clue_kept for j, s in r["think"].items() if j != r["ci"]])
    print(f"THINK after thinking: clue rated Yes {np.mean(cs > 0):.2f}, other kept facts rated Yes {np.mean(other > 0):.2f}", flush=True)
    tag = f"think{args.top}" + ("" if args.src == "rate_sweep.json" else "_" + Path(args.src).stem)
    (K.OUT / f"{tag}.json").write_text(json.dumps({"args": vars(args), "results": out, "rows": [{k: v for k, v in r.items()} for r in rows]}))
    if args.src == "rate_sweep.json":
        sp = K.OUT / "select_scores.json"
        sel = json.loads(sp.read_text())
        for mode in ("think", "think+stage1"):
            sel[f"{tag}{'' if mode == 'think' else '+s1'}"] = {W.row_key(r["q"], r["facts"][r["ci"]], r["facts"]): dict(zip(r["facts"], map(float, final(r, mode)))) for r in rows}
        tmp = sp.with_suffix(".tmp")
        tmp.write_text(json.dumps(sel))
        tmp.replace(sp)


if __name__ == "__main__":
    main()
