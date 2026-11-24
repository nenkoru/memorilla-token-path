"""H40: after the per-fact rating keeps the top K facts, one comparative question per request: "which one of these facts most
changes what a good reply should say or recommend?" The probability of each number (forward and reversed order, averaged) re-ranks
the kept facts. No training. Reads rate_sweep.json (48 facts) or rate_s96/s192.json (padded).
  MPORT_VLLM_URL=... poc_pick.py --src rate_sweep.json --top 8
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
ASK = ('The user asks: "{q}"\n\nHere are some facts about this user:\n\n{facts}\n\nWhich one of these facts most changes what a good reply '
       "should say or recommend? Answer with its number only.")


def number_logprobs(tok, texts, n):
    req = urllib.request.Request(os.environ["MPORT_VLLM_URL"] + "/v1/completions",
                                 json.dumps({"model": "qwen3-8b", "prompt": texts, "max_tokens": 1, "temperature": 0, "logprobs": 20}).encode(),
                                 {"Content-Type": "application/json"})
    out = []
    for c in sorted(json.loads(urllib.request.urlopen(req, timeout=3600).read())["choices"], key=lambda c: c["index"]):
        top = c["logprobs"]["top_logprobs"][0]
        out.append([max([v for k, v in top.items() if k.strip() == str(i + 1)] or [-30.0]) for i in range(n)])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="rate_sweep.json")
    ap.add_argument("--stage1", default="change")
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    K = PF.poc()
    tok = AutoTokenizer.from_pretrained(K.DECODER)
    tx = W.Text(tok)
    rows = json.loads((K.OUT / args.src).read_text())["rows"]
    texts, orders = [], []
    for r in rows:
        cand = [int(j) for j in np.argsort(-np.array(r[args.stage1]), kind="stable")[: args.top]]
        r["cand"] = cand
        for order in (sorted(cand), sorted(cand)[::-1]):  # profile order and reversed
            facts = "\n".join(f"{i + 1}. {r['facts'][j]}" for i, j in enumerate(order))
            texts.append(tx.apply(tok, [{"role": "system", "content": tx.sys}, {"role": "user", "content": ASK.format(q=r["q"], facts=facts)}],
                                  add_generation_prompt=True))
            orders.append(order)
    lp = np.concatenate([number_logprobs(tok, texts[b:b + 1000], args.top) for b in range(0, len(texts), 1000)])
    print(f"PICK {len(texts)} comparative questions at +{time.perf_counter() - T0:.0f}s", flush=True)
    z = lambda x: (np.asarray(x) - np.mean(x)) / (np.std(x) + 1e-6)  # noqa: E731
    ty = np.array([r["type"] for r in rows])
    for i, r in enumerate(rows):
        pk = {}
        for o, l in ((orders[2 * i], lp[2 * i]), (orders[2 * i + 1], lp[2 * i + 1])):
            for pos, j in enumerate(o):
                pk[j] = pk.get(j, 0.0) + l[pos] / 2
        r["pick"] = pk
    sel = {}
    for mode in ("stage1", "pick", "pick+stage1"):
        rks = []
        for r in rows:
            s1 = z(r[args.stage1])
            f = s1 - 1000.0
            for j, v in r["pick"].items():
                f[j] = s1[j] if mode == "stage1" else v + 0.01 * s1[j] if mode == "pick" else v + s1[j]
            rks.append(int((f > f[r["ci"]]).sum()))
            sel.setdefault(mode, {})[W.row_key(r["q"], r["facts"][r["ci"]], r["facts"])] = dict(zip(r["facts"], map(float, f)))
        rk = np.array(rks)
        print(f"PICK {mode:12s} {len(rows[0]['facts'])} facts: clue first {np.mean(rk == 0):.2f}  top 2 {np.mean(rk < 2):.2f}  top 4 {np.mean(rk < 4):.2f}"
              f"  (in the {args.top} kept {np.mean(rk < args.top):.2f})  top 2 by type: " + "  ".join(f"{t} {np.mean(rk[ty == t] < 2):.2f}" for t in sorted(set(ty))),
              flush=True)
    if args.src == "rate_sweep.json":
        sp = K.OUT / "select_scores.json"
        allsel = json.loads(sp.read_text())
        allsel[f"pick{args.top}"] = sel["pick"]
        allsel[f"pick{args.top}+s1"] = sel["pick+stage1"]
        tmp = sp.with_suffix(".tmp")
        tmp.write_text(json.dumps(allsel))
        tmp.replace(sp)


if __name__ == "__main__":
    main()
