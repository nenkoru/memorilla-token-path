"""Does the vLLM server apply our LoRA the way training saw it?

On training examples (distill_*.json) prints, side by side: the teacher's answer (training target), the plain decoder's
answer and the adapter's answer, all from the server at MPORT_VLLM_URL. If the adapter answer is no closer to the
teacher than the plain one on data it was trained on, the adapter is not being applied as trained.
Optional --hf: also the training loss of the adapter on those examples in HF (needs a free GPU).

  MPORT_VLLM_URL=http://127.0.0.1:8137 python experiments/lora_check.py --tag d_adamw --n 6
"""
import argparse
import difflib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import poc_profile as PF  # noqa: E402
import poc_relevance as R  # noqa: E402
import poc_whole as W  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=120)
    args = ap.parse_args()
    K = PF.poc()
    rows = json.loads((K.OUT / "distill_0.json").read_text())["rows"]
    rows = [r for r in rows if r["fact"]][: args.n]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(K.DECODER)
    tx = W.Text(tok)
    msgs = [tx.msgs(r["facts"], r["request"]) for r in rows]
    os.environ.pop("MPORT_VLLM_LORA", None)
    plain = R.generate(None, tok, msgs, 0, args.max_new_tokens, "plain")
    os.environ["MPORT_VLLM_LORA"] = args.tag
    lora = R.generate(None, tok, msgs, 0, args.max_new_tokens, args.tag)
    sim = lambda a, b: difflib.SequenceMatcher(None, a[:400], b[:400]).ratio()  # noqa: E731
    sp, sl = [], []
    for r, p, l in zip(rows, plain, lora):
        t = r["answer"]
        sp.append(sim(p, t))
        sl.append(sim(l, t))
        print(f"\n### {len(r['facts'])} facts | fact: {r['fact'][:90]}\nrequest: {r['request'][:120]}")
        print(f"TEACHER: {t[:240]!r}\nPLAIN   ({sp[-1]:.2f}): {p[:240]!r}\nADAPTER ({sl[-1]:.2f}): {l[:240]!r}")
    print(f"\nLORA_CHECK similarity to the teacher (first 400 chars): plain {sum(sp) / len(sp):.2f}  adapter {sum(sl) / len(sl):.2f}")


if __name__ == "__main__":
    main()
