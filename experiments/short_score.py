"""Would shorter answers give the same scores? Saved answers are cut to their first N generated tokens and scored again
(CPU only). A decoding step costs the same whatever the batch, so an answer run's time is proportional to the token limit.

    short_score.py name=file[+file2] ... [--tokens 100,150,200,300]
Per reader and condition (held-out types): acts at each limit, and the share of answers whose score equals the full one.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import poc_broad as B  # noqa: E402
import poc_profile as PF  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("readers", nargs="+")
    ap.add_argument("--tokens", type=lambda x: [int(v) for v in x.split(",")], default=[100, 150, 200, 300])
    args = ap.parse_args()
    from transformers import AutoTokenizer
    from poc_clue import DECODER
    K = PF.poc()
    tok = AutoTokenizer.from_pretrained(DECODER)
    su, qvec, _, _ = B.load_all()
    _, pmap = B.test_set(su, qvec, True)
    print(f"SHORT acts at the first N generated tokens {args.tokens} (the recite line counts), then the share of answers scored as in full", flush=True)
    for spec in args.readers:
        name, files = spec.split("=")
        rows = [r for f in files.split("+") for r in json.loads((K.OUT / f).read_text())["rows"] if r["group"] == "held-out types"]
        ids = [tok(((r.get("note") or "") + "\n" if r.get("note") else "") + r["answer"], add_special_tokens=False)["input_ids"] for r in rows]
        print(f"SHORT {name}: {len(rows)} answers, tokens median {int(np.median([len(i) for i in ids]))}, "
              f"at 290 or more {np.mean([len(i) >= 290 for i in ids]):.2f}", flush=True)
        acts = {}
        for n in args.tokens:
            got = []
            for r, i in zip(rows, ids):
                text = tok.decode(i[:n])
                if text.startswith("Relevant fact:"):
                    text = text.partition("\n")[2]
                p = pmap[r["pair"]]
                got.append(K.strict_score(text.strip(), p["rubrics"], p["value"])["acts_strict"])
            acts[n] = np.array(got)
        full = np.array([r["acts_strict"] for r in rows])
        for c in dict.fromkeys(r["condition"] for r in rows):
            m = np.array([r["condition"] == c for r in rows])
            print(f"SHORT {name} | {c.replace('tokLr27w_full_change ', ''):28s} n={int(m.sum()):3d} saved {full[m].mean():.2f} | "
                  + "  ".join(f"{n}: {acts[n][m].mean():.2f} (same {np.mean(acts[n][m] == full[m]):.2f})" for n in args.tokens), flush=True)


if __name__ == "__main__":
    main()
