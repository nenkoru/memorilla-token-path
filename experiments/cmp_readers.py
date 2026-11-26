"""Readers compared on the questions they share (held-out types), per condition: acts, and the paired difference to the
first file with its standard error (the same questions, so the difference is far tighter than two separate rates).

    cmp_readers.py name=file[+file2] name=file ...     files under work/runs/poc; + merges files of one reader
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import poc_profile as PF  # noqa: E402


def load(out, spec):
    rows = {}
    for f in spec.split("+"):
        for r in json.loads((out / f).read_text())["rows"]:
            if r["group"] == "held-out types":
                rows[(r["condition"], r["pair"])] = r
    return rows


def main():
    out = PF.poc().OUT
    readers = [(a.split("=")[0], load(out, a.split("=")[1])) for a in sys.argv[1:]]
    conds = list(dict.fromkeys(c for _, rows in readers for c, _ in rows))
    base_name, base = readers[0]
    print(f"CMP paired differences are to {base_name}; +- is one standard error", flush=True)
    for c in conds:
        line = []
        have = [rows for _, rows in readers if any(cc == c for cc, _ in rows)]
        pairs = sorted(set.intersection(*[{p for cc, p in rows if cc == c} for rows in have]))  # questions every reader answered
        for name, rows in readers:
            if rows not in have or (c, pairs[0]) not in base:
                continue
            a = np.array([rows[(c, p)]["acts_strict"] for p in pairs], dtype=float)
            b = np.array([base[(c, p)]["acts_strict"] for p in pairs], dtype=float)
            shown = np.array([rows[(c, p)]["clue_shown"] for p in pairs], dtype=bool)
            when = f" (shown {shown.mean():.2f}, acts when shown {a[shown].mean():.2f})" if 0 < shown.mean() < 1 else ""
            d = a - b
            diff = "" if rows is base else f"  {d.mean():+.2f} +- {d.std(ddof=1) / len(d) ** 0.5:.2f}"
            line.append(f"{name} {a.mean():.2f}{diff}{when}")
        print(f"CMP {c.replace('tokLr27w_full_change ', '')} (n={len(pairs)}): " + " | ".join(line), flush=True)


if __name__ == "__main__":
    main()
