"""Rename tables in select_scores.json: python -I rename_table.py old=new [old=new ...]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import poc_profile as PF  # noqa: E402
import poc_tok as T  # noqa: E402

K = PF.poc()
full = T.load_tables(K)
for x in sys.argv[1:]:
    a, b = x.split("=")
    full[b] = full.pop(a)
    print(f"RENAME {a} -> {b} ({len(full[b])} pairs)", flush=True)
p = K.OUT / "select_scores.json"
tmp = p.with_suffix(".tmp")
import json  # noqa: E402
tmp.write_text(json.dumps(full))
tmp.replace(p)
