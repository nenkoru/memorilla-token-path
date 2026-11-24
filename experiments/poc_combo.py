"""H37b: combine the selectors on the same 48-fact unseen-type profiles (no model; reads saved scores).
attention per token (attn_feats.npz, test 37), per-fact ratings (rate_sweep.json, test 35c), embedding cosine.
Each score is z-normalised within the profile, then summed. -> prints where the clue ranks."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import poc_profile as PF  # noqa: E402


def main():
    K = PF.poc()
    a = np.load(K.OUT / "attn_feats.npz")
    sw = json.loads((K.OUT / "rate_sweep.json").read_text())["rows"]
    assert len(sw) == len(a["te_ci"]) and all(r["ci"] == c for r, c in zip(sw, a["te_ci"]))
    attn = a["te_x"].astype(np.float32).mean(2) / a["te_len"]
    prompts = [k for k in sw[0] if k not in ("type", "q", "facts", "ci", "cos")]
    z = lambda x: (np.asarray(x, dtype=np.float64) - np.mean(x)) / (np.std(x) + 1e-6)  # noqa: E731
    ty = np.array([r["type"] for r in sw])

    def report(key, fn):
        rk = np.array([int((s > s[r["ci"]]).sum()) for i, r in enumerate(sw) for s in [fn(i, r)]])
        print(f"COMBO {key:34s} clue in top 4 {np.mean(rk < 4):.2f}  top 8 {np.mean(rk < 8):.2f}  top 12 {np.mean(rk < 12):.2f}  median {np.median(rk):.0f}"
              "  top 8 by type: " + "  ".join(f"{t} {np.mean(rk[ty == t] < 8):.2f}" for t in sorted(set(ty))), flush=True)

    report("attention per token", lambda i, r: attn[i])
    report("cosine", lambda i, r: np.array(r["cos"]))
    report("attention + cosine", lambda i, r: z(attn[i]) + z(r["cos"]))
    for p in prompts:
        report(f"rating {p}", lambda i, r, p=p: np.array(r[p]))
        report(f"rating {p} + attention", lambda i, r, p=p: z(r[p]) + z(attn[i]))
    report("all ratings + attention", lambda i, r: sum(z(r[p]) for p in prompts) / len(prompts) + z(attn[i]))
    for wa in (0.5, 2.0):
        report(f"rating base + {wa} x attention", lambda i, r, wa=wa: z(r["base"]) + wa * z(attn[i]))
    sel = {"change+attn": lambda i, r: z(r["change"]) + z(attn[i]), "change": lambda i, r: np.array(r["change"]),
           "all+attn": lambda i, r: sum(z(r[p]) for p in prompts) / len(prompts) + z(attn[i]), "attn": lambda i, r: attn[i]}
    import poc_whole as W
    out = {k: {W.row_key(r["q"], r["facts"][r["ci"]], r["facts"]): dict(zip(r["facts"], map(float, fn(i, r)))) for i, r in enumerate(sw)} for k, fn in sel.items()}
    sp = K.OUT / "select_scores.json"
    allsel = json.loads(sp.read_text()) if sp.exists() else {}
    allsel.update(out)
    tmp = sp.with_suffix(".tmp")
    tmp.write_text(json.dumps(allsel))
    tmp.replace(sp)
    print(f"COMBO wrote select_scores.json: {', '.join(out)}", flush=True)


if __name__ == "__main__":
    main()
