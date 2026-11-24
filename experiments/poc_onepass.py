"""H38: the per-fact rating in one forward pass per request instead of one per fact.

The request comes first, then every fact in turn, each followed by the rating question and "Answer:"; the decoder's
log P(Yes) - log P(No) is read at each "Answer:" in a single pass (earlier answers are left blank). Each fact's rating
sees the request, the fact and the facts before it. Run in profile order and reversed, then averaged.
Same 288 unseen-type profiles of 48 facts as tests 35c and 37. -> work/runs/poc/onepass.json, select scores "onepass*".
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import poc_broad as B  # noqa: E402
import poc_profile as PF  # noqa: E402
import poc_whole as W  # noqa: E402

T0 = time.perf_counter()
ASK = {"base": "Should a good reply to this request take it into account?",
       "change": "Does it change what a good reply should say or recommend?"}


def ratings(model, tok, tx, facts, q, ask, yes, no, torch):
    head = (f'The user asks: "{q}"\n\nBelow are facts about this user, one at a time. After each fact, answer Yes or No: {ask}\n\n')
    body = "".join(f"Fact: {f}\n{ask} Answer:\n\n" for f in facts)
    text = tx.apply(tok, [{"role": "system", "content": tx.sys}, {"role": "user", "content": head + body}], add_generation_prompt=False)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    off = enc["offset_mapping"]
    ends, pos = [], text.index(body)
    for f in facts:
        a = text.index(f"Fact: {f}\n", pos)
        b = a + len(f"Fact: {f}\n{ask} Answer:")
        pos = b
        ends.append(max(i for i, (s, e) in enumerate(off) if e <= b))
    ids = torch.tensor([enc["input_ids"]], device=model.device)
    with torch.no_grad():
        h = model.model(ids).last_hidden_state[0, ends]
        lp = torch.log_softmax(model.lm_head(h).float(), -1)
    return (torch.logsumexp(lp[:, yes], -1) - torch.logsumexp(lp[:, no], -1)).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=48)
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    K = PF.poc()
    tok = AutoTokenizer.from_pretrained(K.DECODER)
    tx = W.Text(tok)
    model = AutoModelForCausalLM.from_pretrained(K.DECODER, dtype=torch.float16, attn_implementation="sdpa", device_map="cuda",
                                                 low_cpu_mem_usage=True).eval().requires_grad_(False)
    ids = lambda ws: sorted({tok.encode(w, add_special_tokens=False)[0] for w in ws})  # noqa: E731
    yes, no = ids([" Yes", "Yes", " yes"]), ids([" No", "No", " no"])
    su, qvec, fvec, _ = B.load_all()
    ev, pmap = B.test_set(su, qvec, True)
    rows = []
    for e in ev:
        if e["group"] != "held-out types":
            continue
        lst = W.profile(e, args.size, True, "rank")
        q = pmap[e["pair"]]["question"]
        r = {"type": e["type"], "q": q, "facts": lst, "ci": lst.index(e["clue"])}
        for name, ask in ASK.items():
            fw = ratings(model, tok, tx, lst, q, ask, yes, no, torch)
            bw = ratings(model, tok, tx, lst[::-1], q, ask, yes, no, torch)[::-1]
            r[name + " fwd"], r[name + " rev"], r[name] = fw.tolist(), bw.tolist(), ((fw + bw) / 2).tolist()
        rows.append(r)
    print(f"ONEPASS {len(rows)} profiles, {2 * len(ASK)} passes each, at +{time.perf_counter() - T0:.0f}s", flush=True)
    ty = np.array([r["type"] for r in rows])
    res = {}
    for key in [k for k in rows[0] if k not in ("type", "q", "facts", "ci")]:
        rk = np.array([int((np.array(r[key]) > r[key][r["ci"]]).sum()) for r in rows])
        res[key] = {"top4": float(np.mean(rk < 4)), "top8": float(np.mean(rk < 8)), "median": float(np.median(rk))}
        print(f"ONEPASS {key:12s} clue in top 4 {np.mean(rk < 4):.2f}  top 8 {np.mean(rk < 8):.2f}  top 12 {np.mean(rk < 12):.2f}  median {np.median(rk):.0f}"
              "  top 8 by type: " + "  ".join(f"{t} {np.mean(rk[ty == t] < 8):.2f}" for t in sorted(set(ty))), flush=True)
        pos = np.array([r["ci"] for r in rows])
        if key.endswith("fwd"):  # does a fact's position bias its rating? correlation of rating with position, averaged over profiles
            c = np.mean([np.corrcoef(np.arange(len(r[key])), r[key])[0, 1] for r in rows])
            print(f"ONEPASS {key:12s} mean correlation of rating with position {c:+.2f} (clue position median {np.median(pos):.0f})", flush=True)
    (K.OUT / "onepass.json").write_text(json.dumps({"results": res, "rows": rows}))
    sp = K.OUT / "select_scores.json"
    sel = json.loads(sp.read_text()) if sp.exists() else {}
    for name in ASK:
        sel["onepass_" + name] = {W.row_key(r["q"], r["facts"][r["ci"]], r["facts"]): dict(zip(r["facts"], r[name])) for r in rows}
    tmp = sp.with_suffix(".tmp")
    tmp.write_text(json.dumps(sel))
    tmp.replace(sp)  # atomic: another job may be reading it


if __name__ == "__main__":
    main()
