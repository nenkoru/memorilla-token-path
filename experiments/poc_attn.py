"""H37: the decoder's own attention as the selector. One forward pass over the whole profile and the request;
for each fact, the attention mass that the request tokens put on it, per layer and head (36 x 32 features per fact).

Scores compared on unseen types (where the clue ranks among 48 facts):
  - mean over all heads; each layer's mean, the layer picked on training profiles;
  - a weighting of heads learned on training-type profiles (a known relevant fact among 47 random ones), listwise softmax.
Nothing about the held-out types is used for picking or fitting.
  --train 400 -> work/runs/poc/attn.json
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import poc_broad as B  # noqa: E402
import poc_profile as PF  # noqa: E402
import poc_rater as RT  # noqa: E402
import poc_whole as W  # noqa: E402

T0 = time.perf_counter()
SUFFIX = {"": "", "which": "\n\nBefore answering: which of the facts above should a good reply take into account?",
          "change": "\n\nBefore answering: which facts above change what a good reply should say or recommend?"}


def features(model, tok, tx, facts, q, torch, suffix=""):
    """(n_facts, layers * heads) attention mass from the tokens after the last fact to each fact's tokens."""
    text = tx.apply(tok, tx.msgs(facts, q + suffix), add_generation_prompt=True)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    off = enc["offset_mapping"]
    spans, pos = [], text.index("<|im_start|>user")
    for f in facts:
        a = text.index(f, pos)
        b = a + len(f)
        pos = b
        idx = [i for i, (s, e) in enumerate(off) if s < b and e > a]
        spans.append((idx[0], idx[-1] + 1))
    qstart = spans[-1][1]
    if suffix:  # queries: only the relevance question and the template tokens after it
        sa = text.index(suffix.strip(), pos)
        qstart = next(i for i, (s0, e0) in enumerate(off) if e0 > sa)
    ids = torch.tensor([enc["input_ids"]], device=model.device)
    with torch.no_grad():
        out = model(ids, output_attentions=True, use_cache=False)
    feats = []
    for att in out.attentions:  # (1, H, T, T)
        a = att[0, :, qstart:, :].float().mean(1)  # (H, T): mean over query tokens
        feats.append(torch.stack([a[:, s:e].sum(-1) for s, e in spans], 0))  # (n, H)
    return torch.cat(feats, 1).cpu().numpy(), np.array([e - s for s, e in spans])


def rank(s, ci):
    return int((s > s[ci]).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=400)
    ap.add_argument("--size", type=int, default=48)
    ap.add_argument("--suffix", default="", choices=list(SUFFIX))
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    K = PF.poc()
    tok = AutoTokenizer.from_pretrained(K.DECODER)
    tx = W.Text(tok)
    model = AutoModelForCausalLM.from_pretrained(K.DECODER, dtype=torch.float16, attn_implementation="eager", device_map="cuda",
                                                 low_cpu_mem_usage=True).eval().requires_grad_(False)
    su, qvec, fvec, pool, rel = RT.training_space(K)
    rng = random.Random(0)
    tr = []
    for q in rng.sample(sorted(rel), args.train):
        good = rng.choice(sorted(rel[q]))
        others = [f for f in rng.sample(pool, args.size + 5) if f not in rel[q]][: args.size - 1]
        lst = others + [good]
        rng.shuffle(lst)
        x, ln = features(model, tok, tx, lst, q, torch, SUFFIX[args.suffix])
        tr.append({"x": x, "len": ln, "ci": lst.index(good)})
    print(f"ATTN training profiles: {len(tr)} at +{time.perf_counter() - T0:.0f}s", flush=True)
    ev, pmap = B.test_set(su, qvec, True)
    te = []
    for e in ev:
        if e["group"] != "held-out types":
            continue
        lst = W.profile(e, args.size, True, "rank")
        x, ln = features(model, tok, tx, lst, pmap[e["pair"]]["question"], torch, SUFFIX[args.suffix])
        te.append({"x": x, "len": ln, "ci": lst.index(e["clue"]), "type": e["type"]})
    print(f"ATTN eval profiles: {len(te)} at +{time.perf_counter() - T0:.0f}s", flush=True)
    nl = model.config.num_hidden_layers
    nh = tr[0]["x"].shape[1] // nl

    res = {}

    def report(key, fn):
        rk = np.array([rank(fn(r), r["ci"]) for r in te])
        trk = np.array([rank(fn(r), r["ci"]) for r in tr])
        ty = np.array([r["type"] for r in te])
        o = {"train_top4": float(np.mean(trk < 4)), "top4": float(np.mean(rk < 4)), "top8": float(np.mean(rk < 8)), "median": float(np.median(rk)),
             "top8_by_type": {t: float(np.mean(rk[ty == t] < 8)) for t in sorted(set(ty))}}
        res[key] = o
        print(f"ATTN {key:34s} training types top 4 {o['train_top4']:.2f} | unseen: top 4 {o['top4']:.2f}  top 8 {o['top8']:.2f}  median {o['median']:.0f}"
              "  top 8 by type: " + "  ".join(f"{t} {v:.2f}" for t, v in o["top8_by_type"].items()), flush=True)

    report("random order (chance)", lambda r: np.random.default_rng(len(r["x"]) + r["ci"]).random(len(r["x"])))
    report("fact length", lambda r: r["len"].astype(float))
    report("mean of all heads", lambda r: r["x"].mean(1))
    report("mean of all heads per token", lambda r: r["x"].mean(1) / r["len"])
    lay = [np.mean([rank(r["x"][:, i * nh:(i + 1) * nh].mean(1), r["ci"]) < 4 for r in tr]) for i in range(nl)]
    bl = int(np.argmax(lay))
    report(f"layer {bl} (best on training)", lambda r: r["x"][:, bl * nh:(bl + 1) * nh].mean(1))
    hd = [np.mean([rank(r["x"][:, j], r["ci"]) < 4 for r in tr]) for j in range(nl * nh)]
    bh = int(np.argmax(hd))
    report(f"head {bh // nh}.{bh % nh} (best on training)", lambda r: r["x"][:, bh])

    # learned head weights: listwise softmax over log attention mass, 10% of training profiles for early stopping
    X = torch.tensor(np.log(np.stack([r["x"] for r in tr]) + 1e-6), device="cuda")
    Y = torch.tensor([r["ci"] for r in tr], device="cuda")
    nv = len(tr) // 10
    w = torch.zeros(X.shape[2], device="cuda", requires_grad=True)
    opt = torch.optim.AdamW([w], lr=1e-2, weight_decay=0.0)
    best = (1e9, None)
    for step in range(2000):
        logit = X @ w
        lt = torch.nn.functional.cross_entropy(logit[nv:], Y[nv:]) + 1e-3 * w.abs().sum()
        opt.zero_grad()
        lt.backward()
        opt.step()
        with torch.no_grad():
            lv = torch.nn.functional.cross_entropy((X[:nv] @ w), Y[:nv]).item()
        if lv < best[0]:
            best = (lv, w.detach().clone(), step)
    wb = best[1].cpu().numpy()
    print(f"ATTN head weights: best val {best[0]:.3f} at step {best[2]}; heads with |w| > 0.05: {int((np.abs(wb) > 0.05).sum())}", flush=True)
    report("learned head weights", lambda r: np.log(r["x"] + 1e-6) @ wb)
    top = np.argsort(-np.abs(wb))[:10]
    res["top_heads"] = [(int(j // nh), int(j % nh), float(wb[j])) for j in top]
    print("ATTN top heads: " + "  ".join(f"{a}.{b} {c:+.2f}" for a, b, c in res["top_heads"]), flush=True)
    np.savez_compressed(K.OUT / f"attn_feats{args.tag}.npz", tr_x=np.stack([r["x"] for r in tr]).astype(np.float16), tr_ci=Y.cpu().numpy(),
                        te_x=np.stack([r["x"] for r in te]).astype(np.float16), te_ci=np.array([r["ci"] for r in te]),
                        te_type=np.array([r["type"] for r in te]), te_len=np.stack([r["len"] for r in te]), tr_len=np.stack([r["len"] for r in tr]))
    (K.OUT / f"attn{args.tag}.json").write_text(json.dumps({"args": vars(args), "results": res}, indent=1))


if __name__ == "__main__":
    main()
