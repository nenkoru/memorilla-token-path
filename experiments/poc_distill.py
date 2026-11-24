"""H33: self-distillation against distraction. No picker, no hand-written rules.

Teacher: the plain decoder that sees only the fact that matters (or no fact) and answers (it reaches 0.80 on unseen
types when it gets the clue alone, 0.53 / 0.42 / 0.35 when the clue is buried among 8 / 16 / 48 facts).
Student: the same decoder with a LoRA, reading the whole profile (4-48 facts, the fact buried among others, a few of
them close to the request), trained to give the teacher's answer. Profiles without the fact are trained toward the
teacher's no-fact answer, so the student does not learn to lean on whatever fact is there.

  teach:  --mode teach --shard 0 --shards 2 --n 8000     (MPORT_VLLM_URL set: teacher answers from a vLLM server)
          -> work/runs/poc/distill_<shard>.json
  weigh:  --mode weigh --shard 0   (MPORT_VLLM_URL set) -> pmi_<shard>.json, per-token log-probs under 3 contexts (H34)
  train:  --mode train [--weight pmi|gap] --opt muon|adamw --minutes 25 --ckpt --budget 12000 --tag d_muon
          (data parallel when started with experiments/v100.sh ddp) -> lora_<tag>.pt and a PEFT copy for vLLM (peft_<tag>/)
  Evaluation: poc_whole.py --mode zero with MPORT_VLLM_LORA=<tag> against a vLLM server started with that adapter.
  --dry checks sampling and token counts without a model.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

import fast_train as FT  # noqa: E402
import poc_broad as B  # noqa: E402
import poc_profile as PF  # noqa: E402
import poc_relevance as R  # noqa: E402
import poc_whole as W  # noqa: E402

NAMES = ("q_proj", "k_proj", "v_proj", "o_proj")
T0 = time.perf_counter()


def minutes():
    return (time.perf_counter() - T0) / 60


class Picker(W.Sampler):
    def pick(self, n=None):
        """-> (profile facts, request, the fact that matters or None when it is left out)"""
        rng, a, torch = self.rng, self.args, self.torch
        if rng.random() < a.p_any:  # any training fact with any training request: no topical pairing at all (H35)
            request, fact = rng.choice(self.requests), rng.choice(self.pool)
        elif rng.random() < a.p_full:
            request, fact, _, _ = rng.choice(self.full)
        else:
            request, fact = rng.choice(self.pairs)
        n = n or rng.choice(W.TRAIN_SIZES)
        none = rng.random() < a.p_none
        fi = self.pidx[fact]
        copy = (self.PV @ self.PV[fi]) >= a.dup
        idx = []
        q = torch.from_numpy(B.unit(self.qvec[request])).to(self.dev)
        near = [i for i in (self.PV @ q).topk(16).indices.tolist() if not bool(copy[i])]
        idx = rng.sample(near, min(len(near), rng.choice([0, 1, 2, 3]), n - 1))
        want = n - (0 if none else 1)
        while len(idx) < want:
            cand = rng.sample(range(len(self.pool)), 2 * (want - len(idx)) + 4)
            ok = (~copy[torch.tensor(cand, device=self.dev)]).tolist()
            idx += [c for c, good in zip(cand, ok) if good and c not in idx][: want - len(idx)]
        lst = [self.pool[i] for i in idx] + ([] if none else [fact])
        rng.shuffle(lst)
        return lst, request, None if none else fact


def teach(args, K, torch):
    sm = Picker(args, K, torch, "cpu")
    sm.requests = sorted({q for q, _ in sm.pairs} | {q for q, _, _, _ in sm.full})
    sm.rng = random.Random(1000 + args.shard)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(K.DECODER)
    tx = W.Text(tok)
    items = [sm.pick() for _ in range(args.n // args.shards)]
    print(f"DISTILL teach shard {args.shard}: {len(items)} examples, {sum(f is None for _, _, f in items)} without the fact", flush=True)
    if args.dry:
        for lst, req, f in items[:3]:
            print(len(lst), "|", req, "|", f)
        return
    out = []
    step = 1000
    for b in range(0, len(items), step):
        chunk = items[b:b + step]
        ans = R.generate(None, tok, [tx.msgs([f] if f else [], req) for _, req, f in chunk], 0, args.max_new_tokens, f"DISTILL teacher {b}")
        out += [{"facts": lst, "request": req, "fact": f, "answer": a} for (lst, req, f), a in zip(chunk, ans)]
        (K.OUT / f"{args.data}_{args.shard}.json").write_text(json.dumps({"rows": out}))
    print(f"DISTILL teach done: {len(out)} rows at +{minutes():.1f} min", flush=True)


def score(texts_ids, url):
    """Log-probability of every prompt token (None for the first) from a vLLM server, prompts given as token ids."""
    import urllib.request
    out = []
    for b in range(0, len(texts_ids), 400):
        chunk = texts_ids[b:b + 400]
        body = json.dumps({"model": "qwen3-8b", "prompt": chunk, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0}).encode()
        req = urllib.request.Request(url + "/v1/completions", body, {"Content-Type": "application/json"})
        res = json.loads(urllib.request.urlopen(req, timeout=3600).read())["choices"]
        res.sort(key=lambda c: c["index"])
        for ids, c in zip(chunk, res):
            lp = c["prompt_logprobs"]
            out.append([None if d is None else d[str(t)]["logprob"] for t, d in zip(ids, lp)])
    return out


def score_file(K, stem, shard):
    return K.OUT / (f"pmi_{shard}.json" if stem == "distill" else f"{stem}_pmi_{shard}.json")


def weigh(args, K, torch):
    """H34 step 1: score each teacher answer token under three contexts with the plain decoder:
    lf = the teacher's own context (the fact alone, or nothing for rows without it), ln = no fact, lw = the whole profile.
    lf - ln: how much the fact raises the token (what the fact is for); lf - lw: what the whole profile breaks.
    Answers cut off at --max-new-tokens get no end token. -> work/runs/poc/pmi_<shard>.json, one entry per row."""
    import os
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(K.DECODER)
    tx = W.Text(tok)
    rows = json.loads((K.OUT / f"{args.data}_{args.shard}.json").read_text())["rows"]
    url = os.environ["MPORT_VLLM_URL"]
    t0 = time.perf_counter()
    out = []
    step = 1000
    for b in range(0, len(rows), step):
        chunk = rows[b:b + step]
        jobs, spans = [], []
        for r in chunk:
            t = tok.encode(r["answer"], add_special_tokens=False)
            if len(t) < args.max_new_tokens - 2:
                t = tok.encode(r["answer"] + "<|im_end|>", add_special_tokens=False)
            ctxs = [[r["fact"]] if r["fact"] else [], [], r["facts"]]
            spans.append((t, []))
            for c in ctxs:
                p = tx.encode(c, r["request"], "", False)[0]
                jobs.append(p + t)
                spans[-1][1].append(len(p))
        lps = score(jobs, url)
        for i, (t, starts) in enumerate(spans):
            f, n, w = (lps[3 * i + j][s:s + len(t)] for j, s in enumerate(starts))
            out.append({"t": t, "lf": [round(x, 3) for x in f], "ln": [round(x, 3) for x in n], "lw": [round(x, 3) for x in w]})
        print(f"DISTILL weigh shard {args.shard}: {len(out)} rows at {time.perf_counter() - t0:.0f}s", flush=True)
    score_file(K, args.data, args.shard).write_text(json.dumps(out))
    fr = [o for o, r in zip(out, rows) if r["fact"]]
    pm = np.array([sum(max(0.0, a - b) for a, b in zip(o["lf"], o["ln"])) for o in fr])
    gp = np.array([sum(max(0.0, a - b) for a, b in zip(o["lf"], o["lw"])) for o in out])
    print(f"DISTILL weigh done: {len(out)} rows; per answer, sum of positive (fact - none) {np.mean(pm):.1f} nats"
          f" (median {np.median(pm):.1f}), sum of positive (teacher - whole) {np.mean(gp):.1f} nats; tokens {np.mean([len(o['t']) for o in out]):.0f}",
          flush=True)


def token_weights(r, kind, base, cap):
    if kind == "none" or "lf" not in r:
        return None
    other = r["ln"] if kind == "pmi" else r["lw"]
    return [base + min(cap, max(0.0, a - b)) for a, b in zip(r["lf"], other)]


def tensors_w(batch, pad, torch, dev):
    ids, mask, lab = W.tensors([(p, t) for p, t, _ in batch], pad, torch, dev)
    wt = torch.zeros(lab.shape, dtype=torch.float32)
    for i, (p, t, w) in enumerate(batch):
        wt[i, len(p): len(p) + len(t)] = torch.tensor(w if w is not None else [1.0] * len(t))
    return ids, mask, lab, wt.to(dev)


def loss_w(model, ids, mask, lab, wt, torch):
    """Weighted mean of the target-token loss over the whole batch (weights 1 everywhere = token mean)."""
    h = model.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state[:, :-1]
    tgt, w = lab[:, 1:], wt[:, 1:]
    sel = tgt != -100
    hs, ts, ws = h[sel], tgt[sel], w[sel]
    if not model.training:
        ce = torch.nn.functional.cross_entropy(model.lm_head(hs).float(), ts, reduction="none")
        return (ce * ws).sum() / ws.sum().clamp(min=1e-6)
    # vocabulary layer + loss in checkpointed chunks: the float32 logits (152k per token, 2-3 GB per batch) are never kept whole
    from torch.utils.checkpoint import checkpoint

    def part(hc, tc, wc):
        return (torch.nn.functional.cross_entropy(model.lm_head(hc).float(), tc, reduction="none") * wc).sum()

    tot = sum(checkpoint(part, hs[i:i + 1024], ts[i:i + 1024], ws[i:i + 1024], use_reentrant=False) for i in range(0, len(ts), 1024))
    return tot / ws.sum().clamp(min=1e-6)


def save_peft(state, out_dir, base):
    """Our LoRA (keys '<i>.A' / '<i>.B', i = layer * 4 + projection, scale alpha / r = 2) as a PEFT adapter for vLLM."""
    import torch
    from safetensors.torch import save_file
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = {}
    for key, v in state.items():
        i, k = key.split(".")
        layer, name = divmod(int(i), len(NAMES))
        sd[f"base_model.model.model.layers.{layer}.self_attn.{NAMES[name]}.lora_{k}.weight"] = v.to(torch.float16).contiguous()
    r = next(v for k, v in state.items() if k.endswith(".A")).shape[0]
    save_file(sd, str(out_dir / "adapter_model.safetensors"))
    (out_dir / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": r, "lora_alpha": 2 * r, "lora_dropout": 0.0, "bias": "none",
        "target_modules": list(NAMES), "fan_in_fan_out": False, "base_model_name_or_path": base}, indent=1))


def batches(rows, tx, budget, max_bs, rng, kind="none", base=0.1, cap=5.0):
    """Token-budget batches of similar length, shuffled; each item (prompt ids, target ids, token weights or None)."""
    enc = []
    for r in rows:
        if "t" in r:  # scored rows: the exact target ids that were scored (no end token after a cut-off answer)
            enc.append((tx.encode(r["facts"], r["request"], "", False)[0], r["t"], token_weights(r, kind, base, cap)))
        else:
            enc.append(tx.encode(r["facts"], r["request"], r["answer"], True) + (None,))
    order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]) + len(enc[i][1]) + rng.random())
    out, cur, longest = [], [], 0
    for i in order:
        n = len(enc[i][0]) + len(enc[i][1])
        if cur and ((len(cur) + 1) * max(longest, n) > budget or len(cur) >= max_bs):
            out.append(cur)
            cur, longest = [], 0
        cur.append(enc[i])
        longest = max(longest, n)
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


def train(args, K, torch):
    nw, rank = FT.setup()
    rows = []
    for p in [q for stem in args.data.split(",") for q in sorted(K.OUT.glob(f"{stem}_[0-9]*.json"))]:
        part = json.loads(p.read_text())["rows"]
        stem, shard = p.stem.rsplit("_", 1)
        sc = score_file(K, stem, shard)
        if args.weight != "none" or sc.exists():
            for r, s in zip(part, json.loads(sc.read_text()), strict=True):
                r.update(s)
        rows += part
    rows = [r for r in rows if r["answer"].strip()]
    rng = random.Random(0)
    rng.shuffle(rows)
    import poc_agentmem as A
    from poc_clue import D, P
    runner = D.Runner([], K.DECODER)
    model, tok = runner.model, runner.tok
    tx = W.Text(tok)
    bl = batches(rows, tx, args.budget, args.max_bs, rng, args.weight, args.w_base, args.w_cap)
    bl = bl[rank::nw]  # each process its own share
    if rank == 0:
        print(f"DISTILL train: {len(rows)} rows, {len(bl)} batches per process, {nw} process(es), optimizer {args.opt}, lr {args.lr},"
              f" checkpointing {args.ckpt}, budget {args.budget}", flush=True)
    lora = A.add_lora(model, 16, NAMES)
    lps = [t for m in lora for t in (m.A, m.B)]
    if not args.slow_lora:
        FT.fast_lora(lora)
    opt = FT.optimizer(lps, args.lr, args.opt)
    if args.ckpt:
        FT.checkpointing(model)
    half = next(model.parameters()).dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=half)
    model.train()
    t0, u, seen, toks, run, bi, epoch, last = time.perf_counter(), 0, 0, 0, [], 0, 0, 0.0
    limit = args.minutes * 60
    while True:
        frac = FT.total((time.perf_counter() - t0) / limit) / nw
        if frac >= 1:
            break
        for g in opt.param_groups:
            g["lr"] = args.lr * min(1.0, (u + 1) / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        if bi >= len(bl):
            bi, epoch = 0, epoch + 1
        batch = bl[bi]
        bi += 1
        oom = False
        try:
            loss = loss_w(model, *tensors_w(batch, tok.pad_token_id, torch, P.DEV), torch)
            scaler.scale(loss).backward()
            run.append(float(loss))
        except torch.OutOfMemoryError:
            oom = True
        if FT.agree(oom):
            if not oom:
                run.pop()
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"DISTILL out of memory on a batch of {len(batch)} x {max(len(p) + len(t) for p, t, _ in batch)} tokens, skipped", flush=True)
            continue
        if FT.agree(not math.isfinite(run[-1])):
            run.pop()
            opt.zero_grad(set_to_none=True)
            scaler.update()
            continue
        FT.average_grads(lps)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(lps, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        u, seen, toks = u + 1, seen + len(batch), toks + sum(len(p) + len(t) for p, t, _ in batch)
        if FT.agree(time.perf_counter() - last > 60):  # a line a minute
            last = time.perf_counter()
            el = last - t0
            s_all, t_all = FT.total(seen), FT.total(toks)
            if rank == 0:
                print(f"update {u}  epoch {epoch}  examples {s_all:.0f}  loss {np.mean(run[-20:]):.3f}  {s_all / el:.1f} examples/s  {t_all / el:.0f} tokens/s"
                      f"  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB  {el / 60:.1f} min", flush=True)
    seen, toks = FT.total(seen), FT.total(toks)
    FT.done()
    if rank != 0:
        return
    el = time.perf_counter() - t0
    state = {f"{i}.{k}": v.detach().cpu().clone() for i, m in enumerate(lora) for k, v in (("A", m.A), ("B", m.B))}
    torch.save(state, A.OUT / f"lora_{args.tag}.pt")
    save_peft(state, K.OUT / f"peft_{args.tag}", K.DECODER)
    info = {"tag": args.tag, "opt": args.opt, "lr": args.lr, "gpus": nw, "ckpt": args.ckpt, "budget": args.budget, "weight": args.weight, "w_base": args.w_base, "updates": u,
            "examples": seen, "epochs": seen / len(rows), "tokens": toks, "tokens_per_s": toks / el, "loss": float(np.mean(run[-50:])),
            "minutes": el / 60}
    (K.OUT / f"distill_train_{args.tag}.json").write_text(json.dumps(info, indent=1))
    print(f"DISTILL saved lora_{args.tag}.pt and peft_{args.tag}/  {json.dumps(info)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["teach", "weigh", "train"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--p-full", type=float, default=0.4)
    ap.add_argument("--p-any", type=float, default=0.0, help="share of examples pairing any fact with any request")
    ap.add_argument("--data", default="distill", help="data file stem(s), commas for several in train")
    ap.add_argument("--p-none", type=float, default=0.15)
    ap.add_argument("--dup", type=float, default=0.6)
    ap.add_argument("--max-new-tokens", type=int, default=260)
    ap.add_argument("--opt", default="muon", choices=["muon", "adamw"])
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--minutes", type=float, default=25)
    ap.add_argument("--ckpt", action="store_true")
    ap.add_argument("--budget", type=int, default=2400)
    ap.add_argument("--max-bs", type=int, default=64)
    ap.add_argument("--tag", default="d_muon")
    ap.add_argument("--slow-lora", action="store_true", help="the original float32 LoRA branch with dropout")
    ap.add_argument("--weight", default="none", choices=["none", "pmi", "gap"], help="token weights from --mode weigh")
    ap.add_argument("--w-base", type=float, default=0.1)
    ap.add_argument("--w-cap", type=float, default=5.0)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    import torch
    K = PF.poc()
    torch.manual_seed(0)
    return {"teach": teach, "weigh": weigh, "train": train}[args.mode](args, K, torch)


if __name__ == "__main__":
    main()
