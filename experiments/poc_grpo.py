"""Test 44: GRPO on the reader (on-policy RL, no labels, no teacher). The reward is the benchmark's own regex rubric.

Test 43 found the reader is a bottleneck of its own: with the clue first it acts on it only ~78% of the time, and wrong
picks next to it mislead it. Every earlier reader was trained by imitation (teacher answers). Here the reader LoRA
(start: lora_lsort.pt) samples G answers per context from 16-token memory blocks and is pushed toward the ones the rubric
scores well, relative to the group mean (Dr. GRPO: advantage = reward - group mean, no std division, no KL, one update
per batch, so the policy ratio is 1 and needs no clipping).

Data: the 120 trained-type clue pairs (split "train", 8 types) only; their rubrics are in synth.json. The held-out types
(health, no_car, toddler, flying) and the val pairs are never seen. Each context: k in {1, 2, 4, 8} memory blocks; with
p 0.75 the clue is one of them at a random place (an imperfect rater), the rest are other users' facts of other types
(half picked by cosine to the request, the look-alikes, half random). Reward: clue shown -> acts on it (strict rubric);
clue not shown -> does not act as if it were there. Scored on the answer after the "Relevant fact:" line, as poc_tok.

    poc_grpo.py dry                         no model: builds contexts, prints one, checks the rubric reward
    poc_grpo.py train --minutes 90          -> work/runs/agentmem/lora_grpo.pt (and lora_grpo_s<step>.pt every --save)
Then evaluate exactly like test 43:
    poc_tok.py answer --lora lora_grpo.pt --tables tokL90q8_change --conds ... --stride 3 --out grpo_answer.json
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

T0 = time.perf_counter()
KS, KW = (1, 2, 4, 8), (1, 2, 2, 1)


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def train_entries(su, qvec, fvec):
    """Trained-type pairs with their other-user fact pool (same filter as poc_relevance.eval_sets)."""
    s, cfg, facts, fv, info, clue_of, emap = su
    out = []
    for p in s["pairs"]:
        if p["split"] != "train":
            continue
        _, text, _ = clue_of(p)
        if text not in fvec or p["question"] not in qvec:
            continue
        doc = s["docs"][p["clue"]]["doc"]
        pool = [f["text"] for f in facts if f["doc"] != doc and f["type"] != p["type"] and not f["doc"].startswith("inv:") and f["text"] in fvec]
        sim = unit(np.stack([fvec[t] for t in pool])) @ unit(qvec[p["question"]])
        out.append({"pair": p["pair"], "type": p["type"], "value": p["value"], "question": p["question"], "clue": text,
                    "rubrics": p["rubrics"], "pool": pool, "near": [pool[i] for i in np.argsort(-sim)[:12]]})
    return out


def context(rng, e, p_clue):
    k = rng.choices(KS, KW)[0]
    inc = rng.random() < p_clue
    n = k - inc
    nh = sum(rng.random() < 0.5 for _ in range(n))
    shown = rng.sample(e["near"], nh)
    shown += rng.sample([f for f in e["pool"] if f not in shown], n - nh)
    rng.shuffle(shown)
    if inc:
        shown.insert(rng.randrange(k), e["clue"])
    return shown, inc


def reward(K, answer, e, inc):
    note = ""
    if answer.startswith("Relevant fact:"):
        note, _, answer = answer.partition("\n")
    acts = K.strict_score(answer.strip(), e["rubrics"], e["value"])["acts_strict"]
    return float(acts if inc else not acts), bool(acts)


def cmd_dry(args):
    K = PF.poc()
    su, qvec, fvec, _ = B.load_all()
    ents = train_entries(su, qvec, fvec)
    print(f"GRPO dry: {len(ents)} trained-type pairs; by type " + ", ".join(f"{t} {sum(e['type'] == t for e in ents)}" for t in sorted({e['type'] for e in ents})))
    rng = random.Random(0)
    e = ents[0]
    shown, inc = context(rng, e, args.p_clue)
    print(f"question: {e['question']}\nclue: {e['clue']}\nclue shown {inc}, {len(shown)} facts: {shown}\nnear: {e['near'][:3]}")
    print(f"rubrics: {e['rubrics']}")
    for a in [f"Relevant fact: {e['clue']}\nHere is a {e['value']} plan.", "Relevant fact: none\nAnything works.", "Make a steak."]:
        print(f"reward clue shown {reward(K, a, e, True)}  not shown {reward(K, a, e, False)}  <- {a!r}")


def cmd_train(args):
    import torch
    import poc_agentmem as A
    import poc_core as C
    import poc_fast as FB
    from poc_agentmem import P, OUT
    K = PF.poc()
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    su, qvec, fvec, _ = B.load_all()
    ents = train_entries(su, qvec, fvec)
    st = A.Stack("lora", train=True)
    C.load_lora(st, args.init, eval_mode=True)  # no dropout: the log-probs must be the sampling policy's
    model, tok = st.r.model, st.r.tok
    model.eval()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})  # fits the 15 GB cap
    core = st.core.float().eval()
    dt = next(model.parameters()).dtype
    lps = [t for m in st.lora for t in (m.A, m.B)]
    opt = torch.optim.AdamW(lps, lr=args.lr, weight_decay=0.0)
    gc = model.generation_config
    eos = set(gc.eos_token_id if isinstance(gc.eos_token_id, list) else [gc.eos_token_id]) | {tok.pad_token_id}
    emb = model.get_input_embeddings()
    print(f"GRPO train: {len(ents)} pairs, init {args.init}, {args.prompts} contexts x {args.group} samples per step, "
          f"lr {args.lr}, max_new {args.max_new}, {args.minutes} min", flush=True)
    t0, step, hist = time.perf_counter(), 0, []
    while time.perf_counter() - t0 < args.minutes * 60:
        ts = time.perf_counter()
        ctx = []
        with torch.no_grad():
            for _ in range(args.prompts):
                e = rng.choice(ents)
                shown, inc = context(rng, e, args.p_clue)
                mem = PF.blocks(C, core, np.stack([fvec[f] for f in shown]), qvec[e["question"]], 16, P, torch).to(dt)
                ctx.append((e, inc, mem))
            items = [(e["question"], mem) for e, _, mem in ctx for _ in range(args.group)]
            x, mask, _ = FB.pad_batch(st, [(q, m, None) for q, m in items])
            gen = model.generate(inputs_embeds=x, attention_mask=mask, max_new_tokens=args.max_new, do_sample=True,
                                 temperature=args.temp, top_p=1.0, top_k=0, pad_token_id=tok.pad_token_id)
        tg = time.perf_counter() - ts
        # completion mask: tokens up to and including the first end token
        g = gen.clone()
        endm = torch.zeros_like(g, dtype=torch.bool)
        for t in eos:
            endm |= g == t
        first = torch.where(endm.any(1), endm.float().argmax(1), torch.full_like(g[:, 0], g.shape[1]))
        cm = (torch.arange(g.shape[1], device=g.device)[None] <= first[:, None]).float()
        texts = [tok.decode(o, skip_special_tokens=True).strip() for o in g]
        R, acts = [], []
        for i, a in enumerate(texts):
            e, inc, _ = ctx[i // args.group]
            r, ac = reward(K, a, e, inc)
            R.append(r)
            acts.append(ac)
        R = torch.tensor(R).view(args.prompts, args.group)
        adv = (R - R.mean(1, keepdim=True)).view(-1)
        live = adv.abs() > 0
        # policy gradient on the sequences with a non-zero advantage, in micro-batches
        idx = live.nonzero().flatten().tolist()
        norm = len(items) * args.max_new
        del gen
        torch.mps.empty_cache()  # drop the generation KV cache before the backward passes
        model.train()  # checkpointing is active only in train mode; there is no dropout left (LoRA p = 0, Qwen3 none)
        for b in range(0, len(idx), args.micro):
            j = idx[b:b + args.micro]
            jt = torch.tensor(j, device=x.device)
            full = torch.cat([x[jt], emb(g[jt].masked_fill(cm[jt] == 0, tok.pad_token_id)).to(x.dtype)], 1)
            fm = torch.cat([mask[jt], cm[jt].long()], 1)
            h = model.model(inputs_embeds=full, attention_mask=fm, use_cache=False).last_hidden_state[:, x.shape[1] - 1:-1]
            logits = model.lm_head(h).float()  # completion positions only
            if args.temp != 1.0:
                logits = logits / args.temp
            lp = logits.gather(-1, g[jt][..., None]).squeeze(-1) - torch.logsumexp(logits, -1)  # no [b, t, vocab] copy
            loss = -((lp * cm[jt]).sum(1) * adv[j].to(lp.device)).sum() / norm
            loss.backward()
            del logits, lp, loss, full, h
        model.eval()
        torch.mps.empty_cache()
        if idx:
            torch.nn.utils.clip_grad_norm_(lps, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        step += 1
        inc_m = torch.tensor([c[1] for c in ctx]).repeat_interleave(args.group)
        Rf = R.view(-1)
        hist.append((float(Rf[inc_m].mean()) if inc_m.any() else float("nan"), float(Rf[~inc_m].mean()) if (~inc_m).any() else float("nan")))
        print(f"GRPO step {step}  reward clue shown {hist[-1][0]:.2f}  not shown {hist[-1][1]:.2f}  live groups "
              f"{int((R.std(1) > 0).sum())}/{args.prompts}  mean length {float(cm.sum(1).mean()):.0f}  gen {tg:.0f}s  "
              f"step {time.perf_counter() - ts:.0f}s  +{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
        if step == 1 or step % 10 == 0:
            print(f"GRPO sample: {texts[0][:300]!r}", flush=True)
        if step % args.save == 0:
            torch.save(st.state(), OUT / f"lora_{args.tag}_s{step}.pt")
            h = np.array(hist[-args.save:])
            print(f"GRPO last {args.save} steps: clue shown {np.nanmean(h[:, 0]):.2f}  not shown {np.nanmean(h[:, 1]):.2f}", flush=True)
    torch.save(st.state(), OUT / f"lora_{args.tag}.pt")
    print(f"GRPO saved lora_{args.tag}.pt after {step} steps, +{(time.perf_counter() - T0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dry")
    t = sub.add_parser("train")
    for a in (d, t):
        a.add_argument("--p-clue", type=float, default=0.75)
    t.add_argument("--init", default="lora_lsort.pt")
    t.add_argument("--tag", default="grpo")
    t.add_argument("--minutes", type=float, default=90)
    t.add_argument("--prompts", type=int, default=8)
    t.add_argument("--group", type=int, default=4)
    t.add_argument("--max-new", type=int, default=200)
    t.add_argument("--temp", type=float, default=1.0)
    t.add_argument("--lr", type=float, default=2e-5)
    t.add_argument("--micro", type=int, default=1)
    t.add_argument("--save", type=int, default=10)
    t.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"dry": cmd_dry, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
