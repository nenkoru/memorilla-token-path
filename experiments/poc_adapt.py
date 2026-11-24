"""Follow-up to POC_RESULTS.md: can a small trainable piece make the frozen Qwen3-8B use the clue the frozen core
already carries? (Step 2 showed the fact survives the core: a linear probe reads it, but the decoder never says it.)

Trainable, everything else frozen (encoder, pv4 core, Qwen3-8B weights):
    adapter   residual MLP on the core's 16 output tokens (4096 -> 256 -> 4096, zero-initialised output: ~2.1M)
    lora      rank-8 LoRA on q_proj and v_proj of all 36 Qwen3-8B layers (~3.8M), hand-written (no peft download)
    both      the two together

Subcommands (outputs in $MPORT_SCRATCH/runs/poc/):
    teach     text-teacher targets: Qwen3-8B reads the clue chat AS TEXT and answers briefly (sampled, kept only if the
              strict acts + mentions rubric passes); the twin chat (no clue) gives a negative target. Train/val types.
    train     --variant adapter|lora|both: answer loss on teacher targets, documents = clue + k random fillers
              (k in 0/4/16), plus twin-only negatives. Batch 1, grad accumulation, early stop on val loss.
    eval      --variant base|adapter|lora|both: the 40 held-out-type pairs from steps 2-3 (unseen clue types),
              plus 20 val pairs (train types, unseen chats).

RAM: one process at a time under poc_guard.sh; bf16 frozen weights loaded straight to MPS; trainable params fp32;
torch.mps.empty_cache() every few steps.
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
import torch  # noqa: E402

import poc_clue as C  # noqa: E402  (sets env defaults, path to memorilla)
from poc_clue import D, P, PERSONA_PROMPT, HEAD, DECODER, OUT, save  # noqa: E402
from memorilla.data import apply_chat_template, build_messages  # noqa: E402

N_MEM = 16
TEACH_SUFFIX = "\n\nAnswer in one or two sentences."


# ---------------------------------------------------------------- trainable pieces


class Adapter(torch.nn.Module):
    def __init__(self, dim=4096, hidden=256):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim)
        self.up = torch.nn.Linear(dim, hidden)
        self.down = torch.nn.Linear(hidden, dim)
        torch.nn.init.zeros_(self.down.weight)
        torch.nn.init.zeros_(self.down.bias)
        # memory tokens are much larger than unit norm; learn a per-channel output scale starting at the token RMS
        self.gain = torch.nn.Parameter(torch.ones(1))

    def forward(self, m):  # [B, 16, 4096] fp32
        rms = m.pow(2).mean(-1, keepdim=True).sqrt()
        return m + self.gain * rms * self.down(torch.nn.functional.gelu(self.up(self.norm(m))))


class LoRALinear(torch.nn.Module):
    def __init__(self, base: torch.nn.Linear, r=8, alpha=16, dropout=0.05):
        super().__init__()
        self.base = base
        self.A = torch.nn.Parameter(torch.randn(r, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.B = torch.nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        self.scale = alpha / r
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x):
        y = self.base(x)
        return y + ((self.drop(x).float() @ self.A.t()) @ self.B.t() * self.scale).to(y.dtype)


def add_lora(model, r=8):
    mods = []
    for layer in model.model.layers:
        for name in ("q_proj", "v_proj"):
            lin = LoRALinear(getattr(layer.self_attn, name), r=r)
            setattr(layer.self_attn, name, lin)
            mods.append(lin)
    return mods


def lora_state(mods):
    return {f"{i}.{k}": v.detach().cpu() for i, m in enumerate(mods) for k, v in (("A", m.A), ("B", m.B))}


def load_lora_state(mods, state):
    for i, m in enumerate(mods):
        m.A.data.copy_(state[f"{i}.A"])
        m.B.data.copy_(state[f"{i}.B"])


class Stack:
    """Frozen decoder + frozen core, with the optional adapter and LoRA."""

    def __init__(self, variant, train=False):
        self.r = D.Runner([HEAD], DECODER)
        self.core = self.r.heads[HEAD].eval().requires_grad_(False)
        self.variant = variant
        self.adapter = Adapter().to(P.DEV) if variant in ("adapter", "both") else None
        self.lora = add_lora(self.r.model) if variant in ("lora", "both") else []
        ck = OUT / f"adapt_{variant}.pt"
        if not train and variant != "base":
            st = torch.load(ck, map_location="cpu")
            if self.adapter is not None:
                self.adapter.load_state_dict(st["adapter"])
            if self.lora:
                load_lora_state(self.lora, st["lora"])
        if not train:
            for m in self.lora:
                m.drop.p = 0.0

    def params(self):
        ps = list(self.adapter.parameters()) if self.adapter is not None else []
        for m in self.lora:
            ps += [m.A, m.B]
        return ps

    def state(self):
        return {"adapter": self.adapter.state_dict() if self.adapter is not None else None,
                "lora": lora_state(self.lora) if self.lora else None}

    def tokens(self, d, mask, q):
        with torch.no_grad():
            m = self.core(d, mask, q)
        return self.adapter(m.float()) if self.adapter is not None else m

    def prompt_ids(self, question):
        return self.r.ids(build_messages(question, PERSONA_PROMPT, num_memories=N_MEM))

    def train_ids(self, question, answer):
        msgs = build_messages(question, PERSONA_PROMPT, num_memories=N_MEM, answer=answer)
        prompt = apply_chat_template(self.r.tok, msgs[:2], add_generation_prompt=True)
        full = apply_chat_template(self.r.tok, msgs, add_generation_prompt=False)
        p_ids = self.r.tok.encode(prompt, add_special_tokens=self.r.add_special)
        f_ids = self.r.tok.encode(full, add_special_tokens=self.r.add_special)
        assert f_ids[:len(p_ids)] == p_ids, "prompt is not a prefix of the training text"
        ids = torch.tensor([f_ids], device=P.DEV)
        labels = ids.clone()
        labels[:, :len(p_ids)] = -100
        return ids, labels


def doc_batch(docs, sets):
    n = max(len(x) for x in sets)
    d = torch.zeros(len(sets), n, docs.shape[1])
    mask = torch.ones(len(sets), n, dtype=torch.bool)
    for i, idx in enumerate(sets):
        d[i, : len(idx)] = torch.from_numpy(docs[idx])
        mask[i, : len(idx)] = False
    return d.to(P.DEV), mask.to(P.DEV)


# ---------------------------------------------------------------- teacher targets


def cmd_teach(args):
    s = C.load_synth()
    pairs = [p for p in s["pairs"] if p["split"] in ("train", "val")]
    r = D.Runner([], DECODER)
    tok = r.tok
    tok.padding_side = "left"

    def generate(contents, n, sample, seed):
        texts = [apply_chat_template(tok, [{"role": "system", "content": PERSONA_PROMPT},
                                           {"role": "user", "content": c + TEACH_SUFFIX}], add_generation_prompt=True)
                 for c in contents]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(P.DEV)
        torch.manual_seed(seed)
        kw = dict(do_sample=True, temperature=0.8, top_p=0.95) if sample else dict(do_sample=False)
        with torch.no_grad():
            out = r.model.generate(**enc, max_new_tokens=args.max_new_tokens, num_return_sequences=n,
                                   pad_token_id=tok.pad_token_id, **kw)
        L = enc["input_ids"].shape[1]
        res = [tok.decode(o[L:], skip_special_tokens=True).strip() for o in out]
        del enc, out
        torch.mps.empty_cache()
        return [res[i * n:(i + 1) * n] for i in range(len(contents))]

    rows, t0 = [], time.perf_counter()
    B = args.batch
    for b in range(0, len(pairs), B):
        chunk = pairs[b:b + B]
        clue_c = [D.user_message([s["docs"][p["clue"]]["text"]], p["question"]) for p in chunk]
        twin_c = [D.user_message([s["docs"][p["fillers"][0]]["text"]], p["question"]) for p in chunk]
        pos = generate(clue_c, args.samples, True, 1000 + b)
        neg = generate(twin_c, 1, False, 0)
        for p, cands, (twin_ans,) in zip(chunk, pos, neg):
            kept = []
            for a in cands:
                st = C.strict_score(a, p["rubrics"], p["value"])
                if st["acts_strict"] and st["mentions_strict"] and a not in kept:
                    kept.append(a)
            twin_ok = not C.strict_score(twin_ans, p["rubrics"], p["value"])["mentions_strict"]
            rows.append({"pair": p["pair"], "split": p["split"], "type": p["type"], "value": p["value"],
                         "candidates": cands, "kept": kept[: args.keep], "twin_answer": twin_ans, "twin_ok": twin_ok})
        print(f"--- {min(b + B, len(pairs))}/{len(pairs)} {time.perf_counter() - t0:.0f}s  "
              f"kept so far {sum(len(x['kept']) for x in rows)}", flush=True)
    by = {sp: {"pairs": sum(1 for x in rows if x["split"] == sp),
               "pairs_with_target": sum(1 for x in rows if x["split"] == sp and x["kept"]),
               "targets": sum(len(x["kept"]) for x in rows if x["split"] == sp),
               "twin_ok": sum(x["twin_ok"] for x in rows if x["split"] == sp)} for sp in ("train", "val")}
    print(by)
    save("teach.json", {"suffix": TEACH_SUFFIX, "samples": args.samples, "summary": by, "rows": rows})


# ---------------------------------------------------------------- training


def examples(s, teach, split, rng):
    """(question, doc indices, answer, is_positive); one fresh random filler draw per call."""
    pairs = {p["pair"]: p for p in s["pairs"]}
    out = []
    for row in teach["rows"]:
        if row["split"] != split:
            continue
        p = pairs[row["pair"]]
        others = [i for i in p["fillers"][1:]]  # fillers[0] is the twin
        for a in row["kept"]:
            k = rng.choice([0, 4, 16])
            idx = [p["clue"]] + rng.sample(others, k)
            rng.shuffle(idx)
            out.append((p["question"], idx, a, True))
        if row["twin_ok"] and row["kept"]:
            k = rng.choice([0, 4, 16])
            idx = [p["fillers"][0]] + rng.sample(others, k)
            rng.shuffle(idx)
            out.append((p["question"], idx, row["twin_answer"], False))
    return out


def loss_one(st, docs, qmap, ex):
    question, idx, answer, _ = ex
    d, mask = doc_batch(docs, [idx])
    q = torch.from_numpy(qmap[question])[None].to(P.DEV)
    ids, labels = st.train_ids(question, answer)
    embeds = P.input_embeds(st.r.model, st.r.tok, st.r.memory_id, ids, st.tokens(d, mask, q))
    return st.r.model(inputs_embeds=embeds, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False).loss


def cmd_train(args):
    torch.manual_seed(args.seed)
    s, teach = C.load_synth(), json.loads((OUT / "teach.json").read_text())
    docs, qmap = C.load_emb()
    st = Stack(args.variant, train=True)
    ps = st.params()
    print(f"variant {args.variant}: {sum(p.numel() for p in ps) / 1e6:.2f}M trainable params", flush=True)
    groups = []
    if st.adapter is not None:
        groups.append({"params": list(st.adapter.parameters()), "lr": args.lr_adapter})
    if st.lora:
        groups.append({"params": [t for m in st.lora for t in (m.A, m.B)], "lr": args.lr_lora})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    rng = random.Random(args.seed)
    val = examples(s, teach, "val", random.Random(7))[: args.n_val]
    print(f"val examples {len(val)} ({sum(e[3] for e in val)} positive)", flush=True)

    def val_loss():
        st.r.model.eval()
        with torch.no_grad():
            pos = [loss_one(st, docs, qmap, e).item() for e in val if e[3]]
            neg = [loss_one(st, docs, qmap, e).item() for e in val if not e[3]]
        torch.mps.empty_cache()
        return float(np.mean(pos)), float(np.mean(neg)) if neg else float("nan")

    log, best, step, t0 = [], None, 0, time.perf_counter()
    v = val_loss()
    log.append({"step": 0, "val_pos": v[0], "val_neg": v[1]})
    print(f"step 0  val pos {v[0]:.3f} neg {v[1]:.3f}", flush=True)
    best, bad = v[0], 0
    save_state = st.state()
    total = args.updates * args.accum
    pool = []
    while step < total:
        if not pool:
            pool = examples(s, teach, "train", rng)
            rng.shuffle(pool)
        ex = pool.pop()
        st.r.model.train()
        loss = loss_one(st, docs, qmap, ex) / args.accum
        loss.backward()
        step += 1
        if step % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(ps, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        if step % 8 == 0:
            torch.mps.empty_cache()
        if step % (args.accum * args.eval_every) == 0:
            v = val_loss()
            u = step // args.accum
            log.append({"step": u, "val_pos": v[0], "val_neg": v[1], "train_loss": loss.item() * args.accum})
            print(f"update {u}  {time.perf_counter() - t0:.0f}s  train {loss.item() * args.accum:.3f}  "
                  f"val pos {v[0]:.3f} neg {v[1]:.3f}", flush=True)
            if v[0] < best - 1e-3:
                best, bad, save_state = v[0], 0, st.state()
            else:
                bad += 1
                if bad >= args.patience:
                    print("early stop", flush=True)
                    break
    torch.save(save_state, OUT / f"adapt_{args.variant}.pt")
    save(f"adapt_train_{args.variant}.json", {"variant": args.variant, "args": vars(args), "best_val_pos": best,
                                               "seconds": time.perf_counter() - t0, "log": log})


# ---------------------------------------------------------------- evaluation


@torch.no_grad()
def answer_sets(st, question, q_vec, sets, docs, max_new):
    d, mask = doc_batch(docs, sets)
    q = torch.from_numpy(q_vec)[None].expand(len(sets), -1).to(P.DEV)
    tokens = st.tokens(d, mask, q)
    ids = st.prompt_ids(question).expand(len(sets), -1)
    embeds = P.input_embeds(st.r.model, st.r.tok, st.r.memory_id, ids, tokens)
    out = st.r.model.generate(inputs_embeds=embeds, attention_mask=torch.ones(ids.shape, device=P.DEV),
                              max_new_tokens=max_new, do_sample=False, pad_token_id=st.r.tok.pad_token_id)
    res = [st.r.tok.decode(o, skip_special_tokens=True).strip() for o in out]
    del embeds, out, tokens
    torch.mps.empty_cache()
    return res


def cmd_eval(args):
    s = C.load_synth()
    docs, qmap = C.load_emb()
    probe = json.loads((OUT / "probe.json").read_text())
    learned = probe["topm"]["doc_mlp"]
    st = Stack(args.variant)
    st.r.model.eval()
    rows, t0 = [], time.perf_counter()
    test = C.eval_pairs(s, args.per_value)
    for n, p in enumerate(test):
        L = learned[p["pair"]]
        sets = {"clue only -> core": [p["clue"]],
                "clue + 4 -> core": [p["clue"]] + p["fillers"][:4],
                "clue + 149 -> core": [p["clue"]] + p["fillers"],
                "learned top-4 -> core": L["fillers"][:4],
                "counterfactual 150 -> core": p["counterfactual"],
                "learned top-4, counterfactual -> core": L["counterfactual"][:4]}
        ans = answer_sets(st, p["question"], qmap[p["question"]], list(sets.values()), docs, args.max_new_tokens)
        for (name, idx), a in zip(sets.items(), ans):
            rows.append({"split": "test", "pair": p["pair"], "type": p["type"], "value": p["value"], "condition": name,
                         "clue_selected": p["clue"] in idx, "answer": a, **C.score(a, p["rubrics"]),
                         **C.strict_score(a, p["rubrics"], p["value"])})
        print(f"--- test {n + 1}/{len(test)} {time.perf_counter() - t0:.0f}s", flush=True)
    vals = [p for p in s["pairs"] if p["split"] == "val"]
    random.Random("adapt-val").shuffle(vals)
    for n, p in enumerate(vals[: args.n_val_pairs]):
        sets = {"val: clue only -> core": [p["clue"]], "val: twin only -> core": [p["fillers"][0]]}
        ans = answer_sets(st, p["question"], qmap[p["question"]], list(sets.values()), docs, args.max_new_tokens)
        for (name, idx), a in zip(sets.items(), ans):
            rows.append({"split": "val", "pair": p["pair"], "type": p["type"], "value": p["value"], "condition": name,
                         "clue_selected": p["clue"] in idx, "answer": a, **C.score(a, p["rubrics"]),
                         **C.strict_score(a, p["rubrics"], p["value"])})
    conds = list(dict.fromkeys(r["condition"] for r in rows))
    table = C.table_both(rows, conds)
    by_type = {t: C.table_both([r for r in rows if r["type"] == t and r["split"] == "test"], conds)
               for t in sorted({r["type"] for r in rows if r["split"] == "test"})}
    paired = {c: C.paired_lift(rows, c, ctl) for c, ctl in (("clue + 149 -> core", "counterfactual 150 -> core"),
                                                             ("learned top-4 -> core",
                                                              "learned top-4, counterfactual -> core"))}
    for c, v in table.items():
        print(f"{c:40s} {v}")
    for c, v in paired.items():
        print(f"paired {c:33s} {v}")
    save(f"adapt_eval_{args.variant}.json", {"variant": args.variant, "table": table, "by_type": by_type,
                                              "paired": paired, "rows": rows})


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("teach")
    t.add_argument("--samples", type=int, default=4)
    t.add_argument("--keep", type=int, default=3)
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--max_new_tokens", type=int, default=80)
    tr = sub.add_parser("train")
    tr.add_argument("--variant", choices=["adapter", "lora", "both"], required=True)
    tr.add_argument("--updates", type=int, default=150)
    tr.add_argument("--accum", type=int, default=4)
    tr.add_argument("--eval_every", type=int, default=15)
    tr.add_argument("--patience", type=int, default=3)
    tr.add_argument("--n_val", type=int, default=40)
    tr.add_argument("--lr_adapter", type=float, default=3e-4)
    tr.add_argument("--lr_lora", type=float, default=2e-4)
    tr.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("eval")
    e.add_argument("--variant", choices=["base", "adapter", "lora", "both"], required=True)
    e.add_argument("--per_value", type=int, default=5)
    e.add_argument("--n_val_pairs", type=int, default=20)
    e.add_argument("--max_new_tokens", type=int, default=96)
    args = ap.parse_args()
    {"teach": cmd_teach, "train": cmd_train, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
