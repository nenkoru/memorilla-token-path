"""H11: can a cheap fine-tune of the core make it PICK one fact out of a pile? (the buried-clue behaviour, no RAG)

Today, with the core frozen, facts pooled into one 16-token block blur: ~4 facts -> functions 44%, against 81% with a
block per fact. Here the pile holds the facts of N sessions, the question names one session by its issue title and
asks for one kind of name, and everything goes through ONE 16-token block.

    embed   core_facts.npz: keyed fact sentences ("The function touched for the issue "<title>" was X.") for all
            sessions, and the keyed question per session and kind.
    train   --core 1: unfreeze the core's slot queries, question conditioning and cross-attention (the original
                      checkpoint is not modified; the tuned tensors go to core_<tag>.pt) + reader LoRA.
            --core 0: reader LoRA only (control arm).
    eval    60 held-out target sessions, pile = N in {1, 4, 16, 64, 137} test sessions, plus N=16 without the target.
            Also the share of the core's attention that lands on the target session's facts.

H12 (--bias 1): the fine-tune above left the attention at chance, while cos(question, fact) alone picks an asked fact
95% of the time among 530. So add that similarity to the core's cross-attention logits: logit += beta[layer, head] *
cos(question, fact key). 16 new parameters; every original weight stays frozen. Each fact has a key vector (keyed
sentence, used for the similarity) and a value vector that the attention reads: either the same keyed sentence
(value "keyed") or the plain one-name sentence from atoms_all.npz (value "plain").
"""

import argparse
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import poc_agentmem as A  # noqa: E402
import poc_atoms as T  # noqa: E402
import poc_spell as S  # noqa: E402
from poc_agentmem import D, P, OUT, save  # noqa: E402

SENTK = {"repo": 'The coding session about the issue "{t}" was in the repository {x}.',
         "file": 'The file changed for the issue "{t}" was {x}.',
         "func": 'The function or class touched for the issue "{t}" was {x}.'}
QK = {"repo": 'Which repository was the coding session about the issue "{t}" in? Answer with the repository name only.',
      "file": 'Which file was changed for the issue "{t}"? Answer with the file path only.',
      "func": 'Which function or class was touched for the issue "{t}"? Answer with its name only.'}
TUNED = ("memory_queries", "query_cond_attn.", "cross_attn_layers.", "cross_attn_norms.")
MAX_CHUNKS = 200


def title(s):
    return s["title"][:90]


def cmd_embed(args):
    ss, _, _ = A.load_emb()
    facts = [SENTK[k].format(t=title(s), x=x) for s in ss for k, x in T.facts(s)]
    qs = [QK[k].format(t=title(s)) for s in ss for k in S.KINDS]
    enc = D.Encoder()
    t0, out = time.perf_counter(), []
    texts = facts + qs
    for b in range(0, len(texts), 512):
        out.append(enc(texts[b:b + 512], batch_size=32).astype(np.float16))
        print(f"{b + len(out[-1])}/{len(texts)}  {time.perf_counter() - t0:.0f}s", flush=True)
        torch.mps.empty_cache()
    v = np.concatenate(out)
    np.savez(OUT / "core_facts.npz", vecs=v[: len(facts)], q=v[len(facts):].reshape(len(ss), 3, -1))
    print("done", flush=True)


def load_all():
    ss, _, _ = A.load_emb()
    z = np.load(OUT / "core_facts.npz")
    plain = np.load(OUT / "atoms_all.npz")["vecs"]
    o = 0
    for i, s in enumerate(ss):
        s["facts"] = T.facts(s)
        s["arows"] = np.arange(o, o + len(s["facts"]))
        s["si"] = i
        o += len(s["facts"])
    assert o == len(z["vecs"]) == len(plain)
    return ss, z["vecs"], z["q"], plain


def core_fwd(core, d, q, want_attn=False, bias=None):
    """MemoryModule.forward for one unpadded collection. bias: [layers, heads, docs] added to the attention logits."""
    assert not core.config.retrieval_init and core.config.num_self_attn_layers == 0
    question = q.unsqueeze(1)
    slots = core.memory_queries.expand(d.size(0), -1, -1)
    cond, _ = core.query_cond_attn(slots, question, question)
    slots = core.query_cond_norm(slots + cond)
    docs = core.doc_norm(d)
    attns = []
    for li, (norm, attn, fnorm, ffn) in enumerate(zip(core.cross_attn_norms, core.cross_attn_layers, core.ffn_norms, core.ffn_layers)):
        # hand-written multi-head attention: MPS has no derivative for the fused kernel when a mask is passed
        x, dim, h = norm(slots), d.size(-1), core.config.num_heads
        W, b = attn.in_proj_weight, attn.in_proj_bias
        split = lambda z: z.view(z.size(0), z.size(1), h, dim // h).transpose(1, 2)  # noqa: E731
        qh = split(torch.nn.functional.linear(x, W[:dim], b[:dim]))
        kh = split(torch.nn.functional.linear(docs, W[dim:2 * dim], b[dim:2 * dim]))
        vh = split(torch.nn.functional.linear(docs, W[2 * dim:], b[2 * dim:]))
        scores = qh @ kh.transpose(-1, -2) * (dim // h) ** -0.5
        if bias is not None:
            scores = scores + bias[li][None, :, None, :]
        w = torch.softmax(scores, dim=-1)
        att = attn.out_proj((w @ vh).transpose(1, 2).reshape(x.shape))
        if want_attn:
            attns.append(w[0].mean(0).float().cpu())
        slots = slots + core.attn_dropout(att)
        slots = slots + ffn(fnorm(slots))
    out = core.memory_projection(core.layer_norm(slots))
    return (out, attns) if want_attn else out


def pile(target, others, nrng, include_target=True):
    """-> (fact rows, boolean mask of the target session's rows), shuffled."""
    rows = [target["arows"]] if include_target else []
    rows += [o["arows"] for o in others]
    rows = np.concatenate(rows)
    is_t = np.zeros(len(rows), dtype=bool)
    if include_target:
        is_t[: len(target["arows"])] = True
    if len(rows) > MAX_CHUNKS:
        keep = np.concatenate([np.where(is_t)[0], nrng.permutation(np.where(~is_t)[0])[: MAX_CHUNKS - int(is_t.sum())]])
        rows, is_t = rows[keep], is_t[keep]
    p = nrng.permutation(len(rows))
    return rows[p], is_t[p]


def memory(core, rows, keys, vals, qv, beta=None, want_attn=False):
    """keys / vals: [facts, dim] arrays; qv: [dim] array. beta: [layers, heads] tensor or None."""
    d = torch.from_numpy(vals[rows].astype(np.float32))[None].to(P.DEV)
    qt = torch.from_numpy(qv.astype(np.float32))[None].to(P.DEV)
    bias = None
    if beta is not None:
        k = torch.nn.functional.normalize(torch.from_numpy(keys[rows].astype(np.float32)).to(P.DEV), dim=-1)
        cos = k @ torch.nn.functional.normalize(qt[0], dim=-1)
        bias = beta[:, :, None] * cos[None, None, :]
    return core_fwd(core, d, qt, want_attn, bias)


def load_lora(st, name, eval_mode=False):
    sd = torch.load(OUT / name, map_location="cpu")
    for i, m in enumerate(st.lora):
        m.A.data.copy_(sd[f"{i}.A"])
        m.B.data.copy_(sd[f"{i}.B"])
        if eval_mode:
            m.drop.p = 0.0


def tuned_params(core):
    return {n: p for n, p in core.named_parameters() if n.startswith(TUNED)}


def cmd_train(args):
    torch.manual_seed(0)
    ss, keys, q, plain = load_all()
    train = [s for s in ss if s["split"] == "train"]
    st = A.Stack("lora", train=True)
    load_lora(st, args.init)
    core = st.core.float().eval()
    lps = [t for m in st.lora for t in (m.A, m.B)]
    groups = [{"params": lps, "lr": args.lr, "base": args.lr}]
    cps, beta = [], None
    if args.core:
        cps = list(tuned_params(core).values())
        for p in cps:
            p.requires_grad_(True)
        groups.append({"params": cps, "lr": args.core_lr, "base": args.core_lr})
        print(f"core params tuned: {sum(p.numel() for p in cps) / 1e6:.1f}M of {sum(p.numel() for p in core.parameters()) / 1e6:.1f}M", flush=True)
    if args.bias:
        beta = torch.nn.Parameter(torch.full((len(core.cross_attn_layers), core.config.num_heads), float(args.beta0), device=P.DEV))
        groups.append({"params": [beta], "lr": 0.5, "base": 0.5})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    rng, nrng = random.Random(0), np.random.default_rng(0)
    dt = next(st.r.model.parameters()).dtype
    grad_core = bool(args.core or args.bias)

    def loss_one():
        s = rng.choice(train)
        kind = rng.choice([k for k in S.KINDS if any(fk == k for fk, _ in s["facts"])])
        n = rng.choices([1, 2, 4, 8, 16, 32], [0.15, 0.15, 0.2, 0.2, 0.15, 0.15])[0]
        others = rng.sample([o for o in train if o is not s], n - 1)
        rows, _ = pile(s, others, nrng)
        vals = plain if (args.bias and rng.random() < 0.5) else keys
        with torch.set_grad_enabled(grad_core):
            mem = memory(core, rows, keys, vals, q[s["si"], S.KINDS.index(kind)], beta)
        mem = mem.to(dt)
        answer = ", ".join(x for k, x in s["facts"] if k == kind)
        ids, labels = st.ids(QK[kind].format(t=title(s)), mem.shape[1], answer)
        emb = P.input_embeds(st.r.model, st.r.tok, st.r.memory_id, ids, mem)
        return st.r.model(inputs_embeds=emb, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False).loss

    def checkpoint():
        torch.save(st.state(), OUT / f"lora_{args.tag}.pt")
        sd = {n: p.detach().cpu().half() for n, p in tuned_params(core).items()} if args.core else {}
        if beta is not None:
            sd["beta"] = beta.detach().cpu()
        if sd:
            torch.save(sd, OUT / f"core_{args.tag}.pt")

    st.r.model.train()
    t0, u, run = time.perf_counter(), 0, []
    budget, next_save = args.minutes * 60, 1800
    while time.perf_counter() - t0 < budget:
        frac = min(1.0, (time.perf_counter() - t0) / budget)
        for g in opt.param_groups:
            g["lr"] = g["base"] * min(1.0, (u + 1) / 20) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for _ in range(4):
            loss = loss_one() / 4
            loss.backward()
            run.append(loss.item() * 4)
        torch.nn.utils.clip_grad_norm_(lps, 1.0)
        if cps:
            torch.nn.utils.clip_grad_norm_(cps, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        u += 1
        torch.mps.empty_cache()
        if u % 25 == 0:
            b = "" if beta is None else "  beta " + " ".join(f"{x:.0f}" for x in beta.detach().mean(1).tolist())
            print(f"update {u}  examples {u * 4}  loss {np.mean(run[-100:]):.3f}  {(time.perf_counter() - t0) / 60:.0f} min{b}", flush=True)
        if time.perf_counter() - t0 > next_save:
            next_save += 1800
            checkpoint()
    checkpoint()
    save(f"core_train_{args.tag}.json", {"updates": u, "examples": u * 4, "loss": float(np.mean(run[-100:])),
                                         "minutes": args.minutes, "core": args.core, "bias": args.bias,
                                         "beta": None if beta is None else beta.detach().cpu().tolist()})
    print("saved", flush=True)


@torch.no_grad()
def cmd_eval(args):
    ss, keys, q, plain = load_all()
    test = [s for s in ss if s["split"] == "test"]
    st = A.Stack("lora", train=True)
    load_lora(st, args.lora, eval_mode=True)
    core = st.core.float().eval()
    beta = None
    if args.core:
        sd = torch.load(OUT / args.core, map_location="cpu")
        if "beta" in sd:
            beta = sd.pop("beta").to(P.DEV)
        ps = dict(core.named_parameters())
        for n, t in sd.items():
            ps[n].data.copy_(t.float())
    if args.beta is not None:
        beta = torch.full((len(core.cross_attn_layers), core.config.num_heads), float(args.beta), device=P.DEV)
    st.r.model.eval()
    dt = next(st.r.model.parameters()).dtype
    random.Random(3).shuffle(test)
    sizes = [int(x) for x in args.sizes.split(",")]
    conds = [(n, True) for n in sizes] + [(16, False)]
    global MAX_CHUNKS
    MAX_CHUNKS = args.cap
    values = [("keyed", keys)] + ([("plain", plain)] if beta is not None else [])
    values = [x for x in values if x[0] in args.values.split(",")]
    rows_out, t0, tab = [], time.perf_counter(), {}
    for i, s in enumerate(test[: args.n]):
        rng = random.Random(i)
        pool = [o for o in test if o is not s and o["repo"] != s["repo"]]
        extra = [o for o in ss if o["split"] != "test"]  # only used when a pile asks for more sessions than the test set has
        for kind in S.KINDS:
            truth = [x for k, x in s["facts"] if k == kind]
            if not truth:
                continue
            qv = q[s["si"], S.KINDS.index(kind)]
            for n, inc in conds:
                want = n - 1 if inc else n
                others = rng.sample(pool, min(len(pool), want))
                if want > len(others):
                    others += rng.sample(extra, min(len(extra), want - len(others)))
                rows, is_t = pile(s, others, np.random.default_rng(i), include_target=inc)
                asked = is_t & np.array([_KIND[r] == kind for r in rows])
                for vname, vals in values:
                    mem, attns = memory(core, rows, keys, vals, qv, beta, want_attn=True)
                    ans = A.generate(st, QK[kind].format(t=title(s)), mem.to(dt), 48).strip()
                    got = [t.lower() in ans.lower() for t in truth]
                    name = (f"N={n}" if inc else f"N={n} absent") + f" {vname}"
                    rows_out.append({"session": s["id"], "kind": kind, "condition": name, "n_chunks": int(len(rows)),
                                     "truth": truth, "answer": ans, "hit": float(any(got)), "recall": float(np.mean(got)),
                                     "attn_target": [float(a[:, is_t].sum(-1).mean()) for a in attns],
                                     "attn_asked": [float(a[:, asked].sum(-1).mean()) for a in attns],
                                     "chance": float(is_t.mean())})
        torch.mps.empty_cache()
        if (i + 1) % 5 == 0 or i + 1 == min(args.n, len(test)):
            tab = {}
            for r in rows_out:
                tab.setdefault(r["condition"], {}).setdefault(r["kind"], []).append(r)
            tab = {c: {k: {"any": float(np.mean([r["hit"] for r in x])), "recall": float(np.mean([r["recall"] for r in x])),
                           "attn1": float(np.mean([r["attn_target"][0] for r in x])),
                           "attn2": float(np.mean([r["attn_target"][1] for r in x])),
                           "asked2": float(np.mean([r["attn_asked"][1] for r in x])),
                           "chance": float(np.mean([r["chance"] for r in x]))} for k, x in ks.items()}
                   for c, ks in tab.items()}
            save(args.out, {"lora": args.lora, "core": args.core, "beta": None if beta is None else beta.cpu().tolist(),
                            "rows": rows_out, "table": tab})
            print(f"{i + 1}  {(time.perf_counter() - t0) / 60:.0f} min", flush=True)
    for c, ks in tab.items():
        print(f"CORE {c:18s} " + "  ".join(f"{k} {x['any']:.2f}" for k, x in ks.items())
              + f"  attn on target L1 {np.mean([x['attn1'] for x in ks.values()]):.2f} L2 {np.mean([x['attn2'] for x in ks.values()]):.2f}"
              + f" asked fact L2 {np.mean([x['asked2'] for x in ks.values()]):.2f}"
              + f" (chance {np.mean([x['chance'] for x in ks.values()]):.2f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("embed")
    t = sub.add_parser("train")
    t.add_argument("--minutes", type=float, default=150)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--core-lr", type=float, default=2e-5)
    t.add_argument("--core", type=int, default=1)
    t.add_argument("--bias", type=int, default=0)
    t.add_argument("--beta0", type=float, default=50.0)
    t.add_argument("--init", default="lora_slots.pt")
    t.add_argument("--tag", default="pick")
    e = sub.add_parser("eval")
    e.add_argument("--n", type=int, default=60)
    e.add_argument("--lora", default="lora_pick.pt")
    e.add_argument("--core", default="")
    e.add_argument("--beta", type=float, default=None)
    e.add_argument("--sizes", default="1,4,16,64,137")
    e.add_argument("--out", default="core_eval.json")
    e.add_argument("--cap", type=int, default=200, help="max facts per pile (the first runs were capped at 200)")
    e.add_argument("--values", default="keyed,plain")
    args = ap.parse_args()
    global _KIND
    if args.cmd == "eval":
        ss, _, _ = A.load_emb()
        _KIND = [k for s in ss for k, _ in T.facts(s)]
    {"embed": cmd_embed, "train": cmd_train, "eval": cmd_eval}[args.cmd](args)


_KIND = []

if __name__ == "__main__":
    main()
