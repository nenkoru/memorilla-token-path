"""Greedy decoding for the 4-bit reader, written for a V100 that is shared with other jobs.

Why not model.generate: with 4-bit weights every decoding step unpacks every weight once (about 14 GB of fp16 written
per step, whatever the batch), so the step cost is fixed and the speed is the batch size. model.generate also copies
the whole key/value cache every step (torch.cat) and repeats it 4x for grouped-query attention. This loop:
  - keeps keys and values in buffers allocated once, written in place (no copy, peak memory = the cache itself);
  - runs grouped-query attention without repeating the keys (the 4 query heads of a group become 4 query rows);
  - picks the batch from the free GPU memory, and sorts the prompts by length so little is padded;
  - prefills in sub-batches so the prompt pass has a small footprint.
Same maths as the Qwen3 layer (pre-norm, q/k norm, RoPE, GQA), the layers' own modules, so LoRA wrappers apply.

    generate(st, items, max_new, batch=0)    items = [(question, mem)]; batch 0 = from free memory
    fast_gen.py selftest [--lora L] [--n 16] [--max-new-tokens 60]   compares with poc_fast.generate_batch
    fast_gen.py profile [--batches 1,16] [--steps 30]                bus, unpack and step time on this GPU now
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _kv_bytes(model, T):
    c = model.config
    return len(model.model.layers) * 2 * c.num_key_value_heads * (c.hidden_size // c.num_attention_heads) * T * 2


def pick_batch(model, T, dev, reserve_gb=None):
    """Sequences that fit: free memory (under MPORT_GPU_CAP_GB when set) minus a reserve for the transients (one
    layer's unpacked weights, the output layer, the prefill activations)."""
    import torch
    reserve = float(os.environ.get("MPORT_GEN_RESERVE_GB", "1.2")) if reserve_gb is None else reserve_gb
    free, total = torch.cuda.mem_get_info(dev)
    alloc, held = torch.cuda.memory_allocated(dev), torch.cuda.memory_reserved(dev)
    room = free + (held - alloc)
    cap = float(os.environ.get("MPORT_GPU_CAP_GB", "0"))
    if cap:
        room = min(room, cap * 2**30 - alloc)
    return max(1, int((room - reserve * 2**30) // _kv_bytes(model, T)))


def _layer(lyr, h, n, L, cos, sin, Kb, Vb, r0, t0, mask, rot):
    """h [n * L, dim] at cache positions t0..t0+L of rows r0..r0+n: write their keys and values, attend over [:t0+L]."""
    import torch.nn.functional as F
    a = lyr.self_attn
    hd = a.head_dim
    x = lyr.input_layernorm(h)
    q = a.q_norm(a.q_proj(x).view(n * L, -1, hd))
    k = a.k_norm(a.k_proj(x).view(n * L, -1, hd))
    v = a.v_proj(x).view(n * L, -1, hd)
    q, k = rot(q, k, cos, sin, unsqueeze_dim=1)
    kvh = k.shape[1]
    grp = q.shape[1] // kvh
    Kb[r0:r0 + n, :, t0:t0 + L] = k.view(n, L, kvh, hd).transpose(1, 2)
    Vb[r0:r0 + n, :, t0:t0 + L] = v.view(n, L, kvh, hd).transpose(1, 2)
    # grouped queries without repeating the keys: [n, L, kvh, grp, hd] -> [n, kvh, grp * L, hd]
    qq = q.view(n, L, kvh, grp, hd).permute(0, 2, 3, 1, 4).reshape(n, kvh, grp * L, hd)
    o = F.scaled_dot_product_attention(qq, Kb[r0:r0 + n, :, :t0 + L], Vb[r0:r0 + n, :, :t0 + L], attn_mask=mask, scale=a.scaling)
    o = o.view(n, kvh, grp, L, hd).permute(0, 3, 1, 2, 4).reshape(n * L, -1)
    h = h + a.o_proj(o)
    return h + lyr.mlp(lyr.post_attention_layernorm(h))


def _decode(model, embs, max_new, eos, pad_id, prefill_tokens=2000, temp=0.0, top_p=1.0, gen=None):  # 6000 prefill tokens overran the cap
    """embs: list of [l_i, dim] prompt embeddings. -> list of generated token id lists (cut at the first end token).
    temp 0: greedy. temp > 0: sampled from softmax(logits / temp), kept to the smallest set of tokens whose mass reaches top_p."""
    import torch
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb as rot
    mm = model.model
    emb = model.get_input_embeddings()
    dev, dt, dim = embs[0].device, embs[0].dtype, embs[0].shape[-1]
    B = len(embs)
    lens = torch.tensor([e.shape[0] for e in embs], device=dev)
    T0 = int(lens.max())
    T = T0 + max_new
    c = model.config
    kvh, hd = c.num_key_value_heads, c.hidden_size // c.num_attention_heads
    grp = c.num_attention_heads // kvh
    KV = [(torch.empty(B, kvh, T, hd, dtype=dt, device=dev), torch.empty(B, kvh, T, hd, dtype=dt, device=dev)) for _ in mm.layers]
    ar = torch.arange(T, device=dev)
    off = T0 - lens  # left padding: every row's next token lands on the same cache position
    valid = ar[None] >= off[:, None]  # [B, T] real keys
    pos = (ar[None] - off[:, None]).clamp(min=0)  # positions count from each row's first real token
    last = torch.empty(B, dim, dtype=dt, device=dev)
    rows = max(1, prefill_tokens // T0)
    causal = torch.tril(torch.ones(T0, T0, dtype=torch.bool, device=dev))
    eye = torch.eye(T0, dtype=torch.bool, device=dev)
    for r0 in range(0, B, rows):
        n = min(rows, B - r0)
        h = torch.zeros(n, T0, dim, dtype=dt, device=dev)
        for i in range(n):
            h[i, int(off[r0 + i]):] = embs[r0 + i]
        h = h.view(n * T0, dim)
        cos, sin = (x[0] for x in mm.rotary_emb(h, pos[r0:r0 + n, :T0].reshape(1, -1)))
        # padded query rows see only themselves (an empty row would give NaN); nobody reads them
        m = ((causal[None] & valid[r0:r0 + n, None, :T0]) | eye[None])[:, None].repeat(1, 1, grp, 1)
        for lyr, (Kb, Vb) in zip(mm.layers, KV):
            h = _layer(lyr, h, n, T0, cos, sin, Kb, Vb, r0, 0, m, rot)
        last[r0:r0 + n] = h.view(n, T0, dim)[:, -1]
        del h, m
    torch.cuda.synchronize()
    t_fill = time.perf_counter()
    head = model.lm_head
    out = torch.full((B, max_new), pad_id, dtype=torch.long, device=dev)
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    eos_t = torch.tensor(sorted(eos), dtype=torch.long, device=dev)
    for s in range(max_new):
        logits = head(mm.norm(last)).float()
        nxt = logits.argmax(-1) if temp <= 0 else _sample(logits, temp, top_p, gen)
        nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
        out[:, s] = nxt
        done = done | torch.isin(nxt, eos_t)
        if bool(done.all()) or s == max_new - 1:
            break
        h = emb(nxt).to(dt)
        t = T0 + s
        cos, sin = (x[0] for x in mm.rotary_emb(h, (lens + s)[None]))
        m = valid[:, None, None, :t + 1]
        for lyr, (Kb, Vb) in zip(mm.layers, KV):
            h = _layer(lyr, h, B, 1, cos, sin, Kb, Vb, 0, t, m, rot)
        last = h
    del KV
    torch.cuda.synchronize()
    _decode.steps, _decode.step_s = s + 1, (time.perf_counter() - t_fill) / (s + 1)
    res = []
    for row in out.tolist():
        cut = next((i for i, t in enumerate(row) if t in eos), len(row))
        res.append(row[:cut])
    return res


def _sample(logits, temp, top_p, gen):
    import torch
    p = torch.softmax(logits / temp, -1)
    if top_p < 1.0:
        ps, idx = p.sort(-1, descending=True)
        drop = ps.cumsum(-1) - ps > top_p  # tokens after the mass already reached top_p
        ps = ps.masked_fill(drop, 0.0)
        return idx.gather(-1, torch.multinomial(ps, 1, generator=gen)).squeeze(-1)
    return torch.multinomial(p, 1, generator=gen).squeeze(-1)


def generate(st, items, max_new, batch=0, log=None, temp=0.0, top_p=1.0, seed=0):
    """Answers for items = [(question, mem)], in the order given. batch 0: as many as the free GPU memory holds.
    temp > 0 samples (see _decode); the same seed gives the same answers at the same batches."""
    import torch
    from poc_agentmem import P
    model, tok = st.r.model, st.r.tok
    g = model.generation_config
    assert (getattr(g, "repetition_penalty", None) or 1.0) == 1.0, "greedy only: no repetition penalty"
    eos = g.eos_token_id if isinstance(g.eos_token_id, (list, tuple)) else [g.eos_token_id]
    eos = {int(e) for e in eos if e is not None}
    pad_id = tok.pad_token_id
    lens = [st.ids(q, m.shape[1])[0].shape[1] for q, m in items]
    order = sorted(range(len(items)), key=lambda i: -lens[i])
    ans = [None] * len(items)
    i = 0
    gen = torch.Generator(device=P.DEV).manual_seed(seed) if temp > 0 else None
    with torch.no_grad():
        while i < len(order):
            T = lens[order[i]] + max_new
            fit = pick_batch(model, T, P.DEV)
            b = min(fit, batch) if batch else fit
            part = order[i:i + b]
            embs = []
            for j in part:
                q, m = items[j]
                ids, _ = st.ids(q, m.shape[1])
                embs.append(P.input_embeds(model, tok, st.r.memory_id, ids, m.to(P.DEV))[0])
            t0 = time.perf_counter()
            for j, row in zip(part, _decode(model, embs, max_new, eos, pad_id, temp=temp, top_p=top_p, gen=gen)):
                ans[j] = tok.decode(row, skip_special_tokens=True).strip()
            del embs
            if log:
                log(f"FASTGEN {i + len(part)}/{len(order)}: batch {len(part)} (fits {fit}), prompt {lens[part[0]]} tokens, "
                    f"{time.perf_counter() - t0:.0f}s ({_decode.steps} steps of {_decode.step_s * 1e3:.0f} ms), peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
            i += len(part)
    return ans


def cmd_selftest(args):
    import json
    import numpy as np
    import torch
    import poc_core as C
    import poc_fast as FB
    import poc_profile as PF
    import poc_tok as T
    import q8
    from poc_agentmem import P
    K = PF.poc()
    *_, emap = PF.setup()
    gen = json.loads((K.OUT / "profile_gen.json").read_text())["rows"]
    rows = [r for r in gen if r["fact"] in emap and r["request"] in emap][:: 7][: args.n]
    torch_, A, C, st, core, dt = T.load_stack(args.lora)
    if q8.lean():
        core = q8.place_core(core)
        q8.head_to_gpu(st.r.model)
    rng = np.random.default_rng(0)
    facts = [r["fact"] for r in rows]
    items = []
    with torch.no_grad():
        for r in rows:  # 1-8 blocks per item so the prompts differ in length
            fs = [r["fact"]] + list(rng.choice(facts, int(rng.integers(0, 8)), replace=False))
            items.append((r["request"], PF.blocks(C, core, np.stack([emap[f] for f in fs]), emap[r["request"]], 16, P, torch).to(dt)))
    sync = torch.cuda.synchronize
    sync()
    t0 = time.perf_counter()
    ref = [a for b in range(0, len(items), args.ref_batch) for a in FB.generate_batch(st, items[b:b + args.ref_batch], args.max_new_tokens)]
    sync()
    t1 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    new = generate(st, items, args.max_new_tokens, args.batch, log=lambda s: print(s, flush=True))
    sync()
    t2 = time.perf_counter()
    same = sum(a == b for a, b in zip(ref, new))
    pre = [next((i for i, (x, y) in enumerate(zip(a.split(), b.split())) if x != y), min(len(a.split()), len(b.split()))) for a, b in zip(ref, new)]
    print(f"FASTGEN selftest: {len(items)} prompts, {args.max_new_tokens} new tokens: model.generate (batch {args.ref_batch}) {t1 - t0:.1f}s, "
          f"this loop {t2 - t1:.1f}s ({(t1 - t0) / (t2 - t1):.1f}x); identical answers {same}/{len(items)}, "
          f"words before the first difference: min {min(pre)}", flush=True)
    for a, b in list(zip(ref, new))[:2]:
        print("FASTGEN sample", json.dumps({"generate": a[:200], "loop": b[:200]}), flush=True)


def cmd_profile(args):
    """Where a decoding step goes on this GPU now: the bus (packed weights, offloaded runs), the unpack, the rest."""
    import numpy as np
    import torch
    import poc_tok as T
    import q8
    torch_, A, C, st, core, dt = T.load_stack(args.lora)
    model = st.r.model
    if q8.lean():
        core.cpu()
        q8.head_to_gpu(model)
    dev = next(model.model.norm.parameters()).device
    sync = lambda: torch.cuda.synchronize()  # noqa: E731
    med = lambda xs: float(np.median(xs))  # noqa: E731
    lins = [[m for m in layer.modules() if isinstance(m, q8.Int4Linear)] for layer in model.model.layers]
    cache = getattr(lins[0][0], "cache", None)
    deq = q8._deq4()
    bus, unp = [], []
    for _ in range(args.reps):
        tb = tu = 0.0
        for L, mods in enumerate(lins):
            if cache is not None:
                cache.have.clear()
                sync()
                t0 = time.perf_counter()
                cache._fetch(L)
                sync()
                tb += time.perf_counter() - t0
                got = cache.have[L][1]
                ws = [got[id(m)] for m in mods]
            else:
                ws = [(m.weight, m.scale, m.low) for m in mods]
            sync()
            t0 = time.perf_counter()
            for m, (p, sc, lo) in zip(mods, ws):
                deq(p, sc, lo, m.group, dt)
            sync()
            tu += time.perf_counter() - t0
        bus.append(tb)
        unp.append(tu)
    if cache is not None:
        cache.have.clear()
        cache.warm()
    nbytes = sum(m._buffers[k].numel() * m._buffers[k].element_size() for mods in lins for m in mods for k in ("weight", "scale", "low"))
    print(f"FASTGEN profile: packed layers {nbytes / 2**30:.2f} GiB; bus {med(bus) * 1e3:.0f} ms per pass (best {min(bus) * 1e3:.0f}, "
          f"{nbytes / 2**30 / min(bus) if min(bus) else 0:.1f} GiB/s); unpack {med(unp) * 1e3:.0f} ms per pass (best {min(unp) * 1e3:.0f})", flush=True)
    emb = model.get_input_embeddings()
    g = torch.Generator().manual_seed(0)
    for B in args.batches:
        ids = torch.randint(1000, 20000, (B, args.prompt), generator=g)
        with torch.no_grad():
            embs = [emb(ids[i:i + 1].to(dev))[0].to(dev, dt) for i in range(B)]
            torch.cuda.reset_peak_memory_stats()
            _decode(model, embs, args.steps, set(), st.r.tok.pad_token_id)
        print(f"FASTGEN profile: batch {B:3d}, prompt {args.prompt} tokens: {_decode.step_s * 1e3:.0f} ms per step over {_decode.steps} steps, "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        del embs


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("selftest")
    t.add_argument("--lora", default="lora_kds8.pt")
    t.add_argument("--n", type=int, default=16)
    t.add_argument("--batch", type=int, default=0)
    t.add_argument("--ref-batch", type=int, default=16)
    t.add_argument("--max-new-tokens", type=int, default=60)
    p = sub.add_parser("profile")
    p.add_argument("--lora", default="lora_kds8.pt")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--prompt", type=int, default=100)
    p.add_argument("--batches", type=lambda x: [int(v) for v in x.split(",")], default=[1, 16])
    args = ap.parse_args()
    {"selftest": cmd_selftest, "profile": cmd_profile}[args.cmd](args)


if __name__ == "__main__":
    main()
