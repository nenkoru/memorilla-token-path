"""Int8 (MPORT_Q8=1) or 4-bit (also MPORT_QBITS=4) weight-only decoder: ~1/2 or ~1/3.5 of the ~16 GB of decoder weights.

Every nn.Linear inside the decoder layers (q/k/v/o, gate/up/down) is stored as int8 with one bf16 scale per output row
(symmetric, round to nearest) and dequantized on the fly in forward. Embeddings, norms and lm_head stay bf16, so
code that reads model.lm_head.weight is unchanged. LoRA wraps an Int8Linear like a Linear (in_features,
out_features, weight.device). The model is built on the meta device and filled tensor by tensor from the safetensors
shards, so the full bf16 model never sits in memory.
"""

import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F


class Int8Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, bias, device, dtype):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.int8, device=device))
        self.register_buffer("scale", torch.empty(out_features, dtype=dtype, device=device))
        self.bias = None if not bias else torch.nn.Parameter(torch.empty(out_features, dtype=dtype, device=device), requires_grad=False)

    @torch.no_grad()
    def fill(self, w):
        w = w.float()
        s = w.abs().amax(1).clamp_min(1e-8) / 127
        self.weight.copy_(torch.round(w / s[:, None]).clamp_(-127, 127).to(torch.int8))
        self.scale.copy_(s.to(self.scale.dtype))

    def forward(self, x):
        y = _Int8MM.apply(x, self.weight, self.scale)
        return y if self.bias is None else y + self.bias.to(y.dtype)


class _Int8MM(torch.autograd.Function):
    """x @ (int8 W * scale).T with torch's int8 weight-only kernel (no bf16 copy of W; ~15x faster than dequantizing
    for generation on MPS). Backward to x only (W is frozen), dequantizing W just for that product."""

    @staticmethod
    def forward(ctx, x, w, s):
        ctx.save_for_backward(w, s)
        shape = x.shape
        y = _int8_mm(x.reshape(-1, shape[-1]).contiguous(), w, s)
        return y.reshape(*shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, g):
        w, s = ctx.saved_tensors
        return g @ (w.to(g.dtype) * s[:, None].to(g.dtype)), None, None


def _int8_dequant_mm(x, w, s):
    return x @ (w.to(x.dtype) * s[:, None].to(x.dtype)).t()


_INT8_KERNEL = {}  # device type -> bool: torch's int8 weight-only kernel exists there (MPS, CPU); else dequantize


def _int8_mm(x, w, s):
    ok = _INT8_KERNEL.get(x.device.type)
    if ok is None or ok:
        try:
            y = torch._weight_int8pack_mm(x, w, s.to(x.dtype))
            _INT8_KERNEL[x.device.type] = True
            return y
        except (RuntimeError, NotImplementedError):
            _INT8_KERNEL[x.device.type] = False
    return _int8_dequant_mm(x, w, s)


class Int4Linear(torch.nn.Module):
    """4-bit weight-only Linear, asymmetric per group of GROUP input columns: W ~ q * scale + low, q in 0..15.

    Two codes per byte in `weight` (uint8 [out, in // 2]). Plain torch: W is dequantized for each product and freed
    right after, so it runs on any device (V100 and consumer GPUs, MPS, CPU). The backward pass dequantizes again
    instead of saving W, so memory stays at ~4.5 bits per weight plus one layer's transient copy."""

    def __init__(self, in_features, out_features, bias, device, dtype, group=64):
        super().__init__()
        assert in_features % group == 0 and group % 2 == 0, (in_features, group)
        self.in_features, self.out_features, self.group = in_features, out_features, group
        self.dev = torch.device(device)  # where the product runs; the packed weights may live in pinned CPU memory (offload())
        self.register_buffer("weight", torch.empty(out_features, in_features // 2, dtype=torch.uint8, device=device))
        self.register_buffer("scale", torch.empty(out_features, in_features // group, dtype=dtype, device=device))
        self.register_buffer("low", torch.empty(out_features, in_features // group, dtype=dtype, device=device))
        self.bias = None if not bias else torch.nn.Parameter(torch.empty(out_features, dtype=dtype, device=device), requires_grad=False)

    @torch.no_grad()
    def fill(self, w):
        o, i, g = self.out_features, self.in_features, self.group
        w = w.float().reshape(o, i // g, g)
        lo, hi = w.amin(-1), w.amax(-1)
        s = ((hi - lo) / 15).clamp_min(1e-8)
        q = torch.round((w - lo[..., None]) / s[..., None]).clamp_(0, 15).to(torch.uint8).reshape(o, i)
        self.weight.copy_(q[:, 0::2] | (q[:, 1::2] << 4))
        self.scale.copy_(s.to(self.scale.dtype))
        self.low.copy_(lo.to(self.low.dtype))

    def dequant(self, dtype):
        if _FP16:  # MPORT_Q4_FP16=1: unpack once and keep (2 bytes per weight on the GPU; for a GPU with ~17 GB free)
            w = self.__dict__.get("_w")
            if w is None or w.dtype != dtype:
                w = self.__dict__["_w"] = self._unpack(dtype)
            return w
        return self._unpack(dtype)

    def _unpack(self, dtype):
        p, s, lo = self.weight, self.scale, self.low
        if p.device != self.dev:  # offloaded: this layer's packed weights, copied in ahead of time by the prefetcher
            cache = getattr(self, "cache", None)
            p, s, lo = cache.get(self.L)[id(self)] if cache else (x.to(self.dev, non_blocking=True) for x in (p, s, lo))
        return _deq4()(p, s, lo, self.group, dtype)

    def forward(self, x):
        y = _Int4MM.apply(x, self)
        return y if self.bias is None else y + self.bias.to(y.dtype)


def _deq4_eager(p, s, lo, group: int, dtype: torch.dtype):
    """Unpack two 4-bit codes per byte and apply the per-group scale and offset: W = q * s + lo."""
    o = p.shape[0]
    q = torch.stack((p & 15, p >> 4), -1).reshape(o, -1, group).to(dtype)
    return (q * s[..., None].to(dtype) + lo[..., None].to(dtype)).reshape(o, -1)


_DEQ4 = []


def _deq4():
    """The unpacking function: a fused Triton kernel on CUDA (one pass: read 4.5 bits, write fp16), else plain torch.
    MPORT_Q4_TRITON=0 forces plain torch."""
    if not _DEQ4:
        fn = _deq4_eager
        if os.environ.get("MPORT_Q4_TRITON", "1") == "1" and torch.cuda.is_available():
            try:
                fn = _triton_deq4()
            except ImportError:
                pass
        _DEQ4.append(fn)
    return _DEQ4[0]


def _triton_deq4():
    import triton
    import triton.language as tl

    @triton.jit
    def kernel(P, S, L, OUT, n_bytes, half_in, groups_per_row, GROUP_HALF: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_bytes
        p = tl.load(P + offs, mask=mask, other=0).to(tl.int32)
        g = (offs // half_in) * groups_per_row + (offs % half_in) // GROUP_HALF  # both codes of a byte share a group
        s = tl.load(S + g, mask=mask, other=0).to(tl.float32)
        lo = tl.load(L + g, mask=mask, other=0).to(tl.float32)
        w = tl.interleave((p & 15).to(tl.float32) * s + lo, (p >> 4).to(tl.float32) * s + lo)
        o2 = tl.program_id(0) * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
        tl.store(OUT + o2, w.to(OUT.dtype.element_ty), mask=o2 < 2 * n_bytes)

    def run(p, s, lo, group, dtype):
        o, h = p.shape
        out = torch.empty(o, 2 * h, dtype=dtype, device=p.device)
        n, blk = p.numel(), 1024
        kernel[(triton.cdiv(n, blk),)](p, s, lo, out, n, h, s.shape[1], GROUP_HALF=group // 2, BLOCK=blk)
        return out

    return run


class _Int4MM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, mod):
        ctx.mod = mod
        w = mod.dequant(x.dtype)
        # MPORT_Q4_REUSE=1 (training with every layer checkpointed): the pass that is differentiated keeps the unpacked
        # weight for its own backward, which follows at once, so each layer unpacks 3 times per step instead of 4.
        # Costs one layer's fp16 weights (~0.4 GB) at the peak. Never use it without per-layer checkpointing.
        # It must go through save_for_backward: the checkpoint's first pass runs with gradients on and drops only what is
        # saved that way (a plain attribute on ctx would keep every layer's weights alive: 13 GB for the 8B).
        if _REUSE and ctx.needs_input_grad[0] and getattr(mod, "reuse", True):
            ctx.save_for_backward(w)
        return x @ w.t()

    @staticmethod
    def backward(ctx, g):
        saved = ctx.saved_tensors
        w = saved[0] if saved else ctx.mod.dequant(g.dtype)
        return g @ w.to(g.dtype), None


_REUSE = os.environ.get("MPORT_Q4_REUSE") == "1"
_FP16 = os.environ.get("MPORT_Q4_FP16") == "1"


def bits():
    """MPORT_QBITS=4 or 8 (default 8 when MPORT_Q8=1)."""
    return int(os.environ.get("MPORT_QBITS", "8"))


def _resolve(model_id):
    p = Path(model_id)
    if p.is_dir():
        return p
    from huggingface_hub import try_to_load_from_cache  # the local cache only; never downloads
    cfg = try_to_load_from_cache(model_id, "config.json")
    assert isinstance(cfg, str), f"q8: {model_id} is not in the local Hugging Face cache"
    return Path(cfg).parent


def load(model_id, dev, dtype=None):
    """Decoder with int8 (default) or 4-bit (MPORT_QBITS=4) weight-only layers; dtype from MPORT_DTYPE (float16 on V100)."""
    dtype = dtype or getattr(torch, os.environ.get("MPORT_DTYPE", "bfloat16"))
    dev = torch.device(dev)
    devs = split_devices(dev)
    cap = float(os.environ.get("MPORT_GPU_CAP_GB", "0"))
    if cap and dev.type == "cuda":  # hard cap per GPU: this process fails with out-of-memory instead of growing past it
        for d in dict.fromkeys(devs):
            torch.cuda.set_per_process_memory_fraction(min(1.0, cap * 2**30 / torch.cuda.get_device_properties(d).total_memory), d.index)
    nb = bits()
    assert nb in (4, 8), f"q8: MPORT_QBITS must be 4 or 8, not {nb}"
    Q = Int4Linear if nb == 4 else Int8Linear
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    root = _resolve(model_id)
    cfg = AutoConfig.from_pretrained(root)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype, attn_implementation="sdpa")
    n = len(model.model.layers)
    where = [devs[L * len(devs) // n] for L in range(n)]  # MPORT_SPLIT=1: consecutive layers spread over the GPUs
    for layer, d in zip(model.model.layers, where):
        for parent in (layer.self_attn, layer.mlp):
            for name, m in list(parent.named_children()):
                if isinstance(m, torch.nn.Linear):
                    q = Q(m.in_features, m.out_features, m.bias is not None, "cpu" if offload() else d, dtype)
                    q.dev = torch.device(d)
                    setattr(parent, name, q)
    if lean():  # the embedding table lives in CPU memory and is never put on the GPU, not even while loading
        e = model.model.embed_tokens
        model.model.embed_tokens = CpuEmbedding(torch.empty(e.num_embeddings, e.embedding_dim, dtype=dtype))
    for name, child in model.model.named_children():
        if name != "layers":
            child.to_empty(device=dev)
    for layer, d in zip(model.model.layers, where):
        for m in layer.modules():  # offload(): packed 4-bit weights stay in CPU memory, streamed to the GPU per product
            m.to_empty(device="cpu" if offload() and isinstance(m, Int4Linear) else d, recurse=False)
    if lean():  # the output layer too: CPU memory; raters copy their few rows to the GPU, generation computes on CPU
        model.lm_head = CpuLinear(torch.empty(model.lm_head.out_features, model.lm_head.in_features, dtype=dtype))
    else:
        model.lm_head.to_empty(device=dev)
    idx = root / "model.safetensors.index.json"
    files = sorted(set(json.loads(idx.read_text())["weight_map"].values())) if idx.exists() else ["model.safetensors"]
    mods = dict(model.named_modules())
    done = set()
    for fn in files:
        with safe_open(str(root / fn), framework="pt", device="cpu") as f:
            for k in f.keys():
                owner, _, attr = k.rpartition(".")
                m = mods.get(owner)
                if m is None:
                    continue
                t = f.get_tensor(k)
                if isinstance(m, (Int8Linear, Int4Linear)) and attr == "weight":
                    m.fill(t.to(getattr(m, "dev", m.weight.device)))
                else:
                    dst = getattr(m, attr)
                    dst.data.copy_(t.to(dst.device, dtype))
                done.add(k)
    if offload():
        for m in model.modules():
            if isinstance(m, Int4Linear):
                for k in ("weight", "scale", "low"):
                    m._buffers[k] = m._buffers[k].pin_memory()
        if dev.type == "cuda" and len(set(where)) == 1:
            cache = LayerPrefetch([[m for m in layer.modules() if isinstance(m, Int4Linear)] for layer in model.model.layers], dev)
            for L, layer in enumerate(model.model.layers):
                for m in layer.modules():
                    if isinstance(m, Int4Linear):
                        m.cache, m.L = cache, L
            cache.warm()
    if getattr(cfg, "tie_word_embeddings", False):
        model.lm_head.weight = model.model.embed_tokens.weight
    want = {k for k in model.state_dict() if not k.endswith((".scale", ".low", ".anchor"))}
    if lean():
        want |= {"model.embed_tokens.weight", "lm_head.weight"}
    miss = sorted(want - done - ({"lm_head.weight"} if getattr(cfg, "tie_word_embeddings", False) else set()))
    assert not miss, f"q8: {len(miss)} tensors not in the checkpoint, e.g. {miss[:3]}"
    # non-persistent buffers (rotary inv_freq) were left empty by to_empty: rebuild them on dev
    model.model.rotary_emb = type(model.model.rotary_emb)(config=cfg).to(dev)
    if len(set(where)) > 1:
        for layer, d in zip(model.model.layers, where):
            layer.register_forward_pre_hook(_mover(d), with_kwargs=True)
        model.model.norm.register_forward_pre_hook(_mover(dev), with_kwargs=True)
    try:
        from transformers import GenerationConfig
        model.generation_config = GenerationConfig.from_pretrained(root)
    except OSError:
        pass
    return model


def split_devices(dev):
    """MPORT_SPLIT=1 on CUDA with 2+ GPUs: every visible GPU, layers spread evenly (the rest of the model on dev)."""
    if os.environ.get("MPORT_SPLIT") == "1" and dev.type == "cuda" and torch.cuda.device_count() > 1:
        return [torch.device("cuda", i) for i in range(torch.cuda.device_count())]
    return [dev]


def _to(x, d):
    if torch.is_tensor(x):
        return x.to(d)
    if isinstance(x, (tuple, list)):
        return type(x)(_to(v, d) for v in x)
    if isinstance(x, dict):
        return {k: _to(v, d) for k, v in x.items()}
    return x  # caches and other objects: each layer's cache is created on that layer's device


def _mover(d):
    def hook(module, args, kwargs):
        return _to(args, d), _to(kwargs, d)
    return hook


def enabled():
    return os.environ.get("MPORT_Q8") == "1"


def mem_gib():
    """Accelerator memory for logs: CUDA peak allocated, or MPS driver memory now."""
    if torch.cuda.is_available():
        return "peak " + " / ".join(f"{torch.cuda.max_memory_allocated(i) / 2**30:.2f}" for i in range(torch.cuda.device_count())) + " GiB"
    if torch.backends.mps.is_available():
        return f"mps {torch.mps.driver_allocated_memory() / 2**30:.2f} GiB"
    return "cpu"


class CpuEmbedding(torch.nn.Module):
    """Token embedding table kept in CPU memory (MPORT_LEAN=1): only looked up, so the ~1.2 GB table never sits on the
    accelerator. The weight is a plain attribute, not a parameter, so model.to(device) does not move it back."""

    def __init__(self, weight):
        super().__init__()
        self.table = weight.detach().cpu()
        # empty parameter in the table's dtype: code that reads next(model.parameters()).dtype still gets the model dtype
        self.anchor = torch.nn.Parameter(torch.empty(0, dtype=weight.dtype, device=weight.device), requires_grad=False)
        self.num_embeddings, self.embedding_dim = self.table.shape

    @property
    def weight(self):
        return self.table

    def forward(self, ids):
        return self.table[ids.cpu()].to(ids.device)


class CpuLinear(torch.nn.Module):
    """Bias-free output layer kept in CPU memory (MPORT_LEAN=1); input moved to CPU, logits moved back."""

    def __init__(self, weight):
        super().__init__()
        self.table = weight.detach().cpu()
        self.out_features, self.in_features = self.table.shape
        self.anchor = torch.nn.Parameter(torch.empty(0, dtype=weight.dtype), requires_grad=False)

    @property
    def weight(self):
        return self.table

    def forward(self, x):
        return (x.to("cpu", self.table.dtype) @ self.table.t()).to(x.device)


class LayerPrefetch:
    """GPU copies of a few layers' packed 4-bit weights (MPORT_OFFLOAD=1). Using layer L waits for its copy and starts
    copying L - 1 and L + 1 on a side stream, so the transfer runs behind the compute in both the forward pass and the
    backward pass (which recomputes and differentiates the layers in reverse order). Keeps at most `keep` layers, plus
    the first MPORT_RESIDENT_LAYERS layers, which are copied once and stay (0.10 GiB each for the 8B): for the GPU memory
    that is free beyond the working set, so that fewer bytes cross the bus per pass."""

    def __init__(self, mods, dev, keep=4):
        import collections
        self.mods, self.dev, self.keep = mods, dev, keep
        self.fixed = int(os.environ.get("MPORT_RESIDENT_LAYERS", "0"))
        # other people's jobs share this GPU: when its free memory falls under this, the resident layers are given back
        self.min_free = float(os.environ.get("MPORT_MIN_FREE_GB", "1.5")) * 2**30
        self.stream = torch.cuda.Stream(dev)
        self.have = collections.OrderedDict()  # layer -> (event, {id(module): (packed, scale, low)})

    def warm(self):
        """Copy the resident layers now, so that what is measured as free memory afterwards is really free."""
        for L in range(min(self.fixed, len(self.mods))):
            self._fetch(L)
        torch.cuda.synchronize(self.dev)

    def _yield(self):
        if torch.cuda.mem_get_info(self.dev)[0] < self.min_free:
            print(f"q8: {torch.cuda.mem_get_info(self.dev)[0] / 2**30:.1f} GiB free on the GPU: giving back the {self.fixed} resident layers", flush=True)
            self.fixed = 0
            self.have.clear()
            torch.cuda.empty_cache()

    def _fetch(self, L):
        if L in self.have or not 0 <= L < len(self.mods):
            return
        main = torch.cuda.current_stream(self.dev)
        self.stream.wait_stream(main)  # the memory it reuses may still be read by kernels queued on the main stream
        with torch.cuda.stream(self.stream):
            got = {}
            for m in self.mods[L]:
                ts = tuple(m._buffers[k].to(self.dev, non_blocking=True) for k in ("weight", "scale", "low"))
                for x in ts:
                    x.record_stream(main)
                got[id(m)] = ts
            ev = torch.cuda.Event()
            ev.record(self.stream)
        self.have[L] = (ev, got)

    def get(self, L):
        if L == 0 and self.fixed:  # once per pass
            self._yield()
        self._fetch(L)
        ev, got = self.have[L]
        self.have.move_to_end(L)
        torch.cuda.current_stream(self.dev).wait_event(ev)
        nxt = (L + 1) % len(self.mods)  # after the last layer comes layer 0 again (next decoding step, next forward)
        for n in (nxt, L - 1):
            self._fetch(n)
        while sum(k >= self.fixed for k in self.have) > self.keep:
            old = next(k for k in self.have if k >= self.fixed)
            if old in (L - 1, L, nxt):
                break
            del self.have[old]
        return got


def offload():
    """MPORT_OFFLOAD=1 (4-bit only): keep the packed layer weights in pinned CPU memory, stream each to the GPU per product."""
    return os.environ.get("MPORT_OFFLOAD") == "1" and bits() == 4


def lean():
    return os.environ.get("MPORT_LEAN") == "1"


def place_core(core):
    """Where the memory core runs in a lean run. Default: the CPU (frees ~0.4 GB of GPU). MPORT_CORE_GPU=1 leaves it on
    the GPU: on a box whose CPUs are oversubscribed the CPU core was half of every training step (4.8 s of 9.8 s)."""
    return core if os.environ.get("MPORT_CORE_GPU") == "1" else core.cpu()


def make_lean(model):
    """MPORT_LEAN=1: embedding table to CPU memory. Callers that only need a few lm_head rows (the raters) also call
    drop_lm_head after reading them."""
    if lean() and not isinstance(model.get_input_embeddings(), CpuEmbedding):
        emb = model.get_input_embeddings()
        model.set_input_embeddings(CpuEmbedding(emb.weight))
        del emb
        _empty_cache()
    return model


class SlicedHead(torch.nn.Module):
    """Output layer for the GPU in 4 bits, cut along the vocabulary into slices of Int4Linear: ~0.33 GB resident and
    one slice (~250 MB) dequantized at a time, against 1.2 GB for fp16. The logits are concatenated."""

    def __init__(self, table, dev, slices=10):
        super().__init__()
        n = table.shape[0]
        cuts = [n * i // slices for i in range(slices + 1)]
        self.out_features, self.in_features = table.shape
        self.parts = torch.nn.ModuleList()
        for a, b in zip(cuts, cuts[1:]):
            m = Int4Linear(table.shape[1], b - a, False, "cpu", table.dtype)
            m.fill(table[a:b])  # quantized on the CPU: no fp32 transient on the GPU
            m.to(dev)
            m.dev = torch.device(dev)
            m.reuse = False  # MPORT_Q4_REUSE: the output layer's slices are not kept (10 of them would be 1.2 GB)
            self.parts.append(m)

    def forward(self, x):
        return torch.cat([m(x) for m in self.parts], -1)


def head_to_gpu(model, dev="cuda"):
    """MPORT_LEAN=1, for code that needs the whole output layer (reader training and generation): the CPU table to the
    GPU, as a SlicedHead (4 bits, MPORT_HEAD_BITS=4, the default) or an fp16 layer (MPORT_HEAD_BITS=16, ~1.2 GB).
    The CPU product would take seconds per training step."""
    if isinstance(model.lm_head, CpuLinear):
        t = model.lm_head.table
        if os.environ.get("MPORT_HEAD_BITS", "4") == "4":
            model.lm_head = SlicedHead(t, dev)
        else:
            lin = torch.nn.Linear(t.shape[1], t.shape[0], bias=False, device=dev, dtype=t.dtype)
            with torch.no_grad():
                lin.weight.copy_(t)
            lin.weight.requires_grad_(False)
            model.lm_head = lin
        _empty_cache()


def drop_lm_head(model):
    """MPORT_LEAN=1: free the ~1.2 GB output layer once the needed rows were copied (code that calls model.model only)."""
    if lean():
        model.lm_head = torch.nn.Identity()
        _empty_cache()


def _empty_cache():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
