"""Int8 weight-only decoder for the Mac (MPORT_Q8=1): halves the ~16 GB of bf16 decoder weights.

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
        y = torch._weight_int8pack_mm(x.reshape(-1, shape[-1]).contiguous(), w, s.to(x.dtype))
        return y.reshape(*shape[:-1], w.shape[0])

    @staticmethod
    def backward(ctx, g):
        w, s = ctx.saved_tensors
        return g @ (w.to(g.dtype) * s[:, None].to(g.dtype)), None, None


def _resolve(model_id):
    p = Path(model_id)
    if p.is_dir():
        return p
    from huggingface_hub import try_to_load_from_cache  # the local cache only; never downloads
    cfg = try_to_load_from_cache(model_id, "config.json")
    assert isinstance(cfg, str), f"q8: {model_id} is not in the local Hugging Face cache"
    return Path(cfg).parent


def load(model_id, dev, dtype=torch.bfloat16):
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    root = _resolve(model_id)
    cfg = AutoConfig.from_pretrained(root)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(cfg, dtype=dtype, attn_implementation="sdpa")
    for layer in model.model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name, m in list(parent.named_children()):
                if isinstance(m, torch.nn.Linear):
                    setattr(parent, name, Int8Linear(m.in_features, m.out_features, m.bias is not None, dev, dtype))
    model.to_empty(device=dev)  # allocates the rest (embeddings, norms, lm_head) on dev; int8 shells already there
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
                if isinstance(m, Int8Linear) and attr == "weight":
                    m.fill(t.to(dev))
                else:
                    getattr(m, attr).data.copy_(t.to(dev, dtype))
                done.add(k)
    if getattr(cfg, "tie_word_embeddings", False):
        model.lm_head.weight = model.model.embed_tokens.weight
    want = {k for k in model.state_dict() if not k.endswith(".scale")}
    miss = sorted(want - done - ({"lm_head.weight"} if getattr(cfg, "tie_word_embeddings", False) else set()))
    assert not miss, f"q8: {len(miss)} tensors not in the checkpoint, e.g. {miss[:3]}"
    # non-persistent buffers (rotary inv_freq) were left empty by to_empty: rebuild them on dev
    model.model.rotary_emb = type(model.model.rotary_emb)(config=cfg).to(dev)
    try:
        from transformers import GenerationConfig
        model.generation_config = GenerationConfig.from_pretrained(root)
    except OSError:
        pass
    return model


def enabled():
    return os.environ.get("MPORT_Q8") == "1"
