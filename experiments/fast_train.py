"""Training speed-ups for the LoRA runs on the V100 box (2 x V100 32 GB, no NVLink, no bfloat16, no FlashAttention-2).

- Data parallel over both cards: one process per GPU (each sees only its own card, so P.DEV stays "cuda"),
  gradients averaged with one all-reduce per update. Only the LoRA gradients travel (about 15 M numbers),
  so the slow PCIe link between the cards costs little.
- Fused AdamW (one kernel for all parameters).
- Gradient checkpointing on the frozen decoder: activations are recomputed in the backward pass, so a batch can hold
  several times more tokens; fewer, fuller batches keep the card busy.
Already in the training loops: loss and vocabulary layer on target positions only, length-sorted token-budget batches,
float16 weights with a loss scaler.

Started by experiments/v100.sh ddp, which sets MPORT_WORLD, MPORT_RANK and MPORT_DIST (a rendezvous file).
"""
import os


def world():
    return int(os.environ.get("MPORT_WORLD", "1")), int(os.environ.get("MPORT_RANK", "0"))


def setup():
    """Join the process group when started with MPORT_WORLD > 1. Returns (world size, rank)."""
    n, r = world()
    if n > 1:
        import torch.distributed as dist
        dist.init_process_group("nccl", init_method="file://" + os.environ["MPORT_DIST"], world_size=n, rank=r)
    return n, r


def optimizer(params, lr, kind="muon"):
    """Muon (PyTorch's built-in) for 2-D weight matrices: momentum, then the update is orthogonalized with 5 Newton-Schulz steps.
    adjust_lr_fn="match_rms_adamw" scales each update to AdamW's size, so an AdamW learning rate carries over.
    All our LoRA parameters are 2-D; anything else is refused rather than silently mixed."""
    import torch
    params = list(params)
    if kind == "adamw":
        try:
            return torch.optim.AdamW([{"params": params, "lr": lr}], weight_decay=0.0, fused=True)
        except (RuntimeError, TypeError):
            return torch.optim.AdamW([{"params": params, "lr": lr}], weight_decay=0.0)
    assert all(p.ndim == 2 for p in params), "Muon here takes 2-D matrices only"
    return torch.optim.Muon([{"params": params, "lr": lr}], weight_decay=0.0, adjust_lr_fn="match_rms_adamw")


def fast_lora(mods):
    """LoRALinear computes its branch in float32 with dropout: a float32 copy of every input and a mask per projection,
    about 4.7 GB extra at 2k tokens on Qwen3-8B. Here: the branch in the input's dtype (fp16), no dropout.
    The parameters stay float32 (the optimizer state too); only the matmuls run in half precision."""
    import types

    def forward(self, x):
        return self.base(x) + (x @ self.A.t().to(x.dtype)) @ (self.B.t() * self.scale).to(x.dtype)

    for m in mods:
        m.forward = types.MethodType(forward, m)


def checkpointing(model):
    """Recompute each decoder layer in the backward pass (only while model.train())."""
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False


def average_grads(params):
    """Mean of the gradients over all processes, one flat all-reduce. Must run before unscale/clip/step on every rank."""
    n, _ = world()
    if n == 1:
        return
    import torch
    import torch.distributed as dist
    ps = [p for p in params if p.grad is not None]
    flat = torch.cat([p.grad.reshape(-1) for p in ps])
    dist.all_reduce(flat)
    flat /= n
    i = 0
    for p in ps:
        k = p.grad.numel()
        p.grad.copy_(flat[i:i + k].view_as(p.grad))
        i += k


def agree(flag):
    """True on every process if any process says True (e.g. out of memory or a non-finite loss: all skip together)."""
    n, _ = world()
    if n == 1:
        return bool(flag)
    import torch
    import torch.distributed as dist
    t = torch.tensor([1.0 if flag else 0.0], device="cuda")
    dist.all_reduce(t)
    return bool(t.item() > 0)


def total(x):
    """Sum of a number over all processes."""
    n, _ = world()
    if n == 1:
        return x
    import torch
    import torch.distributed as dist
    t = torch.tensor([float(x)], device="cuda", dtype=torch.float64)
    dist.all_reduce(t)
    return t.item()


def done():
    n, _ = world()
    if n > 1:
        import torch.distributed as dist
        dist.barrier()
        dist.destroy_process_group()
