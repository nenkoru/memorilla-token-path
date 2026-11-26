"""MPORT_Q4_REUSE check (no model): through checkpointed 4-bit Linears the gradient is the same with and without keeping
the unpacked weight for the backward pass, no unpacked weight stays alive after the forward pass (GPU only), and each
layer unpacks twice instead of three times (forward, recomputation; the backward reuses the recomputation's)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

import q8  # noqa: E402


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32
    torch.manual_seed(0)
    n, d = 8, 2048 if dev == "cuda" else 256
    lins = []
    for _ in range(n):
        lin = q8.Int4Linear(d, d, False, dev, dt)
        lin.fill(torch.randn(d, d, device=dev) / d ** 0.5)
        lins.append(lin)
    a = torch.nn.Parameter(torch.randn(d, d, device=dev, dtype=dt) * 0.02)
    x = torch.randn(32, d, device=dev, dtype=dt)
    grads, held, calls = [], [], []
    count = [0]
    for lin in lins:  # count the unpack calls
        lin.dequant = (lambda dtype, f=lin.dequant: (count.__setitem__(0, count[0] + 1), f(dtype))[1])
    for reuse in (False, True):
        q8._REUSE = reuse
        a.grad = None
        count[0] = 0
        base = torch.cuda.memory_allocated() if dev == "cuda" else 0
        h = x.clone().requires_grad_(True)
        for lin in lins:  # one checkpoint per layer, as poc_tok.shared_hidden does
            h = checkpoint(lambda t, lin=lin: torch.nn.functional.silu(lin(t @ a)), h, use_reentrant=False)  # an op after the Linear, as in a layer
        y = h.float().pow(2).sum()
        held.append((torch.cuda.memory_allocated() - base) / 2**20 if dev == "cuda" else 0.0)
        y.backward()
        grads.append(a.grad.float().clone())
        calls.append(count[0] / n)
        del h, y
    diff = float((grads[0] - grads[1]).abs().max())
    one = d * d * 2 / 2**20  # one unpacked weight, MiB
    ok = diff == 0 and held[1] - held[0] < one and calls == [3.0, 2.0]
    print(f"REUSE check on {dev}: max abs gradient difference {diff:.3g} (norm {float(grads[0].norm()):.3g}); held after the forward "
          f"pass {held[0]:.1f} MiB without, {held[1]:.1f} MiB with (one unpacked weight is {one:.0f} MiB, {n} layers); unpacks per layer {calls[0]:.0f} -> {calls[1]:.0f} -> {'OK' if ok else 'WRONG'}", flush=True)
    assert ok


if __name__ == "__main__":
    main()
