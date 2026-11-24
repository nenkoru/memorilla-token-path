"""Where does a 4-bit training step's time go? Times one Qwen3-8B layer's seven products on synthetic weights.

For each weight shape: dequantize (weights on the GPU, or streamed from pinned CPU memory), and the fp16 matmuls of a
training step (forward, the checkpointed recompute, and the input-gradient). Extrapolated to 36 layers per step.

  python -I experiments/q4_bench.py [tokens ...]     (default 1760 = 2 requests x 10 facts x 88 positions)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import q8  # noqa: E402

SHAPES = {"q": (4096, 4096), "k": (4096, 1024), "v": (4096, 1024), "o": (4096, 4096),
          "gate": (4096, 12288), "up": (4096, 12288), "down": (12288, 4096)}  # (in, out)


def timeit(fn, n=10):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def main():
    dev = torch.device("cuda")
    toks = [int(a) for a in sys.argv[1:]] or [1760]
    tot = {k: 0.0 for k in ("deq_gpu", "deq_off", "deq_fused", "copy", *[f"mm{t}" for t in toks])}
    for name, (i, o) in SHAPES.items():
        m = q8.Int4Linear(i, o, False, dev, torch.float16)
        m.fill(torch.randn(o, i, device=dev) * 0.02)
        off = q8.Int4Linear(i, o, False, "cpu", torch.float16)
        off.dev = dev
        for k in ("weight", "scale", "low"):
            off._buffers[k] = getattr(m, k).cpu().pin_memory()
        fused = q8._triton_deq4()
        err = (fused(m.weight, m.scale, m.low, 64, torch.float16).float() - q8._deq4_eager(m.weight, m.scale, m.low, 64, torch.float32)).abs().max()
        print(f"{name:5s} fused vs plain: max abs diff {float(err):.2e}", flush=True)
        r = {"deq_gpu": timeit(lambda: m.dequant(torch.float16)), "deq_off": timeit(lambda: off.dequant(torch.float16)),
             "deq_fused": timeit(lambda: fused(m.weight, m.scale, m.low, 64, torch.float16)),
             "copy": timeit(lambda: [x.to(dev, non_blocking=True) for x in (off.weight, off.scale, off.low)])}
        w = m.dequant(torch.float16)
        for t in toks:
            x, g = torch.randn(t, i, device=dev, dtype=torch.float16), torch.randn(t, o, device=dev, dtype=torch.float16)
            r[f"mm{t}"] = timeit(lambda: x @ w.t()) * 2 + timeit(lambda: g @ w)  # forward + recompute + input-gradient
        for k in tot:
            tot[k] += r[k]
        print(f"{name:5s} " + "  ".join(f"{k} {v:6.2f} ms" for k, v in r.items()), flush=True)
    print("per step (36 layers; dequant x3: forward, recompute, backward):")
    print(f"  (deq_gpu and deq_off use q8's default path: fused when Triton loads, MPORT_Q4_TRITON=0 for plain torch)")
    print(f"  dequant, weights on GPU   {36 * 3 * tot['deq_gpu'] / 1000:.2f} s")
    print(f"  dequant, streamed         {36 * 3 * tot['deq_off'] / 1000:.2f} s   (of which copy {36 * 3 * tot['copy'] / 1000:.2f} s)")
    print(f"  dequant, fused (compiled) {36 * 3 * tot['deq_fused'] / 1000:.2f} s")
    for t in toks:
        print(f"  matmuls at {t} tokens     {36 * tot[f'mm{t}'] / 1000:.2f} s")


if __name__ == "__main__":
    main()
