"""Variants of the 4-bit unpack kernel on this GPU (no model; ~0.4 GB). Each must equal q8's kernel bit for bit.

The unpack runs once per weight per pass, and once per generated token, so it bounds 4-bit decoding on a V100.
    q4_kernel_bench.py [--reps 30]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import q8  # noqa: E402


def variants():
    import triton
    import triton.language as tl

    @triton.jit
    def k_shift(P, S, L, OUT, n_bytes, GROUP_HALF: tl.constexpr, BLOCK: tl.constexpr):
        # the group of a byte is its global offset // GROUP_HALF (rows are whole groups): no division by a runtime value
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_bytes
        p = tl.load(P + offs, mask=mask, other=0).to(tl.int32)
        g = offs // GROUP_HALF
        s = tl.load(S + g, mask=mask, other=0).to(tl.float32)
        lo = tl.load(L + g, mask=mask, other=0).to(tl.float32)
        w = tl.interleave((p & 15).to(tl.float32) * s + lo, (p >> 4).to(tl.float32) * s + lo)
        o2 = tl.program_id(0) * (2 * BLOCK) + tl.arange(0, 2 * BLOCK)
        tl.store(OUT + o2, w.to(OUT.dtype.element_ty), mask=o2 < 2 * n_bytes)

    @triton.jit
    def k_group(P, S, L, OUT, GROUP_HALF: tl.constexpr, NG: tl.constexpr):
        # NG whole groups per program: scale and offset are loaded once per group; sizes are exact multiples, no masks
        pid = tl.program_id(0)
        p = tl.load(P + pid * (NG * GROUP_HALF) + tl.arange(0, NG * GROUP_HALF)).to(tl.int32)
        gi = pid * NG + tl.arange(0, NG)
        s = tl.load(S + gi).to(tl.float32)
        lo = tl.load(L + gi).to(tl.float32)
        s = tl.reshape(tl.broadcast_to(s[:, None], (NG, GROUP_HALF)), (NG * GROUP_HALF,))
        lo = tl.reshape(tl.broadcast_to(lo[:, None], (NG, GROUP_HALF)), (NG * GROUP_HALF,))
        w = tl.interleave((p & 15).to(tl.float32) * s + lo, (p >> 4).to(tl.float32) * s + lo)
        tl.store(OUT + pid * (2 * NG * GROUP_HALF) + tl.arange(0, 2 * NG * GROUP_HALF), w.to(OUT.dtype.element_ty))

    def shift(blk, warps):
        def run(p, s, lo, group, dtype):
            out = torch.empty(p.shape[0], 2 * p.shape[1], dtype=dtype, device=p.device)
            n = p.numel()
            k_shift[(triton.cdiv(n, blk),)](p, s, lo, out, n, GROUP_HALF=group // 2, BLOCK=blk, num_warps=warps)
            return out
        return run

    def grouped(ng, warps):
        def run(p, s, lo, group, dtype):
            out = torch.empty(p.shape[0], 2 * p.shape[1], dtype=dtype, device=p.device)
            n_groups = s.numel()
            assert n_groups % ng == 0
            k_group[(n_groups // ng,)](p, s, lo, out, GROUP_HALF=group // 2, NG=ng, num_warps=warps)
            return out
        return run

    v = {"q8 now": q8._triton_deq4()}
    for blk in (1024, 2048, 4096):
        for warps in (4, 8):
            v[f"shift block {blk} warps {warps}"] = shift(blk, warps)
    for ng in (16, 32, 64, 128):
        for warps in (4, 8):
            v[f"group x{ng} warps {warps}"] = grouped(ng, warps)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    dev, dt = "cuda", torch.float16
    torch.manual_seed(0)
    shapes = [(12288, 4096), (4096, 12288), (4096, 4096), (1024, 4096)]
    mods = []
    for o, i in shapes:
        m = q8.Int4Linear(i, o, False, dev, dt)
        m.weight.copy_(torch.randint(0, 256, m.weight.shape, dtype=torch.uint8, device=dev))
        m.scale.copy_(torch.rand(m.scale.shape, device=dev) * 0.01 + 1e-4)
        m.low.copy_(torch.randn(m.low.shape, device=dev) * 0.02)
        mods.append(m)
    V = variants()
    ok = {}
    for name, fn in list(V.items()):
        try:
            ok[name] = all(torch.equal(fn(m.weight, m.scale, m.low, m.group, dt), V["q8 now"](m.weight, m.scale, m.low, m.group, dt)) for m in mods)
        except Exception as e:  # a variant this Triton cannot compile is skipped
            print(f"KBENCH {name}: failed ({type(e).__name__}: {str(e)[:120]})", flush=True)
            del V[name]
    times = {n: [] for n in V}
    for _ in range(args.reps):  # interleaved, so another job's load hits every variant alike
        for name, fn in V.items():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for m in mods:
                fn(m.weight, m.scale, m.low, m.group, dt)
            torch.cuda.synchronize()
            times[name].append(time.perf_counter() - t0)
    n_w = sum(o * i for o, i in shapes)
    base = sorted(times["q8 now"])[len(times["q8 now"]) // 2]
    for name, ts in sorted(times.items(), key=lambda kv: sorted(kv[1])[len(kv[1]) // 2]):
        med, best = sorted(ts)[len(ts) // 2], min(ts)
        print(f"KBENCH {name:28s} median {med * 1e3:6.2f} ms  best {best * 1e3:6.2f} ms  {n_w / med / 1e9:5.1f} G weights/s  "
              f"x{base / med:4.2f}  exact {ok[name]}", flush=True)
    print(f"KBENCH a full pass over the 8B's layers (6.9 G weights) at the best median: "
          f"{6.9e9 / (n_w / min(sorted(t)[len(t) // 2] for t in times.values())) * 1e3:.0f} ms (now {6.9e9 / (n_w / base) * 1e3:.0f} ms)", flush=True)


if __name__ == "__main__":
    main()
