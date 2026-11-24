"""Quantization check: load Qwen3-8B with int8 or 4-bit weight-only layers (MPORT_QBITS), report weight memory, and
save next-token log-probs on fixed prompts so two runs can be compared (q4_check.py compare a.pt b.pt)."""
import os, sys, time
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
OUT = Path(os.environ.get("MPORT_SCRATCH", Path(__file__).resolve().parents[1] / "work")) / "runs/quant"
PROMPTS = [
    "The user has type 1 diabetes. What should I pack for a long day hike?",
    "Explain in two sentences why the sky is blue.",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "The capital of France is",
    "I don't have a car. Plan a Saturday trip to see the coast.",
]


def run():
    import q8
    from transformers import AutoTokenizer
    dev = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    t0 = time.perf_counter()
    model = q8.load("Qwen/Qwen3-8B", dev).eval().requires_grad_(False)
    tok = AutoTokenizer.from_pretrained(q8._resolve("Qwen/Qwen3-8B"))
    nbytes = sum(b.numel() * b.element_size() for b in list(model.buffers()) + list(model.parameters()))
    layers = sum(b.numel() * b.element_size() for n, b in model.named_buffers() if ".layers." in n)
    print(f"Q bits={q8.bits()} load {time.perf_counter() - t0:.0f}s weights {nbytes / 2**30:.2f} GiB (decoder linears {layers / 2**30:.2f} GiB)", flush=True)
    res = []
    with torch.no_grad():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids.to(dev)
            lp = torch.log_softmax(model(ids).logits[0].float(), -1).cpu()
            res.append((ids.cpu(), lp))
        g = tok(PROMPTS[0], return_tensors="pt").input_ids.to(dev)
        t1 = time.perf_counter()
        out = model.generate(g, max_new_tokens=40, do_sample=False)
        gen = tok.decode(out[0, g.shape[1]:])
        print(f"Q generate 40 tokens {time.perf_counter() - t1:.1f}s: {gen!r}", flush=True)
    if dev.type == "mps":
        print(f"Q mps driver memory {torch.mps.driver_allocated_memory() / 2**30:.2f} GiB", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"res": res, "gen": gen}, OUT / f"q4check_b{q8.bits()}.pt")


def compare(a, b):
    A, B = torch.load(a), torch.load(b)
    for (ia, la), (ib, lb) in zip(A["res"], B["res"]):
        pa, pb = la.exp(), lb.exp()
        kl = (pa * (la - lb)).sum(-1).mean().item()
        top1 = (la.argmax(-1) == lb.argmax(-1)).float().mean().item()
        print(f"Q compare: {ia.shape[1]} positions, mean KL {kl:.4f}, same top-1 {top1:.2f}")
    print("Q gen A:", repr(A["gen"]))
    print("Q gen B:", repr(B["gen"]))


if __name__ == "__main__":
    compare(*sys.argv[2:4]) if sys.argv[1:2] == ["compare"] else run()
