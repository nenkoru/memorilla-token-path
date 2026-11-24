"""Does poc_fast.lean_loss_batch give the same loss and LoRA gradient as poc_fast.loss_batch? Run under the q8 4-bit
lean settings on the box. One example at a time (no padding, so HF's positions match), then a batch of 4 against the
mean of the single losses, the LoRA gradient with and without checkpointing, then the timing of a training-sized lean step.

  python -I experiments/reader_check.py [lora]      (default lora_lsort.pt)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import fast_train as FT  # noqa: E402
import poc_agentmem as A  # noqa: E402
import poc_broad as B  # noqa: E402
import poc_core as C  # noqa: E402
import poc_fast as FB  # noqa: E402
import poc_profile as PF  # noqa: E402
import q8  # noqa: E402
from poc_agentmem import P  # noqa: E402


def main():
    lora = sys.argv[1] if len(sys.argv) > 1 else "lora_lsort.pt"
    su, qvec, fvec, rows = B.load_all()
    gib = lambda: f"{torch.cuda.memory_allocated() / 2**30:.2f} GiB"  # noqa: E731
    st = A.Stack("lora", train=True)
    C.load_lora(st, lora)
    print(f"CHECK loaded: {gib()}", flush=True)
    core = st.core.float().eval().cpu()
    torch.cuda.empty_cache()
    print(f"CHECK core on CPU: {gib()}", flush=True)
    q8.head_to_gpu(st.r.model)
    print(f"CHECK head on GPU: {gib()}", flush=True)
    FT.fast_lora(st.lora)
    dt = next(st.r.model.parameters()).dtype
    ex = [r for r in rows if r["request"] in qvec and r["fact"] in fvec][:40:5]
    items = []
    for i, r in enumerate(ex):
        others = [x["fact"] for x in ex if x["fact"] != r["fact"]][: i % 4]
        mem = PF.blocks(C, core, np.stack([fvec[f] for f in [r["fact"], *others]]), qvec[r["request"]], 16, P, torch).to(dt)
        items.append((r["request"] + PF.SUFFIX, mem, r["answer"]))
    st.r.model.eval()
    with torch.no_grad():
        one = [(float(FB.loss_batch(st, [it])), float(FB.lean_loss_batch(st, [it]))) for it in items[:4]]
        for a, b in one:
            print(f"CHECK single  plain {a:.4f}  lean {b:.4f}  diff {abs(a - b):.4f}", flush=True)
        lb = float(FB.lean_loss_batch(st, items[:4]))
        print(f"CHECK batch of 4  lean {lb:.4f}  mean of singles (lean) {np.mean([b for _, b in one]):.4f}", flush=True)
    q, m, ans = items[0]  # gradient: lean loss without (eval mode) and with (train mode) layer checkpointing; short answer
    short = [(q, m, " ".join(ans.split()[:30]))]
    pick = [st.lora[i].B for i in (0, 70, 140)]
    grads = []
    for train in (False, True):
        st.r.model.train(train)
        for p in pick:
            p.grad = None
        FB.lean_loss_batch(st, short).backward()
        grads.append([p.grad.detach().float().clone() for p in pick])
    for (g1, g2), k in zip(zip(*grads), (0, 70, 140)):
        cos = float((g1 * g2).sum() / (g1.norm() * g2.norm() + 1e-12))
        print(f"CHECK grad lora[{k}].B  cosine {cos:.5f}  norm plain {float(g1.norm()):.4e} checkpointed {float(g2.norm()):.4e}", flush=True)
    for p in st.lora:
        p.A.grad = p.B.grad = None
    torch.cuda.reset_peak_memory_stats()
    for n in range(3):
        torch.cuda.synchronize()
        t = time.perf_counter()
        FB.lean_loss_batch(st, items[4:8]).backward()
        torch.cuda.synchronize()
        toks = sum(m.shape[1] + 60 + len(st.r.tok(a)["input_ids"]) for _, m, a in items[4:8])
        print(f"CHECK lean step {n}: {time.perf_counter() - t:.1f} s, ~{toks} tokens, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    print("CHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
