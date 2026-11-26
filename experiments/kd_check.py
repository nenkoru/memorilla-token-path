"""Check poc_fast.kd_loss_batch on the box (q8 4-bit lean): how many examples align with the text teacher's answer
tokens, that alpha=0 gives lean_loss_batch's loss, the teacher's and the reader's cross-entropy on the same answers
(the gap distillation would close), and the time and peak memory of a training step.

  python -I experiments/kd_check.py [lora]      (default lora_lsort.pt)
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
    K = PF.poc()
    su, qvec, fvec, rows = B.load_all()
    import json
    long = [r for r in json.loads((K.OUT / "broad_long.json").read_text())["rows"] if r["request"] in qvec and r["fact"] in fvec]
    st = A.Stack("lora", train=True)
    C.load_lora(st, lora)
    core = st.core.float().eval().cpu()
    torch.cuda.empty_cache()
    q8.head_to_gpu(st.r.model)
    FT.fast_lora(st.lora)
    dt = next(st.r.model.parameters()).dtype
    ex = long[:64:4]
    items, teach = [], []
    for i, r in enumerate(ex):
        others = [x["fact"] for x in ex if x["fact"] != r["fact"]][: i % 4]
        facts = [r["fact"], *others][::-1] if i % 2 else [r["fact"], *others]
        mem = PF.blocks(C, core, np.stack([fvec[f] for f in facts]), qvec[r["request"]], 16, P, torch).to(dt)
        items.append((r["request"], mem, f"Relevant fact: {r['fact']}\n{r['answer']}"))
        teach.append((facts, r["answer"]))
    st.r.model.eval()
    with torch.no_grad():
        for k in range(0, 16, 4):
            a = float(FB.lean_loss_batch(st, items[k:k + 4]))
            b, n = FB.kd_loss_batch(st, items[k:k + 4], teach[k:k + 4], alpha=0.0)
            c, _ = FB.kd_loss_batch(st, items[k:k + 4], teach[k:k + 4], alpha=1.0)
            print(f"KDCHECK batch {k // 4}: aligned {n} of 4  lean CE {a:.4f}  kd alpha=0 {float(b):.4f}  kd alpha=1 (KL+recite CE) {float(c):.4f}", flush=True)
        # cross-entropy of the teacher (text, LoRA off) and of the reader (16 tokens) on the answer tokens only
        tce, sce = [], []
        for (q, mem, ans), (facts, plain) in zip(items, teach):
            f_ids, ta = FB.teacher_ids(st, facts, q, plain)
            ids = torch.tensor([f_ids], device=P.DEV)
            with FB.lora_off(st):
                lt = torch.log_softmax(st.r.model(input_ids=ids).logits[0, ta - 1:-1].float(), -1)
            tce.append(float(-lt.gather(1, ids[0, ta:, None]).mean()))
            sce.append(float(FB.lean_loss_batch(st, [(q, mem, ans)])))  # includes the recite line
        print(f"KDCHECK answer CE: text teacher {np.mean(tce):.3f}  token reader (with recite line) {np.mean(sce):.3f}", flush=True)
    st.r.model.train()
    torch.cuda.reset_peak_memory_stats()
    for n in range(3):
        torch.cuda.synchronize()
        t = time.perf_counter()
        loss, _ = FB.kd_loss_batch(st, items[4 * n:4 * n + 4], teach[4 * n:4 * n + 4], alpha=1.0)
        loss.backward()
        torch.cuda.synchronize()
        print(f"KDCHECK kd step {n}: {time.perf_counter() - t:.1f} s, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    print("KDCHECK_DONE", flush=True)


if __name__ == "__main__":
    main()
