"""Port Memorilla's pv4 memory head to another decoder and measure it on PersonalizationV4, on a laptop (MPS or CPU).

Subcommands:
    fit        fit token-table converters (ridge and Procrustes) from Qwen3-8B to the target decoder
    generate   answer a seeded pv4 test subset with one decoder + head; also teacher-forced answer NLL
    score      choice accuracy with Qwen3-Embedding-4B, the same rule as Memorilla's evaluate.py
    calibrate  train only the head (memory_projection) on pv4 train, decoder and core frozen

Heads: native (the released Qwen3-8B head), ridge, procrustes, random (same norm as ridge, random directions),
none (closed book: no memory placeholder at all), or a path to a calibrated head .pt.
"""

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn

SCRATCH = Path(os.environ.get("MPORT_SCRATCH", Path(__file__).resolve().parents[1] / "work"))
REPO = Path(os.environ.get("MEMORILLA_REPO", SCRATCH / "repo/memorilla"))
DATA_ROOT = Path(os.environ.get("MPORT_DATA", SCRATCH / "mdata"))
CKPT = Path(os.environ.get("MPORT_CKPT", SCRATCH / "ckpt/pv4"))
sys.path[:0] = [str(REPO), str(Path(__file__).resolve().parents[1])]

from memorilla.benchmarks import PERSONA_PROMPT  # noqa: E402
from memorilla.data import MemoryCollator, apply_chat_template  # noqa: E402
from memorilla.embeddings import load_split_with_embeddings  # noqa: E402
from memorilla.llm import inject_memory  # noqa: E402
from memorilla.memory import MemoryModule  # noqa: E402
from memorilla.metrics import choice_index, choice_predictions, clean_generation  # noqa: E402
from memorilla.utils import prepare_tokenizer  # noqa: E402
from mport.align import LinearMap, fit_map, load_vocab, shared_ids  # noqa: E402
from mport.tables import EmbeddingTable, locate  # noqa: E402

SOURCE = "Qwen/Qwen3-8B"
ENCODER = "Qwen/Qwen3-Embedding-4B"
COLUMNS = ["collection_id", "question", "answer", "choices", "hard"]
DEV = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def slug(model_id: str) -> str:
    return model_id.split("/")[-1]


def snapshot(model_id: str) -> Path:
    """Local snapshot dir (may be partial: e.g. only tokenizer files while the weights are still downloading)."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(model_id, "config.json", local_files_only=True)).parent


def embed_table(model_id: str) -> EmbeddingTable:
    fetched = SCRATCH / "tables" / f"{slug(model_id)}.safetensors"  # written by mport.tables.fetch_tensor
    if fetched.exists():
        return EmbeddingTable(fetched, "model.embed_tokens.weight")
    path = snapshot(model_id)
    name, shard = locate(str(path))
    return EmbeddingTable(path / shard, name)


def maps_dir(decoder: str) -> Path:
    return SCRATCH / "maps" / f"{slug(SOURCE)}__{slug(decoder)}"


# ---------------------------------------------------------------- data


def load_split(split: str):
    return load_split_with_embeddings("pv4", split, DATA_ROOT, COLUMNS)


def subset(dataset, split: str, n: int, seed: int) -> list[dict]:
    rows = sorted(random.Random(seed).sample(range(len(dataset)), min(n, len(dataset))))
    return [{**dataset[r], "config": "pv4", "split": split, "row": r} for r in rows]


def shuffle_collections(items: list[dict], dataset) -> list[dict]:
    """Give every row another user's documents (a derangement over the split's collections)."""
    ids = sorted(set(dataset["collection_id"]))
    other = {c: ids[(i + 1) % len(ids)] for i, c in enumerate(ids)}
    return [{**item, "collection_id": other[item["collection_id"]]} for item in items]


# ---------------------------------------------------------------- models


def load_memory(head: str, decoder: str) -> MemoryModule | None:
    """The released pv4 module with its head replaced for ``decoder``."""
    if head == "none":
        return None
    memory = MemoryModule.from_pretrained(CKPT).float()
    if head == "native":
        return memory.to(DEV).eval()
    w, b = memory.memory_projection.weight.data, memory.memory_projection.bias.data
    if head in ("ridge", "procrustes", "random"):
        mapping = LinearMap.load(maps_dir(decoder) / f"{'ridge' if head == 'random' else head}.safetensors")
        new_w, new_b = mapping.weight @ w, mapping.weight @ b + mapping.bias
        if head == "random":
            g = torch.Generator().manual_seed(0)
            noise = torch.randn(new_w.shape, generator=g)
            new_w = noise * (new_w.norm() / noise.norm())
    else:
        state = torch.load(head, map_location="cpu")
        new_w, new_b = state["weight"], state["bias"]
    projection = nn.Linear(new_w.shape[1], new_w.shape[0])
    projection.weight.data.copy_(new_w)
    projection.bias.data.copy_(new_b)
    memory.memory_projection = projection
    memory.config = dataclasses.replace(memory.config, output_dim=new_w.shape[0])
    return memory.to(DEV).eval()


def load_decoder(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    memory_id = prepare_tokenizer(tokenizer)
    # straight onto the GPU: no full CPU copy first (on 36 GB unified memory that doubled the peak)
    # MPORT_DTYPE=float16 on GPUs without fast bfloat16 (V100)
    if os.environ.get("MPORT_Q8") == "1":  # Mac light mode: int8 weight-only decoder layers (experiments/q8.py)
        import q8
        model = q8.load(model_id, DEV)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, os.environ.get("MPORT_DTYPE", "bfloat16")), attn_implementation="sdpa",
                                                     device_map=str(DEV), low_cpu_mem_usage=True)
    model = model.eval().requires_grad_(False)
    return model, tokenizer, memory_id


def input_embeds(model, tokenizer, memory_id, input_ids, memory_tokens):
    """Embed ``input_ids``; ``<|memory|>`` rows (outside the decoder's vocab) are looked up as pad, then replaced."""
    safe = input_ids.masked_fill(input_ids == memory_id, tokenizer.pad_token_id)
    embeds = model.get_input_embeddings()(safe)
    if memory_tokens is not None:
        embeds = inject_memory(embeds, memory_tokens.to(embeds.dtype), input_ids, memory_id)
    return embeds


class Batcher:
    """Memorilla's collator for memory heads; a plain system prompt (no placeholder) for closed book."""

    def __init__(self, tokenizer, stores, num_memories: int | None, train: bool, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.num_memories = num_memories
        self.collator = MemoryCollator(tokenizer, stores, num_memories=num_memories or 0, max_length=max_length,
                                       system_prompt=PERSONA_PROMPT, train=train)

    def __call__(self, items):
        batch = self.collator(items)
        if self.num_memories is None:  # closed book: re-render without the memory placeholder
            texts, prompt_lengths = [], []
            for item in items:
                messages = [{"role": "system", "content": PERSONA_PROMPT},
                            {"role": "user", "content": str(item["question"]).strip()}]
                prompt = apply_chat_template(self.tokenizer, messages, add_generation_prompt=True)
                if self.collator.train:
                    full = messages + [{"role": "assistant", "content": str(item["answer"]).strip()}]
                    texts.append(apply_chat_template(self.tokenizer, full, add_generation_prompt=False))
                    prompt_lengths.append(len(self.tokenizer.encode(prompt, add_special_tokens=False)))
                else:
                    texts.append(prompt)
            enc = self.tokenizer(texts, padding=True, padding_side="left", return_tensors="pt", truncation=True,
                                 max_length=self.collator.max_length)
            batch["input_ids"], batch["attention_mask"] = enc["input_ids"], enc["attention_mask"]
            if self.collator.train:
                batch["labels"] = MemoryCollator._labels(enc["input_ids"], enc["attention_mask"], prompt_lengths)
        return batch


def memory_tokens(memory, batch):
    if memory is None:
        return None
    return memory(batch["doc_embeds"].to(DEV, torch.float32), batch["doc_padding_mask"].to(DEV),
                  batch["question_embeds"].to(DEV, torch.float32))


def answer_loss(model, tokenizer, memory_id, memory, batch) -> torch.Tensor:
    """Mean next-token loss over the reference answer tokens (training-mode batch)."""
    ids, mask, labels = (batch[k].to(DEV) for k in ("input_ids", "attention_mask", "labels"))
    embeds = input_embeds(model, tokenizer, memory_id, ids, memory_tokens(memory, batch))
    return model(inputs_embeds=embeds, attention_mask=mask, labels=labels).loss


# ---------------------------------------------------------------- subcommands


def cmd_fit(args):
    src, dst = embed_table(SOURCE), embed_table(args.decoder)
    src_ids, dst_ids = shared_ids(load_vocab(snapshot(SOURCE) / "tokenizer.json"),
                                  load_vocab(snapshot(args.decoder) / "tokenizer.json"), src.num_rows, dst.num_rows)
    print(f"shared tokens: {len(src_ids):,}  ({src.shape} -> {dst.shape})")
    out = maps_dir(args.decoder)
    report = {"shared": int(len(src_ids))}
    for method in ("ridge", "procrustes"):
        t0 = time.time()
        mapping = fit_map(src, dst, src_ids, dst_ids, ridge=args.ridge, method=method)
        mapping.save(out / f"{method}.safetensors")
        report[method] = {**mapping.stats, "seconds": round(time.time() - t0, 1)}
        print(method, json.dumps(report[method]))

    # Are memory tokens anywhere near the token manifold the map was fitted on?
    dataset, store = load_split("test")
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(SOURCE)
    prepare_tokenizer(tokenizer)
    batcher = Batcher(tokenizer, {("pv4", "test"): store}, 16, train=False)
    native = load_memory("native", args.decoder)
    with torch.no_grad():
        mem = memory_tokens(native, batcher(subset(dataset, "test", 32, 1))).reshape(-1, src.dim).cpu()
    sample = torch.from_numpy(src_ids[:: max(1, len(src_ids) // 20000)])
    table = src.rows(sample)
    nearest = (nn.functional.normalize(mem, dim=-1) @ nn.functional.normalize(table, dim=-1).T).max(-1).values
    report["memory_vs_tokens"] = {
        "memory_norm": float(mem.norm(dim=-1).mean()),
        "token_norm": float(table.norm(dim=-1).mean()),
        "nearest_token_cosine": float(nearest.mean()),
        "token_to_token_nearest_cosine": float(
            ((nn.functional.normalize(table[:512], dim=-1) @ nn.functional.normalize(table[512:], dim=-1).T)
             .max(-1).values.mean())),
    }
    dst_table = dst.rows(torch.from_numpy(dst_ids[:: max(1, len(dst_ids) // 20000)]))
    for method in ("ridge", "procrustes"):
        mapped = LinearMap.load(out / f"{method}.safetensors")(mem)
        report["memory_vs_tokens"][f"{method}_mapped_norm"] = float(mapped.norm(dim=-1).mean())
    report["memory_vs_tokens"]["target_token_norm"] = float(dst_table.norm(dim=-1).mean())
    print(json.dumps(report["memory_vs_tokens"], indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2))


@torch.no_grad()
def cmd_generate(args):
    dataset, store = load_split("test")
    items = subset(dataset, "test", args.n, args.seed)
    if args.shuffle_docs:
        items = shuffle_collections(items, dataset)
    model, tokenizer, memory_id = load_decoder(args.decoder)
    memory = load_memory(args.head, args.decoder)
    num = None if memory is None else memory.num_memories
    gen = Batcher(tokenizer, {("pv4", "test"): store}, num, train=False)
    tf = Batcher(tokenizer, {("pv4", "test"): store}, num, train=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records, losses, t0 = [], [], time.time()
    for start in range(0, len(items), args.batch_size):
        chunk = items[start:start + args.batch_size]
        losses.append(float(answer_loss(model, tokenizer, memory_id, memory, tf(chunk))) * len(chunk))
        batch = gen(chunk)
        ids, mask = batch["input_ids"].to(DEV), batch["attention_mask"].to(DEV)
        embeds = input_embeds(model, tokenizer, memory_id, ids, memory_tokens(memory, batch))
        output = model.generate(inputs_embeds=embeds, attention_mask=mask, max_new_tokens=args.max_new_tokens,
                                do_sample=False, pad_token_id=tokenizer.pad_token_id)
        texts = tokenizer.batch_decode(output, skip_special_tokens=True)
        for item, text in zip(chunk, texts, strict=True):
            records.append({k: item[k] for k in ("row", "question", "answer", "choices", "hard")}
                           | {"prediction": clean_generation(text)})
        print(f"{len(records)}/{len(items)}  {time.time() - t0:.0f}s  nll={sum(losses) / len(records):.3f}", flush=True)
    with open(out / "predictions.jsonl", "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    meta = vars(args) | {"answer_nll": sum(losses) / len(records), "seconds": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


def cmd_score(args):
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(str(snapshot(ENCODER)), device=str(DEV), model_kwargs={"dtype": torch.float16})
    cache: dict[str, torch.Tensor] = {}

    def embed(texts):
        missing = sorted({t for t in texts if t not in cache})
        if missing:
            vectors = encoder.encode(missing, batch_size=16, convert_to_numpy=True, normalize_embeddings=True)
            cache.update(zip(missing, vectors))
        return [cache[t] for t in texts]

    import numpy as np

    table = []
    for run in args.runs:
        run = Path(run)
        records = [json.loads(line) for line in open(run / "predictions.jsonl")]
        preds = np.stack(embed([r["prediction"].strip() or " " for r in records]))
        choice_vecs = [np.stack(embed([c for c in r["choices"] if c.strip()])) for r in records]
        picked = choice_predictions(preds, choice_vecs, [r["choices"] for r in records])
        correct = np.array([choice_index(r["choices"], r["answer"]) == p for r, p in zip(records, picked)])
        hard = np.array([bool(r["hard"]) for r in records])
        meta = json.loads((run / "meta.json").read_text())
        metrics = {"accuracy": float(correct.mean()), "accuracy_hard": float(correct[hard].mean()) if hard.any() else None,
                   "n": len(records), "n_hard": int(hard.sum()), "answer_nll": meta["answer_nll"]}
        (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
        for r, ok in zip(records, correct):
            r["correct"] = bool(ok)
        (run / "scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        table.append((run.name, metrics))
    print(f"{'run':40s} {'acc':>7s} {'hard':>7s} {'nll':>7s}")
    for name, m in table:
        hard = f"{100 * m['accuracy_hard']:.1f}" if m["accuracy_hard"] is not None else "-"
        print(f"{name:40s} {100 * m['accuracy']:7.1f} {hard:>7s} {m['answer_nll']:7.3f}")


def cmd_calibrate(args):
    torch.manual_seed(args.seed)
    train_ds, train_store = load_split("train")
    val_ds, val_store = load_split("validation")
    model, tokenizer, memory_id = load_decoder(args.decoder)
    memory = load_memory(args.init, args.decoder)
    memory.requires_grad_(False)
    trainable = list(memory.memory_projection.parameters())
    if args.unfreeze_last:
        last = len(memory.cross_attn_layers) - 1
        for module in (memory.cross_attn_layers[last], memory.ffn_layers[last], memory.ffn_norms[last],
                       memory.cross_attn_norms[last], memory.layer_norm):
            trainable += list(module.parameters())
    for p in trainable:
        p.requires_grad_(True)
    print(f"trainable params: {sum(p.numel() for p in trainable):,}")
    stores = {("pv4", "train"): train_store, ("pv4", "validation"): val_store}
    batcher = Batcher(tokenizer, stores, memory.num_memories, train=True, max_length=args.max_length)
    train_items = [{**train_ds[r], "config": "pv4", "split": "train", "row": r}
                   for r in random.Random(args.seed).sample(range(len(train_ds)), min(len(train_ds), args.steps * args.batch_size))]
    val_items = subset(val_ds, "validation", args.val_n, 0)
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "w")

    def validate():
        memory.eval()
        with torch.no_grad():
            total = sum(float(answer_loss(model, tokenizer, memory_id, memory, batcher(val_items[i:i + 8]))) * len(val_items[i:i + 8])
                        for i in range(0, len(val_items), 8))
        return total / len(val_items)

    best = validate()
    print(f"step 0 val_nll={best:.4f}", flush=True)
    log.write(json.dumps({"step": 0, "val_nll": best}) + "\n")
    save = lambda: torch.save({"weight": memory.memory_projection.weight.detach().cpu(),  # noqa: E731
                               "bias": memory.memory_projection.bias.detach().cpu()}, out / "head.pt")
    save()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        chunk = train_items[(step - 1) * args.batch_size: step * args.batch_size]
        loss = answer_loss(model, tokenizer, memory_id, memory, batcher(chunk))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        if step % args.eval_every == 0 or step == args.steps:
            val = validate()
            entry = {"step": step, "train_loss": float(loss), "val_nll": val, "seconds": round(time.time() - t0)}
            log.write(json.dumps(entry) + "\n")
            log.flush()
            print(json.dumps(entry), flush=True)
            if val < best:
                best = val
                save()
    if args.unfreeze_last:
        memory.save(out / "module")
    print(f"best val_nll={best:.4f}; head at {out / 'head.pt'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--decoder", default="LiquidAI/LFM3-1.7B")
    p.add_argument("--ridge", type=float, default=1e-3)
    p = sub.add_parser("generate")
    p.add_argument("--decoder", default="LiquidAI/LFM3-1.7B")
    p.add_argument("--head", default="ridge")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--shuffle_docs", action="store_true")
    p.add_argument("--out", required=True)
    p = sub.add_parser("score")
    p.add_argument("runs", nargs="+")
    p = sub.add_parser("calibrate")
    p.add_argument("--decoder", default="LiquidAI/LFM3-1.7B")
    p.add_argument("--init", default="ridge")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--val_n", type=int, default=64)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--unfreeze_last", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = parser.parse_args()
    {"fit": cmd_fit, "generate": cmd_generate, "score": cmd_score, "calibrate": cmd_calibrate}[args.cmd](args)


if __name__ == "__main__":
    main()
