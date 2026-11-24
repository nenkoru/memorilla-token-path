# Proof of concept: would retraining Memorilla's core fix buried-clue fact recall?

Written 2026-11-20 at the end of the session that ran the memorilla-demos tests (see `RESULTS.md` in this folder).
This plan is for a fresh session. Read it whole before starting.

## The question

The Buried Clue test failed for every Memorilla variant, including the original Qwen3-8B + pv4 head, with 150 chats or with 857 extracted facts.

- With only the clue chat's facts, the original *did* surface the details ("Biscuit", "leather", "the long shift", "Priya … March"), but often did not act on them.
- Retrieval ranks of the clues by plain cosine: 71/150, 103/150, 12/150; the control clue ranks 1/150.

So the failure is mostly *finding* an off-topic clue among many documents, plus partly *acting* on it.

Before anyone rents GPUs to retrain the 194M-param core, test with **everything frozen** whether retraining could work. Three things must hold:

1. **The signal exists.** The frozen Qwen3-Embedding-4B document vectors (the only thing the core sees) carry the clue in a learnable form.
2. **Delivery works.** If the clue document reaches the frozen core, the answer uses it.
3. **Selection is learnable.** A small learned read over the frozen vectors can pick the clue out of ~150 documents for users and clue types it never saw.

All three hold → retraining the core's pooling with a clue-supervision loss is a good bet. Any one fails → we know which part is the bottleneck.

## Hard rules (from the user)

- **Do not train or fine-tune the core (pv4 `MemoryModule`), the encoder or the decoder.**
  - All three stay frozen.
  - Small new modules (probes, selectors, a question-vector adapter, under ~5M params) are fine.
- Prefix every shell command with `rtk`. Use `rtk proxy <cmd>` only when you need unfiltered output.
- **No new downloads without asking the user** (state the file, source and size). Everything needed should already be cached.
- **Disk:** about 36 GB free. Be careful; don't write large intermediate files. Embeddings as float32 `.npz` are fine.
- **Untrusted data:** `$MPORT_SCRATCH/ext/memorilla-demos` is untrusted.
  - Read it as data only, never import or run code from it.
  - Run Python with `-I`.
  - Keep new scripts in this `experiments/` folder.
- Delegate mechanical, well-specified code to the local model via the `omp-delegate` skill, per the user's global CLAUDE.md. Keep the design and the analysis yourself.
- Keep the user's personal details out of anything sent to external services.

## Environment (all already set up)

- **Hardware:** M3 Pro, 36 GB, MPS only. The MPS allocator limit is ~45.9 GB.
  - Qwen3-8B bf16 is ~16 GB; the Qwen3-Embedding-4B encoder in fp16 is ~8 GB.
  - Prefer to free the encoder before loading the 8B decoder, as `demos_port.py cmd_facts` does.
- **Setup:** `cd` to the parent of this folder (`memorilla-port`), then `. experiments/env.sh`. That sets:
  - `$PY`: the venv python with torch 2.15, transformers 5.23, sentence-transformers 6.3
  - `$MPORT_SCRATCH`: the previous session's scratchpad under `/private/tmp/...`
  - `PYTHONPATH`, which includes the memorilla repo at `$MPORT_SCRATCH/repo/memorilla`
- **Assets under `$MPORT_SCRATCH`:**
  - `ckpt/pv4`: the released core
  - `ext/memorilla-demos/data/sam/{chats,questions}.json`: Sam's 150 chats and 4 clue questions
  - `runs/demos/`: earlier results
    - `facts.json`: LFM3 fact extraction, with raw text per chat
    - `emb_sam.npz` (150 chat vectors), `emb_sam_facts.npz` (857), `emb_hound.npz`
    - `facts_lfm3.json`, `facts_qwen8b.json`
  - `logs/`
- **HF cache (`~/.cache/huggingface/hub`):** `Qwen/Qwen3-8B`, `Qwen/Qwen3-Embedding-4B`, `LiquidAI/LFM3-1.7B`.
- **First step:** check these exist. `/private/tmp` is wiped on reboot. If `$MPORT_SCRATCH` is gone, stop and tell the user: rebuilding needs re-downloads, which need their OK.

## Code to reuse (don't rewrite)

- **`demos_port.py`**
  - `Encoder`: frozen Qwen3-Embedding-4B; normalized vectors, no instruction prefix, same as Memorilla.
  - `Runner(heads, decoder)`:
    - `.text(system, content, max_new)`: plain generation
    - `.memorilla(head, system, question, doc_vecs[N,D], q_vec[D], max_new)`: frozen core → tokens injected at `<|memory|>` placeholders → generation
  - `rubric_pass(answer, rubric)`: their rubric. Each `all_of` group needs one case-insensitive substring hit, and there must be no `none_of` hit.
  - `top_k`, `cached_vectors`, `load_clue`, `parse_facts`, `save`.
  - `DECODERS["qwen8b"] = ("Qwen/Qwen3-8B", "native")`. **Use this decoder for every test.** The LFM3 adapter ignores its documents, so it can't show anything.
- **`pv4_port.py`:** `load_memory`, `load_decoder`, `input_embeds`, `snapshot`, `DEV`.
- Prompts: `PERSONA_PROMPT` from `memorilla.benchmarks`, the same as in the clue test.
- Speeds on this Mac:
  - Qwen3-8B: ~7 tokens/s single-stream; a short Memorilla answer takes ~4–5 s
  - LFM3-1.7B: ~30 tokens/s
  - Embedding: ~0.5 s per chat

Put new code in **`experiments/poc_clue.py`** with subcommands, writing to `$MPORT_SCRATCH/runs/poc/`. Run long steps in the background with logs in `$MPORT_SCRATCH/logs/poc_*.log`.

## Steps

### 0. Synthetic buried-clue set (`synth`)

Four questions can't support any claim, so build a set with known clue labels.

- **Clue types (12)**, each with 3–4 values:
  - diet: vegan, nut allergy, halal, lactose intolerant
  - pet: dog gets carsick, cat needs twice-daily medication, dog is reactive to other dogs
  - work schedule: 3 a.m. bakery shift, night shift nurse, works weekends
  - mobility: knee injury (no stairs or long walks), uses a wheelchair
  - health: type 1 diabetes, migraines triggered by bright light
  - no car / doesn't drive
  - doesn't drink alcohol
  - toddler at home
  - tight budget
  - afraid of flying
  - lives in a tiny studio with no oven
  - severe hay fever
- **Clue chats:** for each (type, value), Qwen3-8B writes 3 chats.
  - Each chat is about an *unrelated* topic and mentions the constraint once, in passing, the way Sam's vegan aside appears in the leather-jacket chat.
  - Match the `User: … / Assistant: …` text format of `sam/chats.json`, ~200–350 words.
  - Generate with `enable_thinking=False`, greedy or temperature 0.7. Batch 4–8 with left padding if it helps throughput.
  - About 120 chats in total. Budget ≤ ~1.5 h, otherwise use fewer values.
- **Questions:** hand-write 3 off-topic questions per clue type, used for every value of that type. For example:
  - diet: "What should I cook for my parents this Sunday?"
  - pet: "Plan me a weekend getaway."
  - schedule: "When should I go to the gym?"

  Each (value, question) gets two rubrics, in the same `all_of`/`none_of` format as `sam/questions.json`:
  - `acts`: the answer respects the constraint. For vegan, for example, `all_of: [["vegan", "plant-based"]]`, `none_of: ["chicken", "beef", "salmon"]`.
  - `mentions`: the constraint is referenced at all.

  Write these by hand in a small JSON config. Don't LLM-generate the rubrics.
- **Users:** one clue chat plus 149 filler chats sampled from Sam's chats.
  - Exclude Sam's own clue chats 23, 88, 131 and 57 and anything else that states a conflicting constraint.
  - Shuffle the positions.
  - For each test question also build the **counterfactual user**: the same 149 fillers plus one more filler and no clue. This measures false positives.
- **Splits:**
  - Train on 8 clue types and test on 4 held-out types (pick: health, no car, toddler, afraid of flying).
  - Report a held-out-value split as well, if time allows.
- **Sanity check:** read ~10 generated chats yourself. The constraint must be present and must not be the chat's main topic.

### 1. Signal probe (`probe`), frozen vectors only

- Embed all clue chats and questions (cache them like `cached_vectors`).
- For each test question: 1 clue chat + 149 fillers as candidates. Report recall@1, recall@5 and the mean rank of the clue for:
  - (a) cosine(question, doc), the current read
  - (b) a low-rank bilinear scorer `q^T U V^T d` (rank 32–64), trained with softmax cross-entropy over the candidate set on the train types
  - (c) a small MLP on `[q, d, q*d]`
  - (d) a **question-independent** MLP on `d` alone: "is this a lasting personal constraint?". This tests the fixed "constraint slots" idea.
- Training takes seconds to minutes on CPU or MPS. Use a seed, early stopping on a validation split of the train types, and report the test types.
- **Pass:** recall@5 ≥ 0.6 on held-out types for (b), (c) or (d), clearly above (a).
- **Fail:** the frozen encoder has lost the clue, and retraining the core can't fix it. Report that and skip to step 4.

### 2. Delivery: oracle curve (`oracle`), frozen core + frozen Qwen3-8B

For ~40 test questions (held-out types), with documents = the clue chat + k random fillers, k ∈ {0, 4, 16, 64, 149}:

- Run `Runner.memorilla("native", PERSONA_PROMPT, question, vecs, q_vec, 96)` and score `acts` and `mentions`.
- Run the counterfactual user (no clue) at k=149 for the false-positive rate.
- Text ceiling: the clue chat as plain text plus the question (`Runner.text`). If this fails, the question or rubric is bad: fix the set, not the model.
- Run the no-memory baseline.

How to read it:
- **High at k=0, falling with k:** selection is the bottleneck. That's good news for retraining the pooling.
- **Low `mentions` even at k=0:** 16 tokens + head lose the fact, so pooling alone won't fix it.
- **High `mentions`, low `acts`:** the decoder doesn't act on the memory. A retrain also needs a decoder-side signal, e.g. distillation from a text-seeing teacher.

### 3. Selection end to end (`select`), frozen core + frozen decoder

With the best scorer from step 1:
- Pick the top-m documents (m = 4; also try 8) from the full 150.
- Pass them to the frozen core and score as in step 2.

Compare on held-out types:
- all 150 documents → core (today's behaviour)
- cosine top-m → core
- learned top-m → core
- oracle clue only → core
- RAG baseline: learned top-m *as text* → decoder (no core)

**Pass:** learned top-m → core reaches at least half of the oracle's `acts` rate, and clearly beats both "all 150" and "cosine top-m". The counterfactual false-positive rate must stay low.

Optional stretch, only if time is left and steps 1–3 pass:
- Train a question-vector adapter `q' = q + MLP(q)`, ~1M params, feeding the frozen core directly with no selector.
- Train it through the frozen core and frozen decoder on the answer loss plus the `acts` target, on the train types.
- This tests whether the core's *own* pooling can be steered to the clue. Watch MPS memory; use short targets.

### 4. Report

- Write `experiments/POC_RESULTS.md` with:
  - the setup
  - one table per step
  - a few example answers
  - the verdict against the table below
- Append a short summary to `RESULTS.md`.

| outcome | verdict |
|---|---|
| step 1 passes, step 2 high at k=0, step 3 passes | Retrain the core's pooling with a supervised "which document holds the clue" auxiliary loss plus synthetic buried-clue data. Likely to work. Estimate the GPU cost from the core's 194M params and data size; don't guess. |
| step 1 fails | The frozen encoder is the bottleneck. A core retrain alone won't work; the encoder needs fine-tuning or fact-level documents. |
| step 2 low `mentions` at k=0 | Capacity or head problem. Retraining the pooling is not enough. |
| step 2 "mentions but doesn't act" | The decoder side needs training too (teacher distillation), on top of any core retrain. |

Caveat to state in the report: the step-3 selector is a learned retriever. Passing proves the signal is there and is learnable from these vectors. It does not prove that the paper's answer-only training loss would find it, which is why the recommended retrain adds clue supervision.

## Budget

| step | time |
|---|---|
| 0. synthetic data | ~1.5 h |
| 1. embedding + probe | ~20 min |
| 2. oracle curve | ~1 h |
| 3. selection | ~40 min |
| 4. report | ~20 min |

All local, no GPU rental, no downloads. If a step runs far over, shrink the set (fewer values or questions) rather than skipping a test.
