# Proof of concept: would retraining Memorilla's core fix buried-clue recall?

Plan: [POC_PLAN.md](POC_PLAN.md). Code: [poc_clue.py](poc_clue.py), config and rubrics: [poc_config.json](poc_config.json).
Outputs: `$MPORT_SCRATCH/runs/poc/`, logs: `$MPORT_SCRATCH/logs/poc_*.log`. `$MPORT_SCRATCH` is now `memorilla-port/work/` (see [env.sh](env.sh)). It used to be under `/private/tmp`, which was lost in the crash described under "Crash, rebuild and replication".
Everything frozen: Qwen3-Embedding-4B encoder, pv4 `MemoryModule` (194M), Qwen3-8B with the native head.
The only trained parts are the step-1 scorers (≤ 0.6M params each, CPU).

## Verdict

**Decision-table row: "step 2 low `mentions` at k=0 → capacity or head problem; retraining the pooling is not enough."**

- Step 1 (signal in the frozen vectors): **borderline fail by the letter**, but the reason the plan attaches to a fail does not hold.
  - Best held-out R@5 is 0.53 (doc_mlp), below the 0.6 bar. Cosine is 0.00.
  - Every learned scorer ranks the clue chat above its own twin (the same chat without the constraint) in 100% of held-out pairs.
  - So the encoder has not lost the clue. What falls short is the margin against 149 unrelated chats on clue types never seen in training. It improves slowly as more train types are added.
- Step 2 (delivery through the frozen core): **fails, and the failure is at the decoder end.**
  - With the clue chat as the *only* document (k=0), the answer names the constraint 0 / 40 times.
  - It acts on it 12.5% of the time, against 10% for the no-clue counterfactual (paired difference +2.5 points).
  - The same chat as plain text: 45% `mentions`, 55% `acts`. So the rubrics and questions are fine.
  - A linear probe on the core's 16 output tokens still tells clue from twin on held-out types (100% pairwise). The fact survives the core. The frozen decoder does not read it out.
- Step 3 (learned top-m → frozen core): **fails.**
  - The learned selector gets the clue into the top 4 47.5% of the time and into the top 8 60% (cosine: 0%).
  - When the clue *is* selected, the core path's answers mention it 0 / 19 (top-4) and 0 / 24 (top-8) times.
  - Learned top-8 `acts` 0.225 equals the no-clue counterfactual's 0.225. Learned top-4 at 0.075 ties cosine top-4 and is below all 150 (0.15).
  - That is what step 2 predicts: better selection feeds a core → decoder path that doesn't use the clue.

What this means for the retrain idea:
- A selection-only fix to the core's pooling (aux "which document holds the clue" loss) would not fix recall.
- The step-2 oracle already hands the core exactly the right document, and the answers still don't use it.
- Any retrain has to change what the memory tokens make the decoder *do*: train the core/output projection end to end through the decoder on answers that use the side-constraint, most likely with text-teacher distillation.
  - That pushes towards decision-table row 4 as well ("decoder side needs training too").
- Selection is a second, separate problem (step 1): a learned scorer gets the clue into the top 5 about half the time on unseen clue types.

**Follow-up (adapter / LoRA, core still frozen; details under "Follow-up" below):**
- A 3.8M-param LoRA on Qwen3-8B, trained on 147 teacher answers, makes the decoder read the clue from the frozen core's tokens on the trained clue types: val clue-only `mentions` 5 / 20, against 1 / 20 untrained.
- On unseen clue types it barely transfers (≤ 3 / 40).
- With all 150 chats, every variant is at 0 mentions.
- An adapter on the memory tokens alone does not help.
- So the decoder was one real bottleneck. Generalizing needs many more clue types or a general "read the memory" stage, and the 150-chat case still needs pooling/selection fixed.

## Setup

### Step 0: synthetic buried-clue set

12 clue types, 28 values. Each value has 3 clue chats: 84 clue chats, 84 twins, 57 neutral chats, 225 in all.
Generated with Qwen3-8B (bf16, MPS, `enable_thinking=False`, temperature 0.7, top-p 0.9, seeded, batch 8). All 225 passed the format and length checks on the first attempt (946 s).

Deviations from the plan, and why:
- **Twin-then-rewrite generation.** A first prompt asking for a chat "that mentions X once in passing" produced stilted tacked-on asides ("Oh, and I've been vegan for several years.") and copied the example wording. Instead:
  1. Qwen3-8B writes the chat about the topic with no constraint (the *twin*).
  2. A second call rewrites only the first user message to weave the constraint in mid-sentence.
  The clue chat and its twin are therefore identical apart from that clause. This controls for topic and generator style, and lets "beats its own twin" serve as a clean signal test.
- **Length ~125–140 words** (clue 139, twin 124, neutral 125 on average) instead of 200–350. This matches Sam's real chats (mean 141 words), which make up most of the fillers.
- **Second values added to single-value types** (no_car + no_license, toddler + newborn, fear_flying + climate_flying, alcohol: no_alcohol + sober, budget: tight_budget + debt, studio: no_oven + microwave, hay_fever + bee_sting), so every type has ≥ 2 values.
- **Filler pools are disjoint between train and test.**
  - Sam's 146 eligible chats (excluding 23, 88, 131, 57) are split 73 / 73.
  - Train users draw on train twins plus Sam half A.
  - Test users draw on test twins, the 57 neutrals and Sam half B.
  - Every user has 150 candidates: the clue + 149 fillers, always including the clue's own twin.
  - A per-type conflict regex removes fillers that state a conflicting constraint (41 removed for budget, fewer for others).
- Counterfactual user: the same 149 fillers + 1 more filler, no clue.
- Splits:
  - Test types (held out): health, no_car, toddler, flying. That is 8 values, 24 clue chats, 72 (chat, question) pairs.
  - Train types: the other 8. Chats j=0,1 are train (120 pairs); chat j=2 is validation (60 pairs).
  - The optional held-out-value split was not run.
- Questions and rubrics: hand-written in `poc_config.json` (3 questions per type, `acts` + `mentions` per value).
- Sanity read of 14 clue messages: every constraint is present as a passing aside, and the topics are unrelated. Quirks:
  - some odd rationales ("careful with the installation team's snacks")
  - `night_nurse:2` also says it's "on a budget"
  - `knee:2` mentions a wheelchair
  - `t1_diabetes:0` keeps "My kid asked me" from its twin
- Code was written here, not delegated to the local omp model: no local LLM server was running, the 27B model can't share the MPS GPU with Qwen3-8B, and the work was mostly design.

### Crash, rebuild and replication

- The first full run got through steps 0–2. Partway into step 3 the Mac ran out of RAM and shut down.
  - The cause: Qwen3-8B in bf16 (~16 GB) on MPS, plus a CPU copy made during loading, plus MPS caching that grew across batches of 6–8 sequences at 320 tokens.
  - The scratch folder in `/private/tmp` (venv, repo clone, pv4 checkpoint, generated data) was lost with it.
- With the user's go-ahead, everything was rebuilt in `memorilla-port/work/`, which persists:
  - re-downloaded the memorilla repo, the pv4 core, Sam's chats and the Python packages
  - regenerated the synthetic set with the same seeds
  - reran steps 1–3 with these RAM guards:
    - **One model in memory at a time.** Every step is a separate process, run one after another.
    - **The decoder loads straight onto MPS** (`device_map="mps"`, `low_cpu_mem_usage=True`), so there is no extra CPU copy of the 16 GB weights.
    - **MPS memory is capped** with `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8` and `LOW=0.6`, and `torch.mps.empty_cache()` runs after every batch and every pair.
    - **A watchdog** ([poc_guard.sh](poc_guard.sh)) logs system free memory every 15 s and kills the job if it falls below 12%.
  - Lowest free memory seen: 20% during generation, 18% during the step-2 oracle and 16% during step-3 select. The watchdog never had to act, and the whole chain finished.
- **Why the decoder was not quantized (int8/4-bit):**
  - No quantization library (quanto, torchao, bitsandbytes for MPS) was installed, and adding one meant another download.
  - It would also have changed the decoder's numbers. Steps 2 and 3 would then no longer be comparable with each other or with the first run.
  - The guards above kept bf16 inside the memory budget.
- **Replication:**
  - Generation is seeded, so the regenerated set reproduced the first one.
  - Every number in steps 1 and 2 came out identical to the first run: probe, learning curve, core probe and the oracle table with both rubrics.
  - So the tables below come from one data set, run twice.

### Rubric scoring: two rules

- **substring**: the plan's rule (case-insensitive substring per `all_of` group, no `none_of` hit).
- **strict**: added after reading the first 3 of 40 step-2 pairs, before the rest were scored.
  - Some substring terms leaked: "son" hit "reason", "rest" hit "restaurant", "bus" hit "business", and "Fly from New York…" got past the flying `none_of` list.
  - Strict requires each term to start at a word boundary, with ≤ 4 trailing letters ("diabet" → "diabetes" still counts).
  - For flying and no_car it adds action phrases to `none_of` ("fly from", "flying into", "rent a car", …). Bare nouns like "flights" are not added, so that "instead of flights, take the train" still passes.
- Both are reported. The conclusions don't change between them. Strict numbers are lower for the no-clue baselines, which makes them the fairer of the two.

## Step 1: signal probe (frozen vectors)

Held-out types, 72 pairs, 150 candidates each (chance R@5 = 0.033, chance mean rank 75.5). Mean over 3 seeds. Early stopping on validation R@5.

| scorer | params | val R@5 | test R@1 | test R@5 | test R@10 | test mean rank | test median rank | clue above own twin |
|---|---|---|---|---|---|---|---|---|
| (a) cosine | 0 | 0.017 | 0.000 | 0.000 | 0.000 | 95.9 | 96 | 29% |
| (b) bilinear, rank 48 (+ cosine residual) | 0.25M | 0.983 | 0.259 | 0.482 | 0.597 | 26.2 | 4–10 | 100% |
| (c) MLP on [q, d, q·d] (+ cosine residual) | 2.0M | 0.978 | 0.245 | 0.496 | 0.560 | 23.7 | 4–9.5 | 100% |
| (d) question-independent MLP on d | 0.66M | 1.000 | 0.361 | **0.528** | 0.639 | 24.2 | 2.5–5 | 100% |
| cosine + (d) | 0.66M | 0.994 | 0.245 | 0.509 | 0.565 | 25.1 | 4.5–8.5 | 100% |

Per-seed test R@5:
- bilinear: 0.54 / 0.40 / 0.50
- MLP: 0.42 / 0.51 / 0.56
- doc_mlp: 0.50 / 0.54 / 0.54

Per held-out type, R@5 (mean rank), seed 0:

| type | cosine | bilinear | doc_mlp |
|---|---|---|---|
| health (t1_diabetes, migraine_light) | 0 | 0.83 (3.1) | 1.00 (1.7) |
| no_car (no_car, no_license) | 0 | 0.83 (3.2) | 0.50 (7.5) |
| flying (fear, climate) | 0 | 0.33 (29.6) | 0.33 (30.6) |
| toddler (toddler, newborn) | 0 | 0.17 (57.3) | 0.17 (57.0) |

- Results depend on the chat, not the question. A chat's rank is nearly the same for all 3 questions of its type. With doc_mlp, the newborn chats sit at ranks 148 / 33 / 4 and the toddler chats at 99 / 50 / 8.
- The question-independent scorer (d), "is this a lasting personal constraint?", is the best of the four. So for this task the question adds little beyond what the document vector says on its own.
- Cosine ranks the clue *below* chance (mean 95.9 vs 75.5). The constraint clause makes a chat look less like a generic question.

Sam's real clue chats (150-chat user, doc_mlp seed 0): getaway 4, parents_dinner 2, gym_time 3, control 1, against cosine's 71, 103, 12, 1.
- This is not a clean held-out test: dog_carsick, vegan and bakery_3am are train values, and Sam's half-A fillers were training negatives.
- It does show the synthetic data transfers to the real chats.

Learning curve (extra, no new data): held-out R@5 by the number of train types seen. Random type subsets, 4 repeats each; the full 8 types is a single run.

| train types | 2 | 4 | 6 | 8 |
|---|---|---|---|---|
| doc_mlp R@5 | 0.41 | 0.48 | 0.49 | 0.50 |
| bilinear R@5 | 0.34 | 0.38 | 0.41 | 0.54 |
| doc_mlp R@10 | 0.52 | 0.58 | 0.59 | 0.63 |
| bilinear R@10 | 0.47 | 0.50 | 0.52 | 0.71 |

Still rising at 8 types, so more clue types would probably help, but the slope does not suggest a big jump.

Reading:
- The clue is clearly in the frozen encoder vectors: 100% clue-over-twin, and median rank 2.5–5 among 150.
- Generalizing to unseen clue types against 149 distractors falls short of the bar: R@5 0.53 vs 0.6.
- The plan says to skip to step 4 on a fail, because "the frozen encoder has lost the clue". That explanation is contradicted here, so steps 2 and 3 were still run. Step 2 does not depend on step 1 anyway, and it turned out to be the decisive step.

## Step 2: delivery through the frozen core (oracle curve)

- 40 held-out pairs (5 per value, stratified, seeded). Documents are the clue chat + k random fillers from the same user.
- Greedy decoding with 96 new tokens. The 6 core conditions are batched in one `generate` call.
- Checked first: the padded, batched core output matches a single unpadded run to 1.5e-5.
- Text ceiling and no-memory were rerun with 320 tokens. At 96 tokens, plain Qwen3-8B is still in its preamble ("Sure! Here's a…"); the 96-token numbers are in `oracle.json`.

| condition | acts (strict) | mentions (strict) | acts (substring) | mentions (substring) | clue attention ÷ uniform |
|---|---|---|---|---|---|
| clue only (k=0) → core | 0.125 | **0.000** | 0.125 | 0.000 | 1.00 |
| clue + 4 → core | 0.075 | 0.000 | 0.075 | 0.000 | 1.45 |
| clue + 16 → core | 0.175 | 0.000 | 0.175 | 0.025 | 2.27 |
| clue + 64 → core | 0.125 | 0.000 | 0.150 | 0.025 | 3.88 |
| clue + 149 → core (today) | 0.150 | 0.000 | 0.250 | 0.050 | 4.79 |
| counterfactual, no clue, 150 → core | 0.100 | 0.000 | 0.200 | 0.050 | — |
| no memory (320 tokens) | 0.225 | 0.000 | 0.300 | 0.075 | — |
| **text ceiling: clue chat as text (320 tokens)** | **0.550** | **0.450** | 0.550 | 0.500 | — |

Paired against the counterfactual (strict `acts`, same pair): the clue set acts where the no-clue set doesn't, minus the reverse.

| k | 0 | 4 | 16 | 64 | 149 |
|---|---|---|---|---|---|
| clue-only wins | 0.075 | 0.025 | 0.100 | 0.075 | 0.050 |
| counterfactual-only wins | 0.050 | 0.050 | 0.025 | 0.050 | 0.000 |

Per type, strict (acts / mentions):

| type | clue only → core | clue + 149 → core | counterfactual | text ceiling | no memory |
|---|---|---|---|---|---|
| health | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.5 / 0.5 | 0.0 / 0.0 |
| toddler | 0.0 / 0.0 | 0.0 / 0.0 | 0.0 / 0.0 | 0.6 / 0.8 | 0.0 / 0.0 |
| no_car | 0.3 / 0.0 | 0.3 / 0.0 | 0.1 / 0.0 | 0.9 / 0.1 | 0.6 / 0.0 |
| flying | 0.2 / 0.0 | 0.3 / 0.0 | 0.3 / 0.0 | 0.2 / 0.4 | 0.3 / 0.0 |

The `acts` hits through the core come only from no_car and flying, where "take the train" or "drive" is a common default answer anyway: no-memory scores 0.6 and 0.3 there. For health and toddler, where acting needs the specific fact, the core path scores 0 at every k.

Does the fact survive the core? A linear clue-vs-twin probe (logistic regression, trained on train-type pairs, tested on held-out-type pairs; 252 (chat, question) items; one document per call):

| representation | dim | clue scored above own twin, held-out | accuracy at threshold 0 | cos(clue, own twin) | cos(random pair) |
|---|---|---|---|---|---|
| encoder vector | 2560 | 100% | 0.77 | 0.941 | 0.227 |
| core memory tokens, mean of 16 | 4096 | 100% | 0.72 | 0.984 | 0.785 |
| core memory tokens, all 16 | 65536 | 100% | 0.75 | 0.992 | 0.904 |

- The core's output space is much more concentrated: any two inputs land at cosine 0.90.
- But the clue-vs-twin gap relative to the random-pair gap is the same as in the encoder: (1−0.992)/(1−0.904) ≈ 0.08 vs (1−0.941)/(1−0.227) ≈ 0.08.
- So the 16 tokens carry the constraint as linearly readable information, and Qwen3-8B does not turn it into behaviour.
- (The 3 probe seeds gave identical results because the weights start at zero; treat them as one run.)

Example answers (strict scores):

- t1_diabetes, "What should I pack for a long day hike?"
  - clue only → core: "Packing appropriate clothing, water, snacks, and navigation tools is essential for a long day hike." (acts 0, mentions 0)
  - text ceiling: a packing list that includes "personal medications (like insulin for type 1 diabetes)" (acts 1, mentions 1). The clue chat itself is about how pearls form.
- no_car, "I'm moving to a new apartment across town - how should I plan the move?"
  - clue only → core: "For a move, a car is better if you need to transport large items and have limited public transit, while a Mac is better if you prioritize portability…" The chat's *main topic* (Mac vs. Windows laptop for school) leaks through; the constraint ("since I don't have a car") does not.
- newborn, "How should I redecorate my living room?"
  - clue only → core: "Remove most wall art and keep only a few key pieces, because she prefers a clean, uncluttered look that matches her minimalist slides and slideshows." The clue chat is about shortening a slideshow; again the topic gist comes through, the newborn doesn't.
  - text ceiling: "…make it more comfortable, especially with a newborn on board."
- climate_flying, "My cousin's wedding is in Miami next month - help me plan the trip."
  - clue + 149 → core and counterfactual → core give the identical answer: "You suggest flying into Miami, renting a car, and visiting Everglades and Key West, because you like combining scenic drives…"
  - text ceiling: "Since you've stopped flying for climate reasons, consider a train or bus…"

Reading:
- `mentions` at k=0 is 0.000 against a text ceiling of 0.45. That is the plan's "16 tokens + head lose the fact" case, but the probe narrows it down: the 16 tokens still hold the fact.
- What loses it is the step from memory tokens to decoder behaviour: the core's learned output projection plus the frozen Qwen3-8B reading it.
- pv4 answers keep the training-data style: short, often "You suggest X…", and half of them (100 / 200 core answers) give a "because you like / prefer…" rationale. They carry the topical gist and a generic preference; none named a side-constraint.
- The core does attend to the clue more than to an average filler (4.8× uniform at k=149, about 3% of attention), and more than to its own twin (0.3–0.4×). So pooling already favours the clue somewhat; that attention just doesn't reach the answer.

## Step 3: learned selection → frozen core

Setup:
- Selector: the step-1 doc_mlp scorer (seed 0), the plan's "best scorer from step 1" by validation R@5.
- Pairs: the same 40 held-out pairs as step 2.
- For each pair it picks the top m = 4 or 8 of the user's 150 documents, and the frozen core and decoder answer from those alone. Greedy decoding, 96 new tokens.
- The learned top-m is also taken from the counterfactual user, who has no clue chat, as a false-positive control.
- RAG baseline: the same learned top-4 chats as plain text to Qwen3-8B, no core, 320 new tokens.
- Peak memory: lowest system free memory was 16% (watchdog limit 12%). The run took 39 min.

| condition | clue in the selected set | acts (strict) | mentions (strict) | acts (substring) | mentions (substring) | clue attention ÷ uniform |
|---|---|---|---|---|---|---|
| all 150 → core (step 2, today) | 1.00 | 0.150 | 0.000 | 0.250 | 0.050 | 4.79 |
| oracle: clue only → core (step 2) | 1.00 | 0.125 | 0.000 | 0.125 | 0.000 | 1.00 |
| cosine top-4 → core | 0.00 | 0.075 | 0.000 | 0.150 | 0.000 | — |
| cosine top-8 → core | 0.00 | 0.000 | 0.000 | 0.000 | 0.000 | — |
| **learned top-4 → core** | **0.475** | 0.075 | 0.025 | 0.075 | 0.075 | 1.23 |
| **learned top-8 → core** | **0.600** | 0.225 | 0.000 | 0.225 | 0.025 | 1.07 |
| learned top-4, counterfactual user → core | 0.00 | 0.025 | 0.025 | 0.025 | 0.075 | — |
| learned top-8, counterfactual user → core | 0.00 | 0.225 | 0.000 | 0.225 | 0.050 | — |
| RAG: learned top-4 as text (320 tokens) | 0.475 | 0.225 | 0.075 | 0.275 | 0.100 | — |
| RAG: cosine top-4 as text (320 tokens) | 0.00 | 0.075 | 0.000 | 0.075 | 0.000 | — |

Paired against the counterfactual user (strict `acts`): learned top-4 is +0.05 (clue-only wins 0.05, counterfactual-only 0.00); learned top-8 is +0.00.

Split by whether the selector caught the clue (strict):

| condition | clue selected: n, acts, mentions | clue missed: n, acts, mentions |
|---|---|---|
| learned top-4 → core | 19, 0.158, **0.000** | 21, 0.000, 0.048 |
| learned top-8 → core | 24, 0.208, **0.000** | 16, 0.250, 0.000 |
| RAG: learned top-4 as text | 19, 0.263, 0.158 | 21, 0.190, 0.000 |
| (same 19 pairs) clue chat alone as text, 320 tokens | 19, 0.579, 0.421 | — |

Per type, strict acts / mentions (fraction with the clue selected):

| type | learned top-4 → core | learned top-8 → core | learned top-8, counterfactual | RAG learned top-4 |
|---|---|---|---|---|
| health | 0.0 / 0.0 (1.0) | 0.0 / 0.0 (1.0) | 0.0 / 0.0 | 0.2 / 0.2 (1.0) |
| no_car | 0.3 / 0.0 (0.5) | 0.6 / 0.0 (0.8) | 0.6 / 0.0 | 0.7 / 0.0 (0.5) |
| flying | 0.0 / 0.1 (0.2) | 0.3 / 0.0 (0.3) | 0.3 / 0.0 | 0.0 / 0.1 (0.2) |
| toddler | 0.0 / 0.0 (0.2) | 0.0 / 0.0 (0.3) | 0.0 / 0.0 | 0.0 / 0.0 (0.2) |

**Step 3 fails the plan's pass bar:**
- The bar: learned top-m → core reaches half the oracle's `acts` and clearly beats both "all 150" and "cosine top-m".
- Learned top-4 is at 0.075. That is nominally above half the oracle (0.0625), but the oracle itself is at counterfactual level.
- It ties cosine top-4 (0.075) and is below all 150 (0.15).
- Learned top-8 (0.225) equals its own no-clue counterfactual (0.225), so it is no better than having no clue at all.

Reading:
- **The selector works; the core path doesn't use what it selects.**
  - Learned selection gets the clue into the set 47.5% (top-4) and 60% (top-8) of the time; cosine gets it 0%.
  - On the pairs where the clue *was* selected, the core's answers mention the constraint 0 times out of 19 (top-4) and 0 out of 24 (top-8), the same as the step-2 oracle.
  - Health is the clearest case. The selector caught the clue in 10 / 10 pairs, and the core path scored 0 / 0 on all of them.
- **The `acts` hits through the core are almost all no_car "hire a moving company" answers.**
  - The counterfactual user gets the same answers at the same rate (0.6 at top-8), so they are a default answer, not recall.
  - One answer looks like real use ("research nearby public transit, then schedule a moving company…").
- **Even as plain text, 4 chats dilute the clue.**
  - With the clue in the 4 chats, Qwen3-8B mentions it 16% of the time.
  - For the same 19 pairs, the clue chat on its own gets 42%.
  - So a top-4 RAG over these chats would also need a stronger decoder prompt or a better selector. That is a side finding; it does not change the core verdict.

Example answers (strict scores), learned top-4 with the clue selected:
- t1_diabetes, "What should I pack for a long day hike?"
  - learned top-4 → core: "Pack comfortable shoes, water, snacks, a map, and a small first-aid kit, because she values practical gear and safety…" (acts 0, mentions 0)
  - counterfactual top-4 → core: "Light layers, sturdy boots, a small first‑aid kit, and a water bottle…" (acts 0, mentions 0)
  - Nothing about insulin or blood sugar in either. The text ceiling for this pair names "insulin for type 1 diabetes".
- no_car, "I'm moving to a new apartment across town - how should I plan the move?"
  - learned top-4 → core: "You would first research nearby public transit, then schedule a moving company…" (acts 1)
  - This is the one core answer that plausibly uses the constraint.
- toddler, "Plan me a relaxing Saturday."
  - learned top-4 → core and the counterfactual give the same answer word for word: "She suggests a quiet morning of reading and tea, a walk in a nearby park…"
- migraine_light, "Suggest a fun night out for this weekend."
  - RAG learned top-4 as text: "…a fun and relaxing night out idea that's gentle on your head… Cozy Movie Night at Home" (acts 1, mentions 1)
  - The text path picks the clue out of 4 chats; the core path never does.

## Follow-up: adapter and LoRA on the decoder side (core still frozen)

Step 2 placed the failure at the decoder end. The user asked for an adapter / LoRA test, since retraining the whole model is not an option. Code: [poc_adapt.py](poc_adapt.py), chain: [run_adapt.sh](run_adapt.sh).

### Setup

- Frozen: Qwen3-Embedding-4B, pv4 `MemoryModule`, and Qwen3-8B base weights.
- Three trained variants:
  - **adapter**: a residual MLP on the core's 16 output tokens (LayerNorm → 4096→256 → GELU → 256→4096, zero-init, gain scaled by token RMS). 2.11M params.
  - **lora**: a hand-written LoRA (no new package) on `q_proj` and `v_proj` of all 36 Qwen3-8B layers, rank 8, alpha 16, dropout 0.05. 3.83M params.
  - **both**: adapter and LoRA together. 5.9M params.
- **Teacher targets:**
  - Qwen3-8B reads the clue chat *as text* and answers the question in one or two sentences. It samples 4 answers at T=0.8 and keeps those that pass both strict rubrics.
  - The twin chat's greedy answer is the negative target, which teaches the model not to invent a constraint.
  - Yield on train pairs: 61 / 120 pairs got targets (147 answers); 118 / 120 twin negatives were usable.
  - Yield on val pairs (chat j=2 of the train types): 29 / 60 pairs (67 answers).
  - The targets are skewed: vegan 14, halal 15, tight_budget 13, debt 12, microwave 12, but dog_reactive 1, wheelchair 1, lactose 2.
  - The teacher sometimes misattributes the constraint, e.g. "Since your parents might not be vegan…".
- **Training:**
  - Each example: the clue chat plus k ∈ {0, 4, 16} random filler chats → core → 16 memory tokens → decoder. Loss is on the answer tokens only.
  - batch 1, accumulation 4, AdamW (lr 3e-4 adapter, 2e-4 LoRA), gradient clip 1.0.
  - Early stop on validation loss of the clue targets, patience 3, evaluated every 15 updates.
  - Every run early-stopped at update 120, after ~10 min each on the Mac.
  - Lowest free memory during these runs was ~15%.
- **Evaluation:**
  - Held-out clue types (health, no_car, toddler, flying): 40 rows per condition.
  - Plus 20 val pairs from the train types, given the clue chat alone and the twin chat alone.
  - Same strict rubrics and generation settings as steps 2–3 (greedy, 96 new tokens).
  - The **base** row reruns the untrained stack under the same conditions, so every variant is compared like for like.

Validation loss on the clue targets, starting from 2.16 untrained:

| variant | best val loss |
|---|---|
| adapter | 0.982 |
| lora | 0.967 (update 75) |
| both | 0.942 (update 75) |

### Results: strict `acts` / `mentions`

| condition | base | adapter | lora | both |
|---|---|---|---|---|
| held-out: clue only → core | 0.125 / 0 | 0.125 / 0 | 0.075 / 0.025 | 0.05 / 0 |
| held-out: clue + 4 → core | 0.125 / 0 | 0.075 / 0 | **0.15 / 0.075** | 0.075 / 0 |
| held-out: clue + 149 → core | 0.15 / 0 | 0.125 / 0 | 0.025 / 0 | 0 / 0 |
| held-out: learned top-4 → core | 0.125 / 0 | 0 / 0 | 0.1 / 0 | 0 / 0 |
| held-out: counterfactual 150 (no clue) | 0.10 / 0 | 0.175 / 0 | 0.025 / 0 | 0 / 0 |
| held-out: learned top-4, counterfactual | 0.15 / 0 | 0 / 0 | 0.15 / 0 | 0.025 / 0 |
| **val types: clue only → core** | 0.05 / 0.05 | 0.10 / 0.10 | **0.25 / 0.25** | **0.30 / 0.25** |
| val types: twin only → core (false alarm) | 0 / 0 | 0 / 0 | 0 / 0 | 0.05 / 0.05 |

n = 40 per held-out row and 20 per val row, so one answer is 2.5 or 5 points. Differences under ~10 points are noise.

Examples, all from the 16 memory tokens with no chat text in the prompt:
- lora, held-out `t1_diabetes`, clue + 4: "Since you're on medication for diabetes, … monitor your blood sugar before and after runs".
- lora, val, clue only: "Since you're allergic to nuts, you could prepare … a vegetable stir-fry"; "Since you're allergic to pollen, plan your picnic on a day with low pollen counts…".
- both, val, clue only: "Since you mentioned you're allergic to dairy, I'd suggest a dairy-free … lentil stew". The clue is lactose intolerance; the core tokens carry the topic and the model rounds it to the nearest trained phrasing.
- both, failure modes:
  - It inverted one constraint: "Since you mentioned you can't eat halal food…". The substring rubric counted it as a hit.
  - It invented an allergy on a twin: "Since you're dealing with a cat allergy…". This is the one false alarm.
- adapter: picked up the teacher's style ("I'd recommend…", one-pan meals) but never the clue on held-out types.

### What it shows

1. **The decoder side was a real bottleneck.** The same frozen core tokens that gave 1 / 20 mentions on val types gave 5 / 20 with a 3.8M-param LoRA. The fact was in the tokens (as the step-2 core probe said), and a small LoRA can learn to read it out.
2. **It learned the trained types, not "read the memory" in general.** On held-out types LoRA reaches at most 3 / 40 mentions (clue + 4), and nothing elsewhere. The training set was 147 answers over 8 constraint types, so the model learned those 8.
3. **Reshaping the memory tokens alone (adapter) is not enough.** The adapter fits the teacher's style but not the clue. Training the reader (LoRA) is what moved the numbers. Adding the adapter to LoRA (both) did not help on held-out types and introduced one false alarm.
4. **Dilution is untouched.** With all 150 chats, every variant is at 0 mentions, and acts falls to the no-clue level or below. At 150 documents the clue gets a small share of the core's pooling, and no reader can recover what isn't in the tokens. Fixing the full-history case needs a trained pooling/selection step (step 3's selector, or a core retrain with clue supervision), not only a better reader.
5. **Caveats:** small n; teacher targets skewed toward 5 values; early stopping on a 67-answer val set; one seed per variant.

### How much data a general reader would need (estimate, not measured)

- The run above suggests that how many distinct constraint types the training covers matters more than how many answers there are. Step 1's retrieval curve also kept rising with each added train type.
- Comparable systems that teach a frozen or LoRA'd LLM to read compressed tokens all train a general reading stage first, then task-tune. ICAE does it with autoencoding then ~240k instruction examples; xRAG does paraphrase pretraining then self-distillation.
- For Memorilla that means:
  - **stage A**: restate one chat's facts from its 16 tokens. Self-supervised, ~10k–50k chats or more.
  - **stage B**: constraint-aware answers over ~30–50 clue types, ~1k–5k answers plus twin negatives.
- At the ~5 teacher answers per minute measured here, 10k targets is about 30 hours on the Mac, or about an hour on a rented GPU.
- Cheapest way to measure it: retrain the LoRA on 2 / 4 / 6 / 8 train types (~10 min each) and plot held-out mentions against the number of types. Not run yet.

## Caveat

The step-3 selector is a learned retriever trained with clue labels.
- Its partial success (47.5% / 60% clue-in-set on unseen clue types) shows the signal is in these vectors and partly learnable from them.
- It does not show that the paper's answer-only training loss would find that signal. That is why any retrain recipe should add clue supervision.
- For the frozen core path the selector question is moot. Steps 2 and 3 both show that the core → decoder path ignores the clue even when it is the only document, or one of 4.

## Rough GPU cost of the next experiment (computed, not measured)

The decision-table row that asks for a cost estimate (row 1) is not the outcome. The next experiment would be the row-3/row-4 one:
- retrain the core end to end (194M) through the frozen Qwen3-8B, on pv4 data plus synthetic buried-clue data with constraint-aware target answers;
- add a clue-selection auxiliary loss and/or text-teacher distillation.

Its cost, worked out from the numbers we have:
- Tokens per example ≈ 54 (question) + 36 (answer) + 16 (memory) + ~40 (system/template) ≈ 150.
- The frozen 8.2B decoder needs a forward pass (2N FLOPs/token) and a backward pass for activation gradients only (≈ 2N, since no weight gradients are taken). That is ≈ 4 × 8.2e9 × 150 ≈ 4.9e12 FLOPs per example.
- The core itself (194M, 2 cross-attention layers over 200 docs) is under 1% of that.
- pv4 recipe: 15,058 train rows × 5 epochs = 75k examples ≈ 3.7e17 FLOPs.
- Synthetic clue data at, say, 10k examples × 5 epochs adds ≈ 2.5e17. For distillation, the teacher's text answers are generated once beforehand and are not counted here.
- Total ≈ 6e17 FLOPs: ≈ 0.4 H100-hours at 40% MFU (989 TFLOPS dense bf16), ≈ 1.7 H100-hours at 10% MFU. The paper's 4×H100 setup would finish in well under an hour.
- Training compute is not the constraint. The real costs are generating and checking the synthetic constraint-aware answers, and LoRA on the decoder if the core alone cannot steer it (which adds weight gradients for the adapter only).

## Reproduce

From `memorilla-port/`. Each step runs as its own process under the memory watchdog (log: `work/logs/poc_<step>.log`, memory: `work/logs/poc_mem.log`):

```bash
. experiments/env.sh && G=experiments/poc_guard.sh
$G gen gen && $G build build && $G embed embed
$G probe probe && $G curve curve && $G coreprobe coreprobe
$G oracle oracle && $G ceiling ceiling && $G select select && $G rescore rescore
```

Adapter / LoRA follow-up (teacher targets, then train + eval of adapter, lora and both, then the base eval):

```bash
. experiments/env.sh && SCRIPT=poc_adapt.py INTERVAL=5 experiments/poc_guard.sh teach teach
caffeinate -i experiments/run_adapt.sh > work/logs/run_adapt.log 2>&1
```

Wall clock for the follow-up: teach ~75 min (with a pause while the laptop slept), each train ~10 min, each eval ~6–10 min.

Wall clock on the M-series Mac (rerun): generation 13 min, embedding 3 min, probe 5 min, curve 4 min, core probe 0.5 min, oracle 15.5 min, ceiling 6 min, select 39 min.
