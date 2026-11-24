# Memorilla memory-token path: handoff report (2026-11-24)

This report is written so that someone new, or the owner a few weeks from now, can pick the work up cold. It says what the project is, how it is measured, what has been tried and what worked. It also covers how to run everything, where every file is, the rules, the known traps, and what to do next.

Full per-test tables are in `experiments/AGENTMEM_HYPOTHESES.md` (tests 1–44). Box procedures are in `experiments/HANDOFF_TOKENS.md`. Host names, user names and ports are deliberately left out of this document; they are in `experiments/v100.sh` on the Mac.

AI use: Claude (as a coding agent in Claude Code) designed and ran most experiments, wrote most of the code and wrote this report, under the owner's direction. A local Qwen model (via omp) wrote some routine code from specs. Benchmark clue chats and all teacher labels are model-generated.

---

## 0. Where things stand (read this first)

- **Goal.** Make Memorilla's 16-token memory act on the one stored fact about a user that should change an answer (the buried clue), on fact types it never trained on.
- **Best result so far, off the token path.** The 27B model rates each fact as text, and the 8B answers. Held-out "acts": 0.68 at 8 facts and 0.53 at 48 (test 41). A perfect picker would give about 0.80.
- **Best result on the token path.** A rater LoRA reads each fact's 16-token block. Acts: 0.44 for top 1 of 8, 0.30 for top 4 of 48 (test 43). It is capped by its 8B teacher.
- **Two bottlenecks:**
  1. Picking at 48+ facts. The clue comes first of 48 in 0.35 of profiles for the 27B, 0.16 for the 8B, and 0.09 for the block rater.
  2. The reader. It acts on the clue 0.76–0.79 when the clue is the first block, but only 0.40–0.60 when it is shown further down.
- **Last thing done.** A GRPO (RL) pilot on the reader (test 44). It was stopped at step 19 of about 50 with no learning signal yet. That is inconclusive, not negative.
- **Next action, if a GPU box window is available:** check 4c (§6.1). It is fully prepared.
- **Next action, if only the Mac is available:** either evaluate the saved GRPO checkpoint for damage (§6.2, 40 min), or run the write-time salience check (§6.3).
- **Nothing is running.** No jobs are queued on the Mac or the box.

### Start here (a new person, first hour)

1. Read §1–§3 of this report. Then skim test 41 and test 43 in `AGENTMEM_HYPOTHESES.md`.
2. On the Mac: `cd ~/memorilla-port && . experiments/env.sh && . experiments/light.sh`.
3. Smoke check with no model: `$PY -I experiments/poc_tok.py dry` and `$PY -I experiments/poc_grpo.py dry`.
4. Pick a next step from §6. Read §8 (rules) and §9 (traps) before running anything.

---

## 1. Background

### 1.1 What Memorilla is here

- **Memorilla.** Upstream code is in `work/repo/memorilla` (snap-stanford/memorilla, commit 6864db7). It reads a document collection and gives a frozen LLM **16 query-dependent memory tokens**.
- **Core.** The "pv4" `MemoryModule` for Qwen3-8B, at `work/ckpt/pv4` (370 MB, about 194M parameters):
  - 16 learned slots cross-attend to the request's embedding;
  - then 2 cross-attention blocks over the document embeddings;
  - then a projection from 2560 to 4096 dimensions, into the decoder's embedding space.
- **Encoder.** `Qwen/Qwen3-Embedding-4B` (2560-d vectors). Every fact sentence and every request is embedded once, and the vectors are cached in `*_emb.npz`.
- **Decoder.** `Qwen/Qwen3-8B`, with the "native" head. The memory tokens replace `<|memory|>` placeholder embeddings in the prompt.
- **The token path (what this project is about).** Each stored fact goes through the untouched core alone, conditioned on the request, and comes out as one **16-token block** (`poc_profile.blocks`). The k picked blocks are concatenated, giving 16·k memory tokens.
- **Reader.** Qwen3-8B plus a LoRA (rank 16 on q/k/v/o of all 36 layers). The current reader is `work/runs/agentmem/lora_lsort.pt`. It was trained on blocks sorted best-first, and it first writes `Relevant fact: …`, then the answer.
- **Frozen.** The core, the encoder and the decoder base weights. Only LoRAs, small maps and probes are trained.

### 1.2 The pipeline being optimised

```
stored facts (profile of 8 / 16 / 48 / 192)
   │  each fact + request ──► frozen core ──► 16-token block
   ▼
picker: rate each block alone ("does this fact change a good reply?") ──► keep top k
   ▼
reader (8B + LoRA) with k blocks in memory ──► "Relevant fact: …" + answer
```

### 1.3 Glossary

| Term | Meaning |
|---|---|
| clue | the one stored fact that should change the answer |
| profile | the clue plus 7, 15, 47 or 191 other users' facts (distractors) |
| block | the 16 memory tokens of one fact |
| rater / picker | scores each fact for a request; the top k go to the reader |
| text rater | a decoder rating the fact given as plain text (off the token path) |
| block rater | a decoder plus LoRA rating the fact from its block (on the token path) |
| acts | the regex rubric says the answer acts on the clue (the main metric) |
| held-out / unseen types | health, no_car, toddler, flying: never trained on |
| light mode | the Mac setup with an int8 decoder and capped memory (§7.2) |
| the box | the V100 server with 2 × 32 GB (§7.3) |

---

## 2. Benchmark and metric

- **Config.** `experiments/poc_config.json` has 12 types:
  - diet, pet, schedule, mobility, alcohol, budget, studio, hay_fever (trained);
  - health, no_car, toddler, flying (`test_types`, held out).
  - Each type has 3 questions, 2–4 values, and hand-written `acts` and `mentions` rubrics. They are deliberately not LLM-generated.
- **Pairs.** `work/runs/poc/synth.json` has 252 pairs (train 120, val 60, test 72). A pair id is `value:j:qi`.
- **Eval set.** `poc_broad.test_set(big=True)` gives 318 rows:
  - 288 held-out-type questions: 72 test pairs, plus 9 extra questions (`XQ`) for each held-out clue;
  - 30 trained-type val questions.
  - Distractors come from other users' facts of other types, seeded by the pair id (`poc_relevance.eval_sets`).
- **Stride.** `--stride 3` (the Mac standard) gives 106 rows: 96 held-out and 10 trained-type. `--stride 12` is a smoke run.
- **Metric.** `strict_pass` in `poc_clue.py`: every `all_of` group needs a word-boundary regex hit, and no `none_of` term may appear. "acts" is scored on the text after the `Relevant fact:` line.
- **Floors.** flying and no_car pass in about 20–36% of answers with no clue at all. health and toddler are the clean types, so watch them.
  - Every answer table also has a clue-removed control.
  - About 2% of rows contain another row's clue, so "clue first" tops out near 0.98.
- **Noise.** One seed everywhere. At stride 3, one standard error is about 5 points at acts 0.4. **Differences under about 10 points on the Mac are not conclusive.** At stride 1 (288 held-out questions), one question is worth 0.35 points.

---

## 3. Results

### 3.1 The ceiling and where we are (acts, held-out types)

| Condition | 8 facts | 48 facts | Source |
|---|---|---|---|
| Clue alone (perfect picker) | 0.80–0.83 | – | test 41 |
| Whole profile as text, plain decoder | 0.53–0.58 | 0.32–0.35 | tests 31, 41 |
| 8B text rater, best top k | 0.60 | 0.47 | tests 38b, 41 |
| **27B text rater ("wrong" wording), best top k** | **0.68** | **0.53** | test 41 |
| Block rater (16 tokens, LoRA), top 1 / top 4 | 0.44 | 0.30 | test 43 |

Where the rater ranks the clue:

| Rater | Top 2 of 8 | First of 48 | Top 8 of 48 |
|---|---|---|---|
| plain decoder on the block, "change" | 0.46 | – | 0.38 |
| block rater, 90 min int8 (`tokL90q8_change`) | 0.60 | 0.09 | 0.55 |
| 8B text rater (the block rater's teacher) | 0.75 | 0.16 | 0.72 |
| 27B text rater, "wrong" | 0.84 | 0.35 | 0.82 |

The 27B's "wrong" wording is: "Could a typical reply to this request be wrong, unsafe or unsuitable for this user because of the fact? Answer Yes or No."

### 3.2 Picking: what was tried

| Test | Idea | Result |
|---|---|---|
| 17–22, 26 | Learned map (request × fact embedding) | Good on topical types (flying, no_car). Fails health and toddler. 2.7× more pairs: flat |
| 30 | Write-time "standing" score, request-independent: a ridge fit on hand-written category lists, or label-free gap/spread | Hand-written lists fix health at 8–16 facts, but only for what the list names. Label-free scores don't find health. Not a blind test |
| 32 | The 8B quotes which facts matter (no training) | False: quotes the clue in 49% at 8 facts and 22% at 48 |
| 35b, 35c | **Rate each fact alone with a yes/no, keep the top k** | **True**: beats the map and the whole profile. Wording matters per type |
| 36 | Distil the rating into an embedding scorer | False: below plain cosine on held-out types |
| 37, 38, 38b | Attention as the selector; rating in one pass per request | False as a substitute: rates by topic in a crowd. Attention + rating combined: top 8 of 48 at 0.48 acts |
| 39 | Shortlist, then think or compare | Partly: +2 points at 5× the cost. Comparing in a group loses health |
| 40 | LoRA rater, listwise, hard labels | **False**: learns the training types; health falls to 0 at 192 |
| 41 | **27B rates each fact as text** | **True, the biggest gain.** Lifts every type. An 8B shortlist with 27B re-rating keeps most of it at 25–33% of the cost |
| 43 | Rate from the 16-token block: plain decoder; draft-then-judge; LoRA distilled from the 8B text rater (45 min bf16, then 90 min int8) | Picking improves, but the student is capped by its teacher. Answers gain 2–6 points. Not promoted (bar: first of 48 ≥ 0.20, top 1 of 8 ≥ 0.45) |
| 43 | Abstain (skip memory) when the top score is low | False: AUC 0.51–0.61 for the 8B, 0.59–0.75 for the 27B |

### 3.3 Reading: what was tried

| Test | Idea | Result |
|---|---|---|
| 19, 23, 27, 29 | Restate the relevant fact first, then answer; blocks sorted best-first; trained with distractors | Became the current reader, `lora_lsort.pt` |
| 31 | No picker: whole profile as text, reader trained to restate | The plain decoder with the whole profile beat the picker of that time (0.58 against 0.39) |
| 33 | Self-distillation: the fact-only decoder teaches the whole-profile one | False: it learned the teacher's style |
| 34 | Token weights (learn only the tokens the fact changes) | No gain over the plain decoder |
| 44 | **GRPO on the reader** (on-policy RL, rubric reward, no labels) | Stopped at step 19, no signal yet (§5) |

### 3.4 Lessons

1. **A stronger general judge beats any narrow trained one** (test 41 against tests 36 and 40). Training on 8–100 generated types teaches those types, not "what matters".
2. **Rate each fact alone.** In a crowd, a model rates by topic, and that is where health is lost (35b against 32, 37, 38).
3. **Distillation is capped by its teacher** (43). Longer training on the same teacher adds 1–6 points.
4. **The reader is its own bottleneck** (41, 43, 44). It depends heavily on the clue being the first block, and it restates and follows wrong picks that look relevant.
5. **Light mode is good enough for screening.** Int8 ratings match bf16 at Spearman 0.993, and the first answers are identical word for word.

---

## 4. Infrastructure built in this round

- **Light mode** (`light.sh`, `q8.py`):
  - `MPORT_Q8=1` makes every Linear in the decoder layers int8 weight-only, using torch's `_weight_int8pack_mm` in a custom autograd function. Backward goes to the input only, so a LoRA still trains.
  - It is built on the meta device and filled tensor by tensor from safetensors, read from the local HF cache only. It never downloads.
  - The MPS cap is about 15 GB (watermarks 0.55/0.45), with 4 CPU threads and `MIN_FREE=25` for the guard.
  - Footprint about 12–14 GB instead of 20 GB. Generation is faster than bf16 (58 s against 67 s per batch).
- **Rater tooling:**
  - `poc_tok.py --tag` names the score tables.
  - `poc_tokrater.py --data` chooses the teacher file.
  - `poc_rate27train.py`: the 27B re-rates the rater's training pairs (for check 4c).
- **RL tooling:** `poc_grpo.py` (dry / train) and `run_mac_grpo.sh`.
  - Sampled generation from memory blocks, the rubric reward, Dr. GRPO.
  - Gradient checkpointing, and logits for the answer positions only, to fit 15 GB.

---

## 5. The GRPO pilot (test 44) in detail

- **Why RL.** It removes the teacher cap: the reward is what the benchmark measures. The reader was chosen over the rater because rewarding the rater mostly means "picked the known fact", which is test 40's signal (it failed on new types).
- **Data.** Only the 120 trained-type pairs. Held-out types and val pairs are never seen.
- **Contexts.** 1, 2, 4 or 8 blocks (weights 1:2:2:1). With p 0.75 the clue is in the context at a random place. The other facts are other users' facts of other types, half of them look-alikes by cosine to the request.
- **Reward.**
  - Clue shown: 1 if the answer acts on it.
  - Clue not shown: 1 if the answer does not act as if it were there.
  - Scored after the `Relevant fact:` line, as in `poc_tok.py answer`.
- **Training.** 8 contexts × 4 samples per step at temperature 1 (no top-k/top-p). Advantage = reward − group mean, no std division and no KL. One update per batch, so the policy ratio is 1. AdamW, lr 2e-5, clip 1.0, 200 new tokens.
- **Speed.** About 1.8 min per step on the Mac in light mode: about 55 s generating, the rest backward.
  - The first attempt ran out of memory at step 4.
  - It was fixed with gradient checkpointing, freeing the generation cache, and logits for the answer positions only.

| Steps | Reward, clue shown (acts) | Reward, clue not shown | Groups with any spread |
|---|---|---|---|
| 1–10 | 0.28 | 0.89 | 3.1 / 8 |
| 11–19 | 0.29 | 0.87 | 3.2 / 8 |

- **Reading.**
  - No sign of learning in 19 steps, but that is only about 600 samples.
  - Only about 40% of groups give a gradient (the others: all 4 samples ignore the clue).
  - Per-step noise is ±0.2.
  - Stopped by the owner, before the held-out answer test.
- **Baseline observation.** With the clue at a random place among up to 8 blocks, the reader acts on it in about 28% of samples. That is lower than 0.40–0.60 (shown, not first) and 0.76–0.79 (shown first) in test 43.
  - Confounds: temperature-1 sampling against greedy, and 200 tokens against 300.
  - It fits the reader leaning on the first block, since it was trained best-first.
  - A greedy run on the same contexts would separate the two effects.
- **Saved.** `work/runs/agentmem/lora_grpo_s10.pt` (step 10), not yet evaluated. Log: `work/logs/poc_grpo.log`.

---

## 6. What to try next (ranked)

### 6.1 Check 4c: a 27B teacher for the block rater (prepared; needs a box window)

The cheapest test of whether the teacher cap is the whole story. If it passes, the 27B's judgement survives distillation onto 16-token blocks, and the product keeps a single 8B at inference.

1. `rtk proxy experiments/v100.sh gpus`, to check that the GPUs are free and that it is your window.
2. `experiments/v100.sh serve27`. This is the 27B, read-only, on both GPUs.
3. Run `poc_rate27train.py --requests 1000` on the box, giving about 25k pairs in `work/runs/poc/rater_train27.json`.
   - It needs `MPORT_VLLM_URL=http://127.0.0.1:<serve27 port>`, `MPORT_VLLM_MODEL=qwen3-27b` and `MPORT_RATER_TOK=<27B model dir on the box>`.
   - Pass them through `E='…' experiments/v100.sh run …`, because v100.sh rejects spaces in arguments.
4. `experiments/v100.sh unserve`
5. `poc_tokrater.py --data rater_train27.json --minutes 45 --out lora_tokrater27.pt`
6. `poc_tok.py rate --lora lora_tokrater27.pt --raters tokL --tag 27 --wordings change`, then `poc_tok.py answer --tables tokL27_change --conds "top+1+from+8,top+1+from+48,top+2+from+8,top+4+from+48"`
7. `experiments/v100.sh pull runs/poc/<files>`, then release the window: `touch ~/<hold dir>/release` (see `HANDOFF_TOKENS.md`).

**Promotion bar:** the clue first of 48 ≥ 0.20, and top 1 of 8 acts ≥ 0.50.

### 6.2 GRPO v2 on the reader (a box job)

These changes target the weak signal in v1:

- **More samples per context:** G = 8–16, so more groups have spread.
- **A denser reward:** partial credit when the `Relevant fact:` line restates the clue (`PF.mentions(note, rubrics)`), even if the answer doesn't act.
- **Penalise following a wrong pick:** with the clue not shown, reward a `Relevant fact: none` note.
- **Curriculum:** the clue first or second at the start, then random places.
- **Scale:** 32 contexts per step and 300+ steps, about 10k samples. Use vLLM-style sampling on the box.
- **Evaluate checkpoints** on the val pairs, then run the held-out answer test for the best one. The clue-removed control must not rise.
- **Risk:** the reward comes from 8 narrow types, the same narrowness that sank tests 36 and 40. A reader-side skill may transfer better than a rater-side one, but only the held-out test can tell.
- **Cheap first step on the Mac** (about 40 min): rule out damage with
  `poc_tok.py answer --lora lora_grpo_s10.pt --tables tokL90q8_change --conds "top+1+from+8,top+1+from+48,top+2+from+8,top+4+from+48" --stride 3 --out grpo_answer.json`, and compare with `tokL90q8_answer.json`.

### 6.3 Write-time salience judged by a model (Mac, no training)

Test 30 showed that an always-on set of "facts that matter whatever the topic" fixes health. But its scores came from hand-written category lists, and the label-free scores failed.

- **Untried version:** ask a model once per fact, at write time, how broadly the fact matters. Keep the top few always in context, and fill the rest with the per-request rating.
- **On the token path:** the 8B reads the block, about 30–45 min on the Mac.
- **The 27B on text** would leave the token path and needs the owner's approval.

### 6.4 A System-1 judge (Jev-style decision model)

- **Assessment.** The block rater already has this shape: one forward pass, and the Yes/No logits become a score. A small decision model would be faster, not better, unless it is trained on better labels.
- **The open candidate worth a check:** a ~400M bidirectional cross-encoder trained on the 27B labels from §6.1. Options are Von (`wfzyx/von-1.0`) or Laya (ModernBERT-large).
- **Conditions:**
  - It reads text, so it leaves the token path and needs approval.
  - It needs a download: state the file, source and size first.
  - Its code is untrusted.
- **Sources were thin:** one DataCamp article (2026-09-23) and two arXiv papers (2610.18773, 2609.37647) that could not be read. Nothing was verified.

### 6.5 Owed or lower priority

- Abstention trained on purpose (test 43 showed the untrained version fails).
- A fact-update test (an old fact replaced by a new one).
- A general-ability check of the reader LoRA, on against off.
- Django exact names.
- Tighter flying and no_car rubrics, to remove the floor.
- Test 42 (never run).
- Re-rate or delete the old-key score tables (§9).
- Mac housekeeping: `sudo pmset -a disablesleep 0`, if sleep was disabled for overnight runs.

---

## 7. How to run things

### 7.1 Mac environment

- **Hardware:** M3 Pro, 36 GB unified memory, MPS only.
- **Project:** `~/memorilla-port`. Code is in `experiments/` (scripts and docs) and `mport/` (package). Data, results and logs are in `work/`.
- **Venv:** `work/venv`, Python 3.12.13, with torch 2.15, transformers 5.23 and sentence-transformers 6.3. Modal's client has its own venv, `work/venv-modal`.
- **`. experiments/env.sh` exports:**
  - `MPORT_SCRATCH=$PWD/work`
  - `PYTHONPATH` = project + `work/repo/memorilla`
  - `PY=work/venv/bin/python`
  - MPS watermarks 0.8/0.6
  - `TOKENIZERS_PARALLELISM=false`
  - It holds no secrets.
- **Models** are in the default HF cache and loaded with `local_files_only`:
  - `Qwen/Qwen3-8B` (15 GB)
  - `Qwen/Qwen3-Embedding-4B` (7.5 GB)
  - The pv4 core is at `work/ckpt/pv4`.
- **Disk:** `work/` is 4.2 GB. The Mac volume had 32 GB free (93% used).

### 7.2 Running on the Mac (light mode, alongside other work)

```bash
cd ~/memorilla-port && . experiments/env.sh && . experiments/light.sh
SCRIPT=poc_tok.py experiments/poc_guard.sh <name> answer --stride 3 ...   # log: work/logs/poc_<name>.log
# long chains: write a run_mac_*.sh (see run_mac_q3b.sh, run_mac_grpo.sh) and launch detached:
nohup caffeinate -i nice -n 15 zsh experiments/run_mac_grpo.sh > work/logs/run_mac_grpo.out 2>&1 &
```

- **`poc_guard.sh`** runs `$PY -I experiments/$SCRIPT …`. It logs free memory to `work/logs/poc_mem.log` and kills the job below `MIN_FREE`%. The default is 12; light.sh sets 25.
- **One model per process.** Don't run two decoder jobs at once.
- **Measured Mac runtimes** (stride 3):

| Job | Time |
|---|---|
| `poc_tok.py rate` (plain decoder, 2 wordings, 2,841 pairs, bf16) | 36 min |
| `poc_tok.py rate` (rater LoRA, 1 wording) | 15 min bf16 / 18 min int8 |
| `poc_tokrater.py --minutes 45` / `--minutes 90` | 49 min, 121 steps (bf16) / 94 min, 218 steps (int8) |
| `poc_tok.py answer` | 24 min for 636 answers / 33 min for 1,060 (int8) |
| GRPO | about 1.8 min per step |

### 7.3 The V100 box

- **Hardware:** 2 × V100 32 GB. No bf16 (use `MPORT_DTYPE=float16`), no NVLink, no FlashAttention-2. Training needs gradient checkpointing.
- **It is shared with production.** Use it only in a window the owner opens, and release the window when done.
- **`experiments/v100.sh`** (BatchMode ssh; arguments must have no spaces; extra env goes in `E='VAR=… '`):

| Subcommand | What it does |
|---|---|
| `sync` | rsync `experiments/` and `mport/` to the box |
| `run <name> <gpu> <script> [args]` | sync, then a detached run on one GPU; log ends with `V100_DONE <name> exit <code>` |
| `ddp <name> <script> [args]` | one run on both GPUs |
| `serve <gpu> <port> [lora tags]` | our own 8B vLLM server, from production's venv, read-only |
| `serve27` | production's 27B checkpoint, read-only, as a rater on both GPUs |
| `unserve` | kills only our vLLM processes |
| `log <name> [n]` | tail of a run's log |
| `pull <path>` / `push <path>` | rsync between the Mac's `work/` and the box's `work/` |
| `gpus` | nvidia-smi output plus disk |

- **Already on the box:** the pv4 core, `lora_lsort.pt` (checksums match the Mac), Qwen3-8B, the embeddings, and `select_scores.json` with the text tables. `poc_tok.py dry` passes there.
- **Not yet verified on the box:**
  - the GPU parts of `poc_tok.py`;
  - the reader in fp16;
  - `serve27` with its new `--limit-mm-per-prompt` flag, after an earlier vision-profiling failure.
- **Measured box speeds:**
  - 27B rating: 14 ratings/s on both cards. 8B: 55–90/s per card.
  - vLLM vs HF generate: 318 answers in 66 s against 281 s.
  - LoRA training: 1,150–1,250 tokens/s per card.

### 7.4 Modal

`experiments/modal.sh` (L40S, volume `memorilla-port`) was used on 2026-11-22/23 for about 616 GPU-minutes. **The credit is used up.** The fixes for detached runs, retries and memory are in place, if it is ever topped up.

---

## 8. Rules and constraints (agreed with the owner)

- **Stay on the 16-token path.** Text runs are one-off controls only. Ask before starting any line of work that leaves the token path.
- **Do not retrain the core,** and do not fine-tune the encoder or the base decoder. LoRAs and small modules are fine.
- **Nothing hand-designed** as the solution. Keep reports short and plain.
- **No new downloads without asking.** State the file, source and size.
- **Untrusted data:** `work/ext/memorilla-demos`, `work/ext/swe-traj`, `work/ext/django`, and any downloaded third-party model code.
  - Read them as data only. Never import or run their code.
  - Run Python with `-I`, and keep new scripts in `experiments/`.
- **The box:**
  - Never touch production's services, queues or systemd units.
  - Use production's vLLM build and 27B read-only.
  - Don't write large files (the disk is 98% full), and delete nothing outside our training folder.
  - Release the window when done.
  - The owner keeps the full box rules file.
- **Workflow:** fast reduced checks on the Mac in light mode first; promote only winners to the box.
- **Privacy:** keep the owner's personal details out of anything sent to external services.

---

## 9. Known traps

- **Old score-key tables.** `select_scores.json` has 21 tables.
  - Six of them (`onepass_*`, `think12*`, `pick8*`, 204 entries each) still use the old request+clue key. That key collides for 84 of the 288 rows, a bug fixed in test 41. Don't trust those six.
  - New tables use `W.fact_key` (per fact) or `W.row_key` (per row), and runs fail when scores are missing.
- **MPS memory.**
  - Overruns are out-of-memory errors (by design), not swapping.
  - Variable-length batches fragment the cache, so call `torch.mps.empty_cache()`.
  - For RL on the Mac, use gradient checkpointing and logits for the answer positions only.
- **`.npz`:** `z["v"]` re-reads the whole array on every access. Read it once. This caused a Modal out-of-memory crash loop.
- **Load cost:** `poc_broad.load_all` (used by `poc_tok.py` and `poc_grpo.py`) always loads 430 MB of pair embeddings.
- **V100:**
  - fp16 needs a loss scaler.
  - Production's vLLM crashes with `--enable-lora`, so `vllm_lora_patch.py` patches it in memory.
  - The 27B needs images and video set to 0.
- **q8 loader:** it needs the model in the local HF cache, and it rebuilds the rotary embedding after `to_empty`.
- **Stale docs:**
  - `POC_PLAN.md` still puts `MPORT_SCRATCH` under `/private/tmp`.
  - `HANDOFF_TOKENS.md` line 3 says "nothing run on a GPU yet". The Mac checks have run; the box parts have not.
  - Light mode's `nice` is applied on the launch command line, not in `light.sh`.
- **Benchmark quirks:**
  - flying and no_car have rubric floors.
  - "clue first" tops out at about 0.98.
  - The text tables cover only the 288 unseen rows, so ignore their "trained types" lines.

---

## 10. Data and artifacts

`work/runs/poc` (benchmark, scores, teacher data):

| File | What |
|---|---|
| `synth.json` | the benchmark: 371 docs, 252 pairs with rubrics |
| `select_scores.json` | all picker score tables (21) |
| `rater_train.json` | 75,309 (request, fact, 8B score, known) rows: the block rater's teacher |
| `rater_train27.json` | **not made yet** (check 4c) |
| `broad_gen.json`, `broad_long.json`, `broad_pairs*.json`, `profile_gen.json`, `teach.json` | generated training data (80+ categories) and teacher answers |
| `*_emb.npz` | cached encoder vectors (`broad_pairs*_emb.npz` are 145 and 284 MB) |
| `tokL_answer.json`, `tokL90q8_answer.json` | test 43 answer runs (the baselines for any new reader) |
| `rate27*.json`, `rate_sweep.json`, `zero_*.json`, `whole_eval.json` | tests 31–41 outputs |
| `relmap_*.pt` | learned maps (tests 21–26) |

`work/runs/agentmem` (LoRAs):

| File | What |
|---|---|
| `lora_lsort.pt` | **current reader** |
| `lora_long.pt`, `lora_lrec.pt`, `lora_recite.pt`, `lora_broad.pt`, `lora_prof.pt` | earlier readers |
| `lora_tokrater.pt`, `lora_tokrater90q8.pt` | block raters (test 43, checks 4 and 4b) |
| `lora_grpo_s10.pt` | GRPO step 10 (test 44), not evaluated |
| `lora.pt`, `emb.npz`, `sessions.json`, `core_pick.pt` | early agent-session tests 1–13 |

Logs are in `work/logs/` (`poc_<name>.log`, `poc_mem.log`, `v100_<name>.log` on the box).

Key scripts:

| Script | Role |
|---|---|
| `poc_tok.py` | rate / answer / dry on the token path |
| `poc_tokrater.py` | train the block rater |
| `poc_grpo.py` | GRPO |
| `poc_rate27train.py` | 27B teacher data |
| `poc_broad.py` | reader training (`train`), data and eval sets |
| `poc_whole.py` | text-side helpers and keys |
| `poc_clue.py` | benchmark and rubrics |
| `q8.py` | int8 loader |
| `v100.sh`, `modal.sh`, `poc_guard.sh`, `env.sh`, `light.sh` | running |

---

## 11. Document map

| File (in `experiments/`) | Holds |
|---|---|
| this report | the overview, the next steps, how to run |
| `AGENTMEM_HYPOTHESES.md` | every test 1–44 with full tables; Caveats, Modal and RAM sections |
| `HANDOFF_TOKENS.md` | box procedures, box state, check 4c steps |
| `POC_PLAN.md`, `POC_RESULTS.md`, `RESULTS.md`, `AGENTMEM_RESULTS.md` | earlier phases (environment notes partly stale) |
