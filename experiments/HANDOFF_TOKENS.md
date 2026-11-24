# Handoff: move the fact-picking recipe onto the 16-token memory (test 43)

Written 2026-11-23 for the next session. Nothing below has been run on a GPU yet.

## Why this is the next step

- The idea of the project is a model that works from Memorilla's 16-token memory blocks, not from text.
- Tests 31–41 (last night) gave the facts as plain text. They are a control, not the product.
- **Rule from the user:** stay on the token path. A text run is a one-off control. Report token numbers first. Ask before starting any line of work that leaves the token path.
- Also from the user: no core retraining, nothing hand-designed, keep reports short and plain.

## What the text tests found (worth carrying over)

Test: 288 requests on unseen fact types (health, no car, toddler, flying); one fact in the profile should change the answer. Score = "acts".

| Setup (text) | 8 facts | 16 facts | 48 facts |
|---|---|---|---|
| Whole profile in the prompt | 53% | 42% | 35% |
| 8B rates each fact alone, top few kept | 60% | 53% | 47% |
| 27B rates, 8B answers | 68% | 62% | 53% |
| Only the right fact given | 81% | 81% | 81% |

- Rating each fact **alone** with a yes/no works; rating them together, attention, embeddings and every trained picker did not.
- Best wordings: "change" for the 8B, "wrong" for the 27B.
- Token reader so far (`lora_lsort.pt`): acts on a fact alone in its 16 tokens in 64% (text 80%), 32% with the old picker's top 3 of 8.

## What test 43 does

Script: [poc_tok.py](poc_tok.py). No fact is given as text at any step.

1. **Rate from tokens.** Each fact's 16-token block goes into memory, the request and a yes/no question go in the user turn, score = logit(Yes) − logit(No).
   - Two raters: the plain 8B reading the block (tables `tok_change`, `tok_wrong`) and the reader LoRA (`tokL_change`, `tokL_wrong`).
   - It prints where the right fact ranks among 8 / 16 / 48, next to the text raters (`r8_change`, `r27_wrong`).
2. **Answer from tokens.** The reader gets the top 1, 2 or 4 blocks, best first (the order it was trained on).
   - Controls: the right fact alone, all 8 blocks, the right fact removed.

## How to run (V100 box, after a GPU window is open)

One GPU is enough. Rating first, because the answer runs read its tables.

```bash
cd ~/memorilla-port
rtk proxy experiments/v100.sh gpus
rtk proxy experiments/v100.sh run tokrate 0 poc_tok.py rate
rtk proxy experiments/v100.sh log tokrate 40
```

When `tokrate` prints `TOK rate done`, pick the token table with the best "top 2" and start the answers (second GPU optional):

```bash
rtk proxy experiments/v100.sh run toka 0 poc_tok.py answer --tables tok_change --out tok_a.json
rtk proxy experiments/v100.sh run tokb 1 poc_tok.py answer --tables r27_wrong --no-controls --out tok_b.json
```

- `toka`: token rater + token reader. This is the headline number.
- `tokb`: the best text picker (27B) + token reader. It shows how much is lost in reading alone.
- Rough time: rating about 10–20 min (8,565 ratings × 4 tables), each answer run about 30–45 min. These are guesses; nothing was timed.
- Results: lines starting with `TOK` in the logs; files `tok_a.json`, `tok_b.json` in `~/train/work/runs/poc/` (`v100.sh pull runs/poc/tok_a.json`).

## How to read the result

- **Rating:** compare `tok_*` "top 2 / top 4" with `r8_change (text)`. If close, the 16 tokens keep what the rating needs. Text raters, right fact in the top 2 from 8 / 16 / 48:
  - 8B "change": 71% / 51% / 25%
  - 27B "wrong": 84% / 74% / 45%
- **Answering:** compare with the token ceiling (`clue alone`, expected about 64%) and with the text table above.
- If `tokb` is far above `toka`, picking from tokens is the loss. If both sit far under `clue alone`, the reader breaks with more than one block; then try top 1 only, or train the reader (reader only, core frozen).

## State of the box (gpu-box)

- Window **released** at 12:15 box time on 2026-11-23 (`touch ~/<hold-dir>/release`). Production's 27B holds both GPUs. Do not touch it; wait for the user to open a new window.
- Uploaded to `~/train/work/`: `ckpt/pv4` (memory core, 370 MB) and `runs/agentmem/lora_lsort.pt` (59 MB). Checksums match the Mac copies.
- `poc_tok.py dry` passes on the box (no GPU): all 8,565 request–fact pairs have vectors, and its ranking code reproduces last night's text-rater numbers.
- Already there: Qwen3-8B, all embedding files, `select_scores.json` with the text tables.
- Disk: 23 GB free (98% full). Do not write large files.
- Rules for the box are in the user's `box rules file` (do not touch production, production's queue, systemd units; release with `touch ~/<hold-dir>/release` when done).

## Not verified

- `poc_tok.py` has only been checked without a model (`poc_tok.py dry` on the Mac and on the box: rows, prompt, ranking, selection all fine). The GPU parts are untested. Start with `--stride 12` for a 2-minute smoke run if unsure.
- The reader and core have never run in float16 on the V100 (earlier reader runs were on Modal).
- The plain decoder was never asked a yes/no about a memory block before; it may answer near chance.
- `serve27` in `v100.sh` (27B on the 11-19 build) got a new flag and was not restarted.

## Known quirks

- For the text tables (`r8_change`, `r27_wrong`) only the 288 unseen-type rows are rated; ignore the "trained types" lines of `tokb`.
- About 2% of rows hold a second fact that is the right fact for another row with the same request, so "clue first" tops out near 98%.
- The flying and no-car checks fire without the right fact in 24–36% of answers; read health and toddler first.
- 84 of 288 rows share request + clue; score tables must be keyed with `W.fact_key` / `W.row_key` (already done here).

## After test 43

1. If tokens rate well but the reader loses: reader training on 1–2 picked blocks (core frozen).
2. Draft, check, revise on tokens: rate each block against a draft answer, not a hypothetical one.
3. Let the rating decide how many blocks to show (top 2 won at 8 facts, top 4 at 48 in text).
4. Test 42 (27B writes the answer) is text only and low priority.

Still owed from earlier: abstention training, fact-update test, general-ability check with the LoRA on and off, Django exact names, tighter flying / no-car checks, and reminding the user about `sudo pmset -a disablesleep 0` on the Mac.

Full history: [AGENTMEM_HYPOTHESES.md](AGENTMEM_HYPOTHESES.md) (tests 1–43), [RESULTS.md](RESULTS.md).

## Mac results and the next box job (2026-11-24)

The Mac quick checks are in [AGENTMEM_HYPOTHESES.md](AGENTMEM_HYPOTHESES.md), test 43. In short, a LoRA trained to copy the 8B text rater while reading only the 16-token blocks lifts the clue into the top 2 of 8 from 0.46 to 0.60. Answers only go from 0.39 to 0.45. At 48 facts it is capped by its teacher: the clue is first in 0.09, and the 8B text rater, its teacher, gets 0.16.

Next box job (check 4c): a 27B teacher.
1. `v100.sh serve27`.
2. `MPORT_VLLM_URL=http://127.0.0.1:8150 MPORT_VLLM_MODEL=qwen3-27b MPORT_RATER_TOK=$HOME/models/<qwen-27b> poc_rate27train.py --requests 1000` writes `rater_train27.json` (about 25k pairs), then `v100.sh unserve`.
3. `poc_tokrater.py --data rater_train27.json --minutes 45 --out lora_tokrater27.pt` (fp16 on the V100).
4. `poc_tok.py rate --lora lora_tokrater27.pt --raters tokL --tag 27 --wordings change`, then `answer --tables tokL27_change`.

Bar: clue first of 48 ≥ 0.20; top 1 of 8 acts ≥ 0.50.

Mac light mode: `. experiments/light.sh` after `env.sh` gives an int8 decoder ([q8.py](q8.py), `MPORT_Q8=1`, about 12–14 GB) and capped GPU memory. Not for the V100.

## Report and test 44 (2026-11-24)

Summary of tests 22–44 and the ranked next steps: [REPORT_2026-11-24_token_path.md](REPORT_2026-11-24_token_path.md). Test 44 (GRPO on the reader, [poc_grpo.py](poc_grpo.py)) was stopped at step 19 on the Mac with no signal yet; its v2 is a box job (report section 6.2).
