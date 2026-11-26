# memorilla-token-path

**TL;DR.** [Memorilla](https://github.com/snap-stanford/memorilla) packs each stored fact about a user into 16 memory
tokens for Qwen3-8B. This repo asks whether those tokens can carry the one fact that should change an answer, on fact
types the models never trained on. As of 2026-11-26:

- **One fact in 16 tokens reaches the 8B almost as well as text.** A 27B judge says the reply takes the fact into
  account in 0.99 of answers, against 1.00 with the fact as text.
- **Among other facts, the token reader equals the 8B reading the same facts as text.**
- **Two limits remain.** The first is finding the fact: the clue makes the rater's top 4 of 48 in only 0.49 of
  questions. The second is the 8B itself: with 8 facts shown it uses the clue in 0.54 of answers, where the 27B
  reaches 0.76.

Most of the work was done by an AI agent (Claude) under the author's direction.

- **Write-up:** https://blog.nenkoru.com/posts/memorilla-token-path/ (markdown for agents: `index.md` next to it)
- **Full handoff report:** [`experiments/REPORT_2026-11-24_token_path.md`](experiments/REPORT_2026-11-24_token_path.md).
  Start there for the background. Its section 6 (what to try next) is superseded by *Next steps* below.
- **Weights (LoRAs):** https://huggingface.co/nenkoru/memorilla-token-path-loras
- **Benchmark, labels and result tables:** https://huggingface.co/datasets/nenkoru/memorilla-buried-clue

## Since the write-up (2026-11-25 to 2026-11-26)

Lab notebook tests 45-56 in [`experiments/AGENTMEM_HYPOTHESES.md`](experiments/AGENTMEM_HYPOTHESES.md). The write-up's
numbers used the keyword rubric. From test 54 on, a 27B judge is reported next to it.

### A second score: the 27B as judge (test 54)

- **How it works.** The 27B reads the request, the clue and the reply, and answers Yes or No to "does the reply take
  the fact into account?" ([`judge27.py`](experiments/judge27.py)). A reply passes when P(Yes) at the first token is
  above 0.5.
- **Why trust it, without human labels.** A reply written without the clue cannot use it, so any pass on "clue removed"
  is a false alarm. The table shows the score with the clue alone minus the score with the clue removed:

  | Reader | Keyword | Judge |
  |---|---|---|
  | token reader `kds8` | 0.77 - 0.18 = 0.59 | 0.93 - 0.05 = 0.89 |
  | token reader `kd27a` | 0.81 - 0.13 = 0.67 | 0.99 - 0.05 = 0.94 |
  | 8B reading the facts as text | 0.86 - 0.12 = 0.74 | 0.99 - 0.02 = 0.97 |

- **What the keyword rubric gets wrong.** It misses replies that use the fact in other words, and it passes replies
  that hit a term while doing the wrong thing.
- **Caveats.** The judge sees the clue, and it is the same model that wrote some of the training answers.

### Where things stand (judge score, 144 held-out questions, rater `r27w`)

| Facts shown | Clue among them | token reader `kd27a` | 8B, text | 27B, text |
|---|---|---|---|---|
| clue alone | 1.00 | **0.99** | 1.00 | 1.00 |
| top 3 of 8 | 0.81 | **0.67** | 0.70 (n = 288) | |
| top 2 of 8 | 0.66 | 0.60 | 0.62 | 0.64 |
| top 4 of 48 | 0.49 | 0.40 | 0.37 | 0.47 |
| top 8 of 16 | 0.89 | 0.54 | 0.56 | 0.76 |
| top 8 of 48 | 0.61 | 0.38 | 0.38 | 0.56 |
| top 2 of 8, clue removed | 0 | 0.05 | 0.02 | 0.04 |

- **With 2-4 facts shown, the rater is the limit.** The score is close to how often the clue was picked at all.
- **With 8 facts shown, the 8B is the limit.** The clue is there 0.89 of the time, but the 8B uses it 0.54-0.56.
  The 27B uses it 0.76.
- **The best setting is 3 facts from 8.**

### What changed, test by test

- **Speed on a shared V100 (tests 45, 46, 52).**
  - Training: the frozen 8B is stored in 4 bits and streamed from CPU memory. The rater trains in about 1.5 GiB of GPU
    memory and the reader in about 3 GiB ([`q8.py`](experiments/q8.py)).
  - Building memory blocks on the GPU gives 2.6x more training updates per minute.
  - Decoding: a custom loop ([`fast_gen.py`](experiments/fast_gen.py)) with a preallocated key/value cache and
    batches sized to free memory. It takes 4.1 s per answer, against 7.6 s with `generate`. A step takes about
    340 ms at any batch size: the time to move the 3.6 GiB of packed layers over PCIe.
- **Correction: the 27B-taught rater's gain was overstated (tests 47, 50, 53).**
  - `r27w` is the best checkpoint: the clue ranks first in 0.56 / 0.43 / 0.26 of profiles with 8 / 16 / 48 facts, on
    the full set. The 8B-taught rater gets 0.45 / 0.34 / 0.17.
  - But a second seed of the same recipe is level with the 8B-taught rater (0.46 / 0.35 / 0.17), so `r27w` was partly
    luck.
  - Training on harder candidates (the facts closest to the request) did not help.
  - The rater leans on topical closeness about twice as much as its 27B teacher.
- **Reader: distillation from the 8B reading text (test 49).** On the keyword score, clue alone went from 0.60 to 0.80
  (`kdsort`, n = 288), and top 2 of 8 from 0.40 to 0.52.
  - A teacher that reads only the clue was much worse: on the judge, 0.15-0.21 below `kds8` wherever other facts were
    shown.
- **Reader taught by the 27B's answers (test 51, `kd27a`).** On the judge it is never lower than `kds8`, and it is
  higher with the clue alone (0.99 vs 0.92). It did not move the 8B toward the 27B with many facts shown. It is the
  reader for new runs.
- **Reader trained on its own best-of-4 answers, picked by the judge (test 55).** No gain anywhere, and -0.10 at top 8
  of 48. Not adopted.
- **The question vector does not matter (tests 49, 52, 54).** Blocks built with another question's vector, or with the
  mean vector, score the same for both the rater and the reader. A fact's 16 tokens can be built once, when the fact
  is stored.
- **The Buried Clue test from the Better Stack video, rerun (test 56).**
  - Setup: 150 chats, 562 facts extracted by the 27B, rater `r27w` on every fact's block, reader `kd27a`.
  - Results: 2 of 3 buried clues pass (the vegan dinner at top 3, 4 and 8; the 3 a.m. gym shift at top 3 and 8). The
    original Memorilla passed none.
  - Failures:
    - The carsick dog fails: it ranks 23rd, and the reader plans a drive even when given only that chat's facts.
    - The control (a sister's birthday) gets the name right and the date wrong (March 17 instead of March 14). Exact
      numbers do not survive 16 tokens.
  - Caveat: two of the three questions are training questions of the synthetic data. This checks the whole pipeline
    end to end; it is not a held-out result.

### Next steps

1. **Picking at scale** is the main bottleneck. Off the token path, a block-rater shortlist re-rated by the 27B as
   text lifts "clue in the top 4 of 48" from 0.49 to about 0.63 (test 41). On the path, the open question is how to
   pass the teacher's judgement on without the student falling back on topic.
2. **A held-out version of the Buried Clue test** (new questions and fact types), so the demo becomes a real result.
3. **Exact values** (dates, numbers, names): test whether a fact can carry a short text span next to its 16 tokens.
4. **The 8B's limit with 8 facts shown:** a reader trained on many more 27B answers, or showing fewer and better
   picks.
5. **Build blocks once per fact at storage time** in the pipeline. The controls say this is safe.
6. **Validate the judge** with a small set of human labels.
7. **Continuous batching** in the decoding loop, for longer runs on the shared GPU.

## Status

This is research code, published as-is so someone can pick the work up. It is updated now and then, not maintained.
Results and their caveats are in the report. The lab notebook, with every test numbered, is
[`experiments/AGENTMEM_HYPOTHESES.md`](experiments/AGENTMEM_HYPOTHESES.md).

## Layout

| Path | What |
|---|---|
| `experiments/` | Scripts (`poc_*.py`, `run_*.sh`) and docs (`*.md`). The token-path work is in `poc_tok.py` (rate, answer), `poc_tokrater.py` (rater training), `poc_broad.py` (reader training), `poc_rate27train.py` and `poc_read27.py` (27B labels and answers), `judge27.py` (27B judge), `fast_gen.py` (decoding loop), `demo27.py` (the video's Buried Clue test) and `poc_grpo.py`. |
| `experiments/poc_config.json` | The buried-clue benchmark definition: fact types, requests and rubrics. |
| `mport/` | A small helper package (token-table alignment between decoders). |
| `work/` | Not in the repo. Data, checkpoints, venvs and logs live here at run time (`MPORT_SCRATCH`). |

## Setup (what the scripts expect)

1. Clone upstream Memorilla (used at commit `6864db7`) into `work/repo/memorilla`.
2. Get the released pv4 memory core for Qwen3-8B from upstream into `work/ckpt/pv4` (`config.json` + `memory.pt`).
3. Download `Qwen/Qwen3-8B` and `Qwen/Qwen3-Embedding-4B` into the default Hugging Face cache. The scripts load them
   with `local_files_only`.
4. Make a venv at `work/venv` (Python 3.12 with torch, transformers, sentence-transformers, numpy and pandas). Then:

```bash
. experiments/env.sh && . experiments/light.sh   # light.sh: int8 decoder and memory caps, for a 36 GB Mac
$PY -I experiments/poc_tok.py dry                 # smoke checks that load no model
$PY -I experiments/poc_grpo.py dry
```

The LoRAs from the Hugging Face repo go in `work/runs/agentmem/` under their original names. The current ones are
`lora_tokrater_r27w.pt` (rater) and `lora_kd27a.pt` (reader). The model card there shows how to convert them back;
`experiments/to_peft.py` goes the other way.

`experiments/v100.sh` and `experiments/modal*.py` are helpers for one specific GPU box and for Modal. Host names,
paths and model names in them are placeholders (`gpu-box`, `<qwen-27b>`, `<vllm-venv>`), so adapt them before use.

## AI use

- **Experiments, code and analysis:** an AI agent (Claude, Anthropic) under the author's direction, plus a local
  Qwen model for mechanical edits.
- **This README and the report:** written by Claude.
- **Second score:** answers are also graded by a 27B Qwen-family model (`judge27.py`); no human labels.
- **Benchmark data:** the benchmark's requests and rubrics were written by Claude. The answers and ratings were
  generated by Qwen3-8B and a 27B Qwen-family model.

## License and attribution

Code: Apache 2.0 (see `LICENSE`). Parts are ported from, and the scripts import,
[snap-stanford/memorilla](https://github.com/snap-stanford/memorilla) (Apache 2.0). See `NOTICE`.
