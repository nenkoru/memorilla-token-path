# Agent-session memory: can 16–512 memory tokens recall what happened in earlier coding sessions?

Run overnight 2026-11-21 (04:43–09:20) on the M-series Mac.
- Code: [poc_agentmem.py](poc_agentmem.py) (build / embed / train / eval) and [poc_agentprobe.py](poc_agentprobe.py) (diagnostics).
- Chains: [run_agentmem.sh](run_agentmem.sh), [run_agentprobe.sh](run_agentprobe.sh).
- Outputs: `work/runs/agentmem/`. Logs: `work/logs/poc_am_*.log`, `work/logs/poc_ap_*.log`.
- Follow-up to [POC_RESULTS.md](POC_RESULTS.md). The question moves from hand-made clues to the real goal: does the memory remember *what you worked on*?

## Verdict

**From memory tokens alone, the stack recalls the gist of a past session but not its exact identifiers.** The diagnostics locate the loss: identity survives the encoder and the core, and is lost when the decoder has to turn the tokens back into exact strings.

- **Session alone:** trained LoRA, held-out repos.
  - Repo name 13–20%, file name 3–10%, function names 10–17%.
  - With the target session removed, the same model scores repo 8%, file 3%, functions 6%.
  - The same chunks given *as text* to Qwen3-8B score repo 67%, file 66%, functions 55%.
- **More tokens don't help.** 16, 32, 64 and 128 tokens for one session score about the same. Pooled vs 16-tokens-per-session memories for 8 or 32 sessions also score about the same, all near the no-target baseline.
- **The answers carry the topic.**
  - "Deprecate `self.round`" in enochecker comes back as "Remove deprecated round function".
  - A sqlalchemy `__repr__` task comes back as a "Representer" fix.
  - But the names get swapped for training-set repos it has seen (sqlfluff, asottile/setup-cfg-fmt).
- **The identity is in the vectors:**
  - The frozen chunk vectors pick out the session's own repo name (top-1 of 82 candidates) 88% of the time, its file (of 122) 88%, and its function (of 115) 58%.
  - Two halves of the same session match each other among 137 test sessions **100%** of the time, both as encoder vectors and as the core's 16 output tokens.
- **So the bottleneck is reading the identifiers out, not capacity or the encoder.**
  - The memory can *recognise* the right repo and file but the decoder cannot *spell* them.
  - A 15M-param LoRA trained on 798 sessions learned the output format and the gist. On names, it overfit: train loss 1.0 against val 1.7.

## Setup

- **Data:** nebius/SWE-agent-trajectories (CC-BY-4.0), shards 0–3 of 12 (355 MB), in `work/ext/swe-traj/`. Untrusted data: read with pandas, nothing executed.
  - 26,680 trajectories, 1,213 issue instances, 547 repos.
  - One trajectory per instance, kept if its final patch modifies at least one existing file.
  - That gives **1,037 sessions: 798 train / 102 val / 137 test**, split **by repository** (test repos are never seen in training).
- **Targets**, extracted mechanically, with no hand-made clues and no LLM labelling:
  - repository: from instance_id
  - issue: first line of the issue
  - files changed: from the patch, excluding new scratch files such as reproduce.py
  - functions/classes touched: from the patch's hunk headers and added `def`/`class` lines
  - How often each fact appears in the session's chunk text: repo 100%, files 99%, functions 83%.
- **Documents:** each session is cut into ≤24 chunks (mean 20.7):
  - the issue (≤800 chars)
  - one chunk per agent step and its output (≤500 + 300 chars)
  - the last 3 steps, which include the submit diff
  - Chunks are embedded once by frozen Qwen3-Embedding-4B: 21,421 chunks in 1.8 h at 3.3 chunks/s.
- **Token budget without retraining the core** (which has 16 fixed slots): split the memory's chunks into G groups, run the core per group and concatenate, giving 16·G tokens (`<|memory|>` placeholders sized to match). In "16 tok each" mode, each session gets its own group.
- **Trainable:** LoRA rank 16 (alpha 32) on q/k/v/o of all 36 Qwen3-8B layers, 15.3M params. Encoder, core and base decoder stay frozen.
  - Examples: half "summarize the session in memory" at G ∈ {1, 2, 4, 8}, half "one session was about issue '<title>'; list its repo, files and functions" with 1/3/7 distractor sessions, pooled or per-session.
  - batch 1 × accumulation 4, AdamW lr 2e-4, warmup plus cosine.
  - Val loss went from 3.73 to 1.77 (update 50) and was best at 1.706 (update 250). Early stop at update 400, 55 min.
- **Eval:** 60 test sessions, greedy, 72 new tokens. A field counts as recalled if the exact string appears in the answer:
  - repo short name
  - issue: share of title words
  - files: full path, or file name
  - functions: share of names
  - The base row is the untrained decoder with the same core tokens.
- **RAM:** one model per process under the watchdog; lowest free memory was 13% (during training). No crashes, no kills.

## Results (fraction recalled, test repos)

| condition | tokens | repo | issue words | file name | file path | functions |
|---|---|---|---|---|---|---|
| **text ceiling**: same chunks as text, no memory | ~3.7k text | **0.67** | **0.78** | **0.66** | 0.40 | **0.55** |
| LoRA, single session | 16 | 0.13 | 0.16 | 0.03 | 0.00 | 0.17 |
| LoRA, single session | 32 | 0.20 | 0.17 | 0.05 | 0.03 | 0.12 |
| LoRA, single session | 64 | 0.17 | 0.19 | 0.07 | 0.00 | 0.10 |
| LoRA, single session | 128 | 0.15 | 0.20 | 0.10 | 0.02 | 0.14 |
| LoRA, 8 sessions, pooled | 16 | 0.07 | n/a | 0.08 | 0.02 | 0.07 |
| LoRA, 8 sessions, pooled | 128 | 0.13 | n/a | 0.03 | 0.02 | 0.07 |
| LoRA, 8 sessions, 16 each | 128 | 0.10 | n/a | 0.02 | 0.02 | 0.08 |
| LoRA, 32 sessions, pooled | 16 | 0.08 | n/a | 0.05 | 0.02 | 0.13 |
| LoRA, 32 sessions, 16 each | 512 | 0.12 | n/a | 0.02 | 0.02 | 0.08 |
| LoRA, 8 sessions **without** the target | 112 | 0.08 | n/a | 0.03 | 0.02 | 0.06 |
| base (no LoRA), single | 16 | 0.07 | 0.11 | 0.03 | 0.00 | 0.14 |
| base, single | 128 | 0.07 | 0.07 | 0.00 | 0.00 | 0.09 |
| base, 8 sessions, 16 each | 128 | 0.02 | n/a | 0.00 | 0.00 | 0.10 |
| base, 8 sessions without the target | 112 | 0.00 | n/a | 0.00 | 0.00 | 0.08 |

n = 60 per row, so one session is 1.7 points. The "without the target" rows score above zero because some names can be guessed from the issue title in the question (e.g. a class named in the title).

### Diagnostics

| question | test | result | chance |
|---|---|---|---|
| does the encoder keep the repo? | max-cosine of the session's chunk vectors against all test repo-name strings | top-1 **0.88**, top-5 0.98 (82 candidates) | 0.012 |
| … the file? | same, first changed file path | top-1 **0.88**, top-5 0.99 (122) | 0.008 |
| … the function? | same, first function name | top-1 0.58, top-5 0.83 (115) | 0.009 |
| does the core keep session identity? | odd vs even chunks → core (16 tokens); nearest other half among 137 sessions | **1.00** | 0.007 |
| (same, on the encoder side) | odd vs even chunks → mean chunk vector | 1.00 | 0.007 |

Caveat: the half-split test shows the tokens are session-specific. Topic alone could give that, so it does not prove exact strings are recoverable from the 16 tokens. The retrieval test does show that the encoder side carries identifier-level information.

### Example answers (LoRA, single session, unseen repos)

- **True:** enowars/enochecker, "Deprecate self.round or self.current_round", src/enochecker/enochecker.py, BaseChecker.
  - 16 tok: "Repository: joke2k/changelogchecker / Issue: Remove deprecated round function / Files changed: changelogchecker.py"
  - Text ceiling: "…in the `enochecker` repository. The issue was to deprecate `self.current_round`… Files Changed: `src/enochecker/enochecker.py`… `BaseChecker`".
- **True:** ofek/hatch-vcs, "setuptools-scm 'root' parameter is ignored". 128 tok: "Repository: asottile/setup-cfg-fmt / Issue: version detection is broken / … _get_version_from_scm". Right topic, a training repo's name.
- **True:** manicmaniac/sqlalchemy-repr, "Allow for whitelist/blacklist list of fields" (class Repr). 16 tok: "Repository: sqlfluff/sqlfluff / Issue: Representer should not be a class / … BaseRepr, `__repr__`".

## What it means

1. **"How many tokens for feature-level detail?"** Here the token count was not the limit. 128 tokens for one ~5k-token session (about 30× compression, inside the range where ICAE keeps detail) did no better than 16. The limit is the trained reader: a frozen decoder plus a LoRA trained on ~800 sessions learns the format and the gist, and swaps names for ones it saw in training. ICAE and xRAG train that reading step on hundreds of thousands to millions of examples, so this run is 2–3 orders of magnitude short.
2. **The identity is there to be had.** The encoder picks the right repo and file 88% of the time, and the core's tokens tell sessions apart perfectly. So the "map" use case, the memory *pointing at* the right past session, is already within reach without generation. Recognising the session is the easy half; writing its names out is the hard half.
3. **The practical design that follows:**
   - Use the memory, or the core's tokens and chunk vectors, to *recognise and retrieve* the relevant past session.
   - Then give the decoder the exact identifiers as text: the session's title, repo and files, a few dozen tokens.
   - That is the "map + territory" split, and it costs little: the text ceiling needs ~3.7k tokens for the whole session, while the identifiers alone need ~30.
4. **Training the memory to spell identifiers would mean:**
   - (a) far more sessions (all 12 shards give ~3.5k instances; SWE-smith / SWE-rebench / OpenHands sets give tens of thousands);
   - (b) training the core end to end, not only the decoder LoRA;
   - (c) probably an encoder that keeps token-level detail (ICAE-style LLM encoder) rather than one similarity vector per chunk.
   - That is a GPU job, not an overnight Mac run.

## Caveats

- One seed, n = 60, one dataset (SWE-agent on Python repos), one LoRA size.
- The trajectories come from several agent models and include failed attempts. The target is the final patch even when it is wrong.
- The scorer is substring-based. Short repo names can match by accident, which is part of why the no-target row is above zero.
- The text ceiling sees ≤24 truncated chunks, not the full trajectory, so it is a ceiling for *these chunks*.

## Reproduce

```bash
cd ~/memorilla-port && . experiments/env.sh
"$PY" -I experiments/poc_agentmem.py build
caffeinate -i -s experiments/run_agentmem.sh > work/logs/run_agentmem.log 2>&1
caffeinate -i -s experiments/run_agentprobe.sh > work/logs/run_agentprobe.log 2>&1
SCRIPT=poc_agentprobe.py experiments/poc_guard.sh ap_coreid coreid
```

Wall clock: embed 1.8 h, train 55 min (early stop), eval LoRA 53 min, eval base 29 min, retrieve 1 min, text ceiling 22 min, core-identity probe ~3 min.
