# Porting Memorilla's pv4 head to LFM3-1.7B — results (2026-11-20)

Setup:
- The released pv4 memory module (trained for Qwen3-8B) is used frozen. Only its output head (Linear 2560→hidden) is replaced.
- Decoder: LiquidAI/LFM3-1.7B, frozen, bf16 on an M3 Pro (MPS).
- Test data: 200 seeded pv4 test questions (33 hard). Scoring is Memorilla's choice rule, using Qwen3-Embedding-4B cosine.
- `nll` = mean loss on the reference answer tokens (lower is better).

| run | head | acc % | hard % | nll |
|---|---|---|---|---|
| lfm3_closed | none (no memory) | 60.5 | 33.3 | 4.078 |
| lfm3_random | random, same norm as ridge | 62.5 | 33.3 | 4.087 |
| lfm3_ridge | 8B head ∘ ridge token-table map | 60.0 | 27.3 | 4.136 |
| lfm3_procrustes | 8B head ∘ Procrustes map | 57.0 | 30.3 | 4.123 |
| lfm3_cal_random | head trained 600×4 from random | 61.0 | 36.4 | 2.817 |
| lfm3_cal_ridge | head trained 600×4 from ridge | 66.0 | 48.5 | 2.815 |
| lfm3_cal_ridge_shuffled | same head, **another user's documents** | 65.0 | 48.5 | 2.829 |

Diagnostics (`maps/.../report.json`):
- The 8B memory tokens have norm ~182, against ~1.5 for ordinary tokens.
- Their nearest-token cosine is 0.07, against 0.24 between tokens. So they sit far off the token manifold.
- Map holdout quality:
  - ridge: r2 0.13, nn@1 0.65
  - Procrustes: r2 0.06, nn@1 0.03

Reading:
1. Conversion without training transfers nothing. All three converted or random heads are within noise of closed-book. A 1σ swing on 200 questions is about ±3.4 points.
2. Head-only calibration helps, but not through personalization. It lowers answer NLL 4.08→2.82, and ridge init gives +5.5 accuracy. However, the shuffled-documents control is as good as the real documents (65.0 vs 66.0, NLL 2.829 vs 2.815). The head learned a soft prompt for answer style; it does not read the user's memory.
3. Ridge init vs random init: same NLL (2.815 vs 2.817). Accuracy is 66.0 vs 61.0, but that is about 1.5σ and the hard subset has only 33 questions, so it's suggestive, not established.

Why the answer loss alone is not enough: style alone can minimise it. A user-specific signal needs either:
- a teacher that sees the user's text (logit-KL), or
- a contrastive term that penalises the head when the wrong user's documents do just as well.

Reproduce: `source experiments/env.sh`, then the subcommands in `pv4_port.py` (`fit`, `generate`, `calibrate`, `score`) as chained in `finish.sh`.

## Re-running the memorilla-demos tests (Buried Clue, book, speed) and the finer-documents test

Buried Clue: 4 questions (3 buried clues + 1 control), scored with their rubric (substring match).

| method | LFM3 + cal_ridge | Qwen3-8B + released head |
|---|---|---|
| no memory | 0/4 | 0/4 |
| RAG top-5 facts (as text) | 1/4 (control) | 1/4 (control) |
| Memorilla, 150 whole chats | 0/4 | 0/4 |
| Memorilla, 857 extracted facts | 0/4 | 0/4 |
| Memorilla, clue chat's facts only | 0/4 | 0/4, but the details come through ("Biscuit", "leather", "the long shift", "Priya … March") |
| Memorilla, wrong user's documents | 0/4 | 0/4 |

- Facts: LFM3-1.7B extracted 857 one-line facts from the 150 chats. All four clues survived extraction.
- Retrieval rank of the clue (question vs documents, Qwen3-Embedding-4B):

  | question | clue chat, of 150 | best clue fact, of 857 |
  |---|---|---|
  | getaway | 71 | 274 |
  | parents_dinner | 103 | 114 |
  | gym_time | 12 | 16 |
  | control | 1 | 1 |

  The buried clues are not semantically close to the question, so retrieval cannot find them at any granularity.

Reading:
- Splitting chats into facts does not help Memorilla.
- The original head does read a handful of documents: with only the 4–15 facts from the clue chat, the specifics show up in its answers.
- Pooled with ~850 other facts, the clue is diluted away, exactly as with whole chats.
- The LFM3 head gives the same answer whatever documents it gets.

Speed (LFM3, 32 new tokens):
- full context: 1.2 s at 1 passage, 72 s at 256 passages (66K tokens); out of memory on MPS at 512 passages (132K tokens)
- Memorilla answering: 1.0–2.1 s at every size
- embedding the passages, done once: 118 s for 256 passages

## Proof of concept: would retraining the core fix buried-clue recall?

Full write-up: [POC_RESULTS.md](POC_RESULTS.md). Encoder, pv4 core and Qwen3-8B (native head) all frozen; 225 synthetic chats (clue / twin / neutral), 4 held-out clue types, 150 candidate documents per user.

| step | result | verdict |
|---|---|---|
| 1. signal in frozen encoder vectors | learned scorer R@5 0.53 on unseen clue types (cosine 0.00); clue beats its own twin 100% | borderline fail (bar 0.6), but the clue is in the vectors |
| 2. clue as the only document → core → decoder | `mentions` 0/40, `acts` 0.125 vs 0.10 with no clue; same chat as text: 0.45 / 0.55 | fail at the decoder end: a probe still reads the clue off the 16 memory tokens |
| 3. learned top-4/8 → core | clue in set 47.5% / 60% (cosine 0%), but `mentions` 0 when selected; `acts` = no-clue counterfactual | fail |

- **Decision-table row: "step 2 low `mentions` at k=0 → capacity or head problem; retraining the pooling is not enough"**, leaning to row 4 (the decoder side needs training too).
- A selection-only core retrain (aux "which document holds the clue" loss) won't fix recall. The fact reaches the memory tokens, but frozen Qwen3-8B doesn't act on it.
- A retrain has to go end to end through the decoder, probably with text-teacher distillation and LoRA if needed. Estimated compute is ≈ 6e17 FLOPs, about 0.4–1.7 H100-hours. Data generation and checking is the real cost.
- Every step-1/2 number was reproduced exactly after the Mac crashed from RAM saturation and everything was rebuilt.
  - The rerun used one model per process, the decoder loaded straight to MPS, MPS memory caps, and a watchdog that kills the job below 12% free memory.
  - Lowest free memory was 16%.

### Follow-up: adapter / LoRA on the decoder side

Details: [POC_RESULTS.md](POC_RESULTS.md), "Follow-up" section. The encoder, the pv4 core and Qwen3-8B's base weights all stay frozen. Training data is 147 teacher answers (Qwen3-8B reading the clue chat as text) across 8 train clue types.

Strict `mentions`, base → adapter (2.1M) → LoRA q/v r8 (3.8M) → both:

| condition | base | adapter | lora | both |
|---|---|---|---|---|
| train clue types, clue chat only (n=20) | 1 | 2 | **5** | **5** (+1 false alarm on a twin) |
| unseen types, clue + 4 fillers (n=40) | 0 | 0 | 3 | 0 |
| unseen types, all 150 chats (n=40) | 0 | 0 | 0 | 0 |

- LoRA on the decoder makes the frozen core's tokens readable for clue types it was trained on, so the decoder end was a real bottleneck.
- It doesn't generalize to new types from 8 training types.
- Dilution at 150 chats is unchanged.
- Next steps:
  - more clue types, plus a general "restate the memory" stage, as in ICAE and xRAG;
  - a trained selector or pooling for the full-history case.

## Agent-session memory: recall of past coding sessions

Full write-up: [AGENTMEM_RESULTS.md](AGENTMEM_RESULTS.md).
- Data: 1,037 real SWE-agent sessions (nebius/SWE-agent-trajectories, 4 of 12 shards), split by repository.
- Targets are extracted from each session's final patch: repo, issue, files changed, functions.
- Frozen encoder and core; LoRA (15M) on Qwen3-8B; 16–512 memory tokens.

| held-out repos (n=60) | repo | file name | functions |
|---|---|---|---|
| same chunks as text (ceiling) | 0.67 | 0.66 | 0.55 |
| LoRA, single session, 16 / 128 tokens | 0.13 / 0.15 | 0.03 / 0.10 | 0.17 / 0.14 |
| LoRA, target session absent (guessing) | 0.08 | 0.03 | 0.06 |

- The memory carries the gist; the answer names the right kind of fix. It does not carry exact identifiers: it swaps in repo and file names seen in training.
- More tokens (16 → 128 per session, or 16 per session for 8 or 32 sessions) don't help.
- Diagnostics:
  - The frozen chunk vectors pick out the session's own repo and file name 88% of the time (chance ~1%).
  - The core's 16 tokens tell sessions apart 100% of the time.
  - So the loss is in reading the identifiers out (decoder and training scale), not in capacity or the encoder.
- Practical reading: use the memory to recognise and retrieve the right past session, then give the decoder its identifiers as text.

## Agent-session memory: what is the missing part? (core frozen)

Full write-up: [AGENTMEM_HYPOTHESES.md](AGENTMEM_HYPOTHESES.md). Nine tests on 2026-11-21, core and encoder frozen, only the decoder LoRA trained.
- Names are not lost in the core: a linear read-out from its 16 tokens picks the right repo 73% and file 79% of the time (chance ~1%).
- The decoder recognises the right name: ranking 82 candidate repos puts the true one first 80% of the time after centring.
- Writing the name out is what fails. Specific questions, short notes and contrastive decoding do not fix it (repo 3–23%).
- Spelling is teachable when the memory holds one name: exact match on unseen names 3% → 32%. It does not transfer to memories holding several names.
- One fact per chunk, read alone: functions 67%, repo 25%. Pooling a session's facts into 16 tokens halves that.
- A reader trained on one fact per chunk at 16 tokens per fact recalls function names in 81% of unseen sessions (text ceiling 55%). Repos and file paths stay at about 20%: their parts come back recombined.
- 2 of the 16 tokens per fact are nearly enough: functions 72% at 2 tokens per fact against 78% at 16, cheaper than the name as text (about 5 tokens).
- The core's pooling is the lossy step: four facts pooled into 16 tokens give 50%, the same four at 4 tokens each give 70%.
- The untouched core does not pick: its attention over a pile of facts is uniform whatever the question. Fine-tuning its attention layers (78.7M parameters, 150 min) left the attention at chance.
- Adding the question–fact cosine to the core's attention logits (16 new parameters, original weights frozen, reader trained 90 min) makes the core pick. Function names through one 16-token block: 65% with the target session alone, 63% among 532 facts, 61% among 4,256, 4% with the target removed. Attention on the target session's facts 0.96–1.00.
  - The attention must read a plain one-name vector (65–67%); reading the keyed sentence with the issue title gives 41–44%.
  - Repos and file paths stay near 10%; sessions with several functions return some of them (42% of all asked names).
  - It relies on the question resembling the fact, which the buried-clue questions do not.
- The same mechanism on non-code facts (700 synthetic people, 4,600 facts; test people, a quarter of the values and three attributes held out): 65% with no training, 87–90% after 75 min of mixed training, flat from 8 to 4,600 facts in one 16-token block, 1% with the asked fact removed. Code recall kept (functions 70%).
  - With the asked fact missing the reader invents an answer; it has not been trained to abstain.
- Buried clues are not solved by this: keys written at ingest ("this fact matters when…") rank the clue first in 1% of 252 questions and top 5 in 18% against about 60 other stored personal facts, and the note writer stored the clue in only 67% of cases. A first reading (top 5 in 100%) was an artifact of almost no filler chat having keys.
- An always-on profile (every lasting fact in the prompt at 4 tokens each, reader trained to act on the one that matters) does not solve buried clues either: 11% on unseen fact types in a profile of 8, 14% with the clue removed. The plain decoder given the same profile as text also fails (64–69% with the clue alone, 20–32% among 7 other facts), so something has to select before the decoder.
- Codebase scale (all of Django 5.2, 36,263 facts in one store): picking is not the problem (attention on the asked fact 0.70; name read back 38% with 1 fact, 34–36% among all). Exact names are: 38% against 93% for the top 3 facts as text, and 20 more minutes of reader training did not move it. The frozen core keeps meaning, not spelling.
- The decoder's own yes/no judgement per fact is too permissive to select (yes to 54–67% of unrelated facts; clue first in a profile of 48: 1%).
- A small learned relevance map over the frozen vectors (rank 64, trained on generated request/fact pairs, seconds on CPU) does select, and improves with data: on unseen fact types the clue is in the top 3 of 8 in 70% with 3,363 pairs (similarity 61%), ranked first in 40% (22%).
- A reader trained on broad generated data (80 fact categories) acts on an unseen-type fact alone in 29% (15% before; text 69%), and on trained types reaches 40% with the map picking 3 of 8 (similarity 17%). On unseen types it stops acting once 2 other facts share the memory (6%). Details in [AGENTMEM_HYPOTHESES.md](AGENTMEM_HYPOTHESES.md) tests 15–18.
- The map keeps improving with data: 25,076 generated pairs put the right fact first in 49% of profiles of 8 on unseen types (3,363 pairs: 40%; similarity 22%) and in the top 3 in 75%. At 48 facts it is still 15% / 32%.
- With that map the plain decoder given the top 3 facts as text acts in 50% at 8 facts (old map 44%, clue removed 12%); handing over only the top fact is worse (33%).
- A reader trained on full-length answers acts on an unseen-type fact read from its 16 memory tokens in 62% when it is alone (same fact as text 74–81%); half of the earlier gap was the two-sentence limit in the test (old reader 29% → 50% without it). With three facts in memory it stays at 24%.
- A reader trained only to restate the relevant fact (two-pass answering) collapsed to writing "none" (48 of 72 with the clue alone); likely near-duplicate facts used as "none" examples. Not a valid test of the idea yet.
- The 72-question test has about ±10 points of noise (the same condition scored 81% and 74% in two runs); a larger test set is needed before comparing close variants.
- Larger test (288 unseen-type questions instead of 72): the 25,076-pair map puts the right fact first in 38% / 27% / 14% of profiles of 8 / 16 / 48, not 49% / 36% / 15%. 2.7 times more pairs (67,376) leave it flat (37% / 27% / 18%); a larger rank and masking near-copies do nothing.
- Best reader so far (`lora_lsort.pt`: restate first, long answers, memory blocks ordered best-first): acts on a fact alone in 16 tokens in 64% (text 80%), with the map's top 3 of 8 in 32% (text 39–43%, clue removed 15%).
- Putting the map's score inside the frozen core's attention does not work (profile of 8 in one block: 18%, clue removed 17%). The restate-only two-pass reader collapsed to "none" again without near-copies (61%).
- By fact type the totals mislead: the rubrics for flying and no car fire without the clue (36% and 24%). On the two clean types the map misses the fact that matters whatever the topic (health: in the top 3 of 8 in 22–24%, below a random pick), and the reader acts on a picked health or toddler fact among three blocks in 12% and 24% (text 71% and 59%). End to end at 8 facts: health 3%, toddler 11%.
- Modal: about 594 GPU-minutes in total; the user reported the credits nearly used up on 2026-11-23. Results, maps, pairs, embeddings and five readers are copied to `work/runs/`.
- Standing facts (`poc_standing.py`, no model): a request-independent score learned from 13 hand-listed body-and-dependents categories puts an unseen health clue in its top 3 of 8 / 16 / 48 in 96% / 65% / 24% (map 24% / 14% / 4%) and in its top 12 of 48 in 94%. It does nothing for no car and flying and little for toddler; label-free versions of the score do not find health. With 3 facts handed over the totals barely move (68% / 44% / 23% against 62% / 48% / 24%). The category list was written after seeing the failures.
- No picker (`poc_whole.py`, test 31, Modal run `whole`, 22.1 GPU-minutes): the plain decoder with the whole profile in the prompt as text acts on the clue in 58% at 8 facts and 32% at 48 on unseen types (picker's top 3 as text: 39% and 24%; health at 8 facts 67% against 17%). A LoRA trained for 11 minutes to restate the fact that matters and then answer made it worse (22% and 16%): it restates the clue at chance (12% at 8 facts) and writes "none" in 62%; when it does restate the clue it acts in 91%. The generated pairs teach "the fact on the request's topic", whoever does the choosing.
- The model's own judgment as the selector (`poc_whole.py --mode zero`, test 32, user's V100 box): no. Plain decoder with the whole profile acts on the clue in 53% / 42% / 35% at 8 / 16 / 48 facts (clue alone 80%, clue removed 18–19%). Asked to quote the facts that matter first, it quotes the clue in 49% at 8 and 22% at 48 and acts 53% / 31%; judging groups of 8 and answering with the quoted facts gives 35% at 16 and 33% at 48. Generation through vLLM (production's V100 build, read-only) is 4.3× faster than Hugging Face generate with the same scores.
- Self-distillation (`poc_distill.py`, test 33, V100 box): the plain decoder seeing only the fact (0.80 on unseen types) as teacher, a LoRA reading the whole profile as student, 8,000 teacher answers, 30 minutes per run. Worse than no training: acts 32% / 26% / 24% at 8 / 16 / 48 facts (AdamW) and 29% / 24% / 18% (Muon), against 53% / 42% / 35% for the plain decoder; health 10% at 8 (plain 50%). The student learns the teacher's style and names one profile fact, often the wrong one, because most answer tokens do not depend on the fact. Muon was 8% slower per token on the LoRA matrices and no better. Next (test 34): weight each answer token by how much the fact raises its probability (scored with vLLM).
- Token-weighted distillation (`poc_distill.py --weight pmi|gap`, test 34, V100 box): each teacher token weighted by how much the fact raises it. Acts 43% / 40% / 29% at 8 / 16 / 48 facts (unweighted 32% / 26% / 24%, plain decoder 53% / 42% / 35%); no car and toddler beat the plain decoder, health collapses (18% / 6% / 3%).
- Any fact with any request (test 35): a random fact moves the teacher's answer as much as an on-topic one, so this teaches "use whatever you see": 38% / 33% / 24%. The decoder picking the best of its single-fact answers: 36% at 8 facts (oracle 90%).
- Per-fact rating (test 35b, no training): the plain decoder rates each fact alone with a yes/no ("should a good reply take this into account?"), the top k are kept. Top 4 of 8: 58% (whole profile 53%), clue in the top 4 83%, health 94%; top 8 of 48: 38% (35%). First no-training, nothing-hand-designed method that beats the whole profile.
- Embedding scorer distilled from 75,000 decoder ratings (test 36): below plain cosine on unseen types; the rater's judgment is not in the stored embeddings.
- Rating wording (test 35c): asking whether a fact "changes what a good reply should say or recommend" puts the clue in the top 8 of 48 in 62% (old wording 45%, map 39%).
- Attention as selector (test 37): one pass over the whole profile; the request's attention per fact token finds the clue in the top 8 of 48 in 50% with no training, but misses health (attention follows topic). One-pass rating of all facts (test 38): 56%, health 19%; the fact must be rated alone.
- End to end, no training, nothing hand-designed (tests 35c/37): rate each fact alone ("change" wording, plus attention at 48), show the top few with the best one last. Acts on unseen types 0.60 / 0.53 / 0.48 at 8 / 16 / 48 facts against the whole profile's 0.53 / 0.42 / 0.35 (clue alone 0.80–0.83). Cost about 90 ratings per second on one V100 (0.5 s for 48 facts).
- Two-stage rating (test 39): thinking before the yes/no on the shortlist buys 2 points end to end (0.50 at 48 facts) at 5x the cost; a comparative pick loses health.
- Rater LoRA (test 40): training the yes/no rating listwise on the known types makes it say Yes far less (40% -> 6%) but it learns those types, not relevance: on unseen types top 8 of 48 62% -> 56%, health 71% -> 33%. False.
- Bigger rater (test 41, approved): production's 27B (on disk, read-only, own vLLM process) rates each fact ("could a typical reply be wrong, unsafe or unsuitable for this user because of the fact?"); the 8B still answers. Clue in the top 4 of 48: 67% (8B 39%). Acts on unseen types 0.68 / 0.62 / 0.53 at 8 / 16 / 48 facts (8B rater 0.60 / 0.53 / 0.47, whole profile 0.53 / 0.42 / 0.35, clue alone 0.80–0.83). An 8B shortlist re-rated by the 27B keeps most of it at a third of the cost.
- Bug fixed (2026-11-23): saved selector scores were keyed by request + clue, which 84 of 288 test rows share; those rows were scored against another row's table. Rerun: the 8B's best at 48 facts is 0.47 (was 0.48).
- Tests 31–41 are text only (facts in the prompt as text, no memory tokens). Test 43 (`poc_tok.py`, prepared 2026-11-23, not run) repeats the rate-and-pick recipe on the 16-token blocks; see [HANDOFF_TOKENS.md](HANDOFF_TOKENS.md). Test 42 (27B answerer) did not run.
