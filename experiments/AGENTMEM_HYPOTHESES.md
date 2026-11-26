# Agent-session memory: what is the missing part? (core frozen)

Run 2026-11-21, 14:30 onwards, following [AGENTMEM_RESULTS.md](AGENTMEM_RESULTS.md). The core and encoder stay frozen and only a LoRA on the decoder (the "reader") is trained, except in tests 11 and 12, where the user allowed cheap changes to the core (the original checkpoint is never modified).
- Code: [poc_agenthyp.py](poc_agenthyp.py), [poc_spell.py](poc_spell.py), [poc_atoms.py](poc_atoms.py), [poc_slots.py](poc_slots.py), [poc_core.py](poc_core.py), [poc_general.py](poc_general.py), [poc_keys.py](poc_keys.py), [poc_keys_stress.py](poc_keys_stress.py).
- Chains: [run_agenthyp.sh](run_agenthyp.sh), [run_spell.sh](run_spell.sh), [run_mix.sh](run_mix.sh), [run_atoms.sh](run_atoms.sh), [run_atoms2.sh](run_atoms2.sh), [run_slots.sh](run_slots.sh), [run_core.sh](run_core.sh), [run_bias.sh](run_bias.sh), [run_bias2.sh](run_bias2.sh), [run_general.sh](run_general.sh), [run_keys.sh](run_keys.sh).
- Outputs: `work/runs/agentmem/` (`hyp_*.json`, `spell_*.json`, `mix_*.json`, `atoms_*.json`).
- All tests use the 137 held-out-repo sessions (60 sampled for generation) or names from held-out repos.

## Summary

| # | Hypothesis | Verdict | Key number |
|---|---|---|---|
| 1 | Names are lost inside the core's 16 tokens | **False** | linear read-out from the tokens picks the repo 73%, the file 79% (encoder vectors: 82%, 81%; chance ~1%) |
| 2 | A field-specific question pulls names out | **False** | read-out unchanged (72–74% repo); generation at guessing level |
| 3 | Short name-dense notes instead of raw agent steps | **False** | repo 3–12% for one sentence, one note, or notes for 8 / 32 / 150 sessions |
| 4 | The reader can recognise the right name | **True** | scoring 82 candidate repos: right one first 32% raw, **80%** after subtracting each name's average score (top-5 97%) |
| 6 | Contrastive decoding turns recognition into text | **Mostly false** | repo 13% → 12% on raw steps, 15% → 23% on notes |
| 5 | Spelling can be taught to the reader | **True, for one name per memory** | exact spelling of unseen names: 3% → 28% (600 names, 30 min) → 32% (6,517 names, 60 min) |
| 7 | The spelling skill transfers to real sessions | **False** | note: repo 17%, file 6%, functions 15%, same as before; single-name spelling kept |
| 8 | One fact per chunk, same reader, no new training | **Partly** | one fact alone: functions 67%, repo 25%; all of a session's facts pooled into 16 tokens: functions 28%, repo 10% |
| 9 | Train the reader on one-fact-per-chunk memories, 16 tokens per fact | **True for function names, false for repos and paths** | functions 33% → **81%** (text ceiling 55%); repo 18%, file path 20% |
| 10 | Fewer than 16 tokens per fact is enough | **True** | functions 78% at 16 tokens per fact, 76% at 8, 70% at 4, **72% at 2**; a name as text costs about 5 |
| 11 | A cheap fine-tune of the core's attention makes it pick one fact from a pile | **False** | attention on the right facts stayed at chance (0.07 of 16 sessions' facts, before and after); functions 28% with the target present against 20% absent |
| 12 | A similarity bias inside the core's attention makes it pick | **True** | functions 65% with one session's facts in one 16-token block, 63% with 532 facts, 61% with 4,256; 4% with the target removed; attention on the target 0.96–1.00 |
| 13 | The same core and reader work on non-code facts | **True** | people facts: 65% with no training, 87–90% after 75 min, flat from 8 to 4,600 facts in one 16-token block; 1% with the asked fact removed; code recall kept (functions 70%) |
| 14 | Keys written at ingest ("this fact matters when…") let similarity pick a buried clue | **False** | among about 60 other stored personal facts the clue ranks first in 1% of 252 questions, top 5 in 18%; the note writer stored the clue in only 67% |
| 15 | A small always-on profile (4 tokens per fact, no lookup) plus a reader trained to act on it handles buried clues | **False** | held-out fact types: acts on the clue in 11% of 72 questions with it in a profile of 8, 14% with it removed. The plain decoder with the profile pasted as text fails the same way: 64–69% with the clue alone, 20–32% among 7 other facts, even when those are irrelevant to the request |
| 16 | The same core and reader hold a whole codebase (Django, 36,263 facts) | **Picking true, exact reading false** | attention finds the asked fact among all 36,263 (0.70; similarity ranks it first for 82% of paraphrased questions). Exact names read back: 38% with the fact alone, 34% among all facts, 19% with it removed (8 min of reader training). Plain text retrieval of the top 3 facts: 93% |
| 17 | The decoder's own yes/no judgement, one fact at a time, picks the fact that matters | **False**; a small learned map does better | yes/no says yes to the clue 94% and to 54–67% of the other facts; clue ranked first in a profile of 8 / 48: 26% / 1% (held-out types). A rank-64 map over the frozen vectors trained on 720 known (request, fact) pairs: 47% / 21%; similarity: 22% / 4% |
| 18 | More varied training data (80 fact categories) makes the map and the reader work on unseen fact types | **Map true, reader half** | map on 3,363 pairs: clue in the top 3 of a profile of 8: 70% (similarity 61%), ranked first 40% (22%), unseen types. Reader: acts on an unseen-type fact alone in 29% (was 15%; text 69%); with 2 other facts in memory 6%, same as with the clue removed. Trained types, map picks 3 of 8: 40% (similarity 17%) |
| 19 | The reader acts more often if it first writes out the fact that matters | **True, partly** | unseen types, map picks 3 of 8: acts 19% (6% without; 10% with the clue removed); alone 36% (29%). It restates an unseen-type fact from its 16 tokens in 62%, and in 31% when it is one of 3 |
| 20 | The short-answer limit in the test hides acting | **True** | plain decoder, picked facts as text, unseen types: alone 69% → 81% without the two-sentence limit; 3 of 8: 26% → 44% (clue removed 10%) |
| 21 | The map keeps improving with more (request, fact) pairs | **True, not yet flat** | 25,076 pairs (9,060 facts): clue ranked first in a profile of 8 / 16 / 48 on unseen types 49% / 36% / 15% (3,363 pairs: 40% / 28% / 17%; similarity 22% / 11% / 4%); in the top 3: 75% / 57% / 32%. Each 3× more pairs adds 7 to 17 points at 8 |
| 22 | A better map gives better answers; handing over only the top fact beats the top 3 | **Small gain; top 1 false** | picked facts as text, long answers, unseen types: top 3 of 8 acts 50% (old map 44%; clue removed 12%); top 1 of 8: 33% (clue is first in 46%). Memory tokens with the recite reader: 25% (19%), top 1: 26% |
| 23 | A reader that only restates the fact that matters, then a second pass that answers from that note | **False as trained** | the restate-only reader writes "none" for 48 of 72 unseen-type clues given alone (recite reader: 14 of 72) and restates the clue in 21%; two-pass answers: 32% alone, 28% with 3 of 8 (clue removed 15%). The core-output add-on (rank 256) and picking from the whole profile of 8 change nothing (24%, 26% alone; 6%, 1% from the whole profile) |
| 24 | A reader trained on full-length answers acts on a fact read from memory tokens | **True alone, false among others** | unseen types, fact alone in 16 tokens: acts 62% (old reader, same long test: 50%; same fact as text: 74–81%). Top 3 of 8: 24% (old 21%, clue removed 10%) |
| 25 | The 72-question test flattered the picker | **True** | on 288 unseen-type questions (9 more questions per type) the same 25,076-pair map ranks the clue first in 38% / 27% / 14% of profiles of 8 / 16 / 48 (72-question test: 49% / 36% / 15%); similarity 28% / 18% / 7% |
| 26 | 2.7 times more pairs keep improving the map | **False, flat** | 67,376 pairs (23,439 facts, 30 user groups): 37% / 27% / 18% first, top 3 66% / 47% / 31%; rank 256 and a near-copy mask change nothing. Plain decoder with the top 3 of 8 as text: acts 43% (smaller map 39%, clue removed 15%) |
| 27 | Restating the fact first, with long answers; and memory blocks ordered best-first | **Sorting true, small** | unseen types, acts alone / 3 of 8: long reader 58% / 26%; + restate 52% / 30%; + restate + sorted **64% / 32%** (clue removed 15–17%; same picked facts as text 80% / 39%) |
| 28 | The map's score as a bias inside the core's attention lets one 16-token block carry the fact that matters | **False** | whole profile of 8 in one block: acts 18%, clue removed 17%; profile of 48: 18% / 18%. The bias strength grew from 1 to 1.8–2.8 in training |
| 29 | The restate-only reader collapsed because of near-copies among its "none" examples | **False** | with look-alikes dropped it still writes "none" for 61% of unseen-type clues given alone; two-pass acts 34% alone, 21% at 3 of 8 (clue removed 12%) |
| 30 | A per-fact score for "matters whatever the topic" (standing facts kept always on) finds the facts the map misses | **True for health, not general** | score learned from 13 hand-listed body-and-dependents categories: unseen health clue in its top 3 of 8 / 16 / 48 in 96% / 65% / 24% (map 24% / 14% / 4%, random 44% / 15% / 7%); toddler 67% / 14% / 0%; no car and flying 0%. Scores learned without labels do not find health. All types, 3 facts handed over: 68% / 44% / 23% against the map's 62% / 48% / 24% |
| 31 | No picker: with the whole profile in the prompt as text, a reader trained to restate the fact that matters and then answer uses a crowded memory | **False as trained; the untrained decoder is the best so far** | plain decoder, whole profile as text: acts 58% at 8 facts, 32% at 48 (picker's top 3 as text: 39% and 24%); health 67% at 8 (picker 17%). After 11 minutes of training: 22% and 16%; it restates the clue in 12% at 8 (chance 12.5%), writes "none" in 62%; when it does restate the clue it acts in 91% |
| 32 | The plain decoder's own judgment of which facts matter can select them (no training, nothing hand-designed) | **False** | whole profile, answer directly: acts 53% / 42% / 35% at 8 / 16 / 48 facts (clue removed 18–19%, clue alone 80%). Quoting the facts that matter first: 53% at 8, 31% at 48 (it quotes the clue in 49% and 22%). Judging groups of 8, then answering with the quoted facts: 35% at 16, 33% at 48 (clue quoted in 38%, 13 of 48 facts quoted) |
| 33 | Self-distillation: the plain decoder that sees only the fact teaches a LoRA student that reads the whole profile | **False (worse than no training)** | after 30 min (0.45 epochs of 8,000 teacher answers): acts 32% / 26% / 24% at 8 / 16 / 48 (plain decoder 53% / 42% / 35%); health 10% at 8 (plain 50%). The student learns the teacher's style and names one fact, often the wrong one. AdamW slightly ahead of Muon (Muon 29% / 24% / 18%, and 8% slower per token on LoRA matrices) |
| 34 | Weighting each answer token by how much the fact raises it (scored by the plain decoder) makes distillation learn the fact, not the style | **Partly: much better than unweighted, still below no training** | weights log p(token given the fact) minus log p(token given no fact): acts 43% / 40% / 29% at 8 / 16 / 48 (unweighted 32% / 26% / 24%, plain decoder 53% / 42% / 35%); no car and toddler beat the plain decoder (no car 61% at 8, toddler 40% at 16) but health collapses (18% / 6% / 3%). Weighting by teacher minus whole-profile log-prob: 36% / 31% / 26% |
| 35 | Teacher answers for any fact behind any request (so only the answer's change says whether a fact matters); or the decoder judging its own single-fact answers | **False** | a random fact moves the teacher's answer as much as an on-topic one, so the target says "use whatever you see": acts 38% / 33% / 24%. Best-of judge at 8 facts: 36% (picks the clue's answer 34%, oracle 90%) |
| 35b | The plain decoder rating each fact alone (yes/no: should a reply take it into account?), keeping the top k | **True at 8 and 48 facts (first no-training gain)** | top 4 of 8: acts 58% (whole profile 53%), clue in the top 4 83% (health 94%); top 4 of 16: 43% (42%); top 8 of 48: 38% (35%). Subtracting each fact's mean rating makes it worse (51% at 8) |
| 36 | A small scorer on the stored embeddings, trained on 75,000 decoder ratings, ranks facts like the rater at a fraction of the cost | **False** | clue in the top 4 of 8 / 16 / 48 on unseen types: best scorer 69% / 42% / 18%, cosine 73% / 49% / 25%, rater 83% / 59% / 29%. Learned terms help health and hurt no car |
| 35c | The wording of the rating question matters | **True** | clue in the top 8 of 48: "take into account" 45%, "does it change what a good reply should say" 62%, mean of five wordings 62% (top 12: 77%); map 39%. Wordings compared on the test profiles |
| 37 | The decoder's attention from the request to each fact selects facts in one pass | **Partly** | no training: clue in the top 8 of 48 in 50% (map 39%), top 4 41% with a relevance question appended; health 11%. Learned head weights do not transfer. Rating + attention, end to end: acts 43% at 48 facts (whole profile 35%) |
| 38 | All facts rated in one pass (request first, each fact then "Answer:") | **False** | top 8 of 48: 56% (rated alone 62%), health 19% (alone 71%); ratings fall with position |
| 38b | The per-fact rating holds up as profiles grow | **Partly** | clue in the top 8: 62% / 36% / 25% at 48 / 96 / 192 facts (with cosine added: 62% / 48% / 32%; cosine alone 39% / 25% / 12%). End to end, best of the no-training recipes: acts 60% / 53% / 47% at 8 / 16 / 48 facts (48: 48% before the score-key fix, see 41) (whole profile 53% / 42% / 35%) |
| 39 | Thinking before the yes/no, or one comparative pick among the shortlist, sharpens the rating | **Partly** | thinking on the top 12: acts 50% at 48 facts (cheap rating 48%) at 5x the cost; still Yes to 90% of shortlisted facts. Pick: clue in the top 4 of 48 38% → 48%, but loses health; acts 44% |
| 40 | A LoRA trained listwise on the rating (known relevant fact above 15 random ones) makes the rater precise on unseen types | **False** | Yes share 40% → 6%, but clue in the top 8 of 48 62% → 56%; health 71% → 33% (192 facts: 58% → 0%). It learns the training types |
| 41 | A bigger rater (production's 27B, read-only, no training) picks the facts; the 8B still answers | **True (best so far)** | clue in the top 4 of 48: 67% (8B 39%), top 8 of 192: 51% (8B 29%), every type up. Acts 68% / 62% / 53% at 8 / 16 / 48 facts (8B rater 60% / 53% / 47%, whole profile 53% / 42% / 35%). An 8B shortlist of 16, re-rated by the 27B, keeps most of it at a third of the cost |
| 42 | The 27B also writes the answer (text) | **Not run** | the 27B did not start on the 11-19 build (image-profiling error; `serve27` now passes `--limit-mm-per-prompt`, untested), then production took the GPUs back. Text only, so low priority (see 43) |
| 43 | The same recipe on the 16-token memory blocks: rate each fact from its block, the reader answers from the top blocks (`poc_tok.py`) | **Prepared, not run** | tests 31–42 gave the facts as text. Script, core and reader weights are on the box; dry check passes. See [HANDOFF_TOKENS.md](HANDOFF_TOKENS.md) |

**Reading so far.**
- The name information is in the core's tokens (1) and the decoder can recognise it (4). What fails is writing it out.
- A reader can be taught to spell (5), but one chunk vector, and the 16 tokens made from it, carries about one identifier (7, 8).
- With one fact per chunk, 16 tokens per fact and a reader trained on that layout (9), function and class names come back in 81% of unseen sessions. That is the first condition in this project where the memory beats guessing by a wide margin on exact names.
- Multi-part names still fail: `owner/name` repos and directory paths come back with the right parts recombined or paraphrased (`mir_eval/sonify.py` → `src/mira_eval/sonify.py`, `pydicom/pydicom` → `dicom/pydicom`).
- Token cost: the first 2 of the 16 tokens per fact are nearly enough (10). At 2 tokens per fact the memory is cheaper than the name as text (about 5 tokens), and 4 facts in 8 tokens beat the same 4 facts pooled by the core into 16 (72% against 50%).
- So the frozen core's pooling is the lossy step: it blends facts that the slots could carry separately.
- The core's attention ignores the question, and a fine-tune does not teach it to pick (11). Adding the encoder's own question–fact cosine to the attention logits does (12): 16 new parameters, and recall no longer falls as the pile grows.
- The mechanism is not specific to code (13): on facts about people it reaches 87–90% and transfers to attributes and names it never trained on.
- Still open after 13: with the asked fact missing the reader invents an answer instead of saying it does not know;  `owner/name` repos and file paths (10–20%), sessions with several asked names (only some come back), and questions that are not similar to the fact they need (the buried-clue case, where cosine ranked the clue 12th to 103rd of 150).
- Buried clues after tests 17–29 (288 unseen-type questions): the fact alone as text 80%, alone in 16 memory tokens 64%; picked 3 of 8 by the map and read from memory 32%, as text 39–43%, floor without the clue 15%; at 48 facts text is at 24–26% against a floor of 17%. The largest loss is picking: the map has the clue in its top 3 in 65% at 8 facts and 31% at 48, and more generated pairs no longer help (26). Next is the drop when the clue sits among two other facts: when the clue is among the 3 picked, text acts in 61% and the best reader in 50% (alone: 80% and 64%). By type the picture is worse than the totals: see "Where the losses are" after test 29.

## Details

### 1–2. Probe: are names readable from the core's tokens?

Kernel ridge from a session's representation to the encoder vector of the true name, fitted on train + val repos, tested on held-out repos by nearest candidate (82 repos, 122 files, 115 functions).

| read-out from | repo | file | function |
|---|---|---|---|
| mean encoder chunk vector (reference) | 0.82 | 0.81 | 0.48 |
| core 16 tokens, "summarize" question | 0.73 | 0.79 | 0.31 |
| core 16 tokens, "which repository?" | 0.72 | 0.78 | 0.30 |
| core 16 tokens, "which files?" | 0.74 | 0.79 | 0.34 |
| core 16 tokens, raw steps + session note | 0.71 | 0.81 | 0.40 |

### 2–3. Generation with specific questions and notes

Reader: last night's LoRA; base model in brackets. Fraction of 60 sessions where the repo short name appears in the answer.

| memory holds | tokens | LoRA | base |
|---|---|---|---|
| raw steps, ask repo | 16 | 0.07 | 0.07 |
| one sentence "…in the repository X." | 16 | 0.03 | 0.15 |
| that sentence + raw steps | 16 | 0.08 | 0.05 |
| one session note | 16 | 0.07 | 0.07 |
| notes of 8 / 32 / 150 sessions | 16 | 0.10 / 0.10 / 0.07 | 0.07 / 0.08 / 0.05 |
| notes of 8 / 32 / 150 sessions | 128 | 0.08 / 0.12 / 0.08 | 0.00 / 0.00 / 0.00 |

File names: 0–8% everywhere. The single-sentence answers are blurred versions of the name: beetbox/beets → "beeware/beeware-platform", redhat-… → "redox-os/redox", enochecker → "enforcer" (base). The base model at 128 tokens (8 groups) falls back to generic phrases; the core's head was only ever trained with 16 slots.

### 4. Ranking

LoRA reader, raw steps, 16 tokens. For each of 60 sessions, log-likelihood of "Repository: <cand>" for all 82 test repos.
- Raw: top-1 0.32, top-5 0.40, median rank 14.
- Minus the candidate's mean score over the other sessions: top-1 **0.80**, top-5 0.97.

### 6. Contrastive decoding

Greedy on lp(mem) + a·(lp(mem) − mean lp(3 other sessions' memories)), restricted to tokens with p ≥ 0.1·max.

| memory | a = 0 | a = 1 | a = 3 |
|---|---|---|---|
| raw steps: repo / file name | 0.13 / 0.03 | 0.12 / 0.12 | 0.12 / 0.06 |
| one note: repo / file name | 0.15 / 0.06 | 0.20 / 0.07 | 0.23 / 0.07 |

### 5. Spelling

Memory = one sentence with one name; question asks for it; answer = the name. Names from the local trajectories (repos, patched file paths, def/class names), split by repo. Fresh LoRA r16.

| | 600 names, 30 min (1,372 examples) | 6,517 names, 60 min |
|---|---|---|
| functions / classes | 0.43 | 0.48 |
| file paths | 0.29 | 0.36 |
| repos | 0.10 | 0.11 |
| all (290 held-out names, exact match) | 0.28 | 0.32 |

Misses are near-misses: `test_special_surface_observation` → `test_surface_observation`, `HTTPRequest` → `http_request`, `keras-team/keras-nlp` → `kundajelab/keras-nlp`. Repo owners are usually invented.

### 7. Transfer to sessions

One reader, 90 min, 5,164 examples: 40% spelling, 20% note → all fields, 15% raw session, 10% note + raw, 15% notes of 4–16 sessions with an anchored question.

| memory (60 held-out sessions) | repo | file name | functions |
|---|---|---|---|
| raw steps | 0.17 | 0.07 | 0.13 |
| one note | 0.17 | 0.06 | 0.15 |
| note + raw steps | 0.15 | 0.04 | 0.15 |
| notes of 8 / 32 / 150 sessions | 0.12 / 0.13 / 0.12 | 0.03 | 0.07–0.09 |
| single-name spelling (same reader) | 0.12 | 0.37 | 0.46 |

### 8. One fact per chunk (reader from 7, no new training)

Each fact is its own sentence and chunk. Score: the answer contains the repo / any of the session's files / any of its functions.

| memory | tokens | repo | file | function |
|---|---|---|---|---|
| the asked fact alone | 16 | 0.25 | 0.07 | 0.67 |
| all the session's facts, pooled | 16 | 0.10 | 0.03 | 0.28 |
| all facts, 16 tokens each (layout never trained) | ~100 | 0.13 | 0.05 | 0.33 |
| facts + raw steps, pooled | 16 | 0.07 | 0.00 | 0.26 |

### 9. Reader trained on one-fact-per-chunk memories

Reader initialised from 7, trained 120 min (6,520 examples, loss 0.99 → 0.30): 25% spelling, 40% one fact per chunk at 16 tokens each, 15% all facts pooled, 20% facts + raw steps. Training memories capped at 8 facts. Answer = all names of the asked kind. Same 60 held-out sessions; sessions have 4.1 facts on average.

| memory | tokens | repo | file path (any / share) | function (any / share) |
|---|---|---|---|---|
| all facts, 16 tokens each | 66 | 0.18 | 0.20 / 0.17 | **0.81** / 0.66 |
| all facts, 16 each + raw steps in 16 | 82 | 0.22 | 0.20 / 0.17 | 0.80 / 0.65 |
| all facts pooled | 16 | 0.08 | 0.10 / 0.10 | 0.44 / 0.34 |
| before training (test 8), 16 each | ~100 | 0.13 | 0.05 | 0.33 |

- Looser scoring on the 16-each row: the file's base name appears in 43% of answers and its stem in 60%; the repo's short name in 45%.
- Typical misses: `cdent/gabbi` → "danielfabian/gabbi", `pypa/wheel` → "pypa/warehouse", `simpleflow/utils/json_tools.py` → "simpleflow/utils/json.py", `src/wheel/metadata.py` → "wheel/metadata.py".
- Adding the raw steps as one more 16-token group changes nothing.
- First two attempts at this training crashed with a GPU out-of-memory error: variable-length memories fragment the MPS cache. Fixed by `torch.mps.empty_cache()` after every update.

### 10. Fewer tokens per fact

[poc_slots.py](poc_slots.py), [run_slots.sh](run_slots.sh). Reader initialised from 9, trained 90 min on mixed layouts: the first k of the 16 slots per fact (k = 2, 4, 8, 16), facts pooled in pairs, all facts pooled. Same 60 held-out sessions, 4.1 facts each on average.

| layout | tokens per session | repo | file path | function (any / share) |
|---|---|---|---|---|
| 16 slots per fact | 66 | 0.22 | 0.20 | 0.78 / 0.65 |
| first 8 slots per fact | 33 | 0.17 | 0.18 | 0.76 / 0.64 |
| first 4 slots per fact | 16 | 0.20 | 0.13 | 0.70 / 0.60 |
| first 2 slots per fact | 8 | 0.17 | 0.10 | 0.72 / 0.58 |
| facts pooled in pairs, 16 tokens per pair | 38 | 0.15 | 0.12 | 0.65 / 0.50 |
| all facts pooled | 16 | 0.07 | 0.08 | 0.50 / 0.36 |

- The 16 slots made from one chunk vector are largely redundant: 2 carry almost what 16 do.
- At equal budget (16 tokens), four facts at 4 slots each give functions 70%; the core pooling the same four facts gives 50%.
- Repos and file paths do not improve at any setting.

### 11. Fine-tuning the core to pick

[poc_core.py](poc_core.py), [run_core.sh](run_core.sh). The pile holds keyed facts ("The function or class touched for the issue "<title>" was X.") from N sessions; the question names one issue and asks for one kind of name; everything passes through one 16-token block. Unfrozen: the core's slot queries, question conditioning and both cross-attention layers (78.7M of 194.1M parameters, lr 2e-5) plus the reader LoRA (lr 1e-4). 150 min, 9,668 examples, loss 1.76 → 0.39. The original checkpoint is untouched; tuned tensors are in `core_pick.pt`.

60 held-out target sessions, piles drawn from held-out sessions of other repos. "attn" is the share of the core's attention (second layer, mean over slots and heads) on the target session's facts.

| pile | before: repo / file / func | attn | after: repo / file / func | attn | chance |
|---|---|---|---|---|---|
| 1 session | 0.12 / 0.00 / 0.20 | 1.00 | 0.05 / 0.08 / 0.33 | 1.00 | 1.00 |
| 4 | 0.03 / 0.00 / 0.11 | 0.26 | 0.05 / 0.03 / 0.28 | 0.28 | 0.26 |
| 16 | 0.02 / 0.00 / 0.04 | 0.06 | 0.05 / 0.03 / 0.28 | 0.06 | 0.07 |
| 64 | 0.02 / 0.02 / 0.06 | 0.02 | 0.05 / 0.03 / 0.26 | 0.02 | 0.02 |
| 137 | 0.02 / 0.02 / 0.06 | 0.01 | 0.05 / 0.02 / 0.22 | 0.02 | 0.02 |
| 16, target absent | 0.00 / 0.02 / 0.02 | 0 | 0.05 / 0.02 / 0.20 | 0 | 0 |

- The untouched core does not pick at all: its attention is uniform over the pile whatever the question says. That is the dilution seen since the buried-clue tests.
- The fine-tune did not change that. The gain on functions is the same with the target absent, so it comes from the issue title in the question (titles often name the class), and the low training loss is the reader memorising title → names for the 798 training sessions.
- Why the core cannot easily learn it: the question enters the core through an attention over a single key, which adds the same vector to all 16 slot queries. Picking then has to emerge as a bilinear match between that shift and the document keys, from about 10k examples, while the reader offers an easier shortcut.
- The reader-only control arm was stopped after 5 minutes: with the attention at chance in the main arm it could not change the reading.

### 12. Similarity bias inside the core's attention

[poc_core.py](poc_core.py) with `--bias 1`, [run_bias.sh](run_bias.sh). The encoder's own similarity already picks: cosine(question, keyed fact) ranks an asked fact first in 95% of cases among all 530 held-out facts (93% among 3,281 training facts), and a softmax over 50 × cosine puts 87% of its mass on the asked facts. So the change to the core is: attention logit += beta[layer, head] × cosine(question, fact key). That is 16 new parameters (initialised at 50), every original weight frozen, trained with the reader for 90 min (6,124 examples, loss 0.30; beta drifted to 75–100). Each fact has a key vector (the keyed sentence) and a value vector the attention reads: the same keyed sentence, or the plain one-name sentence from tests 8–10. `core_bias.pt` holds only beta.

60 held-out target sessions, same piles as test 11, one 16-token block. "func" is any asked name / share of asked names. Attention is the second layer's share.

| pile | value vector | repo | file | func | attention on target session | on the asked fact |
|---|---|---|---|---|---|---|
| 1 session | keyed | 0.10 | 0.07 | 0.43 / 0.29 | 1.00 | 0.80 |
| 1 session | plain | 0.12 | 0.10 | **0.65** / 0.43 | 1.00 | 0.70 |
| 16 | keyed | 0.12 | 0.03 | 0.44 / 0.31 | 0.99 | 0.79 |
| 16 | plain | 0.12 | 0.10 | **0.67** / 0.44 | 1.00 | 0.70 |
| 137 (capped at 200 facts) | keyed | 0.10 | 0.08 | 0.41 / 0.29 | 0.97 | 0.78 |
| 137 (capped at 200 facts) | plain | 0.12 | 0.08 | **0.67** / 0.43 | 0.99 | 0.70 |
| 16, target absent | keyed | 0.00 | 0.00 | 0.02 | 0 | 0 |
| 16, target absent | plain | 0.00 | 0.00 | 0.04 | 0 | 0 |

Uncapped piles ([run_bias2.sh](run_bias2.sh), `core_eval_bias_big.json`), plain value vectors, same 60 sessions:

| pile | facts in the pile | repo | file | func | attention on target session | on the asked fact |
|---|---|---|---|---|---|---|
| all 137 held-out sessions | 532 | 0.10 | 0.08 | **0.63** / 0.42 | 0.99 | 0.70 |
| all 1,037 sessions | 4,256 | 0.10 | 0.07 | **0.61** / 0.42 | 0.96 | 0.68 |
| 16, target absent | 61 | 0.00 | 0.00 | 0.04 | 0 | 0 |

The 4,256-fact pile adds the training and validation sessions as distractors. One 16-token block then stands for about 4,256 facts and returns the asked function name 61% of the time, against 65% when the pile is the target session alone.

- Picking works. Recall does not fall as the pile grows, and it is near zero when the target is absent, so the answer is read from the memory and not guessed from the issue title.
- The value vector matters: the plain one-name sentence reads at 65–67%, the keyed sentence (with the issue title in it) at 41–44%. The title crowds out the name in the vector. Key and value should be separate vectors.
- Recall is nearly flat from 4 facts to 4,256 (65% → 61%).
- The level is the single-block level of test 10 (pooled 50%, 4 slots per fact 70%), now held at any pile size. It is below the 78–81% of one block per fact.
- "Share of names" (0.43) is below "any name" (0.67): when a session has several functions, the block returns some of them.
- Repos and file paths stay at about 10% (`enowars/enochecker` → "en0/awstrue", `hatch_vcs/version_source.py` → "src/hatch/version.py").
- The first table's piles were capped at 200 facts by a limit left from test 11, so "137 sessions" there means 200 facts. The second table has no cap. The same cap applied to the 64- and 137-session rows of test 11.

### 13. Non-code facts: people in the user's life

[poc_general.py](poc_general.py), [run_general.sh](run_general.sh). 700 synthetic people and 4,600 one-sentence facts over 14 attributes ("Priya Nair lives in Lisbon." / "Which city does Priya Nair live in?"). Held out for the test: 200 people whose first and last names never occur in training, a quarter of every value list, and three whole attributes (instrument, usual drink, hometown). Same mechanism as test 12, one 16-token block; the fact vector is both key and value. The pile is all facts of N people, so it always holds the same person's other facts and other people's facts of the same attribute.

Cosine alone ranks the asked fact first in 99.9% of 1,600 test questions among all 4,600 facts.

| pile | facts | no training (reader from test 12), n = 60 | after 75 min mixed training, n = 150 | held-out attributes | unseen values | attention on the asked fact |
|---|---|---|---|---|---|---|
| 1 person | 8 | 0.65 | **0.89** | 0.88 | 0.76 | 0.89 |
| 16 people | 128 | 0.65 | **0.89** | 0.88 | 0.82 | 0.89 |
| 200 people (all test) | 1,600 | 0.65 | **0.87** | 0.88 | 0.76 | 0.84 |
| 700 people (all) | 4,600 | n/a | **0.90** | 0.90 | 0.79 | 0.84 |
| 16 people, asked fact removed | 127 | 0.00 | 0.01 | 0.02 | 0.00 | 0 |

The last four columns are after training. Training: reader LoRA and beta only, started from test 12's, 5,476 examples (60% people, 40% code), loss 0.05, beta 80–128. Saved as `lora_gen.pt` and `core_gen.pt` (beta only).

- The reader trained only on code names already reads people facts at 65% with no training, so it learned to read the block, not a code vocabulary.
- After training, recall is 87–90% at every pile size, including 4,600 facts in 16 tokens.
- Attributes never trained on score as well as trained ones (88–90%). Values from the held-out quarter are lower (76–82%).
- Code recall did not suffer: 16 sessions, plain values, functions 0.70 / 0.50 (0.67 / 0.44 before), repos 0.20 (0.12), files 0.08 (0.10), target absent 0.04.
- Misses are near-synonyms, so the vector carries the gist better than the wording: "shellfish" → "shrimp", "chai latte" → "latte", "tattoo artist" → "artist", "black Kia Sportage" → "black Kia Optima", "Driftwood Insurance" → "Dawn Insurance".
- With the asked fact removed, the reader does not abstain. It invents an answer ("Welsh" → "Taliban", "Montevideo" → "Durban"). It was never trained to say "not in memory".
- The questions here name the person and the attribute, so they resemble the fact. The buried-clue questions do not (test 14).
- Lowest free memory 14% (training), no kills.

### 14. Buried clues: "matters when" keys written at ingest

[poc_keys.py](poc_keys.py), [run_keys.sh](run_keys.sh), [poc_keys_stress.py](poc_keys_stress.py). Buried-clue set from the proof of concept: 252 questions, each with one clue chat among 150 candidates ("What should I pack for a long day hike?" needs "has type 1 diabetes"). Qwen3-8B read each of the 375 chats once, without seeing any question, and wrote up to 3 lines "Fact: … / Matters when: a; b; c; d". Each fact and each matters-when item was embedded; a chat's score for a question is its best cosine. No reader involved.

- Note writing: 71 of 375 chats got at least one fact (204 facts). 63 of the 84 clue chats did; 21 got "NONE" (no-car 6, budget 5, flying 3, …). The clue words are in a stored fact for 67% of the questions.
- Among each question's own 150 candidates the clue is in the top 5 every time (plain chat cosine: 1%), first in 57%. That number is an artifact: only about 2 of the 149 fillers have any key, so there is almost nothing to rank against.
- Fair pool: the clue chat against every other chat with stored facts, leaving out other clue chats of the same type (about 60 chats, all lasting personal facts):

| split | questions | clue stored | clue ranked first | top 3 | top 5 | attention share at beta 100 |
|---|---|---|---|---|---|---|
| all | 252 | 0.67 | 0.01 | 0.12 | 0.18 | 0.02 |
| train types | 120 | 0.70 | 0.03 | 0.16 | 0.22 | 0.04 |
| held-out types | 72 | 0.54 | 0.00 | 0.00 | 0.00 | 0.00 |

- Why: keys written without the question are generic ("Outdoor activities", "Providing safety advice") and many facts claim the same situations. "Suggest a fun night out for this weekend." matches another fact's "Suggesting weekend-friendly activities" at 0.82 and the clue's best key at 0.54–0.57.
- Several of the competing facts are relevant too (hay fever does matter for a day hike). For open requests the target is not one fact but the handful of constraints that apply.
- Sam's demo chats: only 7 of 150 chats yield a lasting fact, the 4 clue chats among them. The right clue ranks 3rd, 3rd and 1st of those 7 for the three buried questions.
- Reading: a person's lasting constraints are few. Picking them by similarity fails, but they are few enough to keep in memory always (test 10: 2–4 tokens per fact), with the picked 16-token block used for direct questions.

### 15. Always-on profile: every lasting fact in the prompt, reader trained to act

[poc_profile.py](poc_profile.py), [run_night.sh](run_night.sh). No lookup: each profile fact goes through the untouched core alone, the first 4 of its 16 slots are kept, and the blocks are concatenated (profile of 8 = 32 tokens). Reader: `lora_gen.pt` trained 60 more minutes (3,500 examples, loss 0.83 → 0.3) to answer an open request using the one relevant fact among distractors; targets are Qwen3-8B answers written with the fact as text. Training facts: 114 stored facts of 8 clue types plus 211 invented facts in 22 categories; 20% people-fact questions kept in the mix. Held out: the 4 clue types health, no car, toddler, flying (72 questions), and 30 questions of trained types from unseen chats. "Acts" = the answer follows the constraint (strict rubric); "mentions" = it names the fact.

| group | condition | tokens | before training: mentions / acts | after: mentions / acts |
|---|---|---|---|---|
| held-out types | clue fact alone | 16 | 0.64 / 0.08 | 0.12 / 0.14 |
| held-out types | clue fact alone | 4 | 0.36 / 0.14 | 0.11 / 0.14 |
| held-out types | profile of 4 | 16 | 0.17 / 0.17 | 0.03 / 0.12 |
| held-out types | profile of 8 | 32 | 0.17 / 0.11 | 0.01 / 0.11 |
| held-out types | profile of 16 | 64 | 0.08 / 0.19 | 0.01 / 0.08 |
| held-out types | profile of 8, clue removed | 32 | 0.00 / 0.03 | 0.00 / 0.14 |
| trained types, new chats | clue fact alone | 16 | not run | 0.43 / 0.43 |
| trained types, new chats | clue fact alone | 4 | not run | 0.47 / 0.43 |
| trained types, new chats | profile of 4 | 16 | not run | 0.23 / 0.27 |
| trained types, new chats | profile of 8 | 32 | not run | 0.20 / 0.20 |
| trained types, new chats | profile of 16 | 64 | not run | 0.10 / 0.13 |
| trained types, new chats | profile of 8, clue removed | 32 | not run | 0.00 / 0.00 |

"Before" is 36 of the 72 held-out questions; "after" is all 72.

- Held-out fact types: no effect. The answer follows the constraint as often with the clue removed (14%) as with it present (8–14%); those hits are chance (a New York–Chicago trip by train, a getaway that happens not to need a car).
- Training changed the style, not the use: before, the reader recited the fact (64% mention it alone) without acting; after, it writes a normal answer and ignores an unfamiliar kind of fact ("For a long day hike, pack water, snacks, a map, a first-aid kit…" with the diabetes fact as the only memory).
- Trained types: a real but weak effect (43% alone against 0% removed), and it halves as the profile grows: 27% in 4 facts, 20% in 8, 13% in 16. By type, alone → profile of 8: schedule 67% → 44%, diet 58% → 17%, pet 0% → 0%.
- Two kinds of miss inside a profile: the reader acts on a distractor ("Since you're on a budget…" when the relevant fact is "vegan"), or on a blurred copy of the clue ("Since you don't have a driver's license…" for "doesn't have a car").
- 4 tokens per fact are not the limit here: alone, 4 and 16 tokens score the same (43%).

**Controls with plain text (plain Qwen3-8B, no core, no LoRA; run on Modal, 4.6 min).** Same questions and distractors. `poc_profile.py text`: the profile is pasted as text. `poc_profile.py select`: the model first lists the fact numbers that matter for the request, then answers with only those as text.

| group | condition | fact tokens | acts | with the clue removed |
|---|---|---|---|---|
| held-out types | no profile | 0 | 0.14 | – |
| held-out types | text: clue fact alone | 13 | **0.64** | – |
| held-out types | text: profile of 8 | 128 | 0.32 | 0.10 |
| held-out types | text: profile of 16 | 258 | 0.19 | – |
| held-out types | two-step: profile of 8 / 16 / 48 | – | 0.25 / 0.18 / 0.12 | 0.11 / 0.11 / 0.12 |
| trained types, new chats | no profile | 0 | 0.00 | – |
| trained types, new chats | text: clue fact alone | 14 | **0.73** | – |
| trained types, new chats | text: profile of 8 | 129 | 0.43 | 0.00 |
| trained types, new chats | text: profile of 16 | 262 | 0.43 | – |
| trained types, new chats | two-step: profile of 8 / 16 / 48 | – | 0.40 / 0.33 / 0.17 | 0.03 / 0.00 / 0.03 |

- Compression is not the main loss. With the clue as plain text and nothing else the plain model acts on it in 64–73%; put it among 7 other facts as text and that halves (32–43%). The soft-token profile of 8 reaches 11–20%.
- The two-step selection is poor at this model size without reasoning: the clue is among the picked facts in 53% (of 8), 58% (of 16), 33% (of 48), while it picks 3.7, 7.9 and 7.7 facts. It often returns a run of consecutive numbers; the 24-token limit also cuts long lists.
- So the gap is the decoder's own use of a relevant fact among others, in one or two sentences, under a strict rubric. A strict per-answer rubric may undercount: with eight plausible constraints the answer honours some other ones ("eco-friendly and fits within your budget").
- Followed up in test 17.

### 16. Codebase scale: all of Django in one store

[poc_django.py](poc_django.py). Django 5.2 source parsed with `ast` (never imported or run): 36,263 facts (function, class, method: name, file, signature, first docstring line). Questions: describe a function, ask for its name ("name"), the same in other words ("paraphrased"), or ask for its file ("path"). n = 120 held-out facts. The core is untouched; the similarity bias from test 12 picks inside one 16-token block.

| condition | name | paraphrased | path |
|---|---|---|---|
| reader from test 13, no Django training: 1 fact | 0.23 | – | – |
| same, all 36,263 facts | 0.20 | 0.17 | 0.07 |
| same, asked fact removed | 0.11 | 0.10 | 0.07 |
| after 8 min of reader training on Modal: 1 fact | 0.38 | – | – |
| 1,000 / 10,000 facts | 0.38 / 0.36 | – | – |
| all 36,263 facts | 0.34 | 0.28 | 0.30 |
| asked fact removed | 0.19 | 0.17 | 0.24 |
| plain decoder, no context | 0.13 | 0.11 | 0.26 |
| plain decoder, top 1 fact as text (35 tokens) | 0.87 | 0.81 | – |
| plain decoder, top 3 facts as text (82 tokens) | 0.93 | 0.93 | 0.96 |

- Scale is not the problem: 1 fact 0.38, all 36,263 facts 0.34; attention on the asked fact 0.70 (name), 0.63 (paraphrased), 0.65 (path).
- Exact reading is: with the right fact alone in 16 tokens the name comes back in 38%. Misses are near-names (`get_ordering` for `get_ordering_field_columns`, `_resolve_related_objects` for `prefetch_related_objects`). The core was trained to carry meaning, not spelling.
- Text retrieval over the same vectors gets 93% with 82 tokens. For exact identifiers the compressed route is far behind and costs only 5 times fewer tokens.
- More reader training does not help. After 20 more minutes (`lora_dj28.pt`, run `dj2`): name 0.38 with 1 fact, 0.41 / 0.40 with 1,000 / 10,000, 0.36 with all (0.20 removed); paraphrased 0.31 (0.19); path 0.24 (0.18). The limit is what the frozen core keeps of a name, not the reader.

### 17. Who should pick the fact: the decoder's yes/no, or a learned map

[poc_relevance.py](poc_relevance.py). Same 102 questions as test 15 (72 held-out types, 30 trained types), profiles of 8 / 16 / 48 facts.

**a. Decoder yes/no per fact** (`teach`): "Should an assistant take this fact into account when answering this request?", yes-minus-no logit, one forward pass per pair (129 pairs/s on Modal).

| | held-out types | trained types |
|---|---|---|
| says yes to the clue | 0.94 | 0.87 |
| says yes to the other facts | 0.67 | 0.54 |
| clue ranked first in a profile of 8 / 16 / 48 | 0.26 / 0.14 / 0.01 | 0.43 / 0.33 / 0.27 |
| clue in the top 3 of 8 / 16 / 48 | 0.71 / 0.46 / 0.06 | 0.73 / 0.60 / 0.40 |
| acts, top 3 facts given as text, profile of 8 / 48 | 0.46 / 0.17 | 0.40 / 0.37 |
| same with the clue removed | 0.12 / 0.12 | 0.03 / 0.07 |

The model says yes to most facts; some of those are real constraints (a wheelchair for "plan a weekend getaway"), some are not ("detail-oriented" for "what should I pack for a hike").

**b. Fair profiles** (`fair`): the other facts are only ones the yes/no judges irrelevant to the request.

| | clue alone | profile of 8 | profile of 16 | of 8, clue removed |
|---|---|---|---|---|
| text, held-out types | 0.69 | 0.20 | 0.15 | 0.09 |
| text, trained types | 0.73 | 0.54 | 0.64 | 0.00 |
| soft tokens (reader of test 15), held-out types | 0.22 | 0.11 | 0.10 | 0.14 |
| soft tokens, trained types | 0.43 | 0.11 | 0.09 | 0.00 |

So the drop in test 15 is not only competition between real constraints. Given 8 facts as text the plain decoder tries to use all of them ("pack a durable silicone case for your phone and bring halal-friendly snacks") and the one that matters gets lost. Pasting the whole profile does not work for this decoder; something has to select first.

**c. A learned relevance map** (`distill`, CPU, seconds): score(q, f) = scale · cos(qA, fB) + w · cos(q, f), rank 64, over the frozen 2560-d vectors. Clue ranked first in a profile of 8 / 16 / 48, held-out types:

| selector | 8 | 16 | 48 |
|---|---|---|---|
| similarity | 0.22 | 0.11 | 0.04 |
| decoder yes/no | 0.26 | 0.14 | 0.01 |
| map distilled from the yes/no scores (20,128 pairs) | 0.17 | 0.06 | 0.00 |
| map trained on 629 known (request, fact) pairs, random negatives, 3 seeds | 0.39 | 0.27 | 0.13 |
| map trained on 720 known pairs, in-batch negatives ([poc_broad.py](poc_broad.py) smoke run, 1 seed, 3 epochs) | 0.47 | 0.31 | 0.21 |

- The map trained on known pairs beats both similarity and the decoder's judgement on fact types it never saw, from a few hundred pairs. This is the "car relates to travel" association, learned inside the memory and applied to vectors.
- Distilling the decoder's yes/no is worse than not using it: the teacher is too permissive.
- Test 18 scales the pairs about 5 times (80 categories) and trains the reader on what the map picks.

### 18. Broad data: 80 fact categories for the map and the reader

[poc_broad.py](poc_broad.py), Modal run `broad` (27.3 GPU-minutes). Qwen3-8B invented 1,397 facts in 80 categories (personal, work, company), 2,735 requests where one fact matters, and 1,365 generic answers. Held-out clue types are not among the categories.

**Map** (rank 64, in-batch negatives, 3 seeds), clue ranked first | in the top 3, profile of 8 / 16 / 48, held-out types:

| training pairs | first | top 3 |
|---|---|---|
| similarity only | 0.22 / 0.11 / 0.04 | 0.61 / 0.38 / 0.18 |
| 336 | 0.16 / 0.10 / 0.05 | 0.43 / 0.27 / 0.12 |
| 1,008 | 0.24 / 0.11 / 0.04 | 0.60 / 0.37 / 0.14 |
| 3,363 | 0.40 / 0.28 / 0.17 | 0.70 / 0.56 / 0.32 |

Still rising with data. Trained types at 3,363 pairs: first 0.52 / 0.43 / 0.24, top 3 0.77 / 0.70 / 0.47.

**Reader** (`lora_broad.pt`: 18 min from `lora_prof.pt`, 3,532 examples, 1 to 3 facts at 16 tokens each, a quarter with no relevant fact and a generic answer). Acts, new reader vs the reader of test 15:

| condition | held-out types (72) | trained types (30) |
|---|---|---|
| clue alone, 16 tokens | **0.29** vs 0.15 | 0.40 vs 0.47 |
| map picks 3 of 8 (clue picked 0.68 / 0.77) | 0.06 vs 0.10 | **0.40** vs 0.17 |
| same, clue removed | 0.11 vs 0.12 | 0.00 |
| map picks 3 of 16 (0.53 / 0.67) | 0.03 vs 0.07 | 0.27 vs 0.23 |
| map picks 3 of 48 (0.35 / 0.40) | 0.07 vs 0.12 | 0.23 vs 0.13 |
| similarity picks 3 of 8 (0.61 / 0.57) | 0.06 vs 0.12 | 0.17 vs 0.13 |
| all 8 facts, 4 tokens each | 0.14 vs 0.12 | 0.17 vs 0.17 |

- Broad data doubles acting on an unseen fact type when the fact is alone (0.15 to 0.29). Plain text gets 0.69, so reading one unseen fact from 16 tokens is still less than half of text.
- On trained types the map plus the reader reach the alone level in a profile of 8 (0.40; similarity 0.17).
- On unseen types the reader does not act once 2 other facts share the memory, even when the clue was picked (4 of 49). The answers show two things: it acts on one of the other two facts (hay fever, halal) when those are constraints too, or it blends them ("if you have a food allergy" from diabetes plus lactose intolerance); and it gives a generic answer for "The user said: they don't have a car".

### 19. Restate the fact that matters, then answer

[poc_broad.py](poc_broad.py) `train --recite`, Modal run `recite` (19.9 GPU-minutes). `lora_broad.pt` trained 14 more minutes with the target "Relevant fact: <fact or none>" followed by the answer (`lora_recite.pt`). Acts | restates the clue in the first line:

| condition | held-out types | trained types |
|---|---|---|
| clue alone, 16 tokens | 0.36 (0.29 before) \| 0.62 | 0.60 (0.40) \| 0.70 |
| map picks 3 of 8 | 0.19 (0.06) \| 0.31 | 0.43 (0.40) \| 0.43 |
| same, clue removed | 0.10 \| 0.03 | 0.00 \| 0.00 |
| map picks 3 of 16 / 48 | 0.17 / 0.11 \| 0.31 / 0.24 | 0.23 / 0.13 \| 0.27 / 0.20 |
| all 8 facts, 4 tokens each | 0.12 \| 0.25 | 0.27 \| 0.30 |

- Writing the fact first helps everywhere, most where it was worst (3 facts, unseen types: 0.06 to 0.19).
- The fact itself is readable from its 16 tokens in 62–70% of cases when alone. With 3 facts in memory the reader names the right one in 31% on unseen types (it was among the 3 in 68%), so choosing among the 3 is the larger loss.
- When the clue is removed the reader almost never claims it (0–3%).

### 20. Controls: the same picked facts as plain text

`eval --text` (plain Qwen3-8B, no memory), runs `btext` (2.5 min) and `blong` (6.0 min). Acts, held-out types | trained types:

| condition | two-sentence answers | answers of any length |
|---|---|---|
| clue alone | 0.69 \| 0.73 | 0.81 \| 0.83 |
| map picks 3 of 8 | 0.26 \| 0.57 | 0.44 \| 0.70 |
| same, clue removed | 0.14 \| 0.13 | 0.10 \| 0.03 |
| map picks 3 of 16 | 0.24 \| 0.47 | 0.28 \| 0.63 |
| map picks 3 of 48 | 0.15 \| 0.37 | 0.28 \| 0.43 |
| similarity picks 3 of 8 | 0.25 \| 0.53 | 0.39 \| 0.43 |
| all 8 facts | 0.24 \| 0.53 | 0.42 \| 0.57 |

- "Answer in one or two sentences" was hiding acting: with three facts in the prompt the plain decoder spends the two sentences on the other facts. Without the limit, 3 of 8 goes from 0.26 to 0.44.
- Even so, with the clue among the three in 68% the ceiling of this route is about 0.55; the picker is the limit.
- The decoder's yes/no (test 17) adds little to the map (`combine`, run `comb`): clue ranked first in a profile of 8 / 16 / 48 on held-out types 0.43 / 0.32 / 0.22 with the map alone, 0.49 / 0.35 / 0.25 with the yes/no score added at equal weight. That is 2 to 4 questions of 72, for one decoder pass per stored fact.

### 21. Many more pairs for the map

`poc_broad.py pairs` (Modal run `big`, then `big2`): Qwen3-8B invented 7,348 more facts for 10 kinds of user (student, nurse, small-company owner, …) and 3 requests per fact, 21,716 rows; with the earlier data 25,076 training pairs over 9,060 distinct facts. Held-out clue types are still banned from the data. Map: rank 64, 3 seeds; clue ranked first | in the top 3, profile of 8 / 16 / 48.

| pairs | held-out types, first | top 3 | trained types, first | top 3 |
|---|---|---|---|---|
| similarity only | 0.22 / 0.11 / 0.04 | 0.61 / 0.38 / 0.18 | 0.33 / 0.13 / 0.07 | 0.57 / 0.43 / 0.20 |
| 752 | 0.18 / 0.10 / 0.06 | 0.41 / 0.23 / 0.11 | 0.33 / 0.27 / 0.14 | 0.56 / 0.43 / 0.28 |
| 2,507 | 0.25 / 0.16 / 0.09 | 0.58 / 0.38 / 0.18 | 0.32 / 0.29 / 0.13 | 0.59 / 0.44 / 0.29 |
| 7,522 | 0.32 / 0.16 / 0.06 | 0.65 / 0.44 / 0.22 | 0.37 / 0.31 / 0.24 | 0.68 / 0.52 / 0.37 |
| 25,076 | **0.49 / 0.36 / 0.15** | **0.75 / 0.57 / 0.32** | 0.46 / 0.38 / 0.21 | 0.70 / 0.58 / 0.39 |
| 25,076, rank 256 | 0.50 / 0.39 / 0.21 | 0.71 / 0.61 / 0.36 | 0.46 / 0.38 / 0.27 | 0.72 / 0.62 / 0.38 |

- Range of "first" over 3 seeds at 25,076 pairs, held-out, profile of 8: 0.46–0.53.
- Below about 2,500 pairs the map is worse than plain similarity on unseen types; the gain comes from data, and the curve has not flattened at 8 or 16.
- Rank 256 helps only in the large profile (0.15 → 0.21 at 48; seeds 0.14–0.28, so within noise).
- Generating the pairs took 8.2 GPU-minutes; the four map sizes with 3 seeds each trained in 2.2 minutes in total.

### 22. The new map in the answer tests; top 1 against top 3

Modal run `big2` (26.6 GPU-minutes), map `relmap_big.pt`. Acts on the clue, held-out types | trained types.

| condition | clue among the picked | plain decoder, facts as text, any length | recite reader, 16 tokens per fact |
|---|---|---|---|
| clue alone | 1.00 | 0.74 \| 0.87 | 0.36 \| 0.63 |
| map picks 3 of 8 | 0.75 \| 0.60 | 0.50 \| 0.60 | 0.25 \| 0.33 |
| same, clue removed | 0 | 0.12 \| 0.00 | 0.06 \| 0.03 |
| map picks 3 of 16 | 0.56 \| 0.53 | 0.32 \| 0.57 | 0.19 \| 0.27 |
| map picks 3 of 48 | 0.31 \| 0.37 | 0.35 \| 0.40 | 0.12 \| 0.17 |
| map picks 1 of 8 | 0.46 \| 0.37 | 0.33 \| 0.40 | 0.26 \| 0.33 |
| map picks 1 of 16 | 0.32 \| 0.27 | 0.22 \| 0.27 | 0.22 \| 0.23 |
| map picks 1 of 48 | 0.10 \| 0.13 | 0.14 \| 0.13 | 0.15 \| 0.10 |
| map picks 1 of 8, clue removed | 0 | 0.14 \| 0.07 | 0.07 \| 0.00 |
| whole profile of 8 | 1.00 | 0.54 \| 0.70 | 0.17 \| 0.20 |
| same, clue removed | 0 | 0.25 \| 0.10 | 0.06 \| 0.00 |
| whole profile of 16 | 1.00 | 0.31 \| 0.53 | 0.11 \| 0.23 |

- With the old map the same text test gave 0.44 at 3 of 8, 0.28 at 16 and 48 (test 20); the recite reader gave 0.19 / 0.17 / 0.11 (test 19). Gains of 3 to 6 points are 2 to 4 questions of 72.
- **The test cannot separate close results.** The identical condition "clue alone, text, any length" scored 0.81 in run `blong` and 0.74 here (batches are composed differently, so greedy outputs differ slightly). Treat differences under about 10 points on 72 questions as noise.
- Top 1 is not better than top 3 for text (0.33 against 0.50): the map puts the clue first in only 46%. For the memory reader top 1 and top 3 are equal (0.26, 0.25), because it loses the clue among three facts about as often as the map misses it.
- With the whole profile as text the rubric fires by accident in 25% with the clue removed (other facts such as a dog that is distressed in cars match the no-car rubric), so 0.54 there is a gain of about 29 points, less than top 3 (38 points).
- The trained-types column has 30 questions; the map picked the clue into the top 3 less often here (0.60) than the old map did in test 18 (0.77), within seed range (0.37–0.53 for "first").

### 23. Restate-only reader, two passes

Modal runs `restate` (33.2), `restated` (33.1, `--delta 256`), `restate8` (29.9, `--max-facts 8`). Each: 20 minutes of training from `lora_recite.pt` on all 25,076 known pairs with the target `Relevant fact: <fact or none>`; then pass 1 writes the note from memory tokens, pass 2 is the plain decoder answering at any length with only that note as text.

| held-out types | recite reader (test 22) | restate | + core add-on | max 8 facts |
|---|---|---|---|---|
| clue alone: restates the clue | 0.65 | 0.21 | 0.24 | 0.26 |
| clue alone: writes "none" | 14 of 72 | 48 of 72 | 46 of 72 | 48 of 72 |
| map picks 3 of 8: restates | 0.35 | 0.12 | 0.12 | 0.14 |
| whole profile of 8: restates | 0.29 | 0.04 | 0.06 | 0.01 |
| two-pass answer, clue alone: acts | n/a | 0.32 | 0.26 | 0.32 |
| two-pass, map picks 3 of 8: acts | n/a | 0.28 | 0.25 | 0.26 |
| same, clue removed | n/a | 0.15 | 0.14 | 0.17 |

- All three readers learned to answer "none": two thirds of the time even when the only fact in memory is the clue, and on trained types too (20–21 of 30). Training loss was low (0.20–0.24), so this is what the data taught.
- Likely cause, not verified: the "none" examples take look-alike facts from a pool of 9,065 facts that has many near-duplicates of the true fact (10 user groups × the same categories), so a fact that does matter is often shown with the target "none". The fix is to drop look-alikes that are close to the true fact; not run yet.
- The core-output add-on and training on up to 8 facts made no difference, but with the reader collapsed to "none" this test says nothing about either idea.
- Cost of the lesson: about 96 GPU-minutes on three runs that shared one flaw. A 4-minute training run with the restate rate printed would have shown it; the half-minute smoke run was too short to tell.

### 24. Long-answer reader

Modal run `long` (14.5, out of memory at batch 8 after writing the data) then `long2` (37.4). `poc_broad.py longans` wrote full-length teacher answers for the 2,735 broad rows; the reader trained on them for 22 minutes at batch 4, starting from `lora_prof.pt` like the old reader, default map (`relmap_broad.pt`). Test without the two-sentence limit, 260 new tokens. Acts on the clue, held-out | trained types.

| condition | long-answer reader | old reader (`lora_broad.pt`), same test | plain decoder, text (test 20) |
|---|---|---|---|
| clue alone, 16 tokens | **0.62** \| 0.50 | 0.50 \| 0.60 | 0.81 \| 0.83 |
| map picks 3 of 8 | 0.24 \| 0.27 | 0.21 \| 0.23 | 0.44 \| 0.70 |
| same, clue removed | 0.10 \| 0.00 | 0.15 \| 0.03 | 0.10 \| 0.03 |
| map picks 1 of 8 | 0.29 \| 0.23 | 0.28 \| 0.40 | n/a |
| same, clue removed | 0.12 \| 0.03 | 0.15 \| 0.03 | n/a |
| map picks 3 of 48 | 0.22 \| 0.10 | 0.15 \| 0.10 | 0.28 \| 0.43 |
| same, clue removed | 0.19 \| 0.03 | 0.15 \| 0.03 | 0.15 \| 0.07 |

- Most of the earlier gap was the test: the old reader scored 0.29 with two-sentence answers (test 18) and 0.50 when allowed to answer at length. Training on long answers adds 12 points on unseen types (9 questions) and nothing on trained types.
- Alone, memory tokens are now at 0.62 against 0.74–0.81 for the same fact as text.
- With three facts in memory the reader is at 0.24 against 0.44–0.50 for text, and barely above the clue-removed floor. Reading one fact out of several memory blocks is the reader's open problem; the recite step (test 19) is the only thing so far that moved it.
- Top 1 hand-over: 0.29 with the clue picked first in 43%, which is 0.62 × 0.43 plus the floor. With a perfect picker this route would be at about 0.62.

### 25. A larger test set

[poc_broad.py](poc_broad.py) `--big`. Each held-out clue (4 types × 2 values × 3 clue sentences) now gets 9 more questions written for its type (`XQ`), on top of its 3: 288 unseen-type questions instead of 72 (one question = 0.35 points). The other facts in each profile are the same pools, rotated. Trained types stay at 30 questions. All numbers from here on use this test.

Map on 25,076 pairs, 3 seeds, Modal run `xq`. Clue ranked first | in the top 3, profiles of 8 / 16 / 48:

| | first | top 3 |
|---|---|---|
| similarity | 0.28 / 0.18 / 0.07 | 0.62 / 0.41 / 0.19 |
| map, 25,076 pairs | 0.38 / 0.27 / 0.14 | 0.65 / 0.49 / 0.28 |
| same map on the 72-question test (test 21) | 0.49 / 0.36 / 0.15 | 0.75 / 0.57 / 0.32 |

- The gain of the map over similarity is +10 points first at 8 facts, not +27. The three original questions per clue were easier for the map than the nine new ones.
- Baselines on this test (`broad_eval_text_big.json`, `broad_eval_long_big.json`), acts, unseen types: plain decoder with the fact as text, alone 0.80; top 3 of 8 as text 0.39 (clue removed 0.15); top 1 of 8 0.31 (0.17); top 3 of 48 0.24 (0.17). Long-answer reader `lora_long.pt`: alone 0.58; 3 of 8 0.26 (0.17).

### 26. 2.7 times more pairs for the map

Modal run `pairs2` (35.2 GPU-minutes). 20 more user groups, 1,600 prompts, 14,388 new facts and 42,302 new requests; with the earlier pairs 67,376 training pairs over 23,439 facts. 3 seeds, unseen types, first | top 3:

| | first, 8 / 16 / 48 | top 3 |
|---|---|---|
| 20,212 pairs (30%) | 0.36 / 0.26 / 0.17 | 0.64 / 0.48 / 0.29 |
| 67,376 pairs | 0.37 / 0.27 / 0.18 | 0.66 / 0.47 / 0.31 |
| 67,376, rank 256 | 0.39 / 0.28 / 0.18 | 0.65 / 0.48 / 0.30 |
| 67,376, near-copies masked as negatives (cosine > 0.85) | 0.38 / 0.27 / 0.18 | 0.66 / 0.48 / 0.31 |

- Flat from 20,000 pairs on. The loss on held-out facts of the generated kind still falls (2.13 → 1.84), so the map keeps learning the generated data, and that no longer carries over to the unseen types.
- Trained types (n = 30): 0.37 / 0.32 / 0.11 first, inside the spread of 3 seeds of the smaller map.
- Answers with `relmap_huge.pt` (`broad_eval_text_big_huge.json`), plain decoder, top 3 as text: 0.43 at 8 facts (clue among them 0.65; removed 0.15), 0.26 at 48 (0.31; removed 0.17). With `relmap_big.pt`: 0.39 and 0.24.

### 27. Restate first with long answers; blocks sorted best-first

Modal runs `lrec` (23.8 GPU-minutes) and `lsort` (27.7). Both continue `lora_long.pt` for 18 minutes at batch 4 with the target "Relevant fact: <fact or none>" followed by the full-length answer; `lsort` also orders the memory blocks by map score, best first, in training and in the test. Unseen types | trained types, acts:

| reader | alone | 3 of 8 | 3 of 8, clue removed | restates the clue, alone / 3 of 8 | writes "none", alone |
|---|---|---|---|---|---|
| `lora_long.pt` | 0.58 \| 0.53 | 0.26 \| 0.23 | 0.17 \| 0.00 | – | – |
| `lora_lrec.pt` | 0.52 \| 0.53 | 0.30 \| 0.17 | 0.17 \| 0.00 | 0.47 / 0.24 | 0.31 |
| `lora_lsort.pt` | **0.64 \| 0.70** | **0.32 \| 0.33** | 0.15 \| 0.00 | 0.66 / 0.27 | 0.10 |
| same picked facts as text, plain decoder | 0.80 \| 0.83 | 0.39 \| 0.60 | 0.15 \| 0.03 | – | – |

- The clue is among the 3 picked in 62%. Above the clue-removed floor, text gains 24 points and the sorted reader 17, the long reader 9.
- Sorting helped more with the fact alone (fewer "none": 0.10 against 0.31) than among three; with one block there is nothing to sort, so that part is run-to-run variation of training or a side effect of the changed training mix. One seed.
- When the clue is among the 3, the sorted reader restates it in 0.27 / 0.62 = 44% of cases; alone 66%.

### 28. The map's score inside the core's attention

Modal run `attn` (27.7 GPU-minutes). All facts of a profile go through the core at once into one 16-token block; the map's score for each fact is added to the cross-attention logits, times a learned strength per layer and head (start 1). Reader continued from `lora_long.pt`, 18 minutes, up to 12 facts per training profile. The core stays frozen.

| unseen types, acts | with the clue | clue removed |
|---|---|---|
| clue alone | 0.42 | – |
| profile of 8, one block | 0.18 | 0.17 |
| profile of 48, one block | 0.18 | 0.18 |
| map picks 1 of 8, that fact alone | 0.25 | – |

- No effect of the clue's presence. The strengths grew to 1.8–2.8, so training used the bias, but with a map that puts the clue first in 38% the block mostly carries another fact, and the reader lost single-fact reading (0.58 → 0.42).
- Test 12 worked with a bias because similarity found the asked fact almost always (attention 0.96–1.00). Here the score itself is wrong most of the time.

### 29. Restate-only reader without near-copies

Modal run `restate2` (23.4 GPU-minutes), from `lora_recite.pt`, candidates whose cosine to the right fact is ≥ 0.6 (plain or map space) dropped; 5.1 of 12 look-alikes kept.

- After training, on 64 training-style examples with the fact present: writes "none" 0.20, restates word for word 0.31.
- Unseen types: restates the clue alone 0.30, writes "none" 0.61; 3 of 8: 0.18 / 0.72. Second pass (plain decoder answers from the note): acts 0.34 alone, 0.21 at 3 of 8, 0.12 with the clue removed.
- Near-copies were not the cause of the collapse. Left as a guess: with only one short line to write, "none" is the cheap answer whenever the 16 tokens are not read confidently; the recite readers that also write the answer do not collapse (0.10–0.31 "none").

### Where the losses are (tests 25–29, by fact type)

From the saved answers (`broad_eval_text_big.json`, `broad_eval_text_big_huge.json`, `lsort_eval.json`), unseen types, 72 questions per type.

| | health | toddler | no car | flying |
|---|---|---|---|---|
| rubric fires with the clue removed (text, 3 of 8) | 0.00 | 0.01 | 0.24 | 0.36 |
| fact alone as text, acts | 0.99 | 0.86 | 0.71 | 0.65 |
| fact alone in 16 tokens, sorted reader | 0.51 | 0.51 | 0.74 | 0.78 |
| map (25,076 pairs) has the clue in its top 3 of 8 (chance 0.38) | 0.24 | 0.47 | 0.81 | 0.97 |
| same, 67,376 pairs | 0.22 | 0.61 | 0.78 | 0.99 |
| top 3 of 48, 25,076 / 67,376 pairs (chance 0.06) | 0.04 / 0.03 | 0.06 / 0.08 | 0.31 / 0.26 | 0.54 / 0.85 |
| text acts when the clue is among the 3 of 8 | 0.71 | 0.59 | 0.55 | 0.66 |
| sorted reader acts when the clue is among the 3 of 8 | 0.12 | 0.24 | 0.48 | 0.74 |
| end to end, 3 of 8: text / sorted reader | 0.17 / 0.03 | 0.29 / 0.11 | 0.46 / 0.39 | 0.64 / 0.74 |

- The rubrics for flying and no car fire without the clue in 36% and 24% of answers (generic travel advice matches them). Health and toddler have a clean floor, so they are the types to read.
- The map finds facts that share a topic with the request (fear of flying ↔ a trip: 97–99%) and misses facts that matter whatever the topic (a health condition ↔ a beach day: 22–24% in the top 3 of 8, below the 38% of a random pick; 3–4% of 48). More pairs helped flying at 48 facts (54% → 85%) and toddler at 8 (47% → 61%), not health.
- Among three blocks the reader acts on a picked health or toddler fact in 12% and 24%; the plain decoder with the same three facts as text in 71% and 59%. The totals in test 27 are carried by flying and no car, where the floor is high.
- So for the clean types the full pipeline is at 3% and 11% at 8 facts; text with the same picks 17% and 29%; the fact alone as text 99% and 86%.

### 30. Standing facts: a score that does not depend on the request

[poc_standing.py](poc_standing.py), on the Mac, saved vectors only (no model; Modal credits were used up). `work/runs/poc/standing.json`, log `work/logs/standing.log`. Each stored fact gets one score from a ridge fit on its encoder vector, trained on the 23,117 generated facts. The picker then keeps the best-scoring facts always on and fills the rest with the map (`relmap_big.pt`). Four ways to get the training target:

- **labels narrow**: the fact's generated category is one of 13 I listed as body and dependents (diets, allergies, chronic conditions, medication, mobility, physical limits, injuries, heat and cold, fasting, accessibility, family at home, elderly relatives, pets).
- **labels wide**: those plus 12 more (money, working hours, free time, sleep, religion, alcohol and smoking, fears, dislikes, things never learned, commute, borrowed vehicles).
- **gap**: 1 − cosine between the fact and the requests generated for it. **spread**: 1 − cosine among those requests. No labels.

The lists were written after seeing which test types fail, so this is not a blind test. The held-out types themselves are still absent from the generated facts.

Does the score carry over to categories it was not trained on (5 folds over whole categories)? AUC 0.83 narrow, 0.76 wide. Gap and spread, correlation on held-out facts: 0.72 and 0.49.

Clue in the top 3 by the standing score alone, unseen types, profiles of 8 / 16 / 48 (map alone; a random score):

| | health | toddler | no car | flying |
|---|---|---|---|---|
| map | 0.24 / 0.14 / 0.04 | 0.47 / 0.28 / 0.06 | 0.81 / 0.62 / 0.31 | 0.97 / 0.88 / 0.54 |
| random score | 0.44 / 0.15 / 0.07 | 0.40 / 0.15 / 0.12 | 0.29 / 0.15 / 0.06 | 0.29 / 0.21 / 0.08 |
| labels narrow | **0.96 / 0.65 / 0.24** | 0.67 / 0.14 / 0.00 | 0.00 / 0.00 / 0.00 | 0.14 / 0.00 / 0.00 |
| labels wide | 0.88 / 0.54 / 0.06 | 0.69 / 0.26 / 0.11 | 0.67 / 0.03 / 0.00 | 0.68 / 0.47 / 0.06 |
| gap | 0.44 / 0.17 / 0.17 | 0.54 / 0.15 / 0.06 | 0.22 / 0.00 / 0.00 | 0.94 / 0.64 / 0.22 |
| spread | 0.22 / 0.12 / 0.00 | 0.50 / 0.11 / 0.00 | 0.00 / 0.00 / 0.00 | 0.89 / 0.62 / 0.00 |

With a larger always-on set at 48 facts (labels narrow; random 0.17 for 8, 0.25 for 12): health clue in the top 8 in 0.68, top 12 in 0.94; toddler 0.07 and 0.32.

Three facts handed over, all unseen types, profiles of 8 / 16 / 48:

| picker | all types | health |
|---|---|---|
| map top 3 | 0.62 / 0.48 / 0.24 | 0.24 / 0.14 / 0.04 |
| 1 standing (narrow) + 2 map | 0.68 / 0.44 / 0.23 | 0.60 / 0.32 / 0.19 |
| 2 standing (narrow) + 1 map | 0.64 / 0.39 / 0.16 | 0.78 / 0.56 / 0.24 |
| map + 0.5 × standing (wide), both z-scored | 0.71 / 0.51 / 0.28 | 0.46 / 0.21 / 0.06 |
| map + 0.5 × gap | 0.66 / 0.49 / 0.31 | 0.24 / 0.11 / 0.04 |

- The encoder vector does say "this is a fact about the body" in a way a linear fit picks up, and that carries to conditions never generated (diabetes, migraine): the health failure of the map is fixable at 8–16 facts, and at 48 with an always-on set of about 12.
- It finds only what the list names. Toddler gains at 8 facts only; no car and flying get nothing from the narrow list. The label-free scores do not find health, so "matters whatever the topic" is not something the generated requests reveal by themselves; a person decided it.
- With the budget fixed at 3 facts the totals barely move: slots given to standing facts are taken from the map's topical picks (no car 0.81 → 0.75, flying 0.97 → 0.92 with one standing slot).
- Trained types (n = 30): 1 standing + 2 map 0.73 / 0.57 / 0.40 against the map's 0.60 / 0.53 / 0.37; the random score gives 0.70 / 0.47 / 0.27, so no clear effect there.
- A check on one explanation: the generated pairs are not more on-topic than the test. Cosine between request and fact: generated 0.52 on average (10th percentile 0.38); test questions and their clues 0.44 (health) to 0.52 (no car). So "the pairs are too topical" is not supported by this measure.
- Not tested: answers with these picks (needs the decoder), and how many other always-on facts a reader tolerates.

### A look at the saved answers: the decoder uses whatever fact it is shown

From `broad_eval_text_big.json` (plain decoder, one fact as text, unseen types). Share of answers that contain a third or more of the shown fact's content words (the same count for a fact that was not shown: 0.00–0.01):

| shown fact | n | uses it |
|---|---|---|
| the clue | 288 | 0.85 |
| the map's top pick, when it is not the clue | 189 | 0.76 |
| the map's top pick with the clue removed | 288 | 0.77 |

- Word overlap, not a rubric, so "uses" includes restating the fact in passing.
- Several of the map's "wrong" picks do matter for the question (lactose intolerance for a hike's food, a wheelchair for a night out). The test counts only the planted clue, so "clue first" understates how often the map picks something that matters, and it cannot tell a useful pick from a useless one.
- The decoder's behaviour with one fact in front of it is therefore not a clean teacher for relevance: it bends the answer around nearly any fact.

### 31. No picker: the whole profile as text, reader trained to restate and answer

[poc_whole.py](poc_whole.py), Modal run `whole` (22.1 GPU-minutes), results in `work/runs/poc/whole_eval.json`, reader `lora_whole.pt` (on the volume only). The user rejected hand-designed tiers, so nothing chooses facts here: every fact of the profile is in the prompt as plain text, in the same format as the earlier text controls, and the decoder's LoRA (fresh, rank 16) is trained for 11 minutes to write "Relevant fact: <fact or none>" and then a full-length answer. Training examples: profiles of 4 to 48 facts; 1,614 with an answer from `broad_long.json`, 2,203 restate-only from the 67,376 pairs, 362 with the fact left out (target "none"); 4,170 examples in 1,225 updates at 2,750 tokens/s. Text is the upper bound for memory tokens.

Acts, unseen types (untrained: every second question, 144; trained: all 288):

| Whole profile as text | all | health | toddler | no car | flying |
|---|---|---|---|---|---|
| Plain decoder, 8 facts | **0.58** | 0.67 | 0.58 | 0.53 | 0.53 |
| Plain decoder, 48 facts | **0.32** | 0.22 | 0.14 | 0.44 | 0.47 |
| Trained reader, 8 facts | 0.22 | 0.01 | 0.03 | 0.36 | 0.46 |
| Trained reader, 48 facts | 0.16 | 0.00 | 0.04 | 0.17 | 0.43 |
| Earlier: picker's top 3 of 8 as text | 0.39 | 0.17 | 0.29 | 0.46 | 0.64 |
| Earlier: rubric fires with the clue removed | 0.15–0.17 | 0.00 | 0.01 | 0.24 | 0.36 |

- **The plain decoder with the whole profile beats the picker pipeline.** 0.58 against 0.39 at 8 facts and 0.32 against 0.24 at 48, and most on the two clean types (health 0.67 against 0.17). The picker was removing the fact that matters. Where the clue sits in the profile makes no difference at 8 facts (first, middle, last third: 0.56, 0.58, 0.59).
- **Training made it worse.** On the same 144 questions: 0.58 → 0.17 at 8 facts (65 questions lost, 7 gained) and 0.32 → 0.13 at 48. The trained reader restates the clue in 12% at 8 facts (chance 12.5%) and 4% at 48, writes "none" in 62% and 40%, and names another fact in 25% and 55%. "None" was 9% of the training targets.
- **The restate step works when the choice is right.** Acts 0.91 when it restates the clue, 0.10–0.13 when it writes "none" or another fact.
- Trained types (n = 30): trained reader 0.37 at 8 facts and 0.03 at 48; plain decoder (n = 15) 0.73 and 0.33.

Reading: naming one fact first is a picker inside the model, and it learned the same thing the map learned from the generated pairs: the fact that matters is the one on the request's topic. By that standard nothing in the profile matches a question about a beach day, so it writes "none" and then answers as if there were no profile. The limit is the training signal (pairs made by asking for a request that fits a fact), not where the choosing happens.

Not measured: the clue-removed floor and the profile of 16 for this reader (cut by the 25-minute stop); one seed; 11 minutes of training, so a longer run could differ, though the restate rate on unseen types is at chance.

### 32. The model's own judgment as the selector (no training)

[poc_whole.py](poc_whole.py) `--mode zero`, on the user's V100 box (fp16), results in `work/runs/poc/zero_vllm0.json`, `zero_vllm1.json` (vLLM, all 288 + 30 questions) and `zero_text.json`, `zero_groups.json` (Hugging Face generate, the first conditions). Idea: if the plain decoder can say which facts change its answer, its own judgment is a selector (or a teacher for one) with nothing hand-designed and no generated pairs. Three ways, all with the plain decoder: answer directly; one pass that first writes "Relevant facts:" quoting the facts that should change the answer, then answers; two passes, where every group of 8 facts is judged on its own ("Do not answer yet ... quote the facts ... or say none") and the answer then sees only the quoted facts.

Acts, unseen types (n = 288):

| Plain decoder | 8 facts | 16 | 48 | 48, clue removed |
|---|---|---|---|---|
| Whole profile, answer directly | **0.53** | 0.42 | **0.35** | 0.19 |
| Quote the facts that matter first, then answer | 0.53 | – | 0.31 | 0.18 |
| Judge groups of 8, answer with the quoted facts | – | 0.35 | 0.33 | 0.16 |
| Clue alone (earlier) | 0.80 | | | |

By type at 48 facts, answering directly: health 0.21, toddler 0.14, no car 0.42, flying 0.65 (8 facts: 0.50, 0.44, 0.57, 0.60; clue removed at 8: 0.01, 0.00, 0.26, 0.46).

- **The model cannot tell which fact matters either.** Quoting first, it quotes the clue in 49% at 8 facts and 22% at 48; judging groups of 8 it quotes it in 37–38%, and quotes 13 of 48 facts on average, so the second pass still reads a crowd.
- **When it does quote the clue, the answer is better** (8 facts, quote first: acts 0.64 quoted, 0.44 not; groups of 16: 0.63 against 0.19), but the misses cancel the gain. No way of asking for a judgment beats answering directly.
- The HF and vLLM runs agree (whole profile of 8: 0.53 both, the same by type), so the faster engine does not change results.

Reading: the missing part is not where the choosing happens but that this decoder has no sense of which personal facts change an answer outside the request's topic. Selection by the model itself is ruled out; what is left is to train the reading, with a teacher that does not come from topical pairs.

Speed on the box: Hugging Face generate kept one CPU core at 100% and the GPU at 40–60% (318 answers in 281 s); my own vLLM server from production's V100 build (`~/<vllm-venv>`, used read-only with the user's approval, caches in `~/train/cache`) did the same 318 in 66 s (about 1,900 generated tokens/s per card, 256 requests at a time).

### 33. Self-distillation: the decoder that sees only the fact teaches the one that sees the whole profile

[poc_distill.py](poc_distill.py) `--mode teach` then `--mode train`; adapters `lora_d_muon.pt`, `lora_d_adamw.pt`; results `work/runs/poc/eval_d_muon.json`, `eval_d_adamw.json`. Idea: the plain decoder answers well when it sees only the fact that matters (0.80 on unseen types), so its answers there are labels with nothing hand-designed. Teacher: the plain decoder with only the fact (or no fact, 15% of examples), 8,000 answers from vLLM (about 90 s per 1,000 per card). Student: the same decoder plus a LoRA r16 on attention, reading the whole profile (4–48 facts, 0–3 near-topic distractors, near-copies of the fact left out), trained on the teacher's answer. Two 30-minute runs, one per card: Muon and AdamW, learning rate 2e-4, gradient checkpointing, 12,000-token batches. Evaluated with my vLLM server serving the adapter (the production build's adapter support needed a fix in memory, [vllm_lora_patch.py](vllm_lora_patch.py)).

Acts, unseen types (n = 288):

| | 8 facts | 16 | 48 | 8, clue removed |
|---|---|---|---|---|
| Plain decoder (test 32) | **0.53** | **0.42** | **0.35** | 0.18 |
| Student, AdamW (loss 1.0 → 0.40, 0.45 epochs) | 0.32 | 0.26 | 0.24 | 0.14 |
| Student, Muon (loss 1.0 → 0.43, 0.42 epochs) | 0.29 | 0.24 | 0.18 | 0.15 |

Health: 0.10 / 0.03 / 0.03 (AdamW) against 0.50 / – / 0.21 for the plain decoder.

- **False: distillation made it worse.** Side by side on training examples, the student's answers are closer to the teacher's than the plain decoder's (similarity 0.24 against 0.13; [lora_check.py](lora_check.py)), so the adapter is applied as trained. What it learned in under half an epoch is the teacher's style ("Absolutely! Here are some tips…") and the habit of naming one profile fact, but often the wrong one (herbal tea, poetry, a budget instead of diabetes).
- Why: most answer tokens do not depend on the fact at all, so the plain token loss is mostly style; the few tokens that carry the fact barely count. This is the same failure as test 31 (training to restate the fact: 0.22 at 8).
- Muon vs AdamW here: AdamW a little better on loss and on the test, and Muon 8% slower per token (1,144 against 1,237 tokens/s: 5 Newton–Schulz steps on 288 small matrices each update). With LoRA's thin matrices Muon's advantage does not show.
- Speed: training without gradient checkpointing runs out of memory on V100 even at 2 × 850 tokens (16.4 GB of weights plus about 7 MB per token of activations); with it, 1,150–1,250 tokens/s per card, about 60 TFLOPS of useful work, near what fp16 on V100 gives. 13–22 batches of 450–590 × 20–26 tokens were skipped for memory at the 12,000 budget.

### 34. Token weights: learn the tokens the fact changes, not the style

[poc_distill.py](poc_distill.py) `--mode weigh`, then `--mode train --weight pmi|gap`; scores `work/runs/poc/pmi_<shard>.json`, adapters `lora_d_pmi.pt`, `lora_d_gap.pt`, results `eval_d_pmi.json`, `eval_d_gap.json`. Same 8,000 teacher answers as test 33. Each answer token is scored by the plain decoder (vLLM `prompt_logprobs`, 16 minutes per 4,000 answers per card) under three contexts: the teacher's own (the fact alone), no fact, and the whole profile. Token weight = 0.1 + the positive part of the difference, capped at 5: `pmi` = fact alone minus no fact (what the fact is for), `gap` = fact alone minus whole profile (what the crowd breaks). Answers cut off at 260 tokens no longer get an end token. 30 minutes, AdamW, about 0.47 epochs each.

The weights land on the right words: for "saving for a trip to Europe" the heaviest tokens are *affordable, budget, without*; for a wool allergy *allergy, synthetic, hypo(allergenic)*; the top 10% of tokens hold 68% of the weight. `gap` also weights style tokens (*That, I, Let*), since the whole-profile decoder opens differently.

Acts, unseen types (n = 288):

| | 8 facts | 16 | 48 |
|---|---|---|---|
| Plain decoder | **0.53** | **0.42** | **0.35** |
| Unweighted (test 33) | 0.32 | 0.26 | 0.24 |
| `pmi` weights | 0.43 | 0.40 | 0.29 |
| `gap` weights | 0.36 | 0.31 | 0.26 |

By type, `pmi` (8 / 16 / 48): health 0.18 / 0.06 / 0.03 (plain 0.50 / – / 0.21), toddler 0.39 / 0.40 / 0.17 (plain 0.44 / – / 0.14), no car 0.61 / 0.58 / 0.47 (plain 0.57 / – / 0.42), flying 0.56 / 0.54 / 0.50 (plain 0.60 / – / 0.65).

- The weights fix most of what plain distillation broke and beat the plain decoder on no car and toddler, but health falls almost to zero. In every training example the fact was chosen to be on topic for the request, so the student learns to use the on-topic fact; a diabetes fact behind a beach-day request is not on topic.
- Next (test 35): pair any training fact with any training request, so the only thing that says whether a fact matters is how much it changes the teacher's answer.

### 35. Any fact with any request; the decoder as judge of its own single-fact answers

[poc_distill.py](poc_distill.py) `--p-any 1.0 --data anyfact`, then `--data distill,anyfact --weight pmi`; adapter `peft_d_any`, result `eval_d_any.json`. 8,000 more teacher answers, each for a random training fact behind a random training request, scored like test 34. The lm_head and loss now run in checkpointed chunks of 1,024 positions (peak 28.9 → 21.5 GB, no skipped batches).

- A random fact moves the teacher's answer as much as an on-topic one (about 100 nats of positive `pmi` per answer either way): the teacher that sees only one fact always works it in. So the any-fact target says "use whatever fact you see", which is the wrong lesson.
- Acts 0.38 / 0.33 / 0.24 at 8 / 16 / 48 (topical `pmi` 0.43 / 0.40 / 0.29, plain 0.53 / 0.42 / 0.35); health 0.24 / 0.10 / 0.01.

Best-of judge (`poc_whole.py --mode zero`, "best of single-fact answers"): an answer per fact plus one with none, shuffled, and the plain decoder picks the one that serves the user best. At 8 facts: acts 0.36; it picks the clue's answer 0.34 (chance 0.11), the no-fact answer 0.05; some candidate acts in 0.90 and the clue's own answer in 0.83. The judge prefers answers that use some fact, not the right one.

### 35b. Rate each fact alone, keep the top k (no training)

`poc_whole.py --mode zero`, conditions "each fact rated alone, …"; results `zero_rate.json`, `zero_rate2.json`. The plain decoder sees one fact and the request and answers "Should a good reply to this request take the fact above into account? Yes or No"; score = log P(Yes) − log P(No) from one forward pass per fact. Keep the top k facts in profile order and answer with them as text. "Normalized" subtracts the fact's mean score over 8 fixed unrelated requests.

| Acts, unseen types (n = 288) | 8 facts | 16 | 48 |
|---|---|---|---|
| Whole profile (test 32) | 0.53 | 0.42 | 0.35 |
| Rated, top 4 | **0.58** | **0.43** | 0.31 |
| Rated, top 8 | | | **0.38** |
| Normalized, top 4 | 0.51 | 0.34 | |
| Normalized, top 8 / top 12 | | | 0.26 / 0.27 |

Where the clue ranks (raw): in the top 4 of 8 in 0.83 (health 0.94, flying 1.00, no car 0.74, toddler 0.62; embedding map 0.22–0.24 for health), top 4 of 16 in 0.59, top 8 of 48 in 0.45 (median rank 9).

- The first method that beats the whole profile with no training and nothing hand-designed: +5 points at 8 facts, +3 at 48. The decoder knows a diabetes fact matters for a beach day when asked about that fact alone; it loses it in a crowd.
- Normalizing hurts, most on health (0.94 → 0.58 in the top 4 of 8): a fact's standing importance is part of what the rater gets right.
- Cost: one short forward pass per fact per request, too much for large memories. Test 36 distils it.

### 36. Distil the rating into a scorer on the stored embeddings

[poc_rater.py](poc_rater.py); labels `rater_train.json`, results `rater_fit*.json`. The plain decoder rated 75,309 (training request, training fact) pairs: 3,000 requests × 24 random pool facts plus up to 2 known relevant ones, 18 minutes on one card. Known relevant facts score +17.4 on average, random ones −1.6, but half the random facts still get a Yes.

A scorer on the Qwen3-Embedding vectors learns to match the rater's ranking per request (listwise softmax, temperature 2, 10% of requests held out). Where the clue ranks among the profile's facts on unseen types (top 4 of 8 / of 16 / of 48):

| Scorer | 8 | 16 | 48 |
|---|---|---|---|
| The rater itself | **0.83** | **0.59** | **0.29** |
| Cosine (the map) | 0.73 | 0.49 | 0.25 |
| Bilinear, rank 256 (overfits after 500 steps) | 0.68 | 0.36 | 0.16 |
| Cosine + rank 32 correction + per-fact term, early stop | 0.66 | 0.42 | 0.18 |
| Cosine + per-fact term only | 0.69 | 0.42 | 0.09 |

- **False.** Every learned scorer is below plain cosine on unseen types. It raises health at 8 facts (0.54 → 0.64–0.68) but loses no car (0.89 → 0.46–0.61). The per-fact term learns which training facts the rater likes, and that does not carry to new kinds of facts.
- The rater's judgment is in the decoder, not linearly in the stored embeddings. A cheap selector needs the decoder's own representations (for example a probe on its hidden states for the fact and request), not the retrieval embeddings.

### 35c. The wording of the rating question

[poc_rate_sweep.py](poc_rate_sweep.py); `rate_sweep.json`. Rank only, the same 288 unseen-type profiles of 48 facts, five wordings, 13,824 ratings each (about 2.5 minutes per wording on one card).

| Wording (after `The user asks: "…"`) | clue in top 4 of 48 | top 8 | top 12 | top 8: health / toddler / no car / flying |
|---|---|---|---|---|
| Cosine (the map) | 0.25 | 0.39 | 0.52 | 0.17 / 0.14 / 0.56 / 0.69 |
| "Should a good reply … take the fact above into account?" (test 35b) | 0.29 | 0.45 | 0.59 | 0.56 / 0.07 / 0.39 / 0.81 |
| "Would a reply that ignores the fact above be worse for this user?" | 0.31 | 0.50 | 0.63 | 0.85 / 0.17 / 0.12 / 0.88 |
| "Does the fact above change what a good reply should say or recommend?" | 0.38 | **0.62** | 0.73 | 0.71 / 0.33 / 0.53 / 0.90 |
| "Could a typical reply … be wrong, unsafe or unsuitable … because of the fact above?" | **0.42** | 0.60 | 0.73 | 0.72 / 0.62 / 0.25 / 0.79 |
| "Should a thoughtful assistant mention or adapt to the fact above?" | 0.39 | 0.60 | 0.70 | 0.82 / 0.35 / 0.49 / 0.75 |
| Mean of all five (z-scored) | 0.41 | **0.62** | **0.77** | 0.81 / 0.32 / 0.42 / 0.93 |

- Wording moves the clue into the top 8 of 48 from 45% to 60–62%. Asking whether the fact *changes* the reply beats asking whether to *take it into account* (which the decoder grants to half of all random facts).
- Caveat: the wordings were compared on the test profiles. The mean of all five involves no choice and does as well as the best single one.

### 37. The decoder's attention as the selector (one pass)

[poc_attn.py](poc_attn.py); `attn.json`, `attn_which.json`, features `attn_feats*.npz`. One forward pass with the whole profile and the request (eager attention, fp16); per fact, the attention mass from the tokens after the last fact to the fact's tokens, per layer and head (36 × 32). Head weights are fitted on 400 training-type profiles (a known relevant fact among 47 random ones). About 0.3 s per profile.

| Clue among 48, unseen types | top 4 | top 8 | health top 8 |
|---|---|---|---|
| Mean of all heads, per token | 0.28 | 0.50 | 0.12 |
| Layer 21 (best on training) | 0.29 | 0.49 | 0.10 |
| Learned head weights (0.81 top 4 on training types) | 0.31 | 0.51 | 0.07 |
| Relevance question appended ("which of the facts above should a good reply take into account?"), attention from it, per token | **0.41** | 0.51 | 0.11 |

- With no training, the request's attention finds the clue as well as the per-fact rating does at top 8 (50% vs 45%) and better than the map (39%), in one pass. But it is blind to health: attention follows topic. Learned head weights learn "topical" from the training profiles and do not transfer.

**Combined** ([poc_combo.py](poc_combo.py), z-scores summed): rating "change" + attention: clue in the top 4 of 48 in 0.50, top 8 0.65, median rank 4; all five ratings + attention: 0.47 / 0.65 / top 12 0.75.

End to end (`poc_whole.py --mode zero --conds saved`, `zero_saved.json`): the decoder answers with the selected facts as text. Acts on unseen types, 48 facts:

| Selector | facts shown | clue among them | acts |
|---|---|---|---|
| Whole profile (test 32) | 48 | 1.00 | 0.35 |
| Rating, old wording (35b) | 8 | 0.45 | 0.38 |
| Attention only | 8 | 0.50 | 0.31 |
| Rating "change" | 8 | 0.68 | **0.43** |
| Rating "change" + attention | 4 / 8 | 0.53 / 0.68 | 0.40 / **0.43** |
| All ratings + attention | 8 / 12 | 0.68 / 0.77 | **0.43** / 0.41 |

- +8 points over the whole profile at 48 facts, with no training and nothing hand-designed. By type ("change", top 8): health 0.49 (whole profile 0.21), flying 0.61, no car 0.42, toddler 0.22.
- The remaining loss is in reading: with the clue among 8 facts the decoder acts on it about as often as with a profile of 8 (0.53).

### 38. The rating in one pass per request

[poc_onepass.py](poc_onepass.py); `onepass.json`. The request first, then each fact followed by the rating question and "Answer:"; log P(Yes) − log P(No) read at every "Answer:" in one pass, forwards and reversed, averaged.

- Clue among 48: "change" top 4 0.36, top 8 0.56 (rated alone 0.38 / 0.62); health top 8 **0.19** (alone 0.71). Ratings fall with position (correlation −0.5).
- **False as a cheap substitute.** In a crowd the decoder rates by topic, as its attention does; seeing the fact alone is what catches health.

### 38b. More end-to-end variants, and scale

`poc_whole.py --mode zero` conditions "rated, wording change, …", "saved … best last"; `zero_wchange.json`, `zero_bestlast.json`, `zero_wlast.json`, `zero_slast.json`. "Best last": the kept facts in score order, the best one right before the request.

| Acts, unseen types (n = 288) | 8 facts | 16 | 48 |
|---|---|---|---|
| Whole profile (test 32) | 0.53 | 0.42 | 0.35 |
| Rating "change", top 4 | **0.60** | **0.53** | 0.40–0.42 |
| Rating "change", top 2, best last | 0.59 | | 0.38 |
| Rating "change", top 8, best last | | 0.50 (profile order) | 0.45 |
| Rating "change" + attention, top 8, best last | | | **0.48** |
| All five ratings + attention, top 8 / 12, best last | | | 0.45 / 0.43 |

Order does not matter at 8 and 16 facts; at 48 the best-last order adds 2–5 points.

Scale ([poc_rate_sweep.py](poc_rate_sweep.py) `--size 96|192`; `rate_s96.json`, `rate_s192.json`): each 48-fact test profile padded with random training facts of other users. Clue in the top 8:

| Facts | cosine | rating "change" | rating + cosine | health, rating |
|---|---|---|---|---|
| 48 | 0.39 | 0.62 | 0.62 | 0.71 |
| 96 | 0.25 | 0.36 | 0.48 | 0.60 |
| 192 | 0.12 | 0.25 | 0.32 | 0.50 |

The rating keeps finding health in big profiles but says Yes to too many other facts (some padding facts are truly relevant, so this understates a little). Throughput: 55–90 ratings per second per V100 (fact first, so a fact's prefix is cacheable).

### 39. Two-stage: shortlist with the rating, then think (or compare) before deciding

[poc_think.py](poc_think.py), [poc_pick.py](poc_pick.py); `think12.json`, `think24_rate_s192.json`, `zero_pick.json`.

- **Think**: the decoder in thinking mode (up to 512 tokens, median 250) on each of the top 12 (of 48) or top 24 (of 192), then yes/no. After thinking it still says Yes to 88–91% of shortlisted facts. Clue in the top 4 of 48: 0.38 → 0.46; top 8 of 192: 0.25 → 0.33. End to end, top 8 best last of 48: acts **0.50**; top 4: 0.46. Cost about 2.7 s per request on one V100 (the cheap rating: 0.5 s).
- **Pick**: one comparative question over the top 8 ("which one most changes what a good reply should say?", the number's probability, two orders averaged); 25 s for all 288 requests. Clue in the top 4 of 48: 0.38 → 0.48, but in a group it again prefers topical facts over health (top 2: 0.49 → 0.28). End to end: 0.39 (top 4), 0.44 (top 8).
- **Partly.** Thinking buys 2 points end to end at 5× the cost; comparing in a group loses health. The 8B decoder's agreeableness (Yes to most facts) limits precision; a stronger rater or a trained one is the next lever.

### 40. Train the rater for precision (LoRA, listwise)

[poc_rater_lora.py](poc_rater_lora.py); `rater_lora.json`, `runs/agentmem/lora_rater.pt`.

LoRA r16 on q/k/v/o of the decoder, Muon, 320 steps (30 min on one V100). Each step: 4 training requests, each with its known relevant fact and 15 random pool facts; listwise cross-entropy on the Yes-vs-No log-odds (the "change" wording). Held-out types never seen. Validation (60 unseen requests of the training types): known fact ranked first among 16, 0.45 → 0.75.

| unseen types | rated Yes | top 4 of 48 | top 8 of 48 | health top 8 of 48 | top 8 of 192 | health top 8 of 192 |
|---|---|---|---|---|---|---|
| plain rater | 0.40 | 0.39 | 0.62 | 0.71 | 0.29 | 0.58 |
| trained rater | 0.06 | 0.39 | 0.56 | 0.33 | 0.29 | 0.00 |
| plain + 0.5 × trained (z-scores) | | 0.46 | 0.62 | 0.61 | 0.41 | 0.38 |

(192 facts: every third saved profile, 96 requests.)

- **False.** It stops saying Yes to everything, but learns what mattered for the training types, not relevance in general: flying rises (top 8 of 192: 0.33 → 0.71), health falls to zero. The mix gains 12 points at 192 facts only by trading types. Same lesson as test 36: a few training types are too narrow a signal for "what matters".

### 41. A bigger rater: production's 27B picks the facts, the 8B answers

Approved by the user 2026-11-23. `~/models/<qwen-27b>` (on disk, read-only), my own vLLM process from production's venv, both V100s (TP 2), port 8150 (`experiments/v100.sh serve27`; not production's launch script). The 27B only rates (one yes/no token per fact, thinking off, its own chat template); the 8B decoder writes every answer. [poc_rate_sweep.py](poc_rate_sweep.py) with `MPORT_VLLM_MODEL=qwen3-27b MPORT_RATER_TOK=<27B dir>`; `rate27.json`, `rate27_s192.json`, `zero_r27_fixed.json`, `zero_saved_fixed.json`.

Where the clue ranks, unseen types (192 facts: every third profile, 96 requests):

| rater, wording | top 4 of 48 | top 8 of 48 | top 8 by type (flying / health / no car / toddler) | top 8 of 192 |
|---|---|---|---|---|
| cosine | 0.25 | 0.39 | 0.69 / 0.17 / 0.56 / 0.14 | 0.19 |
| 8B, change | 0.38 | 0.62 | 0.90 / 0.71 / 0.53 / 0.33 | 0.29 |
| 27B, change | 0.51 | 0.74 | 0.83 / 0.83 / 0.60 / 0.69 | |
| 27B, mention | 0.53 | 0.77 | 1.00 / 0.75 / 0.67 / 0.65 | |
| **27B, wrong** ("could a typical reply be wrong, unsafe or unsuitable for this user because of the fact?") | **0.67** | **0.82** | 0.93 / 0.88 / 0.72 / 0.74 | **0.51** |

Adding cosine or the 8B's rating to the 27B's makes it worse; the three 27B wordings averaged are no better than "wrong" alone.

End to end (acts, unseen types, n=288; best k per size):

| facts | whole profile | 8B rater (best) | 27B rater | clue alone |
|---|---|---|---|---|
| 8 | 0.53 | 0.60 (top 4) | **0.68** (top 2 best last; top 4: 0.62) | 0.80–0.83 |
| 16 | 0.42 | 0.53 (top 4) | **0.62** (top 2; top 4: 0.60) | |
| 48 | 0.35 | 0.47 (change + attention, top 8) | **0.53** (top 4; top 2: 0.45, top 8: 0.48) | |

Cost: 14 ratings per second on both V100s with the 27B (3.4 s for 48 facts), against 55–90 per second per card with the 8B. A cascade keeps most of the gain (from the saved scores; clue rank only):

| profile | 8B shortlist, 27B re-rates | 27B share of the full cost | clue in top 4 | top 8 |
|---|---|---|---|---|
| 48 | 12 | 0.25 | 0.61 | 0.72 |
| 48 | 16 | 0.33 | 0.63 | 0.76 |
| 48 | all 48 (27B alone) | 1 | 0.67 | 0.82 |
| 192 | 48 | 0.25 | 0.30 | 0.48 |
| 192 | all 192 | 1 | 0.30 | 0.51 |

- **True.** No training, nothing hand-designed: a stronger general model asked "could a typical reply be wrong for this user because of this fact?" is what lifts every type. The remaining gap is now the 8B answerer (the clue is in the top 2 of 8 in 84% of requests, but it acts on it in 68%) and ranking at 48+ facts.
- **Bug found and fixed while doing this.** `select_scores.json` tables were keyed by request + clue, but 84 of the 288 unseen-type rows share that key with a row whose other facts differ. Those rows read another row's table, most of their facts scored 0, and the clue won too easily. Fixed with a key that includes the fact set (`row_key`) or a per-fact key (`fact_key`) in [poc_whole.py](poc_whole.py), and a coverage check that fails a run when scores are missing. Rerun: 8B change + attention top 8 best last 0.48 → 0.47, all + attention 0.45 → 0.47, change top 8 0.43 → 0.43; the 27B's first run read 0.68 / 0.65 / 0.52 before the fix. The "saved think12+s1" (0.50) and "saved pick8" results in test 39 still use the old keys and are not rerun: likely within a point or two (the 8B's tables barely moved).

### 43. The rating from 16-token blocks, Mac quick checks (2026-11-24)

On the Mac, stride 3: 96 unseen-type questions and 10 trained-type questions. Each fact is rated from its own block, conditioned on the request ([poc_tok.py](poc_tok.py)). The reader is `lora_lsort.pt`. The answers are acts on the clue, unseen types. Checks 4b and later ran in light mode ([light.sh](light.sh)): the decoder layers are int8 ([q8.py](q8.py), torch's int8 weight-only kernel) and the GPU memory is capped. The int8 ratings match bf16 at Spearman 0.993 over 496 pairs, and the first answers are identical word for word.

Where the rater puts the clue in profiles of 8 / 48 facts:

| rater | top 2 of 8 | first of 48 | top 8 of 48 |
|---|---|---|---|
| plain decoder, "change" (`tok_change`) | 0.46 | – | 0.38 |
| plain decoder, "wrong" | 0.41 | – | 0.35 |
| draft reply first, then "is this draft wrong" (check 3, [poc_draft.py](poc_draft.py)) | 0.45 | – | 0.26 |
| LoRA trained to copy the 8B text rater, 45 min, 242 requests (check 4, [poc_tokrater.py](poc_tokrater.py), `tokL_change`) | 0.58 | 0.08 | 0.50 |
| same, 90 min, int8, 436 requests (check 4b, `tokL90q8_change`) | 0.60 | 0.09 | 0.55 |
| 8B rating the fact as text (`r8_change`, the teacher) | 0.75 | 0.16 | 0.72 |
| 27B rating the fact as text (`r27_wrong`) | 0.84 | 0.35 | 0.82 |

Answers from the picked blocks (the column header shows how many facts the reader got; the floor with the clue removed is 0.19–0.25):

| picker | top 1 of 8 | top 2 of 8 | top 1 of 48 | top 4 of 48 |
|---|---|---|---|---|
| `tok_change` (bf16) | – | 0.39 | – | 0.25 |
| check 4, bf16 | – | 0.41 | – | 0.27 |
| check 4, int8 | 0.42 | 0.38 | 0.27 | 0.31 |
| check 4b, int8 | 0.44 | 0.45 | 0.31 | 0.30 |

- When the clue is the first block the reader gets, it acts on it in 0.76–0.79. When the clue is shown but not first, it acts in about 0.40–0.60. When the trained rater's picks are wrong, they look relevant, and the reader restates and follows them: acts 0.03–0.05 with the clue not shown, against 0.17 for the untrained rater. That cancels part of the gain in picking.
- The student is capped by its teacher. At 48 facts the 8B text rater puts the clue first in only 0.16, and the student gets 0.09. More training with the same teacher added 1–6 points. The next step is a 27B teacher: about 1,000 training requests × 10 facts rated by the 27B is more than check 4b used.
- Health and toddler, where the rubric floor is clean, stay low: top 1 of 8 gives 0.25 acts, and top 4 of 48 gives 0.00–0.08.
- Abstention from the top score does not work. The best fact's score barely separates profiles with the clue from profiles without it: AUC 0.51–0.61 for the 8B raters and 0.75 / 0.59 for the 27B at 8 / 48 facts. A threshold that keeps 80% of the profiles with the clue skips memory in only 0.24–0.40 of the profiles without it.
- Not promoted to the V100. The bars were the clue first of 48 ≥ 0.20 (got 0.09) and top 1 of 8 acts ≥ 0.45 (got 0.44).

### 44. GRPO on the reader: on-policy RL with the rubric as the reward (Mac pilot, 2026-11-24, stopped early)

[poc_grpo.py](poc_grpo.py), [run_mac_grpo.sh](run_mac_grpo.sh); log `work/logs/poc_grpo.log`; checkpoint `lora_grpo_s10.pt`.

- **Idea.** Test 43 found the reader acts on the clue only 0.40–0.60 when it is shown but not first. Every reader so far learned by imitation. Here the reader LoRA (start `lora_lsort.pt`) samples answers from 16-token blocks and is pushed toward the ones the benchmark's regex rubric scores well. No teacher, no labels to make.
- **Setup.**
  - Data: only the 120 trained-type clue pairs (8 types, split "train"). The held-out types and the val pairs are never seen.
  - Context: 1, 2, 4 or 8 blocks. With p 0.75 the clue is among them at a random place. The other facts are other users' facts, half look-alikes by cosine to the request.
  - Reward: clue shown → acts (strict rubric, on the text after "Relevant fact:"); clue not shown → does not act.
  - Training: 8 contexts × 4 samples per step, temperature 1. Dr. GRPO advantage (reward − group mean), one update per batch, no KL. AdamW, lr 2e-5. Answers capped at 200 new tokens.
  - Light mode needed gradient checkpointing and logits for the answer positions only to fit the 15 GB cap.
  - Speed: about 1.8 min per step.
- **Result (stopped by the user at step 19 of about 50, before the answer test).** Training reward, clue shown: 0.28 over steps 1–10 and 0.29 over steps 11–19. Clue not shown: 0.89 / 0.87. Only 3.1–3.2 of the 8 groups per step had any spread, so there was any gradient at all. Single steps range from 0.06 to 0.75.
- **Inconclusive, no sign of learning in 19 steps.** The signal is sparse (most groups: all 4 samples ignore the clue) and each step has only 8 contexts. A real run needs more samples per context, a denser reward and hundreds of steps: a V100 job, not a Mac one. See [REPORT_2026-11-24_token_path.md](REPORT_2026-11-24_token_path.md) for the proposed v2.

### 45. The block rater on a 4-bit base (QLoRA-style), on 8 GB-class GPU memory (V100 box, 2026-11-24)

[q8.py](q8.py) (`MPORT_QBITS=4`), [run_box_q4.sh](run_box_q4.sh); logs `v100_q4_rate90q8`, `v100_q4_train`, `v100_q4_rateq4` on the box; checkpoint `lora_tokrater_q4.pt`.

- **Question.** Does the token-path rater still work when the frozen Qwen3-8B is stored in 4 bits, so that someone with a small GPU can train and run it?
- **Setup.**
  - Weights are 4-bit, asymmetric, in groups of 64, and are dequantized for each matmul. Backward dequantizes again instead of keeping the weights.
  - The LoRA and the memory core stay at full precision.
  - The embedding table and the output layer stay in CPU memory (`MPORT_LEAN=1`). The raters copy only the Yes/No rows to the GPU.
  - The layers are split over two V100s (`MPORT_SPLIT=1`), each capped at 4 GB beside production and another job. fp16, because V100 has no bf16.
- **Memory.** Training peaks at 3.53 GiB on GPU 0 and 2.81 GiB on GPU 1, about 6.3 GiB in total. Rating at batch 24 fits; batch 96 ran out of memory.
- **Speed.** Training: 218 steps (436 requests) in 25.6 min, about 7 s per step. Rating: 2,841 pairs in 5–6 min.
- **Results (wording `change`, stride 3, same eval as test 43).**

| Rater | from 8: first / top 2 | from 16: first / top 4 | from 48: first / top 4 / top 8 | val spearman, top in top 3 |
|---|---|---|---|---|
| int8-trained, run on int8 (`tokL90q8`, test 43) | 0.47 / 0.60 | – | 0.09 / 0.42 / 0.55 | 0.674 / 0.83 |
| int8-trained, run on 4-bit (`tokL90q8b4`) | 0.47 / 0.60 | 0.30 / 0.58 | 0.10 / 0.48 / 0.53 | – |
| trained and run on 4-bit (`tokLq4`) | 0.51 / 0.68 | 0.33 / 0.67 | 0.23 / 0.43 / 0.58 | 0.693 / 0.80 |

- **Reading.**
  - Going down to 4 bits costs nothing here. An int8-trained LoRA moves to a 4-bit base unchanged.
  - A LoRA trained on the 4-bit base is as good or slightly better. Clue first of 48 goes from 0.09 to 0.23, but this is one run on 106 eval requests, and top 4 of 48 is slightly lower. Treat it as a tie, not a gain.
  - The 4-bit rater says Yes to 0.10 of pairs, against 0.51 for the int8 one. Only the ranking is used, so calibration does not matter here.
- **Not tested.** The reader (generation) on 4-bit. On the Mac, 4-bit greedy text was repetitive (mean KL 0.09–0.16 against int8), so check reader answers before trusting it there.

### 46. A faster, smaller 4-bit training pipeline for the block rater (V100 box, 2026-11-24)

[q8.py](q8.py), [q4_bench.py](q4_bench.py), `shared_hidden` in [poc_tok.py](poc_tok.py), flags in [poc_tokrater.py](poc_tokrater.py); logs `v100_q4bench`, `v100_tm_sys`, `v100_sh_*`.

- **Measured first (q4_bench.py).** One layer's products on a V100 shared with another job at 100%, extrapolated to a step of 20 prompts:
  - fp16 matmuls: 3.2 s
  - plain-torch 4-bit unpacking: 2.2 s (five kernels per weight, three times per step)
  - streaming from CPU: 1.2 s more
- **Changes.**
  1. **Fused unpack kernel (Triton).** One pass reads 4.5 bits per weight and writes fp16. It matches plain torch (max abs difference 3e-5) and takes 0.38 s per step instead of 2.2 s. It is on by default on CUDA; `MPORT_Q4_TRITON=0` turns it off. `torch.compile` crashed inside dynamo on torch 2.14, so it is hand-written.
  2. **Offload (`MPORT_OFFLOAD=1`).** The packed layers stay in pinned CPU memory. `LayerPrefetch` copies layer L±1 on a side CUDA stream while layer L computes, keeping at most 4 layers on the GPU. This works in both directions, so the backward pass (recompute, then reverse) is covered too.
  3. **CPU-side parts.** The memory core runs on the CPU in lean mode (no gradient, a few vectors per step), which frees 0.8 GB. The output layer stays on the CPU, and only its Yes/No rows go to the GPU.
  4. **Shared-prefix forward (`shared_hidden`, `--shared 1`, default).**
     - The decoder layers are written by hand: pre-norm, q/k norm, RoPE, GQA, sdpa.
     - All tokens stay in one flat matrix, so each weight is unpacked once per pass.
     - Each request's prompt prefix is computed once, and every fact attends to its keys and values.
     - It matches the HF forward within 0.025 log-odds (fp16).
- **Result.**
  - 1.4 GiB peak on one GPU, against 3.5 + 2.8 GiB on two GPUs in test 45.
  - 4.1 s per step against 7.0 s.
  - The step is now matmul-bound (forward 1.6 s, backward 2.5 s; `MPORT_TIMING=1` prints this).
  - With the original layout only about 25 of 141 prompt tokens are shared, so the shared prefix saves little there.
- **Tried and rejected (8B labels, 218 steps, same eval as test 45):**

| 4-bit rater | from 8: first / top 2 | from 48: first / top 4 / top 8 | val spearman | step |
|---|---|---|---|---|
| all layers, original layout (test 45) | 0.51 / 0.68 | 0.23 / 0.43 / 0.58 | 0.693 | 4.1 s |
| request before the memory block (`--layout qfirst`): about 3.5x fewer tokens | 0.28 / 0.46 | 0.05 / 0.23 / 0.40 | 0.480 | 3.5 s |
| LoRA on layers 18-35 only (`--lora-from 18`) | 0.45 / 0.62 | 0.15 / 0.44 / 0.53 | 0.619 | not comparable (other GPU) |

  - The frozen decoder reads memory tokens well only where it was trained to see them, in the system prompt. Question-first would make rating 48 facts about 5x cheaper at serving time, but it would need far more training.
- **Not done.** A W4A16 GEMM with unpacking inside the matmul (about 0.4 s per step left), and CUDA graphs.

### 47. Check 4c: the block rater taught by the 27B (V100 box, 2026-11-24)

[poc_rate27train.py](poc_rate27train.py) (labels), [poc_tokrater.py](poc_tokrater.py) `--data rater_train27.json`, [run_box_ab.sh](run_box_ab.sh); logs `v100_r27train`, `v100_ab_r27_*`, `v100_ab_q4long_*`.

- **Labels.**
  - The 27B scored 1,000 training-type requests: all 25,096 of their (request, fact) pairs, wording "wrong".
  - Source: production's own vLLM server (same checkpoint as test 41), through the OpenAI API.
  - It sent 16 prompts at a time and waited whenever production had queued requests (`MPORT_VLLM_CHUNK`, `MPORT_VLLM_POLITE`).
  - Took 2.4 h. Correlation with the 8B's labels 0.68. Held-out types are never rated.
- **Student.**
  - 4-bit, offloaded, one GPU (test 46): 485 steps (one pass over the 970 training requests), 41 min, peak 1.47 GiB.
  - Val spearman against the 27B: 0.689.
- **Control: is the old student capped by its teacher?** 650 steps on the 8B labels raised val spearman to 0.757, but the eval did not improve: from 48, first 0.11, top 4 0.44, top 8 0.53. Yes, it is capped.

| Rater (wording change, stride 3) | from 8: first / top 2 / top 4 | from 16: first / top 4 / top 8 | from 48: first / top 4 / top 8 |
|---|---|---|---|
| block rater, 8B labels, 218 steps (test 45) | 0.51 / 0.68 / 0.82 | 0.33 / 0.67 / 0.88 | 0.23 / 0.43 / 0.58 |
| block rater, 8B labels, 650 steps | 0.46 / 0.64 / 0.80 | 0.30 / 0.62 / 0.82 | 0.11 / 0.44 / 0.53 |
| block rater, 27B labels, 485 steps from scratch (`r27`) | 0.54 / 0.75 / 0.86 | 0.39 / 0.74 / 0.90 | 0.25 / 0.51 / 0.67 |
| **same, warm start from test 45's LoRA (`r27w`)** | **0.61 / 0.75 / 0.88** | **0.44 / 0.78 / 0.91** | **0.29 / 0.50 / 0.70** |
| 8B text rater (old teacher) | 0.55 / 0.75 / 0.85 | 0.29 / 0.74 / 0.85 | 0.16 / 0.46 / 0.72 |
| 27B text rater (new teacher, test 41) | 0.72 / 0.84 / 0.99 | 0.56 / 0.91 / 0.99 | 0.35 / 0.72 / 0.82 |

- **Reading.**
  - The better teacher helps on every row. Top 4 of 48 rises from 0.43 to 0.51, and top 2 of 8 from 0.68 to 0.75. Toddler (top 2 of 8) goes from 0.38 to 0.54.
  - Starting from the 8B-taught LoRA (`--init lora_tokrater_q4.pt`) helps the first place most: 0.61 / 0.44 / 0.29 at 8 / 16 / 48 facts, val spearman 0.716 (from scratch 0.689). Checkpoint `lora_tokrater_r27w.pt`.
  - The 16-token rater now matches or beats the 8B reading the facts as text, except top 8 of 48.
  - It is still well below its 27B teacher, so the student is no longer the only limit. That leaves room to grow: more labelled requests, or more passes.
  - One run, 106 eval requests.
- **Next (running 2026-11-25).** 2,000 more requests (50,213 pairs) for the 27B to label, none overlapping the first 1,000 (`--exclude`). The reader is tested on `r27w`'s picks: the current reader, and one trained with the clue at a random place among 1-4 blocks ([run_box_reader.sh](run_box_reader.sh); 4-bit training via `poc_fast.lean_loss_batch`, checked against the plain loss in [reader_check.py](reader_check.py)).
- **Result, 3x the labels (`r27all`, 2026-11-25).** 2,950 requests (75,309 pairs), 1,475 steps from test 45's LoRA, 73 min, peak 1.64 GiB.
  - Val spearman 0.736, up from 0.716. The eval got worse: from 8: 0.45 / 0.66 / 0.80; from 16: 0.28 / 0.48 / 0.65 (first / top 2 / top 4); from 48: first 0.15, top 4 0.47, top 8 0.58.
  - This is the same pattern as the 650-step control: more steps fit the training types and hurt the held-out wording types.
  - Next: the same 3x data at r27w's 485 steps (`r27a485`), to separate more data from more steps.
- **Result: same steps, 3x the data (`r27a485`).** From 8 / 16 / 48, clue first 0.49 / 0.35 / 0.24, top 4 0.82 / 0.66 / 0.46. Val spearman 0.686.
- **Result: `r27w` replicated with another training order (`--seed 1`, `r27ws1`).** Clue first 0.48 / 0.33 / 0.22, top 4 0.82 / 0.69 / 0.47. Val spearman 0.724.
  - **`r27w` was a lucky run.** On "clue first", runs of the same recipe differ by about 0.13 at 8 facts.
  - Over the five 27B-taught runs (`r27`, `r27w`, `r27ws1`, `r27a485`, `r27all`), mean clue first from 8 is 0.51. That equals the 8B-taught rater (test 45), so the 27B teacher does not help "first".
  - Top 4 of 48 averages 0.48 against 0.43 for the 8B-taught rater (one run), a small gain at most.
  - Averaging the five tables (`poc_tok.py ensemble`, each z-scored) gives clue first 0.55 / 0.40 / 0.28 and top 4 0.85 / 0.72 / 0.52: about the runs' mean, not better than the best run.
  - Caveats: the stride-3 eval has 106 requests, and runs drift. Comparisons under about 0.1 on "first" are not readable. Test 48's reader rows used the lucky `r27w` table.
  - Next: re-rate the 8B-taught rater, `r27w` and `r27ws1` on the full eval set (stride 1), where two thirds of the questions are new to the comparison ([run_box_s1.sh](run_box_s1.sh)).
- **Result: full eval set (stride 1, about 3x the questions; tables `tokL<tag>_full_change`, [run_box_s1.sh](run_box_s1.sh)).**

| Rater, full eval set | from 8: first / top 2 / top 4 | from 16: first / top 2 / top 4 | from 48: first / top 4 / top 8 |
|---|---|---|---|
| 8B labels (test 45, `q4`) | 0.45 / 0.62 / 0.81 | 0.34 / 0.47 / 0.64 | 0.17 / 0.41 / 0.55 |
| 27B labels, `r27w` | **0.56 / 0.72 / 0.86** | **0.43 / 0.56 / 0.72** | **0.26 / 0.51 / 0.66** |
| 27B labels, same recipe, seed 1 (`r27ws1`) | 0.46 / 0.66 / 0.83 | 0.35 / 0.47 / 0.65 | 0.17 / 0.43 / 0.57 |

  - `r27w` holds up on the two thirds of questions it was not picked on, so it is a better checkpoint. But its recipe is not reliably better: the replicate is level with the 8B-taught rater.
  - The 27B teacher helps some runs, not every run. Val spearman does not tell which (`r27ws1` 0.724, `r27w` 0.716).
  - `r27w` stays the rater for the reader tests.

### 48. The reader on the box, end to end on `r27w`'s picks (V100 box, 2026-11-25)

[run_box_reader.sh](run_box_reader.sh), [poc_tok.py](poc_tok.py) `answer --tables tokLr27w_change --stride 3`; logs `v100_rd_*`.

- **Training on the box.**
  - The reader trains 4-bit and offloaded on one GPU with `poc_fast.lean_loss_batch`: packed forward, loss only at answer positions in checkpointed chunks, 4-bit output layer on the GPU, memory core on the CPU.
  - Against the plain loss: diffs ≤ 0.0005. LoRA gradient with and without checkpointing: cosine 1.00000.
  - A step of 4 examples (~550 tokens) takes 1.5 s, peak 1.32 GiB ([reader_check.py](reader_check.py)).
- **`rpos`.** 75 minutes from `lora_lsort.pt`, the same recipe without `--sort`, so the clue sits at a random place among 1-4 blocks. Half the distractors are map look-alikes.
- **Results.** Acts on the clue, held-out types, n = 96 per row.

| condition | current reader `lsort` | `rpos` |
|---|---|---|
| clue alone | **0.64** | 0.52 (writes "none" 0.27) |
| whole profile of 8 | 0.33 | 0.40 |
| top 1 of 8 / 16 / 48 | 0.48 / 0.39 / 0.31 | 0.43 / 0.39 / 0.35 |
| top 2 of 8 / 16 / 48 | 0.43 / 0.40 / 0.32 | 0.44 / 0.38 / 0.30 |
| top 4 of 8 / 48 | 0.45 / 0.27 | 0.45 / 0.31 |
| top 2 of 8, clue removed | 0.14 | 0.22 |

- **Reading.**
  - End to end, the 27B-taught rater lifts the blog's result. With the current reader, top 1 of 8 gives 0.48 (blog: 0.44), and 48 facts give about 0.31 (blog: about 0.30).
  - The reader is now the limit. It acts on 0.64 with the clue alone, while the plain decoder reading the clue as text gets 0.81 (test 20).
  - Training with the clue at a random place does not help. Each row moves within one standard error (about 0.05), except clue alone, which falls to 0.52.
  - `rpos` writes "none" for a quarter of lone clues. This is the same failure as test 23: look-alike distractors teach "none".
- **Next (running).** Context distillation, `poc_broad.py train --kd 1.0` with `poc_fast.kd_loss_batch`.
  - Same recipe as `rpos`. The answer tokens learn the full next-token distribution of the same 8B with the LoRA off, reading the same facts as text in block order. The KL is computed in the same checkpointed chunks.
  - The recite line keeps cross-entropy.
  - This targets the 0.64 vs 0.81 gap directly. [kd_check.py](kd_check.py) checks the token alignment and that alpha=0 equals the plain loss.

### 49. Context distillation: the token reader learns from the same 8B reading the facts as text (V100 box, 2026-11-25)

[poc_fast.py](poc_fast.py) `kd_loss_batch`, `poc_broad.py train --kd 1.0`, [kd_check.py](kd_check.py), [run_box_kd.sh](run_box_kd.sh); logs `v100_kd_*`, `v100_rd_ans_kd3`.

- **Method.**
  - The same recipe as `rpos` (test 48): from `lora_lsort.pt`, long answers with the recite line, 1-4 blocks in random order, 75 minutes.
  - On the answer tokens the loss is KL(teacher || reader), where the teacher is the same 8B with the LoRA off. It reads the same facts as text, in block order, in the text-control prompt (persona system prompt, facts then the request). The teacher pass is a no-grad packed forward; its hidden states go through the same 4-bit output layer in the loss chunks.
  - The recite line keeps cross-entropy.
- **Check.**
  - Answer tokens align for 16 of 16 examples, and alpha=0 equals the plain loss to 4 decimals.
  - Cross-entropy on these (training) answers: text teacher 1.06, current reader 0.27. The old reader has partly memorised its targets.
  - A step takes 1.8 s, peak 2.2 GiB. Training: 801 updates, loss 1.41 → 0.17. On training-style examples with the fact present it writes "none" 0.05 of the time.
- **Results.** Acts on the clue, held-out types, n = 96, `r27w` picks.

| condition | `lsort` (test 48) | `rpos` (same recipe, no KD) | **`kd`** |
|---|---|---|---|
| clue alone | 0.64 | 0.52 | **0.67** |
| whole profile of 8 | 0.33 | 0.40 | **0.47** |
| top 1 of 8 / 16 / 48 | 0.48 / 0.39 / 0.31 | 0.43 / 0.39 / 0.35 | 0.41 / 0.32 / 0.31 |
| top 2 of 8 / 16 / 48 | 0.43 / 0.40 / 0.32 | 0.44 / 0.38 / 0.30 | **0.58 / 0.46 / 0.35** |
| top 4 of 8 / 48 | 0.45 / 0.27 | 0.45 / 0.31 | **0.49 / 0.36** |
| top 2 of 8, clue removed | 0.14 | 0.22 | 0.18 |

- **Reading.**
  - Distillation helps the reader with several blocks. Top 2 of 8 rises from 0.43-0.44 to 0.58, about 3 standard errors. This is the best end-to-end number on the token path so far (blog: 0.44).
  - The whole profile of 8 rises to 0.47, and top 4 of 48 to 0.36. Mentions of the clue rise everywhere (clue alone 0.81).
  - Top 1 falls slightly (0.41 vs 0.48 at 8, within noise).
  - Once the clue is shown, the reader acts about as often as with the clue alone: top 1 of 8 acts 0.41 with the clue shown 0.61. The limit at 16 and 48 facts is now the rater's recall.
  - Not explained yet: the first sample writes "Relevant fact: none" and still acts on the diabetes fact. The recite line and the answer now disagree at times.
- **More picks (`kd`).**
  - Top 3 of 8: 0.56 (clue shown 0.82). Top 4 of 16: 0.47 (0.78). Top 8 of 16: 0.47 (0.91). Top 8 of 48: **0.43** (0.70).
  - With the clue removed, top 8 of 16 acts 0.22 and top 8 of 48 acts 0.21, so the clue adds about +0.25 at 16 and +0.22 at 48. The reader keeps working with 8 blocks, though it was trained on at most 4.
- **`kdsort`: the same with blocks sorted by score (like `lsort`; `EXTRA=--sort`).** Training check: writes "none" 0.08.

| condition | `lsort` | `kd` | **`kdsort`** |
|---|---|---|---|
| clue alone | 0.64 | 0.67 | **0.78** (text reader, test 20: 0.81) |
| whole profile of 8 | 0.33 | 0.47 | 0.43 (writes none 0.52) |
| top 1 of 8 / 16 / 48 | 0.48 / 0.39 / 0.31 | 0.41 / 0.32 / 0.31 | **0.55 / 0.46 / 0.35** |
| top 2 of 8 / 16 / 48 | 0.43 / 0.40 / 0.32 | 0.58 / 0.46 / 0.35 | **0.58 / 0.53 / 0.39** |
| top 4 of 8 / 48 | 0.45 / 0.27 | 0.49 / 0.36 | 0.53 / 0.39 |
| top 2 of 8, clue removed | 0.14 | 0.18 | 0.15 |

- **Reading, `kdsort`.**
  - Distillation and sorted training add up. On the clue alone, the token reader is now within 0.03 of the 8B reading the fact as text.
  - Every rater-picked row is the best so far. Top 2 of 16 is 0.53 against 0.40, and at 48 it reaches 0.35-0.39 where the blog had about 0.30.
  - Writing "none" with the unsorted whole profile of 8 (0.52) is the sorted reader's known weakness: it expects the best block first.
- **More picks (`kdsort`).**
  - Top 3 of 8: **0.59** (writes none 0.22). Top 4 of 16: 0.46.
  - Top 8 of 16: 0.39, and top 8 of 48: 0.34 (clue removed 0.17); with 8 blocks it writes "none" 0.36.
  - The sorted reader, trained on at most 4 blocks, does not carry over to 8 as `kd` does.
  - Best per profile size so far: 8 facts, `kdsort` top 3 0.59; 16, `kdsort` top 2 0.53; 48, `kd` top 8 0.43 (clue removed 0.21).
- **Next (running).**
  - `kds8`: `kdsort` trained on up to 8 blocks (`--max-facts 8`).
  - `kdsort` on the full eval set (stride 1, about 3x n) for its main rows, with `r27w`'s full table.
- **Result, full eval set (n = 288 held-out per row), `kdsort`.**
  - Clue alone **0.80** (text reader 0.81).
  - Top 2 of 8: 0.52; top 3 of 8: 0.52; top 2 of 16: 0.43; top 4 of 48: 0.34.
  - Top 2 of 8 with the clue removed: 0.14.
  - The stride-3 rows were optimistic by 0.05-0.10 (noise at n = 96).

| full eval set, n = 288, `r27w` full table | `lsort` | **`kdsort`** | gain |
|---|---|---|---|
| clue alone | 0.60 | **0.80** | +0.20 |
| top 2 of 8 | 0.40 | **0.52** | +0.12 |
| top 3 of 8 | 0.39 | **0.52** | +0.13 |
| top 2 of 16 | 0.36 | **0.43** | +0.07 |
| top 4 of 48 | 0.27 | **0.34** | +0.07 |
| top 2 of 8, clue removed | 0.12 | 0.14 | |

  - The standard error of a difference here is about 0.04. The gains at 8 facts and on the clue alone are solid; at 16 and 48 they are about 2 SE.
- **`kds8` (`kdsort` on up to 8 blocks; 1,093 updates; training check writes none 0.03), stride 3.**
  - Clue alone 0.79. Top 1 of 8 / 16 / 48: 0.50 / 0.42 / 0.32. Top 2: 0.55 / 0.47 / 0.38. Top 4 of 8 / 48: 0.55 / 0.34. Clue removed 0.21.
  - Whole profile of 8: 0.44, writing "none" only 0.16 (`kdsort` 0.52). The same as `kdsort` on 1-4 blocks, and the 8-block weakness is gone.
- **Result: `kds8` with more picks, full set (n = 288).**

| picks | clue shown | acts | clue removed |
|---|---|---|---|
| top 8 of 16 | 0.88 | 0.40 | |
| top 8 of 48 | 0.66 | **0.37** | 0.20 |
| top 16 of 48 | 0.79 | 0.35 | |
| (`kdsort` top 2 of 16 / top 4 of 48, for reference) | 0.56 / 0.51 | 0.43 / 0.34 | |

  - Letting the reader filter more blocks barely helps. The clue is shown more often (0.51 → 0.66 → 0.79 at 48), but each extra block dilutes the reader, and acts with the clue removed rise with the number of blocks (0.14 at 2, 0.20 at 8).
  - The reader's act rate when the clue is among 8 blocks is about (0.37 - 0.34 × 0.20) / 0.66 ≈ 0.46, against 0.80 when the clue is alone.
  - The limit at 16-48 facts is now the reader's focus among distractors, more than the rater's recall. The 8B teacher reading the same 8 facts as text is probably just as distracted, so a better teacher (27B answers) is the next lever, not more blocks.
- **Result: `kds16` (up to 16 blocks, 700 updates), stride 3, n = 96.**
  - Clue alone 0.78. Top 1 of 8 / 16 / 48: 0.48 / 0.35 / 0.27. Top 2: 0.61 / 0.53 / 0.43. Top 4 of 8 / 48: 0.60 / 0.33. Whole profile of 8: 0.49. Clue removed 0.23.
  - Within noise of `kds8` and `kdsort` at this n. Training on more blocks alone does not change the picture.
- **Where the gap is (text results from test 41, same 288 questions).**
  - Text pipeline, 27B picks and the 8B reads the picks as text: 0.68 / 0.62 / 0.53 at 8 / 16 / 48. Token path: 0.52 / 0.43 / 0.37. Clue alone: 0.80 both ways, so the 16 tokens lose nothing.
  - Acting on the clue when it is among the facts shown (floor subtracted, estimates): text 8B about 0.78 among 2 and 0.71 among 4; token reader about 0.67 among 2, 0.50 among 4, 0.46 among 8.
  - The reader is below its own teacher once there are several blocks, and that is the larger share of the gap at 48 (reader fixed: about 0.45; rater fixed: about 0.39; both: 0.53).
- **Result: clean teacher (`kdc8`, `--kd-clean`), full set, n = 288. False.**
  - `kdc8` is `kds8` with a teacher that reads only the fact that matters, while the reader gets it among up to 8 blocks. 880 updates, training loss 0.088 (`kds8` 0.163).

| full set | `kds8` | `kdc8` (clean teacher) |
|---|---|---|
| clue alone | 0.77 | 0.75 |
| top 2 of 8 | 0.51 | 0.49 |
| top 3 of 8 | (`kdsort` 0.52) | 0.40 |
| top 2 of 16 | 0.41 | 0.40 |
| top 4 of 48 | **0.34** | 0.28 |
| top 8 of 16 | **0.40** | 0.36 |
| top 8 of 48 | **0.37** | 0.27 |
| clue removed: top 2 of 8 / top 8 of 48 | 0.18 / 0.20 | 0.19 / 0.15 |

  - The clean teacher is worse wherever there are 3 or more blocks.
  - Likely cause: with a clean teacher the LoRA itself must learn which block matters, from 2,735 rows of the training types, and that choice does not carry over to unseen types. It fits the training types better (loss 0.088). The rater has the same problem.
  - With the teacher reading every fact, the reader only has to pass the facts on, and the base model's own judgement picks. That carries over.
  - Lesson: train the reader to be faithful to the text, and leave the judging to the base model.
- **Result: text control on the same picks, full set, n = 288.** The plain 8B (LoRA off) reads `r27w`'s picks as text (`poc_tok.py answer --text`, outputs `tok_answer_text_full_a.json` and `_b.json`). The standard error of a difference is about 0.04.

  | Condition | Clue shown | Text (plain 8B) | Tokens (`kds8`) | Tokens (`kdsort`) |
  |---|---|---|---|---|
  | clue alone | 1.00 | 0.86 | 0.77 | 0.80 |
  | top 2 of 8 | 0.72 | 0.54 | 0.51 | 0.52 |
  | top 3 of 8 | 0.81 | 0.57 | | 0.52 |
  | top 2 of 16 | 0.56 | 0.43 | 0.41 | 0.43 |
  | top 4 of 48 | 0.51 | 0.35 | 0.34 | 0.34 |
  | top 8 of 16 | 0.88 | 0.46 | 0.40 | |
  | top 8 of 48 | 0.66 | 0.35 | 0.37 | |
  | top 2 of 8, clue removed | 0 | 0.12 | 0.18 | 0.14 |
  | top 8 of 48, clue removed | 0 | 0.18 | 0.20 | |

  - The token reader is at its text ceiling for the same picks: within 0.03 in the top-2 and top-4 rows and at top 8 of 48. The open gaps are the clue alone (0.06-0.09) and 8 facts from 16 (0.06), each about two standard errors or less.
  - More distillation data for the reader (plan step 2) can therefore win at most about 0.05. Dropped as the next step.
  - The ceiling itself is low because the plain 8B loses the clue among other facts: with the clue present in 88% of the top 8 of 16 it acts 0.46, against 0.86 when the clue is alone.
  - So two things cap the path now: how often the rater puts the clue first (0.56 / 0.43 / 0.26 at 8 / 16 / 48), and the 8B's own use of a clue that sits among distractors. The 27B picking gives 0.68 / 0.62 / 0.53 with the same 8B reading (test 41).
- **Result: the 27B reads the same picks as text** (`poc_tok.py answer --text-url`, production's 27B as a polite client, `tok_answer_text27.json`; every second question, n = 144 held-out, all three readers on the same questions).

  | `r27w` picks | Clue shown | 27B, text | 8B, text | `kds8`, tokens | 27B / 8B when the clue is shown |
  |---|---|---|---|---|---|
  | clue alone | 1.00 | 0.89 | 0.88 | 0.76 | 0.89 / 0.88 |
  | top 2 of 8 | 0.66 | 0.55 | 0.48 | 0.42 | 0.80 / 0.72 |
  | top 4 of 48 | 0.49 | 0.43 | 0.33 | 0.32 | 0.81 / 0.60 |
  | top 8 of 16 | 0.89 | 0.65 | 0.42 | 0.35 | 0.73 / 0.48 |
  | top 8 of 48 | 0.61 | 0.47 | 0.32 | 0.36 | 0.75 / 0.49 |
  | top 2 of 8, clue removed | 0 | 0.16 | 0.07 | 0.14 | |

  - With the clue alone the two models act equally often (0.89 and 0.88). The 27B's advantage is keeping the clue among other facts: with the clue among 8 shown facts it acts 0.73-0.75 against the 8B's 0.48-0.49.
  - So a reader taught by the 27B's answers (not the 8B's) has room of about +0.10 with 4 facts and +0.15 to +0.23 with 8 facts on the same picks. Whether the 8B's weights can learn it is untested.
  - The 27B's floor is also higher (0.16 against 0.07 with the clue removed), so about 0.05-0.09 of its gain in the top-2 row may be guessing.
- **Result: question-vector control, full set, n = 288, reader `kds8`** (`tok_answer_bq_wrong.json`, `tok_answer_bq_mean.json`).

  | Question vector given to the core | Clue alone | Restates the clue first | Top 2 of 8 | Top 2 of 8, clue removed |
  |---|---|---|---|---|
  | the right question | 0.77 | 0.58 | 0.51 | 0.18 |
  | another question's | 0.74 | 0.56 | 0.57 | 0.14 |
  | the average of all questions | 0.76 | 0.51 | | |

  - The question vector does not matter to the reader: all differences are within noise (about 0.04 per difference).
  - What the reader gets from a block is the fact. The gain over the floor (0.14-0.18) comes from the fact's content, not from the question re-encoded in the tokens, which is the doubt the Gemma port raised for PV4.
  - Consequence: a fact's 16 tokens can be built once when the fact is stored, with a fixed vector, and no core pass is needed per question. Not yet tested for the rater, whose table here was made from right-question blocks.
- **Background.** Question-vector control (`poc_tok.py answer --block-q wrong|mean`), prompted by a public port of Memorilla to Gemma 4 that found wrong documents score as well as right ones on PV4. Each of our blocks is built by the core from the question vector and one fact. The control gives the core another question's vector (or the average of all) while the decoder still reads the right question as text.

### 50. Why the rater ranks the clue low (2026-11-26, no model run)

[rank_diag.py](rank_diag.py), CPU only, the 288 held-out questions. "Cosine" is the cosine between the question vector and the fact vector, a measure of plain topical closeness.

| Rater | Clue first from 8 / 16 / 48 | Facts above the clue at 48 | Score follows cosine among distractors (Spearman) |
|---|---|---|---|
| cosine alone | 0.28 / 0.18 / 0.07 | | |
| `r8_change` (8B, fact as text) | 0.48 / 0.28 / 0.16 | 9.0 | +0.16 to +0.20 |
| `tokLq4_full_change` (blocks, 8B labels) | 0.45 / 0.34 / 0.17 | 10.9 | +0.19 to +0.23 |
| `tokLr27w_full_change` (blocks, 27B labels) | 0.56 / 0.43 / 0.26 | 8.9 | +0.22 to +0.25 |
| `r27_wrong` (27B, fact as text, the teacher) | 0.67 / 0.52 / 0.28 | 3.9 | +0.12 to +0.14 |

- **The training set is easier than the eval in a specific way.** In `rater_train.json` a known-relevant fact has cosine 0.538 to its request and the other candidates 0.405, so cosine alone puts a relevant fact first in 0.50 of the requests (25 candidates). In the eval the clue has 0.477 and the distractors 0.444, and cosine alone puts the clue first in 0.07 of 48.
- **The block rater leans on topical closeness about twice as much as its teacher** (+0.22-0.25 against +0.12-0.14), and it puts 8.9 facts above the clue at 48 where the teacher puts 3.9.
- The two block raters agree with each other more (Spearman 0.83 over 48 facts) than either agrees with the 27B teacher (0.64-0.67).
- The teacher is itself the ceiling of this recipe for "first": 0.67 / 0.52 / 0.28. At 48 the student already matches it on "first" and loses in the top 4 and top 8 (0.51 and 0.66 against 0.67 and 0.82).
- Reading: candidates for training should be the facts closest to the request (hard negatives), not random pool facts. Tested in 53: no gain.

### 51. A reader taught by the 27B's answers (V100 box, 2026-11-26)

[poc_read27.py](poc_read27.py), `poc_broad.py train --mix27 FILE --p27 P`; logs `v100_read27`, `v100_read27far`, `v100_rd27_train_a|b`, `v100_rd_kd27a|b`.

- **Why.** Test 49: with the clue among 8 shown facts the 27B acts 0.73-0.75 and the 8B 0.48-0.49; the token reader, taught by the 8B, equals the 8B.
- **Data.** Training-type requests only (the 1,000 of `rater_train27h.json`). Per request two sets of 2-8 facts that are close to the request: one the 27B rated relevant (log-odds above 1) among ones it rated not relevant (below -1); 15% of the sets have no relevant fact. The 27B answers from the facts as text in the eval's prompt (200 new tokens, cut back to the last finished sentence).
- **Training.** From `lora_kds8.pt`, 50 minutes, the `kds8` recipe. A share of the batches uses the 27B sets with plain cross-entropy on "Relevant fact: ..." plus the 27B's answer; blocks are in the 27B's score order, with the relevant block moved to a random place half the time.
  - `kd27a`: half the batches use `read27.json` (the other half keeps the 8B distillation).
  - `kd27b`: the same with `read27far.json` (below). The planned "all batches" run was dropped for it.
- **What the 27B wrote (`read27.json`, 1,806 sets, 1,444 with a relevant fact, mean 5.0 facts, mean 108 words).** Words of 6+ letters that a fact has and the request does not, two or more of them in the answer:
  - the relevant fact: 0.66 of the answers;
  - at least one of the other facts: 0.67 (a given other fact: 0.35); with no relevant fact in the set: 0.84.
  - 0.61 of the answers reached the 200-token limit and were cut back to their last finished sentence.
  - Reading: the other facts are the closest ones to the request, and the 27B uses them although as a rater it scored them below -1. This set teaches "use what is close", not "find the one fact".
- **`read27far.json` (`--far 1.0`).** Same requests and relevant facts; the other facts are drawn from the whole pool, so they are unrelated to the request, as in the buried-clue test. `kd27a` against `kd27b` then differs only in how close the other facts are.
- **Check.** The rows of the 27B table in test 49 (every second question, n = 144), `r27w` picks. To beat: `kds8` 0.42 top 2 of 8, 0.32 top 4 of 48, 0.35 top 8 of 16, 0.36 top 8 of 48; the 27B itself 0.55 / 0.43 / 0.65 / 0.47.

- **Training.** 50 minutes each: `kd27a` 742 updates, `kd27b` 802 (3.7 s per update). Training-style check: restates the fact word for word 0.47 and 0.42, never writes "none" when the fact is there.
- **Result** ([cmp_readers.py](cmp_readers.py), log `v100_cmp`; the same 144 held-out questions for every reader, `r27w` picks; in brackets the paired difference to `kds8`, one standard error 0.04-0.05).

| Picks shown | Clue shown | `kds8` | `kd27a` | `kd27b` | 8B text | 27B text |
|---|---|---|---|---|---|---|
| clue alone | 1.00 | 0.76 | 0.81 (+0.04) | 0.83 (+0.06) | 0.88 | 0.89 |
| top 2 of 8 | 0.66 | 0.42 | 0.50 (+0.08) | 0.49 (+0.06) | 0.48 | 0.55 |
| top 4 of 48 | 0.49 | 0.32 | 0.35 (+0.03) | 0.33 (+0.01) | 0.33 | 0.43 |
| top 8 of 16 | 0.89 | 0.35 | 0.44 (+0.08) | 0.40 (+0.04) | 0.42 | 0.65 |
| top 8 of 48 | 0.61 | 0.36 | 0.28 (-0.08) | 0.30 (-0.06) | 0.32 | 0.47 |
| top 2 of 8, clue removed | 0 | 0.14 | 0.13 | 0.14 | 0.07 | 0.16 |

- **Reading.** No reliable gain. Three conditions move up by one to two standard errors and top 8 of 48 moves down by as much; the mean over the four conditions with the clue among other facts is +0.03 for `kd27a` and +0.01 for `kd27b`. Both readers land on the 8B reading the same facts as text (0.48 / 0.33 / 0.42 / 0.32), not on the 27B (0.55 / 0.43 / 0.65 / 0.47). 1,800 answers from the 27B and 50 minutes did not take the 8B past its own reading of text.
- **Close or unrelated other facts** (`kd27a` against `kd27b`): no difference beyond one standard error in any condition.
- Not adopted on the keyword score. Scored again by the 27B judge in 54: `kd27a` is never lower and is higher with the clue alone, so it is the reader for new runs.

### 52. Speed on the shared V100 (2026-11-26)

[fast_gen.py](fast_gen.py), [q4_kernel_bench.py](q4_kernel_bench.py), [q4_reuse_check.py](q4_reuse_check.py), `q8.py` (`MPORT_Q4_REUSE`, `MPORT_Q4_FP16`), `box_chain.sh` (wait for several jobs); logs `v100_kbench`, `v100_q4reuse`, `v100_fg_selftest`.

- **State of the box.** Production's 27B holds 23.8 GB on each GPU and other jobs 1.2-4.4 GB more (it changes within the hour), so 4-7 GB per GPU is free for this work; the cap for one job is 4.0 GB so that other people's jobs keep room. Both GPUs show 100% use whenever the 27B answers, including for my own data jobs.
- **27B as a writer.** 16 requests at a time instead of 4 (still waiting whenever anyone else is queued): 1.1-1.6 s per answer against 2.6 s.
- **Unpack kernel.** 15 Triton variants (no runtime division, scale loaded once per group, block sizes, warps), all bit-exact. Medians are all 5.4 ms for the four layer shapes, set by the GPU's time-slicing between processes; best runs are 1.0 ms (123 G weights/s, about 56 ms for the whole 8B), which is memory-bound. The kernel is not the limit; left unchanged.
- **Backward pass keeps the unpacked weight (`MPORT_Q4_REUSE=1`).** With every layer checkpointed, a step unpacked each weight 4 times (first pass, recomputation, backward, teacher). The recomputation now hands its weight to the backward that follows: 3 times. Gradient identical (difference 0), about 0.4 GB more at the peak.
  - First version was wrong: it kept the weight as a plain attribute, and the checkpoint's first pass (which runs with gradients on) kept every layer's weights, so the first training step ran out of memory. It now goes through `save_for_backward`, which the checkpoint drops in the first pass. `q4_reuse_check.py` checks all three: same gradient, nothing held after the forward pass, 3 -> 2 unpacks per layer.
- **Decoding loop (`fast_gen.py`, `MPORT_FASTGEN=1` in `poc_tok.py answer`).** With 4-bit weights every decoding step unpacks (and, when offloaded, streams 4.6 GB of) every weight once whatever the batch, so speed is the batch size. `model.generate` also copies the whole key/value cache every step and repeats it 4x for grouped attention. The loop keeps keys and values in buffers allocated once, attends without repeating the keys, sorts prompts by length and takes the batch the free memory allows.
  - Self-test, 16 prompts, 60 new tokens, same batch of 16: 16 of 16 answers identical to `model.generate`; 29.5 s against 35.6 s (1.2x), peak 1.75 GiB. 41 sequences fit where the old loop ran 16.
  - A decoding step takes 425 ms whether the batch is 11 or 16, so answers per second is the batch size. With another batch split 14 of 16 answers are identical (first difference after 43 words: fp16 rounding with other padding).
  - Real answer runs (300 new tokens, cap 4.0 GB): 25-35 answers per batch at 350-380 ms per step; 318 answers in 21.9 min, 4.1 s per answer against 7.6 s with the old loop (954 answers in 121.5 min).
  - The memory core leaves the GPU once the blocks are built (25 -> 35 answers per batch). Prefilling 6,000 prompt tokens at a time overran the cap with 190-token prompts; it is 2,000 now.
- **Where a decoding step goes (`fast_gen.py profile`, logs `v100_fg_profile`, `v100_fg_prof36`, `v100_fg_prof24`; 27B idle).** The packed layers are 3.64 GiB and cross the bus at 10.9 GiB/s: 341 ms per pass. Unpacking every weight takes 87 ms. A step takes 337 / 344 / 341 ms at batch 1 / 16 / 32, so an offloaded step is the bus transfer.
- **Some layers kept on the GPU (`MPORT_RESIDENT_LAYERS=N`).** All 36 resident (+3.6 GiB): 251-265 ms per step, 1.35x. The floor without any transfer is about 255 ms at batch 1, so about 10 resident layers (1 GiB) already reach it. The same 1 GiB as batch gives 18 more sequences on 35 (1.5x), so memory goes to the batch first; left at 0. (A first test with 10 layers showed no change, 426 against 425 ms: the 27B was busy for my data job then and the step was limited by sharing, not by the bus.)
  - Resident layers are given back when the GPU's free memory falls under `MPORT_MIN_FREE_GB` (1.5): this fired during the 24-layer test when another job took memory.
- **Fewer generated tokens ([short_score.py](short_score.py), log `v100_short`).** Saved answers cut to their first N tokens and scored again. Not usable: at 200 tokens 4-9% of the answers score differently and acts falls by 0.02-0.08; at 150 by 0.10 or more. 0.98 of `kds8`'s answers, 0.65 of `kd27a`'s and 0.86 of the 27B's reach the 300-token limit, and acts still rises between 200 and 300, so the score depends on the limit. Every comparison must use the same limit.
- **Memory core on the GPU (`MPORT_CORE_GPU=1`).** Reader training printed its time per update: building the blocks 4.82 s, forward 2.63 s, backward 2.35 s. The core ran on the CPU, and the box's CPUs are oversubscribed (load 30 on 16 cores). On the GPU (+0.8 GiB, peak 3.09 GiB): 0.15 s + 1.66 s + 1.89 s, so 3.7 s per update against 9.8 s (2.6x more updates in the same minutes).
- **Kept unpacked (`MPORT_Q4_FP16=1`).** Unpack once and keep fp16 weights on the GPU: for a GPU with about 17 GB free. Not usable while production holds 24 GB.
- **Blocks built once per fact (`s1_r27wmq`).** `r27w` rating blocks that were built with the mean question vector instead of each question's own: clue first 0.57 / 0.43 / 0.26 from 8 / 16 / 48, against 0.56 / 0.43 / 0.26. The rater does not need the question inside the block, so a fact's block can be computed once when the fact is stored and reused for every question. Not yet used in the pipeline, and not checked for the reader.
- **Chains.** `box_chain.sh` waits for several jobs (`a,b`), so a job can wait for its data and for its GPU.

### 53. Rater trained on the closest candidates, labelled by the 27B (V100 box, 2026-11-26)

`rater_train27h.json` (log `v100_r27hard`), `poc_tokrater.py --data rater_train27h.json --steps 485 --init lora_tokrater_q4.pt`, two seeds; logs `v100_ab_r27h_train`, `v100_ab_r27hs1_train`, `v100_s1_r27h`, `v100_s1_r27hs1`.

- **Why.** Test 50: the training candidates were easier than the eval's (cosine alone finds the relevant fact in 0.50 of the training requests, 0.07 of the eval's).
- **Data.** The 1,000 training requests, each with the facts closest to it: 25,096 pairs rated by the 27B in 3,880 s. Mean cosine of the candidates 0.591. Mean log-odds: known-relevant facts -0.56, the close candidates -1.22; the 27B says Yes to 0.30 of the close candidates.
- **Result, all 318 questions, clue first from 8 / 16 / 48 (and in the top 8 of 48).**

| Rater | Seed 0 | Seed 1 |
|---|---|---|
| `r27w` (random candidates) | 0.56 / 0.43 / 0.26 (0.66) | 0.46 / 0.35 / 0.17 (0.57) |
| `r27h` (closest candidates) | 0.47 / 0.34 / 0.20 (0.56) | 0.44 / 0.31 / 0.19 (0.55) |

- Validation on its own data: Spearman 0.65 and 0.64, top fact in the top 3 for 0.83 and 0.80.
- **Reading.** No gain: both seeds sit at the level of `r27w`'s weaker seed. Seeds of one recipe differ by 0.10, so neither recipe is separated from the other, and harder candidates did not move the rater toward its teacher (0.67 / 0.52 / 0.28). Not adopted.

### 54. A 27B judge as the second score (production's 27B as a client, 2026-11-26)

[judge27.py](judge27.py), log `v100_judge27`, cache `work/runs/poc/judge27.json`. The 27B reads the request, the clue and the reply and answers Yes or No to "does the reply take the fact into account" (Yes only if the reply would have to be different for a user without the fact, and nothing in it goes against the fact). The score is Yes at more than 0.5 of the Yes/No probability at the first token. 7,240 replies in 57 minutes, 8 at a time, production busy.

- **Which score is better, without human labels.** A reply written without the clue cannot take it into account, so every pass on "clue removed" is a false alarm. Score with the clue alone minus score with the clue removed:

| Reader | Keyword score | Judge |
|---|---|---|
| `kds8` | 0.77 - 0.18 = 0.59 | 0.93 - 0.05 = 0.89 |
| `kd27a` | 0.81 - 0.13 = 0.67 | 0.99 - 0.05 = 0.94 |
| `kd27b` | 0.83 - 0.14 = 0.69 | 0.98 - 0.03 = 0.95 |
| 8B text | 0.86 - 0.12 = 0.74 | 0.99 - 0.02 = 0.97 |
| 27B text | 0.89 - 0.16 = 0.73 | 1.00 - 0.04 = 0.96 |

  - The judge is between 0.2 and 0.8 on 1-6% of the replies. The two scores agree on 0.75-0.95 of the replies in a condition.
  - `kds8`'s 2,592 replies: judge Yes and keyword No with the clue shown 240; judge No and keyword Yes 165, of which 96 had no clue shown (keyword false alarms). Samples read: the judge's Yes replies use the fact in other words (a newborn, trains to avoid flying); the keyword's Yes replies hit a term while planning flights for a user afraid of flying.
- **Judge score, the 144 questions every reader answered (difference to `kds8`, one standard error).**

| Picks shown | Clue shown | `kds8` | `kd27a` | `kd27b` | 8B text | 27B text |
|---|---|---|---|---|---|---|
| clue alone | 1.00 | 0.92 | 0.99 (+0.06 +- 0.02) | 0.98 (+0.06 +- 0.03) | 1.00 (+0.08 +- 0.02) | 1.00 |
| top 2 of 8 | 0.66 | 0.58 | 0.60 (+0.03 +- 0.03) | 0.58 | 0.62 (+0.05 +- 0.03) | 0.64 (+0.06 +- 0.03) |
| top 4 of 48 | 0.49 | 0.38 | 0.40 | 0.36 | 0.37 | 0.47 (+0.08 +- 0.03) |
| top 8 of 16 | 0.89 | 0.48 | 0.54 (+0.06 +- 0.05) | 0.52 | 0.56 (+0.08 +- 0.05) | 0.76 (+0.28 +- 0.05) |
| top 8 of 48 | 0.61 | 0.37 | 0.38 | 0.38 | 0.38 | 0.56 (+0.19 +- 0.04) |
| top 2 of 8, clue removed | 0 | 0.06 | 0.05 | 0.03 | 0.02 | 0.04 |

  - 288 questions: top 2 of 16 `kds8` 0.51, 8B text 0.52; top 16 of 48 `kds8` 0.36; top 3 of 8 8B text 0.70; top 8 of 48 with the clue removed `kds8` 0.09, 8B text 0.06.
- **Reading.**
  - One fact in 16 memory tokens reaches the 8B decoder almost as well as the same fact as text: 0.92 (`kds8`), 0.99 (`kd27a`) against 1.00. The keyword score put this gap at 0.76 against 0.88.
  - Among other facts the memory reader equals the 8B reading text in every condition (largest difference 0.08 +- 0.05).
  - With 2 or 4 facts shown the score is close to the share of questions where the clue was shown (0.58 of 0.66, 0.38 of 0.49): the rater is the limit, and the 27B as reader adds 0.06-0.08.
  - With 8 facts shown the 8B is the limit: it reaches 0.48-0.56 with the clue shown in 0.89, the 27B 0.76. From 48 facts, showing 4, 8 or 16 gives the same 0.36-0.38.
  - The fall of `kd27a` at top 8 of 48 on the keyword score (-0.08) is not there on the judge (+0.01).
- **How many facts to show (`kd27a`, logs `v100_rd_k8`, `v100_rd_k16`, `v100_judge_k`; 144 questions; clue shown / keyword / judge).**

| Facts stored | top 2 | top 3 | top 4 | top 8 |
|---|---|---|---|---|
| 8 | 0.66 / 0.50 / 0.60 | 0.81 / 0.51 / 0.67 | 0.86 / 0.51 / 0.65 | |
| 16 | (`kds8`: 0.41 / 0.51) | 0.62 / 0.43 / 0.53 | 0.68 / 0.41 / 0.56 | 0.89 / 0.44 / 0.54 |
| 48 | | | 0.49 / 0.35 / 0.40 | 0.61 / 0.28 / 0.38 |

  - 3 facts from 8 is the best point: 0.67, the same as the 8B reading those 3 facts as text (0.68). From 16 and 48 the count hardly matters (0.51-0.56 and 0.36-0.40): each added fact shows the clue more often and is used less often when shown (judge over clue shown: 0.83 at 3 of 8, 0.82 at 4 of 16, 0.61 at 8 of 16).
- **Earlier answers scored again (log `v100_judge_old`; judge, difference to `kds8`).**
  - The reader with the clean teacher, `kdc8` (plan step 1): clue alone 0.85 (-0.08 +- 0.04), top 2 of 8 0.37 (-0.21), top 2 of 16 0.36 (-0.15), top 4 of 48 0.17 (-0.21), top 8 of 16 0.30 (-0.18), top 8 of 48 0.21 (-0.16). The keyword score had it level with `kds8` (0.44 against 0.42 at top 2 of 8); it is far worse. Its rejection stands.
  - Blocks built with the mean question vector (`bq_mean`), clue alone: 0.92, the same as `kds8`'s own 0.92. With another question's vector (`bq_wrong`): clue alone 0.92, top 2 of 8 0.60 (+0.02 +- 0.03). The question vector given to the core does not matter to the reader either (the rater: `s1_r27wmq` in 52), so a fact's 16 tokens can be computed once when the fact is stored.
- **Caveats.** The judge is the model that wrote the 27B text replies and the training answers of `kd27a` / `kd27b`, so it may favour that style. It sees the clue, so it may pass a reply that names the fact without acting on it. No human labels.
- **Use.** Report both scores from here on; `judge27.py` after any answer run costs about 0.5 s per reply.

### 55. The reader trained on its own best answers, picked by the 27B judge (V100 box, 2026-11-26)

[poc_selfgen.py](poc_selfgen.py) (sampling added to `fast_gen.py`: temperature 0.8, top-p 0.95), `judge27.py --rows`, logs `v100_sg_0`, `v100_sg_1`, `v100_sg_judge`, `v100_st_train`, `v100_ev_s1a`, `v100_ev_s1b`, `v100_judge_s1`.

- **Data.** 600 fact sets from `read27.json` + `read27far.json` with a relevant fact and 5-8 facts (mean 6.6), in the order training uses; `kd27a` wrote 4 samples each (2,400 answers, 106 min on two GPUs sharing with production). The judge said Yes to 0.55 of the samples and to the best of the 4 in 0.82 of the sets; 492 sets kept (the best sample of each).
- **Training.** `poc_broad.py train --init lora_kd27a.pt --long --recite --sort --max-facts 8 --batch 4 --minutes 30 --lr 5e-5 --kd 1.0 --mix27 selfkeep.json --p27 0.5 --tag self1`: 863 updates.
- **Result, judge, 144 questions, difference to `kd27a`:** clue alone 0.97 (-0.01 +- 0.02), top 2 of 8 0.59 (-0.01), top 3 of 8 0.64 (-0.03 +- 0.03), top 4 of 48 0.39 (-0.01), top 8 of 16 0.52 (-0.02 +- 0.04), top 8 of 48 0.28 (-0.10 +- 0.04); clue removed 0.02.
- **Reading.** No gain anywhere, and a loss at top 8 of 48. Best of 4 at 0.82 against 0.55 for one sample shows room on the training sets, but it does not carry over to the held-out types. Not adopted; `kd27a` stays.

### 56. The video author's Buried Clue test with the current pipeline (V100 box, 2026-11-26)

[demo27.py](demo27.py), logs `v100_demo_extract`, `modal_demo_embed`, `v100_demo_rate`, `v100_demo_answer`; results `work/runs/poc/demo_answers.json`. Sam's 150 chats (memorilla-demos); production's 27B extracts the facts per chat with the prompt of `demos_port.py extract` (562 facts, every clue kept: 3 / 6 / 4 / 6 facts from the clue chats); Qwen3-Embedding-4B on Modal; rater `r27w` on every fact's block; reader `kd27a`.

- **Contamination.** Two of the three buried-clue questions are training questions of the synthetic data ("What should I cook for my parents this Sunday?" with `vegan`, the gym question with `bakery_3am`, both train types); pets is a train type too. So this is a check of the pipeline end to end, not a held-out result.
- **Rank of the clue chat's best fact among 562** (rater / cosine): getaway 23 / 447, parents dinner 2 / 89, gym 1-3 / 10, control (sister) 1 / 1.
- **Answers** (rubric of the demo, 27B judge on the clue summary):

| Question | top 1 | top 3 | top 4 | top 8 | clue chat's facts only | top 8, clue chat removed |
|---|---|---|---|---|---|---|
| getaway (dog carsick) | fail | fail | fail | fail | fail (judge 0.05) | fail |
| parents dinner (vegan) | fail | PASS | PASS | PASS | PASS | fail |
| gym (3 a.m. shift) | fail | PASS | fail (judge 0.67) | PASS | fail (judge 0.81) | fail |
| control (Priya, March 14) | fail | fail | fail | fail | fail | fail |

- **Reading.**
  - Original Memorilla, the same test: 0 of 4 at every setting. Now 2 of 3 buried clues pass at the top 3 and at the top 8 of 562.
  - The control fails on the date: "Priya ... March 17th" at every setting, also from the clue chat's facts only. The name comes back, the exact number does not (as with identifiers in tests 7-10).
  - The dog clue fails twice: the rater ranks it 23rd, and with only the clue chat's facts the reader names the dog but plans a drive.
  - Without the clue chat the answers fail, so the passes come from the memory.

## Caveats

- One seed everywhere; n = 60 sessions (one session = 1.7 points) or 290 names.
- Notes and facts are built from the ground-truth patch, so they are the best case for a note-writing step.
- The people facts in test 13 are templated sentences with one sentence pattern per attribute; real extracted facts are messier.
- Substring scoring; short names can match by accident.
- Ranking needs a candidate list and reference memories.

## Modal

Since 22:20 on 2026-11-22 heavy runs go to Modal (L40S) through [modal.sh](modal.sh) / [modal_app.py](modal_app.py); data on the volume `memorilla-port`, results pulled back with `modal.sh pull`. Same code: training 31.65 examples/s at batch 8 (Mac 1.09), generation 4.97 answers/s batched (Mac 0.82). Batched helpers in [poc_fast.py](poc_fast.py). GPU minutes so far: fast 1.4, pf 4.6, dj 12.1, rel 5.0, fair 2.3, smoke 2.6, a first `broad` + `dj2` lost at about 16 (the launcher blocked on the client and the app stopped when the Mac went offline; fixed with `run.spawn()` and volume commits every 60 s; read logs with `modal.sh vlog <name>`), broad 27.3, dj2 21.9, recite 19.9, btext 2.5, blong 6.0, psmoke 1.9, comb 1.1. Then: long 14.5 (out of memory in training), rsmoke 4.3, big2 26.6, restate 33.2, restated 33.1, restate8 29.9, long2 37.4. About 130 GPU-minutes were lost before that to a crash loop: `poc_broad.py` indexed an `.npz` array once per row, which re-reads the whole array each time and ran the container out of memory; Modal restarted each killed call from the top, so `big` regenerated its pairs three times and three restate runs never got past loading. Fixed by reading the arrays once, `retries=0`, a per-call marker in `modal_app.py` so a killed call is not re-run, and an explicit 48 GB memory request. Total so far about 435 GPU-minutes, roughly $14 at an unverified $1.95/h. Round of 2026-11-23 (tests 25–29): s5 0.8 (out of memory at batch 8 with long answers; use batch 4), s6 3.8, xq 16.8, lrec 23.8, lsort 27.7, attn 27.7, restate2 23.4, pairs2 35.2, together 159 GPU-minutes. Total about 594 GPU-minutes, roughly $19 by the same unverified rate, which leaves out CPU and memory charges; the user reported at 02:25 that the credits were almost used up, so the real cost is higher than this estimate. Results, maps, generated pairs, embeddings and the readers `lora_long`, `lora_lrec`, `lora_lsort`, `lora_recite`, `lora_broad` are copied to `work/runs/` on the Mac. Test 31: `whole` 22.1 GPU-minutes (the user's last ~$1.5 of credit), total about 616; the app kept a container after `MODAL_DONE` and was stopped by hand. Training with whole-profile text: 2,750 tokens/s, 31.9 GB peak for a batch of 2 × 1,035 tokens.

## RAM

One model per process under the watchdog (kill below 12% free). Lowest free memory today: 13% (18:00, spelling run). No kills.
