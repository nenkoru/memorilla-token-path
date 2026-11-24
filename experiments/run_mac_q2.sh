#!/bin/zsh
# Mac quick-check queue 2 (2026-11-24): after queue 1, check 4: train a LoRA to rate from blocks like the text rater, then
# rate the stride-3 unseen-type profiles with it (table tokL_change, compare with tok_change 46% / 38%).
cd ~/memorilla-port && . experiments/env.sh
while pgrep -f "run_mac_q1.sh|poc_draft.py" >/dev/null; do sleep 20; done
SCRIPT=poc_tokrater.py experiments/poc_guard.sh tokrater --minutes 45 || exit 1
SCRIPT=poc_tok.py experiments/poc_guard.sh tokrater_eval rate --lora lora_tokrater.pt --raters tokL --wordings change --stride 3 --batch 16 || exit 1
echo MAC_Q2_DONE >> work/logs/poc_tokrater_eval.log
