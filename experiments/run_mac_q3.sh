#!/bin/zsh
# Check 4b, light mode (int8 decoder): the block rater trained 90 min (twice check 4), its rating on the stride-3 test,
# then answers from its picks and from check 4's picks (tokL_change), both on the int8 decoder.
cd ~/memorilla-port && . experiments/env.sh && . experiments/light.sh
SCRIPT=poc_tokrater.py experiments/poc_guard.sh tokrater90 --minutes 90 --out lora_tokrater90q8.pt || exit 1
SCRIPT=poc_tok.py experiments/poc_guard.sh tokrater90_eval rate --lora lora_tokrater90q8.pt --raters tokL --tag 90q8 --wordings change --stride 3 --batch 16 || exit 1
SCRIPT=poc_tok.py experiments/poc_guard.sh tokL90_answer answer --tables tokL90q8_change,tokL_change --conds "top+1+from+8,top+1+from+48,top+2+from+8,top+4+from+48" --stride 3 --out tokL90q8_answer.json || exit 1
echo MAC_Q3_DONE >> work/logs/poc_tokL90_answer.log
