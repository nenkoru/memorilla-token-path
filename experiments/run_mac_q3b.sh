#!/bin/zsh
# Check 4b answers (light mode, int8 kernel): picks of the 90-min rater (tokL90q8) and of check 4's rater (tokL_change).
cd ~/memorilla-port && . experiments/env.sh && . experiments/light.sh
SCRIPT=poc_tok.py experiments/poc_guard.sh tokL90_answer answer --tables tokL90q8_change,tokL_change --conds "top+1+from+8,top+1+from+48,top+2+from+8,top+4+from+48" --stride 3 --out tokL90q8_answer.json || exit 1
echo MAC_Q3_DONE >> work/logs/poc_tokL90_answer.log
