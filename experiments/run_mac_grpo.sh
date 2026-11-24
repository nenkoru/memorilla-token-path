#!/bin/zsh
# Test 44 (light mode): GRPO on the reader for 90 min, then the test-43 answer run with the new reader on the same picks.
cd ~/memorilla-port && . experiments/env.sh && . experiments/light.sh
SCRIPT=poc_grpo.py experiments/poc_guard.sh grpo train --minutes 90 || exit 1
SCRIPT=poc_tok.py experiments/poc_guard.sh grpo_answer answer --lora lora_grpo.pt --tables tokL90q8_change --conds "top+1+from+8,top+1+from+48,top+2+from+8,top+4+from+48" --stride 3 --out grpo_answer.json || exit 1
echo MAC_GRPO_DONE >> work/logs/poc_grpo_answer.log
