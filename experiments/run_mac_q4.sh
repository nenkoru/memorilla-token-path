#!/bin/zsh
# 4-bit check on the Mac (light mode + MPORT_QBITS=4, experiments/q8.py):
#   1. the int8-trained block rater (lora_tokrater90q8) rating at 4-bit: what inference-only quantization costs
#   2. the block rater trained at 4-bit (QLoRA-style), same 218 steps as lora_tokrater90q8
#   3. that rater rating at 4-bit; compare with tokL90q8_change (int8 trained, int8 rated)
cd ~/memorilla-port && . experiments/env.sh && . experiments/light.sh
export MPORT_QBITS=4
SCRIPT=poc_tok.py experiments/poc_guard.sh q4_rate90q8 rate --lora lora_tokrater90q8.pt --raters tokL --tag 90q8b4 --wordings change --stride 3 || exit 1
SCRIPT=poc_tokrater.py experiments/poc_guard.sh q4_train --steps 218 --out lora_tokrater_q4.pt || exit 1
SCRIPT=poc_tok.py experiments/poc_guard.sh q4_rateq4 rate --lora lora_tokrater_q4.pt --raters tokL --tag q4 --wordings change --stride 3 || exit 1
echo MAC_Q4_DONE >> work/logs/poc_q4_rateq4.log
