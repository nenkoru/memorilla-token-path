#!/bin/zsh
# H12b: the same biased core and reader on uncapped piles: all 137 test sessions (~530 facts) and all 1,037 (~4,300).
cd "${0:A:h}/.." && . experiments/env.sh
export SCRIPT=poc_core.py INTERVAL=5
step() { n=$1; shift; echo "== $n $(date +%H:%M)"; experiments/poc_guard.sh $n "$@"; echo "   exit $?"; }
step cb_eval2 eval --lora lora_bias.pt --core core_bias.pt --sizes 137,1037 --cap 100000 --values plain --out core_eval_bias_big.json
echo "ALL_DONE $(date +%H:%M)"
