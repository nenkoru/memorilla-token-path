#!/bin/zsh
# H12: similarity bias inside the core's attention (16 new parameters) + reader; original core weights frozen.
cd "${0:A:h}/.." && . experiments/env.sh
export SCRIPT=poc_core.py INTERVAL=5
R=work/runs/agentmem
step() { n=$1; shift; echo "== $n $(date +%H:%M)"; experiments/poc_guard.sh $n "$@"; echo "   exit $?"; }
step cb_train train --core 0 --bias 1 --minutes 90 --tag bias
[ -f $R/lora_bias.pt ] && step cb_eval eval --lora lora_bias.pt --core core_bias.pt --sizes 1,16,137 --out core_eval_bias.json
echo "ALL_DONE $(date +%H:%M)"
