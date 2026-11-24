#!/bin/zsh
# H10: how few memory tokens per fact (first k of the core's 16 slots, or facts pooled in pairs).
cd "${0:A:h}/.." && . experiments/env.sh
export SCRIPT=poc_slots.py INTERVAL=5
step() { n=$1; shift; echo "== $n $(date +%H:%M)"; experiments/poc_guard.sh $n "$@"; echo "   exit $?"; }
step sl_train train --minutes 90
[ -f work/runs/agentmem/lora_slots.pt ] && step sl_eval eval --n 60
echo "ALL_DONE $(date +%H:%M)"
