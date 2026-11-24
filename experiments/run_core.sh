#!/bin/zsh
# H11: fine-tune the core to pick one fact out of a pile (one 16-token block), with a reader-only control arm.
cd "${0:A:h}/.." && . experiments/env.sh
export SCRIPT=poc_core.py INTERVAL=5
R=work/runs/agentmem
step() { n=$1; shift; echo "== $n $(date +%H:%M)"; experiments/poc_guard.sh $n "$@"; echo "   exit $?"; }
until grep -q ALL_DONE work/logs/run_slots.log; do sleep 20; done
[ -f $R/core_facts.npz ] || step co_embed embed
step co_eval0 eval --lora lora_slots.pt --out core_eval_before.json
step co_train train --core 1 --minutes 150 --tag pick
[ -f $R/core_pick.pt ] && step co_eval eval --lora lora_pick.pt --core core_pick.pt --out core_eval_pick.json
step co_train_ctl train --core 0 --minutes 75 --tag pickctl
[ -f $R/lora_pickctl.pt ] && step co_eval_ctl eval --lora lora_pickctl.pt --out core_eval_ctl.json
echo "ALL_DONE $(date +%H:%M)"
