#!/bin/zsh
# Overnight chain for poc_agentmem.py: embed -> train -> eval lora -> eval base, each step its own process under
# the memory watchdog (poc_guard.sh kills a step below 12% free memory). Evals resume if a step dies.
cd "${0:A:h}/.." && . experiments/env.sh
export SCRIPT=poc_agentmem.py INTERVAL=5
G=experiments/poc_guard.sh
R=work/runs/agentmem
BASE_CONDS=("single 16 tok" "single 128 tok" "8 sessions, pooled 16 tok" "8 sessions, 16 tok each (128)" "8 sessions without it, 16 tok each")
echo "== embed $(date +%H:%M)"; $G am_embed embed; echo "   exit $?"
[ -f $R/emb.npz ] || { echo "no emb.npz, stopping"; exit 1; }
echo "== train $(date +%H:%M)"; $G am_train train --max_hours 3.5; echo "   exit $?"
[ -f $R/lora.pt ] || { echo "no lora.pt, stopping"; exit 1; }
for try in 1 2; do
  echo "== eval lora try $try $(date +%H:%M)"; $G am_eval_lora eval --variant lora --n 60 --resume && break
  echo "   failed, retrying in 60 s"; sleep 60
done
for try in 1 2; do
  echo "== eval base try $try $(date +%H:%M)"; $G am_eval_base eval --variant base --n 60 --resume --conditions "${BASE_CONDS[@]}" && break
  echo "   failed, retrying in 60 s"; sleep 60
done
echo "ALL_DONE $(date +%H:%M)"
