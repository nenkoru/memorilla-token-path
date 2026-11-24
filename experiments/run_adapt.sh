#!/bin/zsh
# Unattended adapter/LoRA run: waits for `teach` to finish, then train+eval each variant, then the base eval.
# Each step is its own process under the RAM watchdog. Variants are independent (a failure skips to the next).
cd "${0:A:h}/.." && . experiments/env.sh
export SCRIPT=poc_adapt.py INTERVAL=5
G=experiments/poc_guard.sh
while pgrep -f "poc_adapt.py teach" >/dev/null; do sleep 20; done
[ -f "$MPORT_SCRATCH/runs/poc/teach.json" ] || { echo "no teach.json"; exit 1; }
for v in adapter lora both; do
  echo "== $v $(date +%H:%M)"
  $G train_$v train --variant $v && $G eval_$v eval --variant $v
  echo "   exit $?"
done
echo "== base $(date +%H:%M)"; $G eval_base eval --variant base; echo "   exit $?"
echo "ALL_DONE $(date +%H:%M)"
