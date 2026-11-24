#!/bin/zsh
# H9: train the reader on one-fact-per-chunk memories (16 tokens per fact), then test on held-out sessions.
cd "${0:A:h}/.." && . experiments/env.sh
export SCRIPT=poc_atoms.py INTERVAL=5
step() { n=$1; shift; echo "== $n $(date +%H:%M)"; experiments/poc_guard.sh $n "$@"; echo "   exit $?"; }
[ -f work/runs/agentmem/atoms_all.npz ] || step at_embedall embedall
step at_train train --minutes 120
[ -f work/runs/agentmem/lora_atoms.pt ] && step at_eval2 eval2 --n 60
echo "ALL_DONE $(date +%H:%M)"
