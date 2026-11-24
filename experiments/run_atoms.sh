#!/bin/zsh
# H8: one fact per chunk, read by the spelling reader (lora_mix.pt).
cd "${0:A:h}/.." && . experiments/env.sh
export SCRIPT=poc_atoms.py INTERVAL=5
step() { n=$1; shift; echo "== $n $(date +%H:%M)"; experiments/poc_guard.sh $n "$@"; echo "   exit $?"; }
[ -f work/runs/agentmem/atoms_emb.npz ] || step at_embed embed
step at_eval eval --n 60
echo "ALL_DONE $(date +%H:%M)"
