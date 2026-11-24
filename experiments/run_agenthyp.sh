#!/bin/zsh
# Hypothesis tests with the core frozen: extra embeddings, core probe (CPU), generation tests, decoder ranking.
cd "${0:A:h}/.." && . experiments/env.sh
export SCRIPT=poc_agenthyp.py INTERVAL=5
step() { n=$1; shift; echo "== $n $(date +%H:%M)"; experiments/poc_guard.sh $n "$@"; echo "   exit $?"; }
[ -f work/runs/agentmem/hyp_emb.npz ] || step ah_embed embed
step ah_probe probe
step ah_gen_lora gen --variant lora --n 60 --resume
step ah_rank rank --n 60
step ah_gen_base gen --variant base --n 60 --resume
echo "ALL_DONE $(date +%H:%M)"
