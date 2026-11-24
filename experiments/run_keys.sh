#!/bin/zsh
# H14: ingest-time "matters when" keys for buried clues. Generation (Qwen3-8B), embedding, then a CPU ranking check.
cd "${0:A:h}/.." && . experiments/env.sh
export INTERVAL=5 SCRIPT=poc_keys.py
step() { n=$1; shift; echo "== $n $(date +%H:%M)"; experiments/poc_guard.sh $n "$@"; echo "   exit $?"; }
step ky_gen gen
step ky_embed embed
step ky_rank rank
echo "ALL_DONE $(date +%H:%M)"
