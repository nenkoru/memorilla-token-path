#!/bin/zsh
# After the hypothesis chain: H6 contrastive decoding, then H5 spelling test (600 vs ~6500 distinct names).
cd "${0:A:h}/.." && . experiments/env.sh
export INTERVAL=5
while pgrep -f "run_agenthyp.sh" >/dev/null; do sleep 20; done
step() { n=$1; shift; echo "== $n $(date +%H:%M)"; experiments/poc_guard.sh $n "$@"; echo "   exit $?"; }
SCRIPT=poc_agenthyp.py step ah_cad cad --n 60
export SCRIPT=poc_spell.py
[ -f work/runs/agentmem/spell_emb.npz ] || step sp_embed embed
step sp_k600 run --k 600 --tag k600 --minutes 30
step sp_kall run --k 9000 --tag kall --minutes 60
echo "ALL_DONE $(date +%H:%M)"
