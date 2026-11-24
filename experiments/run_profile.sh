#!/bin/zsh
# H15: always-on profile. Generate (fact, request, answer) triples, embed, baseline eval, train the reader, eval.
cd "${0:A:h}/.." && . experiments/env.sh
export INTERVAL=5 SCRIPT=poc_profile.py
step() { n=$1; shift; echo "== $n $(date +%H:%M)"; experiments/poc_guard.sh $n "$@"; echo "   exit $?"; }
step pf_gen gen
step pf_embed embed
step pf_eval0 eval --lora lora_gen.pt --stride 2 --n-val 0 --out profile_eval_before.json
step pf_train train --minutes 60
step pf_eval eval
echo "ALL_DONE $(date +%H:%M)"
