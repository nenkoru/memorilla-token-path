#!/bin/zsh
# H16: codebase-scale memory (one fact per Django class / function / method). Waits for the H15 chain to finish.
cd "${0:A:h}/.." && . experiments/env.sh
until grep -q ALL_DONE work/logs/run_profile.log 2>/dev/null; do sleep 30; done
export INTERVAL=5 SCRIPT=poc_django.py
step() { n=$1; shift; echo "== $n $(date +%H:%M)"; experiments/poc_guard.sh $n "$@"; r=$?; echo "   exit $r"; return $r; }
step dj_embed embed
step dj_para para
step dj_paraemb paraemb
step dj_cos cos
step dj_self selftest || { echo "selftest failed"; echo "ALL_DONE $(date +%H:%M)"; exit 1; }
step dj_eval0 eval --lora lora_gen.pt --out django_eval_zeroshot.json
step dj_train train --minutes 40
step dj_eval eval --lora lora_dj.pt --out django_eval.json
echo "ALL_DONE $(date +%H:%M)"
