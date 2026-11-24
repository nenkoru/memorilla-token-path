#!/bin/zsh
# H16 baseline: the same Django questions answered by ordinary retrieval (top-k facts pasted as text). Waits for run_django.sh.
cd "${0:A:h}/.." && . experiments/env.sh
until grep -q ALL_DONE work/logs/run_django.log 2>/dev/null; do sleep 30; done
export INTERVAL=5 SCRIPT=poc_django.py
step() { n=$1; shift; echo "== $n $(date +%H:%M)"; experiments/poc_guard.sh $n "$@"; echo "   exit $?"; }
step dj_text textrag
echo "ALL_DONE $(date +%H:%M)"
