#!/bin/zsh
# Rest of the night queue with batched training / evaluation (poc_fast.py). Replaces run_night.sh and run_after.sh.
# Marker lines are appended to the old logs so the existing waiters still fire.
cd "${0:A:h}/.." && . experiments/env.sh
while pgrep -f "poc_guard.sh|experiments/demo.py" > /dev/null; do sleep 10; done
export INTERVAL=5
step() { n=$1; shift; echo "== $n $(date +%H:%M)"; experiments/poc_guard.sh $n "$@"; r=$?; echo "   exit $r"; return $r; }
export SCRIPT=poc_django.py
step dj_train train --minutes 40 --batch 4 --single
step dj_eval eval --lora lora_dj.pt --out django_eval.json
step dj_text textrag
echo "DJANGO_DONE $(date +%H:%M)"
export SCRIPT=poc_profile.py
step pf_text text
step pf_select select
echo "AFTER_DONE $(date +%H:%M)" | tee -a work/logs/run_after.log
export SCRIPT=demo.py
step dm_embed embed
step dm_examples examples
echo "ALL_DONE $(date +%H:%M)" | tee -a work/logs/run_night.log
