#!/bin/zsh
# Canned demo transcript, queued after the Django baseline (run_django2.sh).
cd "${0:A:h}/.." && . experiments/env.sh
until grep -q ALL_DONE work/logs/run_django2.log 2>/dev/null; do sleep 30; done
export INTERVAL=5 SCRIPT=demo.py
step() { n=$1; shift; echo "== $n $(date +%H:%M)"; experiments/poc_guard.sh $n "$@"; echo "   exit $?"; }
step dm_embed embed
step dm_examples examples
echo "ALL_DONE $(date +%H:%M)"
