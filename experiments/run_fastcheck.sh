#!/bin/zsh
# Waits for the running step to finish, then checks the batched loss / generation helpers (poc_fast.py).
cd "${0:A:h}/.." && . experiments/env.sh
while pgrep -f "poc_guard.sh" > /dev/null; do sleep 5; done
export INTERVAL=5 SCRIPT=poc_fast.py
echo "== fast_self $(date +%H:%M)"; experiments/poc_guard.sh fast_self selftest; echo "   exit $?"
echo "FASTCHECK_DONE $(date +%H:%M)"
