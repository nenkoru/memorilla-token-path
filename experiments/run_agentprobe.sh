#!/bin/zsh
# Runs after run_agentmem.sh: encoder retrieval probe, then the text-in-context ceiling. Watchdog as usual.
cd "${0:A:h}/.." && . experiments/env.sh
export SCRIPT=poc_agentprobe.py INTERVAL=5
while pgrep -f "run_agentmem.sh" >/dev/null; do sleep 20; done
echo "== retrieve $(date +%H:%M)"; experiments/poc_guard.sh ap_retrieve retrieve; echo "   exit $?"
echo "== text $(date +%H:%M)"; experiments/poc_guard.sh ap_text text --n 60; echo "   exit $?"
echo "ALL_DONE $(date +%H:%M)"
