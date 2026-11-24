#!/bin/zsh
# Local: canned demo transcript (the demo is meant to run on the Mac).
cd "${0:A:h}/.." && . experiments/env.sh
while pgrep -f "poc_guard.sh" > /dev/null; do sleep 5; done
export INTERVAL=5 SCRIPT=demo.py
echo "== dm_embed $(date +%H:%M)"; experiments/poc_guard.sh dm_embed embed; echo "   exit $?"
echo "== dm_examples $(date +%H:%M)"; experiments/poc_guard.sh dm_examples examples; echo "   exit $?"
echo "DEMO_DONE $(date +%H:%M)"
