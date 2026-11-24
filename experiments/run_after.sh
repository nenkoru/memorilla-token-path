#!/bin/zsh
# Runs after run_night.sh: H15 control with the profile as plain text (plain decoder, no core, no LoRA).
cd "${0:A:h}/.." && . experiments/env.sh
until grep -q "ALL_DONE\|selftest failed" work/logs/run_night.log 2>/dev/null; do sleep 30; done
while pgrep -f "poc_guard.sh|experiments/demo.py" > /dev/null; do sleep 20; done
export INTERVAL=5 SCRIPT=poc_profile.py
echo "== pf_text $(date +%H:%M)"; experiments/poc_guard.sh pf_text text; echo "   exit $?"
echo "== pf_select $(date +%H:%M)"; experiments/poc_guard.sh pf_select select; echo "   exit $?"
echo "== fast_self $(date +%H:%M)"; SCRIPT=poc_fast.py experiments/poc_guard.sh fast_self selftest; echo "   exit $?"
echo "AFTER_DONE $(date +%H:%M)"
