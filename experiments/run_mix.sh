#!/bin/zsh
# H7: after the spelling runs, train one reader on spelling + session tasks, then test it on real held-out sessions.
cd "${0:A:h}/.." && . experiments/env.sh
export SCRIPT=poc_spell.py INTERVAL=5
while pgrep -f "run_spell.sh" >/dev/null; do sleep 20; done
step() { n=$1; shift; echo "== $n $(date +%H:%M)"; experiments/poc_guard.sh $n "$@"; echo "   exit $?"; }
step sp_mix mix --minutes 90
[ -f work/runs/agentmem/lora_mix.pt ] && step sp_mixeval mixeval --n 60
echo "ALL_DONE $(date +%H:%M)"
