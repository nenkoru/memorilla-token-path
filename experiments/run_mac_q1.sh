#!/bin/zsh
# Mac quick-check queue 1 (2026-11-24): waits for the stride-3 token rating, then check 3 (draft, then rate blocks
# against the draft). One model at a time, each step under the RAM guard.
cd ~/memorilla-port && . experiments/env.sh
while pgrep -f "poc_tok.py rate --stride 3" >/dev/null; do sleep 20; done
SCRIPT=poc_draft.py experiments/poc_guard.sh tokdraft draft --stride 3 || exit 1
SCRIPT=poc_draft.py experiments/poc_guard.sh tokdrate rate --stride 3 || exit 1
echo MAC_Q1_DONE >> work/logs/poc_tokdrate.log
