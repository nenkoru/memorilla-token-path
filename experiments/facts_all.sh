#!/bin/sh
# Finer-documents test: wait for the speed test, extract facts with LFM3, run LFM3, then Qwen3-8B once downloaded.
cd "$(dirname "$0")/.." && . experiments/env.sh
L=$MPORT_SCRATCH/logs
DL_DONE=$1  # file that gets "EXIT <code>" when the Qwen3-8B download finishes
until grep -qE "(DONE|FAILED) speed" "$L/demos_all.log"; do sleep 20; done
run() { $PY -I experiments/demos_port.py "$@" > "$L/facts_$3.log" 2>&1 && echo "DONE $*" >> "$L/facts_all.log" || echo "FAILED $*" >> "$L/facts_all.log"; }
$PY -I experiments/demos_port.py extract > "$L/facts_extract.log" 2>&1 && echo "DONE extract" >> "$L/facts_all.log" || { echo "FAILED extract" >> "$L/facts_all.log"; exit 1; }
run facts --decoder lfm3
until grep -q "EXIT" "$DL_DONE"; do sleep 30; done
grep -q "EXIT 0" "$DL_DONE" && run facts --decoder qwen8b || echo "FAILED download" >> "$L/facts_all.log"
