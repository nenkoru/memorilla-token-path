#!/bin/sh
# Re-run the memorilla-demos tests on LFM3-1.7B + frozen pv4 core + adapter-only head, one after another.
cd "$(dirname "$0")/.." && . experiments/env.sh
L=$MPORT_SCRATCH/logs
for t in clue diagnose book speed; do
  $PY -I experiments/demos_port.py $t > "$L/demo_$t.log" 2>&1 || echo "FAILED $t" >> "$L/demos_all.log"
  echo "DONE $t" >> "$L/demos_all.log"
done
