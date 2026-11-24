#!/bin/sh
# After the first two queues: evaluate calibrated heads (+ user-specificity control), then score everything once.
cd "$(dirname "$0")/.." && . experiments/env.sh
R=$MPORT_SCRATCH/runs
L=$MPORT_SCRATCH/logs
until grep -q QUEUE_DONE "$L/queue1.log" && grep -q CAL_DONE "$L/queue2.log"; do sleep 30; done
$PY experiments/pv4_port.py generate --head "$R/cal_ridge/head.pt" --out "$R/lfm3_cal_ridge" > "$L/lfm3_cal_ridge.log" 2>&1
$PY experiments/pv4_port.py generate --head "$R/cal_random/head.pt" --out "$R/lfm3_cal_random" > "$L/lfm3_cal_random.log" 2>&1
$PY experiments/pv4_port.py generate --head "$R/cal_ridge/head.pt" --shuffle_docs --out "$R/lfm3_cal_ridge_shuffled" > "$L/lfm3_cal_ridge_shuffled.log" 2>&1
$PY experiments/pv4_port.py score "$R/lfm3_closed" "$R/lfm3_random" "$R/lfm3_ridge" "$R/lfm3_procrustes" \
    "$R/lfm3_cal_random" "$R/lfm3_cal_ridge" "$R/lfm3_cal_ridge_shuffled" > "$L/score_all.log" 2>&1
echo FINISH_DONE >> "$L/score_all.log"
