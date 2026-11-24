#!/bin/bash
# Check 4c on the box: wait for the 27B labels, then train the 4-bit block rater on them (two arms, one per GPU) and rate.
cd ~/memorilla-port
while ! experiments/v100.sh log r27train 3 2>/dev/null | grep -q "V100_DONE r27train"; do sleep 60; done
experiments/v100.sh log r27train 3 | grep -q "V100_DONE r27train exit 0" || { echo LABELS_FAILED; exit 1; }
experiments/run_box_ab.sh r27 0 2.2 "--data rater_train27.json --steps 485" "" > work/logs/ab_r27.log 2>&1 &
sleep 120
experiments/run_box_ab.sh r27w 1 1.8 "--data rater_train27.json --steps 485 --init lora_tokrater_q4.pt" "--batch 48" > work/logs/ab_r27w.log 2>&1 &
wait
echo BOX27_DONE
