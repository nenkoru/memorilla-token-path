#!/bin/bash
# Night of 2026-11-25: once the reader training frees GPU1, train the block rater on all 75k 27B labels there
# (3,000 requests, one pass, warm start from the 8B-taught 4-bit LoRA as r27w did), then rate the eval set.
cd ~/memorilla-port
while ! experiments/v100.sh log rd_train 3 2>/dev/null | grep -q "V100_DONE rd_train"; do sleep 60; done
experiments/run_box_ab.sh r27all 1 2.4 "--data rater_train27all.json --steps 1475 --init lora_tokrater_q4.pt" "" > work/logs/ab_r27all.log 2>&1
echo NIGHT_DONE
