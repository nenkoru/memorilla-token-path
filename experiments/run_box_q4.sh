#!/bin/bash
# 4-bit (QLoRA-style) block-rater chain on the V100 box, split over GPUs 0,1, capped at 4 GB each.
cd ~/memorilla-port
export E='MPORT_Q8=1 MPORT_QBITS=4 MPORT_LEAN=1 MPORT_SPLIT=1 MPORT_GPU_CAP_GB=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True'
step() {  # name script args...
  n=$1; shift
  experiments/v100.sh run "$n" 0,1 "$@" || return 1
  while ! experiments/v100.sh log "$n" 3 2>/dev/null | grep -q "V100_DONE $n"; do sleep 60; done
  experiments/v100.sh log "$n" 3 | grep -q "V100_DONE $n exit 0"
}
step q4_rate90q8 poc_tok.py rate --lora lora_tokrater90q8.pt --raters tokL --tag 90q8b4 --wordings change --stride 3 --batch 24 || exit 1
step q4_train poc_tokrater.py --steps 218 --out lora_tokrater_q4.pt || exit 2
step q4_rateq4 poc_tok.py rate --lora lora_tokrater_q4.pt --raters tokL --tag q4 --wordings change --stride 3 --batch 24 || exit 3
echo BOX_Q4_DONE
