#!/bin/bash
# Re-rate block raters on the full eval set (stride 1, 3x the questions of the stride-3 A/B runs), one after another
# on one GPU. Tables tokL<tag>_full_change; the stride-3 tables are kept.
#   run_box_s1.sh <gpu> <cap GB> <tag>=<lora file> ...
cd ~/memorilla-port
gpu=$1; cap=$2; shift 2
export E="MPORT_Q8=1 MPORT_QBITS=4 MPORT_LEAN=1 MPORT_OFFLOAD=1 MPORT_GPU_CAP_GB=$cap PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=\$HOME/train/cache/triton"
for x in "$@"; do
  tag=${x%%=*}; lora=${x#*=}
  experiments/v100.sh run s1_$tag "$gpu" poc_tok.py rate --lora $lora --raters tokL --tag ${tag}_full --wordings change --stride 1 --batch 48 || exit 1
  sleep 20; while ! experiments/v100.sh log s1_$tag 3 2>/dev/null | grep -q "V100_DONE s1_$tag"; do sleep 60; done
done
echo S1_DONE
