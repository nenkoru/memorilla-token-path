#!/bin/bash
# One A/B arm on the box: train the block rater (4-bit, offloaded, one GPU), then rate the eval set with it.
#   run_box_ab.sh <tag> <gpu> <cap GB> "<train args>" "<rate args>"
cd ~/memorilla-port
tag=$1; gpu=$2; cap=$3; targs=$4; rargs=$5
export E="MPORT_Q8=1 MPORT_QBITS=4 MPORT_LEAN=1 MPORT_OFFLOAD=1 MPORT_GPU_CAP_GB=$cap PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=\$HOME/train/cache/triton"
step() {
  n=$1; shift
  experiments/v100.sh run "$n" "$gpu" "$@" || return 1
  while ! experiments/v100.sh log "$n" 3 2>/dev/null | grep -q "V100_DONE $n"; do sleep 30; done
  experiments/v100.sh log "$n" 3 | grep -q "V100_DONE $n exit 0"
}
step ab_${tag}_train poc_tokrater.py $targs --out lora_tokrater_$tag.pt || exit 1
step ab_${tag}_rate poc_tok.py rate --lora lora_tokrater_$tag.pt --raters tokL --tag $tag --wordings change --stride 3 --batch 48 $rargs || exit 2
echo AB_DONE $tag
