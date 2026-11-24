#!/bin/bash
# The reader on the box (4-bit, offloaded, one GPU): check the lean loss, answers with the current reader
# (lora_lsort.pt) on a rater's picks, train a reader with the clue at a random place among 1-4 blocks
# (poc_broad.py train without --sort), answers with it.
#   run_box_reader.sh <gpu> <train cap GB> <generation cap GB> <rater table, e.g. tokLr27w_change> <training minutes> [first step: check|base|train|ans]
cd ~/memorilla-port
gpu=$1; cap=$2; gcap=$3; table=$4; mins=$5; from=${6:-check}
base="MPORT_Q8=1 MPORT_QBITS=4 MPORT_LEAN=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=\$HOME/train/cache/triton"
OFF="$base MPORT_OFFLOAD=1 MPORT_GPU_CAP_GB=$cap"  # training: weights streamed from CPU memory (matmul-bound, ~1.5 GiB)
RES="$base MPORT_GPU_CAP_GB=$gcap"                  # generation: weights stay on the GPU (decoding would wait on PCIe)
step() {
  n=$1; shift
  experiments/v100.sh run "$n" "$gpu" "$@" || return 1
  while ! experiments/v100.sh log "$n" 3 2>/dev/null | grep -q "V100_DONE $n"; do sleep 30; done
  experiments/v100.sh log "$n" 3 | grep -q "V100_DONE $n exit 0"
}
ans() {  # ans <reader tag>; AGPU=<n> answers on another GPU than training
  local gpu=${AGPU:-$gpu}
  export E=$RES; step rd_ans_$1 poc_tok.py answer --lora lora_$1.pt --tables $table --stride 3 --batch 32 --out tok_answer_$1_$table.json
}
go=0  # bash 3.2 (macOS): no ;& fall-through, so run every step from $from on
[ $from = check ] && go=1; [ $go = 1 ] && { export E=$OFF; step rd_check reader_check.py || exit 1; }
[ $from = base ] && go=1; [ $go = 1 ] && { ans lsort || exit 2; }
[ $from = train ] && go=1; [ $go = 1 ] && { export E=$OFF; step rd_train poc_broad.py train --init lora_lsort.pt --tag rpos --long --recite --max-facts 4 --batch 4 --minutes $mins || exit 3; }
[ $from = ans ] && go=1; [ $go = 1 ] && { ans rpos || exit 4; }
echo RD_DONE
