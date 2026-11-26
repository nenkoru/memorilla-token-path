#!/bin/bash
# Distilled reader on the box: wait for a box job to finish (optional), check kd_loss_batch, train the rpos recipe
# from lora_lsort.pt with --kd (answer tokens: KL to the plain 8B reading the same facts as text), answers with it.
# Env: EXTRA="<more train args, e.g. --sort>"; ANS_OFF=1 answers offloaded at batch 16 (when the GPU is shared);
# ANS_ARGS="<all poc_tok.py answer args after --lora>" replaces the default stride-3 answer run.
#   run_box_kd.sh <gpu> <train cap GB> <generation cap GB> <rater table> <minutes> <alpha> <tag> [wait-for-job] [first step: check|train|ans]
cd ~/memorilla-port
gpu=$1; cap=$2; gcap=$3; table=$4; mins=$5; alpha=$6; tag=$7; waitfor=$8; from=${9:-check}
base="MPORT_Q8=1 MPORT_QBITS=4 MPORT_LEAN=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=\$HOME/train/cache/triton"
OFF="$base MPORT_OFFLOAD=1 MPORT_GPU_CAP_GB=$cap"
RES="$base MPORT_GPU_CAP_GB=$gcap"
done_() { experiments/v100.sh log "$1" 3 2>/dev/null | grep -q "V100_DONE $1"; }
step() {
  n=$1; shift
  experiments/v100.sh run "$n" "$gpu" "$@" || return 1
  sleep 20; while ! done_ "$n"; do sleep 30; done
  experiments/v100.sh log "$n" 3 | grep -q "V100_DONE $n exit 0"
}
[ -n "$waitfor" ] && [ "$waitfor" != - ] && while ! done_ "$waitfor"; do sleep 60; done
go=0
[ $from = check ] && go=1; [ $go = 1 ] && { export E=$OFF; step kd_check kd_check.py || exit 1; experiments/v100.sh log kd_check 40 | grep -q "aligned [34] of 4" || { echo KD_ALIGN_FAIL; exit 2; }; }
[ $from = train ] && go=1; [ $go = 1 ] && { export E=$OFF; step kd_train_$tag poc_broad.py train --init lora_lsort.pt --tag $tag --long --recite --max-facts 4 --batch 4 --minutes $mins --kd $alpha $EXTRA || exit 3; }
[ $from = ans ] && go=1; [ $go = 1 ] && { if [ -n "$ANS_OFF" ]; then export E=$OFF; bs=16; else export E=$RES; bs=32; fi; step rd_ans_$tag poc_tok.py answer --lora lora_$tag.pt ${ANS_ARGS:---tables $table --stride 3 --batch $bs --out tok_answer_${tag}_$table.json} || exit 4; }
echo KD_DONE
