#!/bin/bash
# Shared-prefix check and speed: original layout, then question-first; 40 steps each, one GPU, 4-bit offloaded.
cd ~/memorilla-port
export E='MPORT_Q8=1 MPORT_QBITS=4 MPORT_LEAN=1 MPORT_OFFLOAD=1 MPORT_GPU_CAP_GB=2.2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=$HOME/train/cache/triton'
step() {
  n=$1; shift
  experiments/v100.sh run "$n" 0 "$@" || return 1
  while ! experiments/v100.sh log "$n" 3 2>/dev/null | grep -q "V100_DONE $n"; do sleep 20; done
}
step sh_sys poc_tokrater.py --steps 40 --val 4 --check --layout sys --out lora_sh_sys.pt
step sh_qf poc_tokrater.py --steps 40 --val 4 --layout qfirst --out lora_sh_qf.pt
echo SHARED_DONE
