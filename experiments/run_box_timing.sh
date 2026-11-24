#!/bin/bash
# Where a step's time goes: original layout with shared prefix; offloaded vs prefetch off (MPORT_PREFETCH=0 not wired: one run).
cd ~/memorilla-port
export E='MPORT_TIMING=1 MPORT_Q8=1 MPORT_QBITS=4 MPORT_LEAN=1 MPORT_OFFLOAD=1 MPORT_GPU_CAP_GB=2.2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=$HOME/train/cache/triton'
experiments/v100.sh run tm_sys 0 poc_tokrater.py --steps 20 --val 2 --layout sys --out lora_tm.pt
while ! experiments/v100.sh log tm_sys 3 2>/dev/null | grep -q "V100_DONE tm_sys"; do sleep 15; done
echo TIMING_DONE
