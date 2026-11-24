# source me from memorilla-port/: points the experiments at the persistent work/ folder (rebuilt 2026-11-20 after
# /private/tmp was wiped by a reboot). Assets: work/venv, work/repo/memorilla, work/ckpt/pv4, work/ext/memorilla-demos.
export MPORT_SCRATCH="$PWD/work"
export PYTHONPATH="$PWD:$MPORT_SCRATCH/repo/memorilla"
export TOKENIZERS_PARALLELISM=false
# RAM guard (36 GB unified memory, shared with everything else): cap MPS allocations at ~0.8 x the recommended
# working set (~21 GB) so an overrun raises an out-of-memory error instead of swapping the machine to death.
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8
export PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6
export PY="$MPORT_SCRATCH/venv/bin/python"
