# source after env.sh: Mac light mode, so the runs leave room for other work.
# int8 decoder layers (~11 GB on the GPU instead of ~20), GPU allocations capped at ~15 GB (an overrun is an
# out-of-memory error, not swapping), 4 CPU threads, and poc_guard kills a run when free memory drops below 25%.
export MPORT_Q8=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.55 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.45
export MIN_FREE=25
