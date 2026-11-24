#!/bin/zsh
# Helper for the user's V100 box (gpu-box, 2 x V100 32 GB, no bfloat16: MPORT_DTYPE=float16).
# Layout there: ~/train/{proj (code), work (data, results, logs), hf (models), .venv}. Nothing else is touched.
#   experiments/v100.sh sync                               copy experiments/ and mport/ to the box
#   experiments/v100.sh run <name> <gpu> <script> [args]   start on one GPU in the background; log work/logs/v100_<name>.log
#   experiments/v100.sh log <name> [lines]                 tail of that log
#   experiments/v100.sh pull <path under work/>            copy a result back
#   experiments/v100.sh push <path under work/>            copy a file or folder to the box
#   experiments/v100.sh gpus                               GPU memory and my processes
#   experiments/v100.sh ddp <name> <script> [args]         one run on both GPUs (data parallel, see fast_train.py)
#   experiments/v100.sh serve <gpu> <port> [lora tags]  my vLLM server (production's build, read-only); unserve stops mine
# Extra environment: E='MPORT_GEN_TOKENS=56000' experiments/v100.sh run ...
# Arguments must not contain spaces or quotes (use _ in --conds).
cd "${0:A:h}/.." || exit 1
H=user@gpu-box
S=(ssh -o BatchMode=yes -o ConnectTimeout=15)
cmd=$1; shift
case $cmd in
  sync)
    rsync -a --exclude __pycache__ --exclude .DS_Store -e "ssh -o BatchMode=yes" experiments mport ${H}:train/proj/ ;;
  run)
    name=$1; gpu=$2; script=$3; shift 3
    $0 sync || exit 1
    $S $H "cd ~/train/proj || exit 1; $E CUDA_VISIBLE_DEVICES=$gpu MPORT_SCRATCH=\$HOME/train/work MEMORILLA_REPO=\$HOME/train/proj/repo/memorilla HF_HOME=\$HOME/train/hf HF_HUB_OFFLINE=1 MPORT_DTYPE=float16 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 nohup sh -c '../.venv/bin/python -I experiments/modal_entry.py $script $*; echo V100_DONE $name exit \$?' > ../work/logs/v100_$name.log 2>&1 < /dev/null & echo started $name on gpu $gpu" ;;
  ddp)
    # one training run on both GPUs (data parallel): ranks 0 and 1, logs v100_<name>.log and v100_<name>_r1.log
    name=$1; shift
    d=\$HOME/train/work/dist_$name
    $S $H "rm -f ~/train/work/dist_$name" || exit 1
    E="$E MPORT_WORLD=2 MPORT_RANK=0 MPORT_DIST=$d" $0 run $name 0 "$@" || exit 1
    E="$E MPORT_WORLD=2 MPORT_RANK=1 MPORT_DIST=$d" $0 run ${name}_r1 1 "$@" ;;
  serve)
    # my own vLLM server from production's V100 build, used read-only (approved by the user 2026-11-23); caches in my folder.
    # serve <gpu> <port> [lora tags ...]: LoRA adapters from work/runs/poc/peft_<tag>, requested as model <tag>
    gpu=$1; port=$2; shift 2
    lora=""; [ $# -gt 0 ] && lora="--enable-lora --max-lora-rank 16 --max-loras $# --lora-modules $(for t in "$@"; do printf '%s=$HOME/train/work/runs/poc/peft_%s ' $t $t; done)"
    $0 sync || exit 1
    $S $H "cd ~/train || exit 1; M=\$HOME/train/hf/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218; CUDA_VISIBLE_DEVICES=$gpu CUDA_DEVICE_ORDER=PCI_BUS_ID LD_LIBRARY_PATH=\$HOME/nvlibs CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:\$PATH VLLM_CACHE_ROOT=\$HOME/train/cache/vllm XDG_CACHE_HOME=\$HOME/train/cache/xdg TORCHINDUCTOR_CACHE_DIR=\$HOME/train/cache/inductor TRITON_CACHE_DIR=\$HOME/train/cache/triton HF_HOME=\$HOME/train/hf HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 nohup ~/<vllm-venv>/bin/python proj/experiments/vllm_lora_patch.py serve \$M --served-model-name qwen3-8b --host 127.0.0.1 --port $port --dtype half --max-model-len 4096 --gpu-memory-utilization 0.90 --enable-prefix-caching --max-num-seqs 256 --max-num-batched-tokens 8192 $lora > work/logs/vllm_$port.log 2>&1 < /dev/null & echo serving on gpu $gpu port $port"
    for i in $(seq 1 40); do sleep 10; $S $H "curl -s -m 3 127.0.0.1:$port/v1/models" 2>/dev/null | grep -q '"id"' && { echo up; exit 0; }; done; echo "not up after 400 s"; exit 1 ;;
  serve27)
    # H41 (approved by the user 2026-11-23): production's 27B (on disk, read-only) as a fact rater, both GPUs (TP 2), port 8150.
    # My own minimal command with production's venv; not production's launch script (that one writes into production's folders).
    # 1.5.3 fails at startup with "Found 1 <|image_pad|> tokens ... no images" (vision profiling); text only, so images 0 (untested yet).
    # Venv: the newest build, prod-2026-11-19 (a V100 vLLM build 1.5.3; ~/<vllm-venv> still points to 11-03, 1.5.1). VENV=<vllm-venv> to override.
    $S $H "cd ~/train || exit 1; M=\$HOME/models/<qwen-27b>; CUDA_VISIBLE_DEVICES=0,1 CUDA_DEVICE_ORDER=PCI_BUS_ID NCCL_P2P_LEVEL=SYS VLLM_SM70_NVFP4_OUTPUT_ALIGNMENT=64 LD_LIBRARY_PATH=\$HOME/nvlibs CUDA_HOME=/usr/local/cuda-12.9 PATH=/usr/local/cuda-12.9/bin:\$PATH VLLM_CACHE_ROOT=\$HOME/train/cache/vllm XDG_CACHE_HOME=\$HOME/train/cache/xdg TORCHINDUCTOR_CACHE_DIR=\$HOME/train/cache/inductor TRITON_CACHE_DIR=\$HOME/train/cache/triton HF_HOME=\$HOME/train/hf HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 nohup \$HOME/${VENV:-<vllm-venv>}/bin/vllm serve \$M --served-model-name qwen3-27b --host 127.0.0.1 --port 8150 --dtype half --tensor-parallel-size 2 --max-model-len 4096 --max-num-seqs 64 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.88 --kv-cache-dtype fp8 --enable-prefix-caching --limit-mm-per-prompt '{\"image\":0,\"video\":0}' > work/logs/vllm_8150.log 2>&1 < /dev/null & echo serving 27B on both gpus port 8150"
    for i in $(seq 1 60); do sleep 10; $S $H "curl -s -m 3 127.0.0.1:8150/v1/models" 2>/dev/null | grep -q '"id"' && { echo up; exit 0; }; done; echo "not up after 600 s"; exit 1 ;;
  unserve)
    # stop my vLLM servers (only processes started with --port 8100-8199 by serve)
    $S $H "pgrep -u nenkoru -f '[p]ort 81[0-9][0-9] --dtype half' | xargs -r kill; sleep 12; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader" ;;
  log)
    $S $H "tr '\r' '\n' < ~/train/work/logs/v100_$1.log | grep -av 'Loading weights' | tail -${2:-25}" ;;
  pull)
    mkdir -p "work/${1:h}"; rsync -a -e "ssh -o BatchMode=yes" "${H}:train/work/$1" "work/${1:h}/" ;;
  push)
    # copy one of my files or folders to the box: push <path under work/>  (e.g. ckpt/pv4, runs/agentmem/lora_lsort.pt)
    $S $H "mkdir -p ~/train/work/${1:h}" && rsync -a -e "ssh -o BatchMode=yes" "work/$1" "${H}:train/work/${1:h}/" ;;
  gpus)
    $S $H "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader; df -h / | tail -1" ;;
  *) sed -n 2,12p "$0" ;;
esac
