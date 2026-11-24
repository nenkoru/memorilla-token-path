#!/bin/zsh
# Modal helper. Run from anywhere.
#   experiments/modal.sh push [dir...]          upload work/<dir> to the volume (default: ckpt runs ext/memorilla-demos)
#   experiments/modal.sh models                 download the two Qwen models into the volume (once)
#   experiments/modal.sh run <name> <script> [args...]   start a detached GPU run; log in work/logs/modal_<name>.log
#   experiments/modal.sh pull <path under work/>          copy a result file or folder back from the volume
#   experiments/modal.sh vlog <name>                       print the run's log from the volume
#   experiments/modal.sh ls [path]
# GPU type: MPORT_GPU=L40S (default) | A100-40GB | H100 | L4 ...
cd "${0:A:h}/.." || exit 1
M=work/venv-modal/bin/modal
V=memorilla-port
cmd=$1; shift
case $cmd in
  push)
    [ $# -eq 0 ] && set -- ckpt runs ext/memorilla-demos
    for d in "$@"; do echo "== push work/$d"; $M volume put --force $V "work/$d" "/work/$d" || exit 1; done ;;
  models) $M run experiments/modal_fetch.py ;;
  run)
    name=$1; script=$2; shift 2
    echo "== modal run $name: $script $* (gpu ${MPORT_GPU:-L40S}) $(date +%H:%M)"
    $M run -d experiments/modal_app.py --script "$script" --args "$*" --name "$name" > "work/logs/modal_$name.launch" 2>&1
    grep -a "spawned" "work/logs/modal_$name.launch" || { tail -5 "work/logs/modal_$name.launch"; exit 1; }
    id=$(grep -ao "ap-[A-Za-z0-9]*" "work/logs/modal_$name.launch" | head -1)
    nohup $M app logs "$id" > "work/logs/modal_$name.log" 2>&1 &   # live copy; the run itself does not need this Mac
    echo "app $id  log: work/logs/modal_$name.log  (volume copy: /work/logs/modal_$name.log)" ;;
  vlog)  # the log as saved on the volume (updated every minute), for after the Mac was asleep
    mkdir -p work/logs/vol; $M volume get --force $V "/work/logs/modal_$1.log" work/logs/vol/ >/dev/null && tr '\r' '\n' < "work/logs/vol/modal_$1.log" ;;
  pull)
    mkdir -p "work/${1:h}"; $M volume get --force $V "/work/$1" "work/${1:h}/" ;;
  ls) $M volume ls $V "${1:-/work}" ;;
  *) sed -n 2,9p "$0" ;;
esac
