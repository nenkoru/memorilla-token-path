#!/bin/bash
# Runs ON the box, detached: waits for one box job to finish, then starts the next one with the same environment
# v100.sh run uses. The chain then survives the Mac side dropping (waiters there die when the session restarts).
#   box_chain.sh <wait-for-job|-> <name> <gpu> "<extra env, space separated>" <script> <args...>
# Start it from the Mac with: experiments/v100.sh chain <wait-for-job|-> <name> <gpu> <script> <args...>   (E= as for run)
wait=$1; name=$2; gpu=$3; extra=$4; shift 4
R=$HOME/train; L=$R/work/logs
if [ "$wait" != - ]; then  # one job name, or several joined with commas: all must have finished
  all_done() { for w in ${wait//,/ }; do grep -q "V100_DONE $w" "$L/v100_$w.log" 2>/dev/null || return 1; done; }
  for i in $(seq 1 720); do all_done && break; sleep 30; done
  all_done || { echo "V100_DONE $name exit 97 (gave up waiting for $wait)" > "$L/v100_$name.log"; exit 97; }
fi
cd "$R/proj" || exit 1
env $extra CUDA_VISIBLE_DEVICES=$gpu MPORT_SCRATCH=$R/work MEMORILLA_REPO=$R/proj/repo/memorilla HF_HOME=$R/hf HF_HUB_OFFLINE=1 \
  MPORT_DTYPE=float16 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
  ../.venv/bin/python -I experiments/modal_entry.py "$@" > "$L/v100_$name.log" 2>&1
echo "V100_DONE $name exit $?" >> "$L/v100_$name.log"
