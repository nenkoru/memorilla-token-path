#!/bin/zsh
# Run one poc_clue.py command under a RAM watchdog: log free memory every 15 s; kill the job if the system-wide
# free percentage drops below $MIN_FREE (default 12) so a runaway job can't take the machine down.
# usage: . experiments/env.sh && [SCRIPT=poc_adapt.py] [INTERVAL=5] experiments/poc_guard.sh <name> <args...>
name=$1; shift
MIN_FREE=${MIN_FREE:-12}
log=$MPORT_SCRATCH/logs/poc_$name.log; mem=$MPORT_SCRATCH/logs/poc_mem.log
"$PY" -I experiments/${SCRIPT:-poc_clue.py} "$@" > "$log" 2>&1 &
pid=$!
while kill -0 $pid 2>/dev/null; do
  free=$(memory_pressure | awk -F': ' '/free percentage/ {gsub("%","",$2); print $2}')
  rss=$(ps -o rss= -p $pid | awk '{printf "%.1f", $1/1048576}')
  echo "$(date +%H:%M:%S) $name free=${free}% rss=${rss}GB" >> "$mem"
  if [ -n "$free" ] && [ "$free" -lt "$MIN_FREE" ]; then
    echo "$(date +%H:%M:%S) $name KILLED: free memory ${free}% < ${MIN_FREE}%" | tee -a "$mem" >> "$log"
    kill $pid; sleep 5; kill -9 $pid 2>/dev/null; exit 99
  fi
  sleep ${INTERVAL:-15}
done
wait $pid
