#!/bin/zsh
# Memory demo. `experiments/demo.sh` = interactive chat; `experiments/demo.sh examples` = canned transcript.
# Extra arguments go to demo.py (e.g. --lora lora_gen.pt, --force).
cd "${0:A:h}/.." && . experiments/env.sh
mode=chat; [ "$1" = examples ] && { mode=examples; shift; }
while true; do
  caffeinate -i "$PY" -I experiments/demo.py embed || exit 1
  caffeinate -i "$PY" -I experiments/demo.py $mode "$@"; r=$?
  [ $r -eq 7 ] || exit $r
done
