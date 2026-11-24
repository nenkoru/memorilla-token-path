#!/bin/zsh
# Everything that was queued on 22 Nov, restarted after the demo was opened mid-run: H15 training + eval, the Django
# test (H16), its plain-retrieval baseline, and the canned demo transcript. Waits while the interactive demo is open.
cd "${0:A:h}/.." && . experiments/env.sh
while pgrep -f "experiments/demo.py" > /dev/null; do sleep 20; done
export INTERVAL=5
step() { n=$1; shift; echo "== $n $(date +%H:%M)"; experiments/poc_guard.sh $n "$@"; r=$?; echo "   exit $r"; return $r; }
export SCRIPT=poc_profile.py
step pf_train train --minutes 60
step pf_eval eval
echo "PROFILE_DONE $(date +%H:%M)"
export SCRIPT=poc_django.py
step dj_embed embed
step dj_para para
step dj_paraemb paraemb
step dj_cos cos
if step dj_self selftest; then
  step dj_eval0 eval --lora lora_gen.pt --out django_eval_zeroshot.json
  step dj_train train --minutes 40
  step dj_eval eval --lora lora_dj.pt --out django_eval.json
else
  echo "selftest failed"
fi
step dj_text textrag
echo "DJANGO_DONE $(date +%H:%M)"
export SCRIPT=demo.py
step dm_embed embed
step dm_examples examples
echo "ALL_DONE $(date +%H:%M)"
