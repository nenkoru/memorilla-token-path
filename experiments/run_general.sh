#!/bin/zsh
# H13: people facts (non-code) through the biased core, one 16-token block. Zero-shot first, then 75 min of mixed training.
cd "${0:A:h}/.." && . experiments/env.sh
export INTERVAL=5
R=work/runs/agentmem
step() { s=$1; n=$2; shift 2; echo "== $n $(date +%H:%M)"; SCRIPT=$s experiments/poc_guard.sh $n "$@"; echo "   exit $?"; }
until grep -q ALL_DONE work/logs/run_bias2.log; do sleep 20; done
[ -f $R/general_emb.npz ] || step poc_general.py ge_embed embed
step poc_general.py ge_cos cos
step poc_general.py ge_eval0 eval --lora lora_bias.pt --core core_bias.pt --n 60 --sizes 1,16,200 --out general_eval_zeroshot.json
step poc_general.py ge_train train --minutes 75 --tag gen
[ -f $R/lora_gen.pt ] && step poc_general.py ge_eval eval --lora lora_gen.pt --core core_gen.pt --n 150 --sizes 1,16,200,700 --out general_eval.json
[ -f $R/lora_gen.pt ] && step poc_core.py ge_code eval --lora lora_gen.pt --core core_gen.pt --sizes 16 --cap 100000 --values plain --out core_eval_gen_code.json
echo "ALL_DONE $(date +%H:%M)"
