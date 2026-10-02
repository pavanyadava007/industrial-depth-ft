#!/usr/bin/env bash
# Full GPU queue (one job at a time on the NVIDIA L4). Logs in logs/. Re-running skips finished results.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
STEPS=${STEPS:-3000}
t() { date -u +%T; }
run() {  # run <result-json> <log-name> <cmd...>
  local res=$1 name=$2; shift 2
  if [ -f "$res" ]; then echo "$(t) skip $name"; return; fi
  echo "$(t) start $name"
  local s=$(date +%s)
  "$@" > "logs/$name.log" 2>&1 || echo "$(t) FAILED $name"
  echo "{\"job\": \"$name\", \"seconds\": $(( $(date +%s) - s ))}" > "results/runtime_$name.json"
  echo "$(t) done $name ($(( $(date +%s) - s )) s)"
}
run results/zeroshot_baseline_constant.json baselines $PY scripts/eval_baselines.py
for m in dav2_small_rel dav2_metric_indoor_small dav2_metric_indoor_large dav2_metric_outdoor_large moge2 depth_pro; do
  run results/zeroshot_$m.json zs_$m $PY scripts/eval_zeroshot.py --model $m
done
# B: three fine-tuning configurations, all training images, seed 0
run results/ft_B_head.json ft_B_head $PY scripts/train_ft.py --name B_head --mode head --lr-decoder 2e-4 --steps $STEPS --save-preds --save-ckpt
run results/ft_B_full.json ft_B_full $PY scripts/train_ft.py --name B_full --mode full --steps $STEPS --save-preds --save-ckpt
run results/ft_B_lora.json ft_B_lora $PY scripts/train_ft.py --name B_lora --mode lora --lr-decoder 1e-4 --steps $STEPS --save-preds --save-ckpt
# best configuration by VALIDATION RMSE (never by test)
BEST=$($PY - <<'PYEOF'
import json
r = {m: json.load(open(f"results/ft_B_{m}.json"))["best_val_rmse_mm"] for m in ("head", "full", "lora")}
print(min(r, key=r.get))
PYEOF
)
echo "$(t) best config by val: $BEST"
case $BEST in head) LR="--lr-decoder 2e-4";; lora) LR="--lr-decoder 1e-4";; *) LR="";; esac
echo "{\"best_config_by_val\": \"$BEST\"}" > results/best_config.json
# C: data requirement curve (seed 0 of N=all is the B run)
# 3 seeds for the smallest N (and for N = all below); one seed for the middle points (GPU budget)
for n in 10 50 200; do
  seeds="0"; [ $n = 10 ] && seeds="0 1 2"
  for s in $seeds; do
    run results/ft_C_n${n}_s${s}.json ft_C_n${n}_s${s} $PY scripts/train_ft.py --name C_n${n}_s${s} --experiment C --mode $BEST $LR --n-train $n --seed $s --steps $STEPS
  done
done
for s in 1 2; do
  run results/ft_C_nall_s${s}.json ft_C_nall_s${s} $PY scripts/train_ft.py --name C_nall_s${s} --experiment C --mode $BEST $LR --seed $s --steps $STEPS
done
# D: train on 7 seen categories, test on all (held-out reported separately)
for s in 0 1 2; do
  extra=""; [ $s = 0 ] && extra="--save-preds"
  run results/ft_D_seen_s${s}.json ft_D_seen_s${s} $PY scripts/train_ft.py --name D_seen_s${s} --experiment D --split seen --mode $BEST $LR --seed $s --steps $STEPS $extra
done
echo "$(t) queue finished"
