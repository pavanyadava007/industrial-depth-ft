#!/usr/bin/env bash
# After run_queue.sh: optional synthetic track (S), quality analysis (E) and ONNX export (F).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
STEPS=${STEPS:-3000}
t() { date -u +%T; }
run() {
  local res=$1 name=$2; shift 2
  if [ -f "$res" ]; then echo "$(t) skip $name"; return; fi
  echo "$(t) start $name"
  local s=$(date +%s)
  "$@" > "logs/$name.log" 2>&1 || echo "$(t) FAILED $name"
  echo "{\"job\": \"$name\", \"seconds\": $(( $(date +%s) - s ))}" > "results/runtime_$name.json"
  echo "$(t) done $name ($(( $(date +%s) - s )) s)"
}
BEST=$($PY -c "import json;print(json.load(open('results/best_config.json'))['best_config_by_val'])")
case $BEST in head) LR="--lr-decoder 2e-4";; lora) LR="--lr-decoder 1e-4";; *) LR="";; esac
run results/ft_S_synth_only.json ft_S_synth_only $PY scripts/train_ft.py --name S_synth_only --experiment S --mode $BEST $LR --synth-only --synth 1000 --steps $STEPS
run results/ft_S_synth_real_all.json ft_S_synth_real_all $PY scripts/train_ft.py --name S_synth_real_all --experiment S --mode $BEST $LR --synth 1000 --steps $STEPS
for s in 0; do
  run results/ft_S_synth_real50_s$s.json ft_S_synth_real50_s$s $PY scripts/train_ft.py --name S_synth_real50_s$s --experiment S --mode $BEST $LR --synth 1000 --n-train 50 --seed $s --steps $STEPS
done
run results/quality.json quality $PY scripts/analyse_quality.py \
  --variants B_$BEST D_seen_s0 moge2 dav2_metric_outdoor_large depth_pro dav2_small_rel \
  --kinds metric metric metric metric metric disparity
run results/export_onnx.json export_onnx $PY scripts/export_onnx.py --ckpt data/ckpt/B_$BEST.pt
echo "$(t) post finished"
