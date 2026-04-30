#!/bin/bash
set -euo pipefail

SEQ_LEN=${SEQ_LEN:-336}
PRED_LEN=${PRED_LEN:-96}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-1}
ITR=${ITR:-1}
GPU=${GPU:-0}
ONLY_TEST=${ONLY_TEST:-true}

mkdir -p logs prefetch plots

run_prefetch() {
  local dataset=$1
  echo "[PREFETCH] ${dataset}"
  python3 run_prefetch.py \
    --model DLinear \
    --dataset "${dataset}" \
    --features M \
    --seq_len "${SEQ_LEN}" \
    --pred_len "${PRED_LEN}" \
    --leader_num 4 \
    --state_num 8 \
    --tag _max \
    > "logs/prefetch_${dataset}.log" 2>&1
}

run_exp() {
  local model=$1
  local dataset=$2
  local features=$3
  local logfile=$4
  local extra=${5:-}

  echo "[RUN] ${model} ${dataset} ${features} -> ${logfile}"
  python3 run_longExp.py \
    --model "${model}" \
    --dataset "${dataset}" \
    --features "${features}" \
    --seq_len "${SEQ_LEN}" \
    --pred_len "${PRED_LEN}" \
    --train_epochs "${TRAIN_EPOCHS}" \
    --itr "${ITR}" \
    --gpu "${GPU}" \
    --checkpoints ./checkpoints/ \
    --des test \
    $( [[ "${ONLY_TEST}" == "true" ]] && echo "--only_test" ) \
    ${extra} \
    > "${logfile}" 2>&1
}

# Full comparison matrix:
# - AirQuality / Weather: DLinear, DLinear+LIFT, LACFNet, PatchTST, Autoformer
# - ECL / Exchange / Sales: DLinear, DLinear+LIFT, LACFNet
CORE_DATASETS=("AirQuality" "Weather")
EXTRA_DATASETS=("ECL" "Exchange" "Sales")

for ds in "${CORE_DATASETS[@]}"; do
  run_prefetch "${ds}"
  run_exp DLinear "${ds}" M "logs/cmp_${ds}_DLinear_M.log" "--individual --learning_rate 0.0001"
  run_exp DLinear "${ds}" M "logs/cmp_${ds}_LIFT_M.log" "--individual --lift --leader_num 4 --state_num 8 --tag _max --learning_rate 0.0001"
  run_exp LACFNet "${ds}" M "logs/cmp_${ds}_LACFNet_M.log" "--learning_rate 0.001"
  run_exp PatchTST "${ds}" M "logs/cmp_${ds}_PatchTST_M.log" "--learning_rate 0.0001"
  run_exp Autoformer "${ds}" M "logs/cmp_${ds}_Autoformer_M.log" "--learning_rate 0.0001"
done

for ds in "${EXTRA_DATASETS[@]}"; do
  run_prefetch "${ds}"
  run_exp DLinear "${ds}" M "logs/cmp_${ds}_DLinear_M.log" "--individual --learning_rate 0.0001"
  run_exp DLinear "${ds}" M "logs/cmp_${ds}_LIFT_M.log" "--individual --lift --leader_num 4 --state_num 8 --tag _max --learning_rate 0.0001"
  run_exp LACFNet "${ds}" M "logs/cmp_${ds}_LACFNet_M.log" "--learning_rate 0.001"
done

echo "All comparison experiments finished."
