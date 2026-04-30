#!/bin/bash
set -euo pipefail

SEQ_LEN=${SEQ_LEN:-336}
PRED_LEN=${PRED_LEN:-96}
TRAIN_EPOCHS=${TRAIN_EPOCHS:-1}
ITR=${ITR:-1}
GPU=${GPU:-0}
ONLY_TEST=${ONLY_TEST:-true}

mkdir -p logs prefetch plots

ABL_DATASETS=("AirQuality" "Weather")
LACF_TOPKS=(1 3 5)
LIFT_LEADERS=(2 4 8)
LIFT_STATES=(4 8 16)

run_prefetch() {
  local dataset=$1
  echo "[PREFETCH] ${dataset}"
  python3 run_prefetch.py \
    --model DLinear \
    --dataset "${dataset}" \
    --features M \
    --seq_len "${SEQ_LEN}" \
    --pred_len "${PRED_LEN}" \
    --leader_num 8 \
    --state_num 16 \
    --tag _max \
    > "logs/prefetch_ablation_${dataset}.log" 2>&1
}

run_exp() {
  local model=$1
  local dataset=$2
  local features=$3
  local logfile=$4
  local des_tag=$5
  local extra=${6:-}

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
    --des "${des_tag}" \
    $( [[ "${ONLY_TEST}" == "true" ]] && echo "--only_test" ) \
    ${extra} \
    > "${logfile}" 2>&1
}

for ds in "${ABL_DATASETS[@]}"; do
  run_prefetch "${ds}"

  # Ablation A: LACFNet top_k
  for k in "${LACF_TOPKS[@]}"; do
    run_exp LACFNet "${ds}" M "logs/abl_${ds}_LACF_topk_${k}.log" "abl_lacf_topk_${k}" "--override_hyper false --top_k ${k} --learning_rate 0.001"
  done

  # Ablation B: LIFT leader_num (fix state_num=8)
  for k in "${LIFT_LEADERS[@]}"; do
    run_exp DLinear "${ds}" M "logs/abl_${ds}_LIFT_leader_${k}.log" "abl_lift_leader_${k}" "--individual --lift --leader_num ${k} --state_num 8 --tag _max --learning_rate 0.0001"
  done

  # Ablation C: LIFT state_num (fix leader_num=4)
  for s in "${LIFT_STATES[@]}"; do
    run_exp DLinear "${ds}" M "logs/abl_${ds}_LIFT_state_${s}.log" "abl_lift_state_${s}" "--individual --lift --leader_num 4 --state_num ${s} --tag _max --learning_rate 0.0001"
  done

done

echo "All ablation experiments finished."
