#!/bin/bash

# Define parameters
SEQ_LEN=336
PRED_LEN=96
EPOCHS=3 
ITR=1
RESULT_FILE="results/comparison_metrics.txt"

# Ensure results directory exists and clear previous metrics
mkdir -p results
> $RESULT_FILE

run_experiment() {
    MODEL=$1
    DATASET=$2
    FEATURES=$3
    DESC=$4
    EXTRA_ARGS=$5
    
    echo "----------------------------------------------------------------"
    echo "Running $MODEL on $DATASET ($FEATURES) - $DESC"
    echo "----------------------------------------------------------------"
    
    python3 run_longExp.py \
      --model $MODEL \
      --dataset $DATASET \
      --features $FEATURES \
      --seq_len $SEQ_LEN \
      --pred_len $PRED_LEN \
      --train_epochs $EPOCHS \
      --itr $ITR \
      --gpu 0 \
      $EXTRA_ARGS
      
    echo "Experiment finished."
}

# 1. Weather Dataset Comparison
echo "================================================================"
echo "STARTING WEATHER DATASET COMPARISON"
echo "================================================================"

# A. Univariate Baseline (Single Variable Prediction using DLinear on 'S' mode)
# "以单变量预测模型(仅使用核心变量建模)"
run_experiment DLinear Weather S "Univariate Baseline"

# B. Pure CI Baseline (Channel Independent using DLinear on 'M' mode)
# "纯 CI 模型(对所有变量独立预测...)"
run_experiment DLinear Weather M "Pure CI Baseline"

# C. Non-CI / Global Baseline (Using Autoformer as proxy for iTransformer/Non-CI)
# "与 iTransformer ... 进行横向对比" - Using Autoformer as Non-CI representative
run_experiment Autoformer Weather M "Non-CI Baseline (Autoformer)"

# D. SOTA CI Baseline (PatchTST)
run_experiment PatchTST Weather M "SOTA CI Baseline"

# E. LACF-Net (Proposed)
run_experiment LACFNet Weather M "Proposed Method (LACF-Net)"


# 2. AirQuality Dataset Comparison
echo "================================================================"
echo "STARTING AIR QUALITY DATASET COMPARISON"
echo "================================================================"
# Note: Reducing learning rate for AirQuality to avoid NaN

# A. Univariate Baseline
run_experiment DLinear AirQuality S "Univariate Baseline" "--learning_rate 0.0001"

# B. Pure CI Baseline
run_experiment DLinear AirQuality M "Pure CI Baseline" "--learning_rate 0.0001"

# C. Non-CI Baseline
run_experiment Autoformer AirQuality M "Non-CI Baseline" "--learning_rate 0.0001"

# D. SOTA CI Baseline
run_experiment PatchTST AirQuality M "SOTA CI Baseline" "--learning_rate 0.0001"

# E. LACF-Net
run_experiment LACFNet AirQuality M "Proposed Method" "--learning_rate 0.0001"

echo "================================================================"
echo "ALL EXPERIMENTS FINISHED"
echo "Results saved to $RESULT_FILE"
echo "================================================================"
