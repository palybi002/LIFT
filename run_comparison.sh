#!/bin/bash

# Define common parameters
SEQ_LEN=336
PRED_LEN=96
EPOCHS=3  # Reduced for demonstration speed, increase for real results
ITR=1

# Create logs directory
mkdir -p logs

run_experiment() {
    MODEL=$1
    DATASET=$2
    FEATURES=$3
    DESC=$4
    
    echo "----------------------------------------------------------------"
    echo "Running $MODEL on $DATASET ($FEATURES) - $DESC"
    echo "----------------------------------------------------------------"
    
    LOG_FILE="logs/${MODEL}_${DATASET}_${FEATURES}.log"
    
    python3 run_longExp.py \
      --model $MODEL \
      --dataset $DATASET \
      --features $FEATURES \
      --seq_len $SEQ_LEN \
      --pred_len $PRED_LEN \
      --train_epochs $EPOCHS \
      --itr $ITR \
      --gpu 0 \
      > $LOG_FILE 2>&1
      
    echo "Finished. Log saved to $LOG_FILE"
    
    # Extract results
    echo "Results:"
    grep -E "mse:|mae:|Number of Params:|Forward Time|cost time" $LOG_FILE | tail -n 5
}

# 1. Weather Dataset Comparison
echo "================================================================"
echo "STARTING WEATHER DATASET COMPARISON"
echo "================================================================"

# A. LACFNet (Proposed)
run_experiment LACFNet Weather M "Proposed Method"

# B. DLinear (CI Baseline)
run_experiment DLinear Weather M "CI Baseline"

# C. PatchTST (SOTA CI Baseline)
run_experiment PatchTST Weather M "SOTA CI Baseline"

# D. Autoformer (Transformer Baseline)
run_experiment Autoformer Weather M "Transformer Baseline"

# E. Univariate Baseline (Using DLinear with features=S as proxy for 'Single variable prediction')
# The user asked for "Single variable prediction model (Only use core variable)"
# We simulate this by running DLinear with features='S' (Univariate)
run_experiment DLinear Weather S "Univariate Baseline"


# 2. AirQuality Dataset Comparison
echo "================================================================"
echo "STARTING AIR QUALITY DATASET COMPARISON"
echo "================================================================"

# A. LACFNet
run_experiment LACFNet AirQuality M "Proposed Method"

# B. DLinear
run_experiment DLinear AirQuality M "CI Baseline"

# C. PatchTST
run_experiment PatchTST AirQuality M "SOTA CI Baseline"

# D. Autoformer
run_experiment Autoformer AirQuality M "Transformer Baseline"

echo "================================================================"
echo "ALL EXPERIMENTS FINISHED"
echo "================================================================"
