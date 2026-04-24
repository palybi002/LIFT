#!/bin/bash

# Define common parameters
SEQ_LEN=336
PRED_LEN=96
EPOCHS=3  # Standard number of epochs for quick comparison
ITR=1     # Number of iterations per experiment

# Create directories
mkdir -p logs
mkdir -p prefetch
mkdir -p plots

# Function to run prefetch (required for LIFT)
run_prefetch() {
    DATASET=$1
    echo ">>>> Running Prefetch for $DATASET..."
    
    python3 run_prefetch.py \
      --model DLinear \
      --dataset $DATASET \
      --features M \
      --seq_len $SEQ_LEN \
      --pred_len $PRED_LEN \
      --leader_num 4 \
      --tag _max \
      > "logs/Prefetch_${DATASET}.log" 2>&1
}

# Function to run experiment
run_experiment() {
    MODEL_NAME=$1      # Name used for logging (e.g., LIFT, DLinear, LACFNet)
    ACTUAL_MODEL=$2    # Actual model class name (e.g., DLinear, LACFNet)
    DATASET=$3
    FEATURES=$4
    EXTRA_ARGS=$5
    
    LOG_FILE="logs/${MODEL_NAME}_${DATASET}_${FEATURES}.log"
    
    echo ">>>> Running $MODEL_NAME on $DATASET..."
    
    python3 run_longExp.py \
      --model $ACTUAL_MODEL \
      --dataset $DATASET \
      --features $FEATURES \
      --seq_len $SEQ_LEN \
      --pred_len $PRED_LEN \
      --train_epochs $EPOCHS \
      --itr $ITR \
      --checkpoints ./checkpoints/ \
      --des 'test' \
      $EXTRA_ARGS \
      > "$LOG_FILE" 2>&1
      
    echo "Finished $MODEL_NAME on $DATASET. Log saved to $LOG_FILE"
}

# List of datasets to process
# Only including dataset names defined in settings.py AND present in dataset directory
DATASETS=("AirQuality" "Weather" "ECL" "Exchange" "Sales")

for DATASET in "${DATASETS[@]}"; do
    echo "========================================================"
    echo "Processing Dataset: $DATASET"
    echo "========================================================"

    # 1. Run Prefetch (Required for LIFT)
    run_prefetch "$DATASET"

    # 2. Run DLinear (Baseline)
    run_experiment "DLinear" "DLinear" "$DATASET" "M" "--individual"

    # 3. Run LIFT (DLinear + LIFT)
    # LIFT requires prefetch to be done first
    # Using specific args for LIFT based on project guide
    run_experiment "LIFT" "DLinear" "$DATASET" "M" "--individual --lift --tag _max --leader_num 4"

    # 4. Run LACFNet (Your Model)
    # Assuming standard parameters for LACFNet
    run_experiment "LACFNet" "LACFNet" "$DATASET" "M" "--learning_rate 0.001"

done

echo "========================================================"
echo "All Experiments Completed."
echo "Running Analysis and Visualization..."
echo "========================================================"

# Analyze logs to update CSV
python3 analyze_results.py

# Visualize results with timestamp
python3 visualize_results.py
