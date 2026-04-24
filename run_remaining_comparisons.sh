#!/bin/bash

# Define common parameters
SEQ_LEN=336
PRED_LEN=96
EPOCHS=3 
ITR=1

# Create logs directory
mkdir -p logs
mkdir -p prefetch

run_prefetch() {
    DATASET=$1
    echo "Running Prefetch for $DATASET..."
    python3 run_prefetch.py \
      --model DLinear \
      --dataset $DATASET \
      --features M \
      --seq_len $SEQ_LEN \
      --pred_len $PRED_LEN \
      --leader_num 4 \
      --tag _max
}

run_experiment() {
    MODEL=$1
    DATASET=$2
    FEATURES=$3
    DESC=$4
    EXTRA_ARGS=$5
    
    echo "----------------------------------------------------------------"
    echo "Running $MODEL on $DATASET ($FEATURES) - $DESC"
    echo "----------------------------------------------------------------"
    
    LOG_FILE="logs/${MODEL}_${DATASET}_${FEATURES}.log"
    
    # Check if log already exists and is complete (simple check)
    if [ -f "$LOG_FILE" ]; then
        echo "Log file $LOG_FILE exists. Skipping..."
        # return
    fi
    
    python3 run_longExp.py \
      --model $MODEL \
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
      
    echo "Finished $MODEL on $DATASET. Log saved to $LOG_FILE"
}

# --- ECL (Electricity) ---
# 1. Prefetch for LIFT
run_prefetch "ECL"

# 2. DLinear (Baseline)
# run_experiment "DLinear" "ECL" "M" "Standard DLinear" "--individual"

# 3. LIFT (DLinear + Lift) - Interpreted as "LIFE"
run_experiment "DLinear" "ECL" "M" "LIFT Model" "--individual --lift --tag _max --leader_num 4"
# Rename log to LIFT for clarity in dashboard
mv logs/DLinear_ECL_M.log logs/LIFT_ECL_M.log 2>/dev/null || true

# 4. LACFNet (My Model)
run_experiment "LACFNet" "ECL" "M" "Standard LACFNet" "--learning_rate 0.001"

# Re-run DLinear to ensure we have its log (since we renamed the previous one or if it was skipped)
run_experiment "DLinear" "ECL" "M" "Standard DLinear" "--individual"


# --- Exchange (Exchange Rate) ---
# 1. Prefetch
run_prefetch "Exchange"

# 2. LIFT
run_experiment "DLinear" "Exchange" "M" "LIFT Model" "--individual --lift --tag _max --leader_num 4"
mv logs/DLinear_Exchange_M.log logs/LIFT_Exchange_M.log 2>/dev/null || true

# 3. LACFNet
run_experiment "LACFNet" "Exchange" "M" "Standard LACFNet" "--learning_rate 0.001"

# 4. DLinear
run_experiment "DLinear" "Exchange" "M" "Standard DLinear" "--individual"


# --- Sales (Custom Sales Data) ---
# 1. Prefetch
run_prefetch "Sales"

# 2. LIFT
run_experiment "DLinear" "Sales" "M" "LIFT Model" "--individual --lift --tag _max --leader_num 4"
mv logs/DLinear_Sales_M.log logs/LIFT_Sales_M.log 2>/dev/null || true

# 3. LACFNet
run_experiment "LACFNet" "Sales" "M" "Standard LACFNet" "--learning_rate 0.001"

# 4. DLinear
run_experiment "DLinear" "Sales" "M" "Standard DLinear" "--individual"

echo "All remaining experiments completed!"

# After running, analyze results
python3 analyze_results.py
