#!/bin/bash

SEEDS=(42)

DIFFICULTY="medium"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=================================================="
echo "Starting QUICK TEST Batch"
echo "Difficulty: $DIFFICULTY"
echo "Group Timestamp ID: $TIMESTAMP"
echo "=================================================="

for S in "${SEEDS[@]}"; do
    echo "----------------------------------"
    echo "Running PPO with seed = $S (Group: $TIMESTAMP)"

    uv run python train.py \
        --difficulty "$DIFFICULTY" \
        --seed "$S" \
        --total_steps 300000 \
        --eval_episodes 5 \
        --group_timestamp "$TIMESTAMP"

    echo "Finished seed $S"
done

echo "----------------------------------"
echo "Test run completed."
echo "Results are stored in checkpoints/ folder under group: $TIMESTAMP"