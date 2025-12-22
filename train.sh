#!/bin/bash

SEEDS=(7 9)
DIFFICULTY="medium"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=================================================="
echo "Starting Experiment Batch"
echo "Difficulty: $DIFFICULTY"
echo "Group Timestamp ID: $TIMESTAMP"
echo "=================================================="

for S in "${SEEDS[@]}"; do
    echo "----------------------------------"
    echo "Running PPO with seed = $S (Group: $TIMESTAMP)"

    uv run python train.py \
        --difficulty "$DIFFICULTY" \
        --seed "$S" \
        --total_steps 1000000 \
        --eval_episodes 3 \
        --group_timestamp "$TIMESTAMP"

    echo "Finished seed $S"
done

echo "----------------------------------"
echo "All trainings completed."
echo "Results are stored in checkpoints/ folder under group: $TIMESTAMP"