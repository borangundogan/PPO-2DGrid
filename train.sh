#!/bin/bash

SEEDS=("123", "124", "125")
DIFFICULTY="easy"

for S in "${SEEDS[@]}"; do
    echo "Running PPO with seed = $S"

    uv run python train.py \
        --difficulty "$DIFFICULTY" \
        --seed "$S" \
        --total_steps 100000 \
        --eval_episodes 2

    echo "Finished seed $S"
    echo "----------------------------------"
done

echo "All trainings completed."
