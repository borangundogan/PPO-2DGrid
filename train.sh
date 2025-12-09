#!/bin/bash

SEEDS=("0" "1" "2" "3")
DIFFICULTY="easy"

for S in "${SEEDS[@]}"; do
    echo "Running PPO with seed = $S"

    uv run python train.py \
        --difficulty "$DIFFICULTY" \
        --seed "$S"

    echo "Finished seed $S"
    echo "----------------------------------"
done

echo "All trainings completed."
