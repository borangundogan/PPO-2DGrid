#!/bin/bash

SEEDS=(777)

DIFFICULTY="mediumhard"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "=================================================="
echo "Starting FOMAML Meta-Training Batch"
echo "Difficulty : $DIFFICULTY"
echo "Batch Start: $TIMESTAMP"
echo "=================================================="

for S in "${SEEDS[@]}"; do
    echo "----------------------------------"
    echo "Running FOMAML with seed = $S"

    uv run python fomaml/fomaml_train.py \
        --difficulty "$DIFFICULTY" \
        --seed "$S" \
        --iterations 1000 \
        --tasks_per_batch 16 \
        --k_steps 256

    echo "Finished FOMAML seed $S"
done

echo "----------------------------------"
echo "Meta-Training run completed."
echo "Results are stored in the checkpoints/ folder."