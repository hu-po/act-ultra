#!/bin/bash

# This script runs multiple Weights & Biases sweep agents in parallel across different GPUs.
# Usage: ./sweep-multi.sh <sweep_id>
# Example: ./sweep-multi.sh abc123

# Check if sweep ID is provided
if [ $# -eq 0 ]; then
    echo "Error: Sweep ID is required"
    echo "Usage: ./sweep-multi.sh <sweep_id>"
    exit 1
fi

SWEEP_ID=$1

# Function to run a sweep agent on a specific GPU
run_agent() {
    local gpu_id=$1
    CUDA_VISIBLE_DEVICES=$gpu_id poetry run wandb agent $SWEEP_ID
}

# Launch 8 agents in parallel, each on a different GPU
for i in {0..7}; do
    run_agent $i &
done

# Wait for all background processes to complete
wait 