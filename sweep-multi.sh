#!/bin/bash

# Create the sweep
SWEEP_ID=$(poetry run wandb sweep sweep-diff.yaml | grep -o "wandb: Created sweep with ID: .*" | cut -d' ' -f6)

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