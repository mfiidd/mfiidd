#!/bin/bash
#SBATCH --job-name=pmcmc_chains
#SBATCH --array=0-1
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=pmcmc_%A_%a.out
#SBATCH --error=pmcmc_%A_%a.err

# Generate both sets of PMMH chains, one model per array task.
#
# The two models ran back to back in a single job until a run overran the wall
# clock. An array gives each model its own 48 hours, and the two run side by
# side.
#
# Four CPUs for four chains, which is the axis that parallelises almost
# linearly. Threading inside a single filter saturates instead, because only
# propagation parallelises and at 256 particles the blocks are small: six
# threads return about 1.3 times the serial speed. With more CPUs to spare,
# `run_particle_filter(...; threaded = true)` can take the extra.
#
# Julia is installed via juliaup on the cluster, so it is already on PATH.

set -euo pipefail

MODELS=(seitl seit4l)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

cd "$SLURM_SUBMIT_DIR"

echo "Starting $MODEL PMMH chain generation at $(date)"
echo "Running on node: $(hostname)"
echo "Julia version: $(julia --version)"

julia --project=. --threads="$SLURM_CPUS_PER_TASK" "scripts/generate_pmcmc_${MODEL}.jl"

echo "Finished $MODEL at $(date)"
