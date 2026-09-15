#!/bin/bash
#SBATCH --job-name=pmcmc_chains
#SBATCH --array=0-1
#SBATCH --time=48:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --output=pmcmc_%A_%a.out
#SBATCH --error=pmcmc_%A_%a.err

# Generate both PMMH chains, one model per array task.
#
# The two models ran back to back in a single job until a run overran the wall
# clock. At the rate a busy node sustains they need about 15 and 24 hours, which
# fits 48 hours together only while the node stays quiet. It does not: one
# earlier run saw the per-iteration cost double as the number of jobs sharing
# the node went from 6 to 35, because the filter is bound by memory bandwidth.
# An array gives each model its own 48 hours, and the two run side by side.
#
# One CPU is deliberate. GeneralisedFilters propagates particles serially, so
# more cores would sit idle. Threading needs a filter that splits the particles
# and gives each task its own JumpProblem.
#
# Julia is installed via juliaup on the cluster, so it is already on PATH.

set -euo pipefail

MODELS=(seitl seit4l)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

cd "$SLURM_SUBMIT_DIR"

echo "Starting $MODEL PMMH chain generation at $(date)"
echo "Running on node: $(hostname)"
echo "Julia version: $(julia --version)"

julia --project=. "scripts/generate_pmcmc_${MODEL}.jl"

echo "Finished $MODEL at $(date)"
