#!/bin/bash
#SBATCH --job-name=pmcmc_abc
#SBATCH --time=04:00:00   # four chains in parallel, around half an hour
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=pmcmc_abc_%j.out
#SBATCH --error=pmcmc_abc_%j.err

# The three-parameter SEIT4L chain for sessions/abc.qmd, one chain per CPU.
# Submit from the repository root: sbatch scripts/run_pmcmc_seit4l_abc.slurm

# Julia is installed via juliaup on the cluster, so it is already on PATH
cd $SLURM_SUBMIT_DIR

echo "Starting the three-parameter PMMH chains at $(date) on $(hostname)"
julia --project=. --threads=$SLURM_CPUS_PER_TASK scripts/generate_pmcmc_seit4l_abc.jl
echo "Finished at $(date)"
