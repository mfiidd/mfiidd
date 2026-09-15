# Generate the pre-computed SEITL PMMH chains loaded by sessions/pmcmc.qmd.
# Run with: julia --project=. --threads=4 scripts/generate_pmcmc_seitl.jl
#
# Takes many hours at 256 particles, which is why the chains it produces are
# committed to data/ and the session loads those instead of running this.
#
# Give Julia at least as many threads as there are chains. With fewer, the
# chains time-share and the wall clock rises in proportion.

include(joinpath(@__DIR__, "pmmh_setup.jl"))

Random.seed!(5678)

frames = run_pmmh_chains(() -> pmmh_seitl(flu_observations(), N_PARTICLES), "SEITL")

output_path = datadir("pmcmc_seitl_chain.csv")
println("Saving $(length(frames)) SEITL chains to $output_path")
save_chains_csv(frames, output_path)

print_chain_diagnostics(frames, "SEITL")

println("\n" * "="^60)
println("Done!")
println("="^60)
