# Generate the pre-computed three-parameter SEIT4L PMMH chain loaded by
# sessions/abc.qmd. It estimates R_0, D_inf and ρ with D_lat, α and D_imm fixed
# at the values the ABC session fixes, so it has the same target as the ABC
# posteriors there, and any difference between them is what ABC approximates.
# Run with:
#   julia --project=. --threads=4 --heap-size-hint=4G scripts/generate_pmcmc_seit4l_abc.jl
#
# Takes about three quarters of an hour on four threads at 256 particles. Four
# independent chains are run so that R-hat can be computed across them. The heap size hint
# keeps the four filters' garbage in check on a laptop, and each chain is written
# to the temporary directory as soon as it finishes, so a run stopped part way
# keeps the chains that completed.

include(joinpath(@__DIR__, "pmmh_setup.jl"))

Random.seed!(1234)

name = "SEIT4L, three parameters"

tasks = [
    Threads.@spawn begin
        chain = run_pmmh(
            pmmh_seit4l_abc(flu_observations(), N_PARTICLES),
            "$name, chain $c";
            n_warmup = 5_000,
            n_samples = 20_000,
            thinning = 5,
        )
        ## `run_pmmh` already returns a frame
        CSV.write(joinpath(tempdir(), "pmcmc_seit4l_abc_chain_$c.csv"), chain)
        chain
    end for c in 1:N_CHAINS
]
frames = fetch.(tasks)

output_path = datadir("pmcmc_seit4l_abc_chain.csv")
println("Saving $N_CHAINS $name chains to $output_path")
save_chains_csv(frames, output_path)

print_chain_diagnostics(frames, name, ABC_PARAMETERS)
