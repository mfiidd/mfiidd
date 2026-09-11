# Generate the pre-computed three-parameter SEIT4L PMMH chain loaded by
# sessions/abc.qmd. It estimates R_0, D_inf and ρ with D_lat, α and D_imm fixed
# at the values the ABC session fixes, so it has the same target as the ABC
# posteriors there, and any difference between them is what ABC approximates.
# Run with:
#   julia --project=. --threads=4 --heap-size-hint=4G scripts/generate_pmcmc_seit4l_abc.jl
#
# Takes about an hour and a half on four threads of an Intel Mac mini at 256
# particles. Four independent
# chains are run so that R-hat can be computed across them. The heap size hint
# keeps the four filters' garbage in check on a laptop, and each chain is written
# to the temporary directory as soon as it finishes, so a run stopped part way
# keeps the chains that completed.

include(joinpath(@__DIR__, "pmmh_setup.jl"))

Random.seed!(1234)

const N_CHAINS = 4
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
        frame = chain_frame(chain)
        CSV.write(joinpath(tempdir(), "pmcmc_seit4l_abc_chain_$c.csv"), frame)
        frame
    end for c in 1:N_CHAINS
]
frames = fetch.(tasks)

# Keep the chain index, which chain_frame drops, so R-hat can be recomputed
# from the saved file
output = vcat([insertcols(f, 1, :chain => c) for (c, f) in enumerate(frames)]...)
output_path = datadir("pmcmc_seit4l_abc_chain.csv")
println("Saving $name chain to $output_path")
CSV.write(output_path, output)

combined = symchain(frames, ABC_PARAMETERS)
println("\n$name summary statistics, across $N_CHAINS chains:")
show(stdout, MIME("text/plain"), summarystats(combined))
println()
