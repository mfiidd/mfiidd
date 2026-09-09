# Does threading the particle propagation inside one filter run pay?
#
# The bootstrap filter spends nearly all of its time propagating particles, and
# the particles within a day are independent, so the propagation is the obvious
# thing to thread. Resampling is a synchronisation point every day, and the work
# per particle-day is one Gillespie day on eight compartments, so whether the
# threading pays is a question about this model at this population rather than a
# question about particle filters.
#
# Run with: julia --project=. --threads=auto scripts/benchmark_threaded_filter.jl

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using Distributions
using DataFrames
using CSV
using DrWatson
using Statistics
using Printf
using MFIIDD

const INIT_STATE = [279.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0]

"""
    run_particle_filter_threaded(θ, obs, n_particles; init_state, nchunks, seed)

Bootstrap filter for SEIT4L with the propagation and weighting of each day
split across threads, and systematic resampling done serially.

This is a transcription of the filter `GeneralisedFilters.BF` runs, not a second
model: it calls the same `gillespie_step` and the same Poisson observation
density. It exists to measure what threading buys, and it is checked against
`run_particle_filter` before it is timed.

Each chunk draws from its own `Xoshiro`, seeded from `seed` and the chunk index,
so a run is reproducible for a fixed `nchunks` — which is what threading with
`Random.default_rng()` would cost you.
"""
function run_particle_filter_threaded(
    θ,
    obs,
    n_particles;
    init_state = INIT_STATE,
    nchunks = Threads.nthreads(),
    seed = rand(UInt32),
)
    ρ = θ[:ρ]
    particles = [copy(init_state) for _ in 1:n_particles]
    buffer = [copy(init_state) for _ in 1:n_particles]
    log_w = Vector{Float64}(undef, n_particles)
    w = Vector{Float64}(undef, n_particles)
    log_lik = 0.0

    ## one generator per chunk, so the run does not depend on thread scheduling
    chunks = collect(Iterators.partition(1:n_particles, cld(n_particles, nchunks)))
    rngs = [Random.Xoshiro(seed + UInt64(k)) for k in 1:length(chunks)]
    resample_rng = Random.Xoshiro(seed + 0xF00D)

    for y in obs
        Threads.@threads for k in 1:length(chunks)
            rng = rngs[k]
            for i in chunks[k]
                new_state, inc = gillespie_step(rng, particles[i], θ)
                particles[i] = new_state
                log_w[i] = logpdf(Poisson(max(ρ * inc, 1e-10)), y)
            end
        end

        mx = maximum(log_w)
        isfinite(mx) || return -Inf
        total = 0.0
        @inbounds for i in 1:n_particles
            w[i] = exp(log_w[i] - mx)
            total += w[i]
        end
        log_lik += log(total) + mx - log(n_particles)

        ## systematic resampling, as GeneralisedFilters does by default
        u = rand(resample_rng) / n_particles
        cumulative = w[1] / total
        j = 1
        @inbounds for i in 1:n_particles
            target = u + (i - 1) / n_particles
            while target > cumulative && j < n_particles
                j += 1
                cumulative += w[j] / total
            end
            buffer[i] = copy(particles[j])
        end
        particles, buffer = buffer, particles
    end

    return log_lik
end

θ_cal =
    Dict(:R_0 => 6.0, :D_lat => 1.3, :D_inf => 2.0, :α => 0.5, :D_imm => 10.5, :ρ => 0.7)
obs = CSV.read(datadir("flu_tdc_1971.csv"), DataFrame).obs

println("Julia threads available: ", Threads.nthreads())
println()

## --- does it give the same answer? -------------------------------------------
Random.seed!(1234)
n_check = 60
serial = [run_particle_filter(θ_cal, obs, 128) for _ in 1:n_check]
threaded =
    [run_particle_filter_threaded(θ_cal, obs, 128; seed = UInt64(i)) for i in 1:n_check]
@printf("agreement at 128 particles over %d runs\n", n_check)
@printf("  GeneralisedFilters BF : mean %8.2f  sd %5.2f\n", mean(serial), std(serial))
@printf("  threaded transcription: mean %8.2f  sd %5.2f\n", mean(threaded), std(threaded))
@printf(
    "  difference in means   : %6.2f  (standard error %.2f)\n",
    mean(threaded) - mean(serial),
    sqrt(var(serial) / n_check + var(threaded) / n_check)
)
println()

## --- one filter run, serial against threaded ---------------------------------
println("one filter run over 59 days, best of 5")
@printf("%10s %12s %12s %9s\n", "particles", "BF (s)", "threaded (s)", "speed-up")
for n in (64, 256, 1024, 4096)
    ts = minimum(@elapsed(run_particle_filter(θ_cal, obs, n)) for _ in 1:5)
    tt = minimum(
        @elapsed(run_particle_filter_threaded(θ_cal, obs, n; seed = UInt64(7))) for
        _ in 1:5
    )
    @printf("%10d %12.4f %12.4f %8.2fx\n", n, ts, tt, ts / tt)
end
println()

## --- the same, inside a short PMMH chain -------------------------------------
include(joinpath(@__DIR__, "pmmh_setup.jl"))

n_iter = parse(Int, get(ENV, "MFIIDD_BENCH_ITERS", "300"))
println("PMMH, $n_iter iterations at 256 particles (no warmup, RAM adapting)")
threaded_filter(θ, obs, n) = run_particle_filter_threaded(θ, obs, n; seed = rand(UInt32))

for (label, filt) in (("BF, serial", run_particle_filter), ("threaded", threaded_filter))
    Random.seed!(1234)
    model = pmmh(obs, 256, filt)
    t = @elapsed chain = sample(
        model,
        externalsampler(AdvancedMH.RobustAdaptiveMetropolis()),
        n_iter;
        num_warmup = 0,
        chain_type = MCMCChains.Chains,
        progress = false,
        check_model = false,
    )
    accepted = length(unique(chain[:R_0])) / n_iter
    @printf(
        "  %-12s %7.1f s  (%.3f s/iteration, %.0f%% of proposals accepted)\n",
        label,
        t,
        t / n_iter,
        100 * accepted
    )
end
