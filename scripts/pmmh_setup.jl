# Shared setup for the PMMH chain generation scripts.
#
# The priors below must match those in sessions/pmcmc.qmd. The session presents
# the saved chains as the posteriors of the model it defines, so a prior changed
# in one place and not the other silently puts the wrong posterior on the page.
# The likelihood is shared rather than copied: both models call the package's
# GeneralisedFilters runners, which is what the session calls too.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using Random
using Distributions
using DataFrames
using Turing
using FlexiChains
using FlexiChains: Parameter
using CSV
using DrWatson
using StatsBase
using AdvancedMH
using ForwardDiff

using MFIIDD

# Sampler settings shared by every chain we generate
const N_PARTICLES = 256
const N_WARMUP = 50_000     # RAM adapts here, and these draws are discarded
const N_SAMPLES = 450_000   # kept, then thinned
const THINNING = 50         # → 9000 final samples

const PARAMETERS = [:R_0, :D_lat, :D_inf, :α, :D_imm, :ρ]
const N_CHAINS = 4         # run side by side, and give R-hat

"""
Serialises printing, because chains running side by side otherwise interleave
mid-line. Left alone, four chains printing their opening banner at once produced
a log in which counting the warm-up notices returned three.
"""
const PRINTING = ReentrantLock()

"""
    say(lines...)

Print `lines` as one uninterrupted block and flush.

`flush` matters as much as the lock: Julia block-buffers a redirected stdout, so
without it a batch job's log stays empty for the length of the run.
"""
function say(lines...)
    lock(PRINTING) do
        for line in lines
            println(line)
        end
        flush(stdout)
    end
end
const PROGRESS_EVERY = 10_000  # iterations between progress lines

"""
    pmmh(obs, n_particles, particle_filter)

PMMH model: weakly informative priors on the six parameters, with the
log-likelihood estimated by `particle_filter` and added through `@addlogprob!`.

`ForwardDiff.value` strips the Duals that Turing's gradient-based initialisation
probe pushes through the model. The particle filter is not differentiable, and
its resampling step errors on Dual-valued weights.
"""
@model function pmmh(obs, n_particles, particle_filter)
    R_0 ~ truncated(Normal(3.0, 2.0), lower = 1.0)
    D_lat ~ truncated(Normal(2.0, 1.0), lower = 0.5)
    D_inf ~ truncated(Normal(3.0, 2.0), lower = 0.5)
    α ~ Beta(2, 2)
    D_imm ~ truncated(Normal(15.0, 10.0), lower = 1.0)
    ρ ~ Beta(2, 2)

    θ = Dict(
        :R_0 => ForwardDiff.value(R_0),
        :D_lat => ForwardDiff.value(D_lat),
        :D_inf => ForwardDiff.value(D_inf),
        :α => ForwardDiff.value(α),
        :D_imm => ForwardDiff.value(D_imm),
        :ρ => ForwardDiff.value(ρ),
    )

    log_lik = particle_filter(θ, obs, n_particles)
    Turing.@addlogprob! log_lik
end

# Both models are estimated with the same bootstrap filter the sessions use, from
# GeneralisedFilters, so the saved chains are the posterior of the model the page
# defines rather than of a second implementation that happens to live in scripts.
## One filter serves both models. The initial state carries the compartment
## count, so both are named here and neither depends on a default.
const SEITL_INIT = [279.0, 0.0, 2.0, 3.0, 0.0]
const SEIT4L_INIT = [279.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0]

"""
    remembering(filter)

Wrap a particle filter so a repeated evaluation of the same parameters returns
the estimate already drawn for them.

Turing evaluates the model twice per iteration: once for the proposal, and again
for the current state when it records the draw. The second evaluation runs the
whole particle filter and never reaches the acceptance ratio, so it doubles the
cost of a chain for nothing. Serving it from store takes the SEITL chain from
71.8 to 40.2 ms an iteration.

Two entries, evicting the least recently used, because the calls alternate
proposal, current, proposal, current. A single entry is evicted by the proposal
just before the current state is asked for again, which recovers only a quarter
of the work instead of half.

Returning the stored estimate is what pseudo-marginal MCMC asks for in any case.
The acceptance ratio keeps the estimate drawn when a state was accepted, so
serving that same number again makes the recorded log-density agree with the one
the chain actually used, where a fresh call reports a second, unrelated draw.

`filter_for` builds one of these per model, so each chain has its own store and
concurrent chains share nothing. `pmmh_seit4l_abc` calls the filter directly and
is deliberately left alone, since its four chains run in spawned tasks.
"""
function remembering(filter)
    seen = Vector{NTuple{6, Float64}}()
    drawn = Vector{Float64}()
    return function (θ, obs, n)
        key = (θ[:R_0], θ[:D_lat], θ[:D_inf], θ[:α], θ[:D_imm], θ[:ρ])
        i = findfirst(==(key), seen)
        if i !== nothing  ## move the hit to the end, so it is not the next evicted
            value = drawn[i]
            deleteat!(seen, i)
            deleteat!(drawn, i)
            push!(seen, key)
            push!(drawn, value)
            return value
        end
        value = filter(θ, obs, n)
        push!(seen, key)
        push!(drawn, value)
        if length(seen) > 2
            popfirst!(seen)
            popfirst!(drawn)
        end
        return value
    end
end

filter_for(init) =
    remembering((θ, obs, n) -> run_particle_filter(θ, obs, n; init_state = init))

pmmh_seitl(obs, n_particles) = pmmh(obs, n_particles, filter_for(SEITL_INIT))
pmmh_seit4l(obs, n_particles) = pmmh(obs, n_particles, filter_for(SEIT4L_INIT))

# sessions/abc.qmd estimates three of the six parameters and fixes the rest at
# these values. The ABC posteriors there are only comparable with a likelihood
# posterior that fixes them too, so this chain has the same target as they do.
const ABC_FIXED = Dict(:D_lat => 2.0, :α => 0.5, :D_imm => 13.0)
const ABC_PARAMETERS = [:R_0, :D_inf, :ρ]

"""
    pmmh_seit4l_abc(obs, n_particles)

PMMH model for the SEIT4L parameters the ABC session estimates, with the same
priors on them as `pmmh` and the others fixed at `ABC_FIXED`.
"""
@model function pmmh_seit4l_abc(obs, n_particles)
    R_0 ~ truncated(Normal(3.0, 2.0), lower = 1.0)
    D_inf ~ truncated(Normal(3.0, 2.0), lower = 0.5)
    ρ ~ Beta(2, 2)

    θ = merge(
        ABC_FIXED,
        Dict(
            :R_0 => ForwardDiff.value(R_0),
            :D_inf => ForwardDiff.value(D_inf),
            :ρ => ForwardDiff.value(ρ),
        ),
    )

    Turing.@addlogprob! run_particle_filter(θ, obs, n_particles; init_state = SEIT4L_INIT)
end

"""
    flu_observations()

Daily incidence from the 1971 Tristan da Cunha outbreak.
"""
flu_observations() = CSV.read(datadir("flu_tdc_1971.csv"), DataFrame).obs

"""
    chain_frame(chain)

Return `chain` as a data frame with one row per iteration and one column per
parameter, dropping the iteration and chain indices. Going through
`DataFrame(chain)` keeps this working across chain types rather than reaching
into internal fields whose layout is not part of the API.
"""
function chain_frame(chain)
    df = DataFrame(chain)
    return select(df, Not(intersect(["iteration", "iter", "chain"], names(df))))
end

"""
    save_chain_csv(chain, path)

Write the chain to `path`, one row per retained iteration.
"""
save_chain_csv(chain, path) = CSV.write(path, chain_frame(chain))

"""
    symchain(frames, keys)

A `SymChain` from one data frame per chain, each with a column for every key in
`keys`. The frames are stacked into the iterations × chains × parameters array that
the FlexiChains array constructor takes.
"""
symchain(frames, keys) =
    SymChain(stack([Matrix(f[:, keys]) for f in frames]; dims = 2), Tuple(Parameter.(keys)))

"""
    acceptance_rate(chain)

Proportion of iterations at which the sampler moved. A Metropolis chain repeats
the previous parameter vector whenever a proposal is rejected, so the iterations
where the parameters changed are the accepted ones.
"""
function acceptance_rate(chain)
    x = chain_frame(chain)[!, first(PARAMETERS)]
    return mean(x[2:end] .!= x[1:(end - 1)])
end

"""
    progress_every(n, name, t_start, total)

A callback that prints one line every `n` kept iterations.

`sample` can show a progress meter, but it writes to stderr and repaints in
place, so a batch job redirecting its output to a file records nothing useful.
`flush` matters as much as the `println`: Julia block-buffers a redirected
stdout, so without it the lines sit unwritten for the length of the run.

The callback is not called during warmup, so nothing appears until the kept
iterations begin. `run_pmmh` says so before it starts.
"""
function progress_every(n, name, t_start, total)
    kept_start = Ref(0.0)
    return function (rng, model, sampler, sample, state, i; kwargs...)
        ## The callback first runs once warm-up is over, so this is when the
        ## kept iterations began. Timing them against `t_start` instead divides
        ## kept iterations by a span that includes warm-up, and the estimate of
        ## the time left then comes out far too pessimistic.
        kept_start[] == 0.0 && (kept_start[] = time())
        if i % n == 0
            elapsed = (time() - t_start) / 60
            rate = i / max((time() - kept_start[]) / 60, eps())
            left = (total - i) / rate
            say(
                "[$name] $i/$total kept, $(round(elapsed, digits = 1)) min elapsed, " *
                "$(round(rate, digits = 0))/min, about $(round(left, digits = 0)) min left",
            )
        end
        return nothing
    end
end

"""
    run_pmmh(model, name; n_warmup, n_samples, thinning)

Sample `model` with Robust Adaptive Metropolis, reporting the acceptance rate of
the kept draws, then thin. Returns the thinned draws as a data frame, one row per
retained iteration and one column per parameter.

`num_warmup` is what makes RAM adaptive: the sampler only updates its proposal
covariance in the warmup phase, and those draws are discarded rather than kept.
Without it the proposal stays at the identity matrix and acceptance sits near
0.5% instead of the 23% RAM aims for.
"""
function run_pmmh(
    model,
    name;
    n_warmup = N_WARMUP,
    n_samples = N_SAMPLES,
    thinning = THINNING,
)
    say(
        "="^60,
        "Running PMMH for $name with RAM",
        "  Particles: $N_PARTICLES",
        "  Warmup (adaptation, discarded): $n_warmup",
        "  Samples kept: $n_samples",
        "  Thinning: $thinning",
        "  Final samples: $(n_samples ÷ thinning)",
        "="^60,
        "Warming up. No progress lines until the $n_warmup warmup iterations finish.",
    )

    t_start = time()
    chain_full = sample(
        model,
        externalsampler(AdvancedMH.RobustAdaptiveMetropolis()),
        n_samples;
        num_warmup = n_warmup,
        check_model = false,
        progress = true,
        callback = progress_every(PROGRESS_EVERY, name, t_start, n_samples),
    )
    t_elapsed = time() - t_start
    say(
        "$name sampling took $(round(t_elapsed/60, digits=1)) minutes",
        "$name acceptance rate: $(round(acceptance_rate(chain_full) * 100, digits=1))%",
    )

    # Thin the data frame: FlexiChains, which Turing now returns by default, does
    # not support `end` inside an index, and a data frame thins the same way
    # whatever chain type the sampler produced
    chain = chain_frame(chain_full)[1:thinning:end, :]
    say("$name after thinning: $(nrow(chain)) samples")
    return chain
end

"""
    run_pmmh_chains(build_model, name; n_chains, n_warmup, n_samples, thinning)

Run `n_chains` chains side by side and return one data frame per chain.

Chains are the axis worth parallelising. They scale almost linearly, where
threading inside one filter saturates: only propagation parallelises, and at 256
particles the blocks are small enough that six threads return about 1.3 times
the serial speed. Four chains also give R-hat, which one long chain cannot.

`n_samples` defaults to a quarter of the single-chain total, so four chains keep
the same 9000 draws after thinning that the session has always loaded.

`build_model` is called once per chain rather than once and shared, so each
chain gets its own model and its own likelihood store and the chains share
nothing mutable.
"""
function run_pmmh_chains(
    build_model,
    name;
    n_chains = N_CHAINS,
    n_warmup = N_WARMUP,
    n_samples = N_SAMPLES ÷ n_chains,
    thinning = THINNING,
)
    println(
        "Running $n_chains chains of $name side by side on $(Threads.nthreads()) threads",
    )
    flush(stdout)
    tasks = [
        Threads.@spawn run_pmmh(
            build_model(),
            "$name, chain $c";
            n_warmup = n_warmup,
            n_samples = n_samples,
            thinning = thinning,
        ) for c in 1:n_chains
    ]
    return fetch.(tasks)
end

"""
    save_chains_csv(frames, path)

Write every chain to `path`, one row per retained iteration, with a leading
`chain` column so R-hat can be recomputed from the saved file.
"""
function save_chains_csv(frames, path)
    output = vcat([insertcols(f, 1, :chain => c) for (c, f) in enumerate(frames)]...)
    return CSV.write(path, output)
end

"""
    print_chain_diagnostics(frames, name, keys = PARAMETERS)

Print the across-chain summary, which carries R-hat and the effective sample
size, then the pooled quantiles.
"""
function print_chain_diagnostics(frames, name, keys = PARAMETERS)
    println("\n$name summary statistics, across $(length(frames)) chains:")
    show(stdout, MIME("text/plain"), summarystats(symchain(frames, keys)))

    pooled = vcat(frames...)
    println("\n\n$name 2.5%, 50% and 97.5% quantiles, pooled:")
    for k in keys
        println("  $k: ", round.(quantile(pooled[!, k], [0.025, 0.5, 0.975]); digits = 3))
    end
    println()
    return nothing
end

"""
    print_diagnostics(chain, name)

Print the posterior summary and credible intervals, from the same six columns
the session reads back out of the saved CSV.
"""
function print_diagnostics(chain, name)
    df = chain_frame(chain)
    mcmc_chain = symchain([df], PARAMETERS)

    println("\n$name summary statistics:")
    show(stdout, MIME("text/plain"), summarystats(mcmc_chain))
    println("\n\n$name 2.5%, 50% and 97.5% quantiles:")
    for k in PARAMETERS
        println("  $k: ", round.(quantile(df[!, k], [0.025, 0.5, 0.975]); digits = 3))
    end
    println()
end
