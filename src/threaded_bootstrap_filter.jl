using Base.Threads: nthreads, @threads
using GeneralisedFilters:
    GeneralisedFilters,
    AbstractParticleFilter,
    ParticleDistribution,
    initialise_particle,
    log_weights,
    num_particles,
    predict_particle,
    resampler,
    update_particle
using Random: Xoshiro
using SSMProblems: LatentDynamics, ObservationProcess, StatePrior

"""
    ThreadedBF(pf; nchunks = Threads.nthreads())

A bootstrap filter that propagates its particles across threads.

The particles within a day do not interact, so propagating them is the one part
of a filter run that parallelises without changing the algorithm. Weighting,
resampling and the likelihood accumulation are left to `GeneralisedFilters`,
which owns them: this type overrides one method, `predict`, and forwards
everything else to the filter it wraps.

`nchunks` blocks of particles are propagated in parallel. Each block draws from
its own `Xoshiro`, seeded from the filter's own generator before the parallel
region starts, so a run is reproducible from a single `Random.seed!` **for a
fixed `nchunks`**. Change the number of chunks and the stream changes, in the
way that changing the number of particles does.

# Example

```julia
using GeneralisedFilters: BF
algo = ThreadedBF(BF(256))
_, log_lik = GeneralisedFilters.filter(rng, model, algo, obs)
```

Start Julia with `--threads=auto` for this to do anything.
"""
struct ThreadedBF{PF <: AbstractParticleFilter} <: AbstractParticleFilter
    pf::PF
    nchunks::Int
end

function ThreadedBF(pf::AbstractParticleFilter; nchunks::Integer = nthreads())
    nchunks ≥ 1 || throw(ArgumentError("nchunks must be at least 1"))
    return ThreadedBF(pf, Int(nchunks))
end

GeneralisedFilters.num_particles(algo::ThreadedBF) = num_particles(algo.pf)
GeneralisedFilters.resampler(algo::ThreadedBF) = resampler(algo.pf)

## the per-particle work is the wrapped filter's, unchanged
function GeneralisedFilters.initialise_particle(
    rng::AbstractRNG,
    prior::StatePrior,
    algo::ThreadedBF,
    ref_state;
    kwargs...,
)
    return initialise_particle(rng, prior, algo.pf, ref_state; kwargs...)
end

function GeneralisedFilters.predict_particle(
    rng::AbstractRNG,
    dyn::LatentDynamics,
    algo::ThreadedBF,
    iter::Integer,
    particle,
    observation,
    ref_state;
    kwargs...,
)
    return predict_particle(
        rng,
        dyn,
        algo.pf,
        iter,
        particle,
        observation,
        ref_state;
        kwargs...,
    )
end

function GeneralisedFilters.update_particle(
    obs::ObservationProcess,
    algo::ThreadedBF,
    iter::Integer,
    particle,
    observation;
    kwargs...,
)
    return update_particle(obs, algo.pf, iter, particle, observation; kwargs...)
end

## the one method that differs: propagate the particles in parallel blocks
function GeneralisedFilters.predict(
    rng::AbstractRNG,
    dyn::LatentDynamics,
    algo::ThreadedBF,
    iter::Integer,
    state,
    observation;
    ref_state::Union{Nothing, AbstractVector} = nothing,
    kwargs...,
)
    N = num_particles(algo)
    blocks = collect(Iterators.partition(1:N, cld(N, algo.nchunks)))

    ## seeds are drawn here, in order, so the run does not depend on the
    ## order the threads happen to finish in
    seeds = rand(rng, UInt64, length(blocks))

    particles = Vector{eltype(state.particles)}(undef, N)
    @threads for b in eachindex(blocks)
        block_rng = Xoshiro(seeds[b])
        for i in blocks[b]
            ref = !isnothing(ref_state) && i == 1 ? ref_state[iter] : nothing
            particles[i] = predict_particle(
                block_rng,
                dyn,
                algo.pf,
                iter,
                state.particles[i],
                observation,
                ref;
                kwargs...,
            )
        end
    end

    ## the same baseline GeneralisedFilters' own `predict` accumulates
    baseline = GeneralisedFilters.logsumexp(log_weights(state)) + state.ll_baseline
    return ParticleDistribution(particles, baseline)
end
