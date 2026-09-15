using Base.Threads: nthreads
using GeneralisedFilters: GeneralisedFilters, BF
using LogExpFunctions: logsumexp
using Random: AbstractRNG, default_rng
using SSMProblems: SSMProblems

"""
Latent dynamics split into independent blocks, one per task.

A `JumpProblem` carries mutable aggregation state, so two tasks cannot drive one
between them. Threading the propagation therefore needs one set of dynamics per
block rather than one set shared across blocks. Each block holds its own problem,
and each draws from the random stream of whichever task runs it, so blocks share
nothing.

The serial path uses the first block, so one model serves `BF` and `ThreadedBF`
alike.
"""
struct BlockedDynamics{D <: SSMProblems.LatentDynamics} <: SSMProblems.LatentDynamics
    blocks::Vector{D}
end

function SSMProblems.simulate(
    rng::AbstractRNG,
    dyn::BlockedDynamics,
    step::Integer,
    prev_state;
    kwargs...,
)
    return SSMProblems.simulate(rng, first(dyn.blocks), step, prev_state; kwargs...)
end

"""
    ThreadedBF(n_particles; n_blocks = Threads.nthreads(), kwargs...)

A bootstrap filter that propagates its particles in `n_blocks` parallel tasks.

Only the propagation is threaded. Weighting, resampling and the log-likelihood
accumulation are the serial filter's own, and each block calls the same
`predict_particle` the serial filter calls, so this returns the estimate `BF`
returns, drawn from a different random stream.

Propagation is 88% of the filter's time at 256 particles, which is why it is the
part worth threading. The speed-up falls well short of the block count, because
the rest is serial and because propagation allocates, which presses on a garbage
collector that every thread shares.

The model must carry `BlockedDynamics` with at least `n_blocks` blocks.
`run_particle_filter` builds that for you.
"""
struct ThreadedBF{RS} <: GeneralisedFilters.AbstractParticleFilter
    filter::GeneralisedFilters.BootstrapFilter{RS}
    n_blocks::Int
end

function ThreadedBF(n_particles::Integer; n_blocks::Integer = nthreads(), kwargs...)
    n_blocks >= 1 || throw(ArgumentError("n_blocks must be at least 1, got $n_blocks"))
    return ThreadedBF(BF(n_particles; kwargs...), n_blocks)
end

## Everything but the propagation loop is the serial filter's, forwarded here so
## there is one implementation of it rather than two that can drift apart
GeneralisedFilters.num_particles(algo::ThreadedBF) =
    GeneralisedFilters.num_particles(algo.filter)

GeneralisedFilters.resampler(algo::ThreadedBF) = GeneralisedFilters.resampler(algo.filter)

GeneralisedFilters.initialise_particle(
    rng::AbstractRNG,
    prior,
    algo::ThreadedBF,
    ref_state;
    kwargs...,
) = GeneralisedFilters.initialise_particle(rng, prior, algo.filter, ref_state; kwargs...)

GeneralisedFilters.update_particle(
    obs,
    algo::ThreadedBF,
    iter::Integer,
    particle,
    observation;
    kwargs...,
) = GeneralisedFilters.update_particle(
    obs,
    algo.filter,
    iter,
    particle,
    observation;
    kwargs...,
)

GeneralisedFilters.predict_particle(
    rng::AbstractRNG,
    dyn,
    algo::ThreadedBF,
    iter::Integer,
    particle,
    observation,
    ref_state;
    kwargs...,
) = GeneralisedFilters.predict_particle(
    rng,
    dyn,
    algo.filter,
    iter,
    particle,
    observation,
    ref_state;
    kwargs...,
)

"""
    particle_blocks(n_particles, n_blocks)

Partition `1:n_particles` into contiguous ranges of near-equal length.

A static partition is enough here. Per-particle cost varies about sevenfold,
because a particle whose outbreak has died does almost no work, but a block
holds many particles and that averages out: at 256 particles in 12 blocks the
slowest block takes 4.0 ms against a mean of 3.7, which is 92% of ideal.
"""
function particle_blocks(n_particles::Integer, n_blocks::Integer)
    n_blocks = min(n_blocks, n_particles)
    return [
        (((b - 1) * n_particles) ÷ n_blocks + 1):((b * n_particles) ÷ n_blocks) for
        b in 1:n_blocks
    ]
end

function GeneralisedFilters.predict(
    rng::AbstractRNG,
    dyn::BlockedDynamics,
    algo::ThreadedBF,
    iter::Integer,
    state,
    observation;
    ref_state::Union{Nothing, AbstractVector} = nothing,
    kwargs...,
)
    ref_state === nothing || throw(
        ArgumentError(
            "ThreadedBF does not take a reference trajectory; conditional " *
            "filtering keeps particle 1 fixed, which a block does not know about",
        ),
    )

    n_particles = GeneralisedFilters.num_particles(algo)
    ranges = particle_blocks(n_particles, algo.n_blocks)
    length(dyn.blocks) >= length(ranges) || throw(
        ArgumentError(
            "the dynamics carry $(length(dyn.blocks)) blocks and the filter asks " *
            "for $(length(ranges)); build the model with the filter's block count",
        ),
    )

    particles = similar(state.particles)
    Base.Threads.@sync for (b, range) in enumerate(ranges)
        Base.Threads.@spawn begin
            block = dyn.blocks[b]
            for i in range
                ## default_rng() inside the task is that task's own stream, so
                ## the blocks neither share a generator nor race on one
                particles[i] = GeneralisedFilters.predict_particle(
                    default_rng(),
                    block,
                    algo.filter,
                    iter,
                    state.particles[i],
                    observation,
                    nothing;
                    kwargs...,
                )
            end
        end
    end

    ## The serial filter accumulates the same baseline after propagation and
    ## before the update; propagation leaves the weights untouched
    return GeneralisedFilters.ParticleDistribution(
        particles,
        logsumexp(GeneralisedFilters.log_weights(state)) + state.ll_baseline,
    )
end
