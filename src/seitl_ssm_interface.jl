using Random
using Distributions
using SSMProblems

"""
SEITL latent dynamics, for any number of temporary immunity stages.

State vector: `[S, E, I, T_1 ... T_k, L, daily_inc]`.

The last element carries daily incidence for the observation process. SEITL is
the `k = 1` case and SEIT4L the `k = 4` case; the number of stages comes from
the length of the state the filter is given, so one type serves both and
nothing here needs to be told which model it is running.
"""
struct SEITLDynamics <: SSMProblems.LatentDynamics
    θ::Dict{Symbol, Float64}
end

function SSMProblems.simulate(
    rng::AbstractRNG,
    dyn::SEITLDynamics,
    step::Integer,
    prev_state;
    kwargs...,
)
    ## Drop the incidence slot, leaving the compartments the simulator works on
    state = collect(prev_state[1:(end - 1)])
    ## Built here rather than held, so the problem carries the generator the
    ## filter is drawing from and the population size the state actually has
    problem = seitl_jump_problem(dyn.θ, state; rng = rng)
    return vcat(state, seitl_jump_step!(problem, state))  ## re-append incidence
end

"""
Poisson observation process.

Observes daily incidence, the last element of the state, with reporting rate ρ.
"""
struct PoissonObservation <: SSMProblems.ObservationProcess
    ρ::Float64
end

function SSMProblems.distribution(obs::PoissonObservation, step::Integer, state; kwargs...)
    daily_inc = state[end]
    return Poisson(max(obs.ρ * daily_inc, 1e-10))
end

"""
Initial state distribution, which is deterministic here: the island as the ship
lands.

The length of `init_state` is what selects the model. Five compartments
`[S, E, I, T, L]` give SEITL, eight `[S, E, I, T1, T2, T3, T4, L]` give SEIT4L,
and any other number of temporary immunity stages works the same way.
"""
struct SEITLInitial <: SSMProblems.StatePrior
    init_state::Vector{Float64}
end

function SSMProblems.simulate(rng::AbstractRNG, prior::SEITLInitial; kwargs...)
    return vcat(prior.init_state, 0.0)  ## append 0 for initial daily incidence
end
