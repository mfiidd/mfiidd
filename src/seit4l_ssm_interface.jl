using Random
using Distributions
using SSMProblems

"""
SEIT4L latent dynamics.
State vector: [S, E, I, T1, T2, T3, T4, L, daily_inc]
The last element tracks daily incidence for the observation process.
"""
struct SEIT4LDynamics <: SSMProblems.LatentDynamics
    θ::Dict{Symbol, Float64}
end

function SSMProblems.simulate(
    rng::AbstractRNG,
    dyn::SEIT4LDynamics,
    step::Integer,
    prev_state;
    kwargs...,
)
    ## One 9-element vector per particle per day, and no more: the compartments
    ## are copied into it, the stepper advances them in place, and the incidence
    ## goes in the ninth slot. Building it with `collect`, `copy` and `vcat`
    ## instead costs three vectors on every one of those calls.
    state = Vector{Float64}(undef, 9)
    @inbounds for i in 1:8
        state[i] = prev_state[i]
    end
    @inbounds state[9] = gillespie_step!(rng, state, dyn.θ)
    return state
end

"""
Poisson observation process.
Observes daily incidence (last element of state) with reporting rate ρ.
"""
struct PoissonObservation <: SSMProblems.ObservationProcess
    ρ::Float64
end

function SSMProblems.distribution(obs::PoissonObservation, step::Integer, state; kwargs...)
    daily_inc = state[end]
    Poisson(max(obs.ρ * daily_inc, 1e-10))
end

"""
Initial state distribution (deterministic).
"""
struct SEIT4LInitial <: SSMProblems.StatePrior
    init_state::Vector{Float64}
end

function SSMProblems.simulate(rng::AbstractRNG, prior::SEIT4LInitial; kwargs...)
    vcat(prior.init_state, 0.0)  # Append 0 for initial daily incidence
end
