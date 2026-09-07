using Random
using Distributions
using SSMProblems

"""
SEITL latent dynamics.

SEITL is the compartment structure S → E → I → T → L: susceptible, exposed,
infectious, temporarily immune and long-term immune, with waning from T back to
S. `k` is the number of sub-stages the T compartment is split into, which sets
the shape of the immunity duration: exponential at `k = 1`, Erlang at higher
`k`. The sessions fit `k = 1` and `k = 4`, the latter written SEIT4L.

State vector: [S, E, I, T_1 ... T_k, L, daily_inc]

The last element tracks daily incidence for the observation process. `k` is
read from the state that `SEITLInitial` supplies rather than fixed by the
type.
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
    ## Drop the incidence slot, leaving the compartments the stepper works on
    state = collect(prev_state[1:(end - 1)])
    step! = length(state) == 5 ? gillespie_step_seitl! : gillespie_step_seit4l!
    return vcat(state, step!(rng, state, dyn.θ))  ## re-append daily incidence
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

The length of `init_state` sets `k`: five compartments [S, E, I, T, L] give
one temporary immunity stage, eight [S, E, I, T1, T2, T3, T4, L] give four.
"""
struct SEITLInitial <: SSMProblems.StatePrior
    init_state::Vector{Float64}
end

function SSMProblems.simulate(rng::AbstractRNG, prior::SEITLInitial; kwargs...)
    vcat(prior.init_state, 0.0)  # Append 0 for initial daily incidence
end
