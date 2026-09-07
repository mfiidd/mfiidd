using Random
using Distributions
using SSMProblems

"""
SEITL latent dynamics, for any number of temporary immunity stages.

State vector: [S, E, I, T_1 ... T_k, L, daily_inc]

The last element tracks daily incidence for the observation process. SEITL is
the k = 1 case and SEIT4L the k = 4 case, so the number of stages is read from
the state that `SEITLInitial` supplies rather than fixed by the type.
"""
struct SEITLDynamics{S} <: SSMProblems.LatentDynamics
    θ::Dict{Symbol, Float64}
    step!::S
end

"""
    SEITLDynamics(θ, init_state)

Pick the Gillespie stepper matching `init_state` and hold it, so the choice is
made once when the dynamics are built rather than on every particle at every
step. `S` is the stepper's own type, which keeps the field concrete.
"""
function SEITLDynamics(θ::Dict{Symbol, Float64}, init_state::AbstractVector{<:Real})
    step! = length(init_state) == 5 ? gillespie_step_seitl! : gillespie_step_seit4l!
    return SEITLDynamics{typeof(step!)}(θ, step!)
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
    return vcat(state, dyn.step!(rng, state, dyn.θ))  ## re-append daily incidence
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

The length of `init_state` is what selects the model: five compartments
[S, E, I, T, L] give SEITL, eight [S, E, I, T1, T2, T3, T4, L] give SEIT4L.
"""
struct SEITLInitial <: SSMProblems.StatePrior
    init_state::Vector{Float64}
end

function SSMProblems.simulate(rng::AbstractRNG, prior::SEITLInitial; kwargs...)
    vcat(prior.init_state, 0.0)  # Append 0 for initial daily incidence
end
