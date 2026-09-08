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
struct SEITLDynamics{K} <: SSMProblems.LatentDynamics
    θ::Dict{Symbol, Float64}
end

"""
    SEITLDynamics(θ, k)

Build dynamics with `k` temporary immunity sub-stages. `k` is a type parameter,
so [`seitl_stepper`](@ref) resolves to the right Gillespie stepper when the
method is compiled and no test of the state survives into the filter's inner
loop.

`k` must agree with the compartment count of the initial state the filter is
given, which is `k + 4`. `run_particle_filter` derives one from the other so
they cannot disagree; construct the two by hand and it is on you to match them.

Only `k = 1` and `k = 4` have steppers, so any other `k` is rejected here
rather than at the first step of the filter.
"""
function SEITLDynamics(θ::Dict{Symbol, Float64}, k::Integer)
    ## Reject an unsupported k where the mistake is, which is the length of the
    ## initial state, and not deep inside the filter
    k in (1, 4) || throw(
        ArgumentError(
            "SEITLDynamics supports k = 1 (SEITL) and k = 4 (SEIT4L); got k = $k. " *
            "k is the number of temporary immunity stages, so the initial state " *
            "must be [S, E, I, T_1 ... T_k, L] and have k + 4 compartments.",
        ),
    )
    return SEITLDynamics{Int(k)}(θ)
end

"""
    seitl_stepper(dyn)

The Gillespie stepper for the number of sub-stages `dyn` declares.
"""
seitl_stepper(::SEITLDynamics{1}) = gillespie_step_seitl!
seitl_stepper(::SEITLDynamics{4}) = gillespie_step_seit4l!

function SSMProblems.simulate(
    rng::AbstractRNG,
    dyn::SEITLDynamics,
    step::Integer,
    prev_state;
    kwargs...,
)
    ## Drop the incidence slot, leaving the compartments the stepper works on
    state = collect(prev_state[1:(end - 1)])
    return vcat(state, seitl_stepper(dyn)(rng, state, dyn.θ))  ## re-append incidence
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
