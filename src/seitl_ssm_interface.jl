using Random
using Distributions
using SSMProblems

"""
SEITL latent dynamics, for any number of temporary immunity stages.

State vector: [S, E, I, T_1 ... T_k, L, daily_inc]

The last element tracks daily incidence for the observation process. SEITL is
the k = 1 case and SEIT4L the k = 4 case, and `k` is a type parameter so the
number of stages is fixed when the dynamics are built.
"""
struct SEITLDynamics{K} <: SSMProblems.LatentDynamics
    θ::Dict{Symbol, Float64}
end

"""
    SEITLDynamics(θ, k)

Build dynamics with `k` temporary immunity sub-stages. Any `k` from 1 upwards
works, SEITL being `k = 1` and SEIT4L `k = 4`, because the transitions are
assembled from `k` rather than written out one model at a time.

`k` must agree with the compartment count of the initial state the filter is
given, which is `k + 4`. `run_particle_filter` derives one from the other so
they cannot disagree; construct the two by hand and a mismatch throws.
"""
SEITLDynamics(θ::Dict{Symbol, Float64}, k::Integer) = SEITLDynamics{Int(k)}(θ)

function SSMProblems.simulate(
    rng::AbstractRNG,
    dyn::SEITLDynamics{K},
    step::Integer,
    prev_state;
    kwargs...,
) where {K}
    ## Drop the incidence slot, leaving the compartments the simulator works on
    state = collect(prev_state[1:(end - 1)])
    length(state) == K + 4 || throw(
        DimensionMismatch("state has $(length(state)) compartments, expected $(K + 4)"),
    )
    ## Built here rather than held, so the problem carries the generator the
    ## filter is drawing from and the population size the state actually has
    problem = seitl_jump_problem(dyn.θ, state; rng = rng)
    return vcat(state, seitl_jump_step!(problem, state))  ## re-append incidence
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
