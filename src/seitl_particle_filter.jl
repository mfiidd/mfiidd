using Random
using Distributions
using StatsBase

"""
    gillespie_step_seitl!(rng, state, θ, dt)
    gillespie_step_seitl!(state, θ, dt)

Simulate SEITL for `dt` time units and return the incidence over the interval.

The two-argument form draws from the global random number generator; pass an
`rng` explicitly when the caller needs to control randomness, as the particle
filter does.

# Arguments
- `rng`: Random number generator (defaults to the global one)
- `state`: Vector [S, E, I, T, L] (modified in place)
- `θ`: Parameter dictionary with keys :R_0, :D_lat, :D_inf, :α, :D_imm
- `dt`: Time step (typically 1.0 for daily)

# Returns
- `incidence`: Number of new cases (E→I transitions)
"""
function gillespie_step_seitl!(
    rng::AbstractRNG,
    state::Vector{Float64},
    θ::Dict,
    dt::Float64 = 1.0,
)
    return seitl_jump_step!(seitl_jump_problem(θ, state; rng = rng), state, dt)
end

function gillespie_step_seitl!(state::Vector{Float64}, θ::Dict, dt::Float64 = 1.0)
    return gillespie_step_seitl!(Random.default_rng(), state, θ, dt)
end
