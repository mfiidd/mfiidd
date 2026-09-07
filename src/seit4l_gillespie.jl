using Random

"""
    gillespie_step(rng, state, θ, dt=1.0)

Simulate SEIT4L for `dt` time units and return the state at the end of the
interval together with the incidence over it.

# Arguments
- `rng`: Random number generator
- `state`: Vector [S, E, I, T1, T2, T3, T4, L]
- `θ`: Parameter dictionary with keys :R_0, :D_lat, :D_inf, :α, :D_imm
- `dt`: Length of the simulation interval (default 1.0, i.e. one day)

# Returns
- `(new_state, incidence)`: Updated state and number of new cases over `dt`
"""
function gillespie_step(
    rng::AbstractRNG,
    state::Vector{Float64},
    θ::Dict,
    dt::Float64 = 1.0,
)
    new_state = copy(state)
    incidence = seitl_jump_step!(
        seitl_jump_problem(θ, new_state; rng = rng),
        new_state,
        dt,
    )
    return new_state, incidence
end

"""
    gillespie_step_seit4l!(rng, state, θ, dt)
    gillespie_step_seit4l!(state, θ, dt)

In-place version for the bootstrap filter, which updates `state` and returns the
incidence rather than returning a new vector.

The two-argument form draws from the global random number generator; pass an
`rng` explicitly when the caller needs to control randomness, as the particle
filter does. This mirrors `gillespie_step_seitl!`, so the two steppers can be
used interchangeably by `SEITLDynamics`.
"""
function gillespie_step_seit4l!(
    rng::AbstractRNG,
    state::Vector{Float64},
    θ::Dict,
    dt::Float64 = 1.0,
)
    return seitl_jump_step!(seitl_jump_problem(θ, state; rng = rng), state, dt)
end

function gillespie_step_seit4l!(state::Vector{Float64}, θ::Dict, dt::Float64 = 1.0)
    return gillespie_step_seit4l!(Random.default_rng(), state, θ, dt)
end
