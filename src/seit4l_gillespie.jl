using Random

"""
Which compartment each transition takes an individual from, and which it takes
them to, as indices into [S, E, I, T1, T2, T3, T4, L].

In order: infection, becoming infectious, recovery, the three steps through
temporary immunity, immunity waning, and immunity becoming long term.
"""
const SEIT4L_MOVES = ((1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 1), (7, 8))

"""
    seit4l_rates(s, N, β, ϵ, ν, τ, α)

The eight transition rates at state `s` in a population of `N`, in the order of
`SEIT4L_MOVES`, as a tuple.

A tuple stays on the stack. `N` is an argument because every transition
conserves it, so it is summed once a day rather than once an event.
"""
function seit4l_rates(s, N, β, ϵ, ν, τ, α)
    @inbounds S, E, I, T1, T2, T3, T4 = s[1], s[2], s[3], s[4], s[5], s[6], s[7]
    return (
        β * S * I / N,
        ϵ * E,
        ν * I,
        τ * T1,
        τ * T2,
        τ * T3,
        (1 - α) * τ * T4,
        α * τ * T4,
    )
end

"""
    gillespie_step(rng, state, θ, dt=1.0)

Simulate SEIT4L for `dt` time units using the Gillespie algorithm.

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
    s = copy(state)
    return s, gillespie_step!(rng, s, θ, dt)
end

"""
    gillespie_step!(rng, state, θ, dt=1.0)

Simulate SEIT4L for `dt` time units, writing the new compartments into the first
eight elements of `state`, and return the incidence.

The caller owns the vector, so the filter can advance a particle without
allocating one. `state` may be longer than eight: the state-space interface
passes a nine-element vector and keeps incidence in the last slot.
"""
function gillespie_step!(
    rng::AbstractRNG,
    state::AbstractVector{Float64},
    θ::Dict,
    dt::Float64 = 1.0,
)
    β = θ[:R_0] / θ[:D_inf]
    ϵ = 1.0 / θ[:D_lat]
    ν = 1.0 / θ[:D_inf]
    τ = 4.0 / θ[:D_imm]
    α = θ[:α]

    # Every transition conserves the population, so this is a constant of the
    # whole day rather than of each event
    N = @inbounds state[1] +
              state[2] +
              state[3] +
              state[4] +
              state[5] +
              state[6] +
              state[7] +
              state[8]

    t, daily_inc = 0.0, 0
    @inbounds while t < dt
        r = seit4l_rates(state, N, β, ϵ, ν, τ, α)
        total = sum(r)
        total ≤ 0 && break

        # Time to the next event, and stop if it falls beyond the interval
        wait = randexp(rng) / total
        t + wait > dt && break
        t += wait

        # Choose the event in proportion to its rate
        u, cumulative, event = rand(rng) * total, 0.0, 8
        for i in 1:8
            cumulative += r[i]
            if u ≤ cumulative
                event = i
                break
            end
        end

        # Apply it: one individual leaves a compartment and joins another
        from, to = SEIT4L_MOVES[event]
        state[from] -= 1
        state[to] += 1

        # E → I is what counts as a new case
        event == 2 && (daily_inc += 1)
    end

    return daily_inc
end

"""
    gillespie_step_seit4l!(state, θ, dt)

In-place version for simple bootstrap filter (no RNG argument, uses global RNG).
"""
function gillespie_step_seit4l!(state::Vector{Float64}, θ::Dict, dt::Float64 = 1.0)
    return gillespie_step!(Random.default_rng(), state, θ, dt)
end
