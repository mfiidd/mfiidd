using Random

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

    @inbounds begin
        S, E, I, T1, T2, T3, T4, L =
            state[1], state[2], state[3], state[4], state[5], state[6], state[7], state[8]

        # Every transition conserves the population, so β/N and the two rates out
        # of T4 are constants of the whole day rather than of each event
        N = S + E + I + T1 + T2 + T3 + T4 + L
        βN = β / N
        τ_wane = (1 - α) * τ
        τ_long = α * τ

        # The eight rates: infection, becoming infectious, recovery, three steps
        # through temporary immunity, immunity waning, immunity becoming long term
        r1 = βN * S * I
        r2 = ϵ * E
        r3 = ν * I
        r4 = τ * T1
        r5 = τ * T2
        r6 = τ * T3
        r7 = τ_wane * T4
        r8 = τ_long * T4
        total = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8

        t, daily_inc = 0.0, 0
        while t < dt
            total ≤ 0 && break

            # Time to the next event, and stop if it falls beyond the interval
            wait = randexp(rng) / total
            t + wait > dt && break
            t += wait

            # Choose the event, apply it, and recompute only the rates it changed.
            # A transition moves one individual between two compartments, so at
            # most three of the eight rates depend on what it touched.
            u = rand(rng) * total
            if u ≤ r1                                   # S → E, infection
                S -= 1
                E += 1
                r1 = βN * S * I
                r2 = ϵ * E
            elseif u ≤ r1 + r2                          # E → I, counted as a case
                E -= 1
                I += 1
                daily_inc += 1
                r1 = βN * S * I
                r2 = ϵ * E
                r3 = ν * I
            elseif u ≤ r1 + r2 + r3                     # I → T1, recovery
                I -= 1
                T1 += 1
                r1 = βN * S * I
                r3 = ν * I
                r4 = τ * T1
            elseif u ≤ r1 + r2 + r3 + r4                # T1 → T2
                T1 -= 1
                T2 += 1
                r4 = τ * T1
                r5 = τ * T2
            elseif u ≤ r1 + r2 + r3 + r4 + r5           # T2 → T3
                T2 -= 1
                T3 += 1
                r5 = τ * T2
                r6 = τ * T3
            elseif u ≤ r1 + r2 + r3 + r4 + r5 + r6      # T3 → T4
                T3 -= 1
                T4 += 1
                r6 = τ * T3
                r7 = τ_wane * T4
                r8 = τ_long * T4
            elseif u ≤ r1 + r2 + r3 + r4 + r5 + r6 + r7 # T4 → S, immunity wanes
                T4 -= 1
                S += 1
                r1 = βN * S * I
                r7 = τ_wane * T4
                r8 = τ_long * T4
            else                                        # T4 → L, immunity is lasting
                T4 -= 1
                L += 1
                r7 = τ_wane * T4
                r8 = τ_long * T4
            end
            total = r1 + r2 + r3 + r4 + r5 + r6 + r7 + r8
        end

        state[1], state[2], state[3], state[4] = S, E, I, T1
        state[5], state[6], state[7], state[8] = T2, T3, T4, L
        return daily_inc
    end
end

"""
    gillespie_step_seit4l!(state, θ, dt)

In-place version for simple bootstrap filter (no RNG argument, uses global RNG).
"""
function gillespie_step_seit4l!(state::Vector{Float64}, θ::Dict, dt::Float64 = 1.0)
    return gillespie_step!(Random.default_rng(), state, θ, dt)
end
