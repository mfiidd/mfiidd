using Random

"""
    gillespie_step(rng, state, θ; k, dt = 1.0)

Simulate `dt` time units of the SEITL model with `k` stages of temporary
immunity, using the Gillespie algorithm.

The state is `[S, E, I, T_1 ... T_k, L]`, so `k = 1` is SEITL and `k = 4` is
SEIT4L. Individuals leave each immunity stage at rate `τ = k / D_imm`, which
holds the mean duration of temporary immunity at `D_imm` whatever `k` is, and
leave the last one for `S` with probability `1 - α` and for `L` with
probability `α`.

`state` is left untouched. [`gillespie_step!`](@ref) updates it in place, which
is what the particle filter wants once per particle per day.

# Arguments
- `rng`: Random number generator
- `state`: Vector [S, E, I, T_1 ... T_k, L]
- `θ`: Parameter dictionary with keys :R_0, :D_lat, :D_inf, :α, :D_imm
- `k`: Number of temporary immunity stages, so `length(state) == k + 4`
- `dt`: Length of the simulation interval (default 1.0, i.e. one day)

# Returns
- `(new_state, incidence)`: the state after `dt` and the new cases over it
"""
function gillespie_step(
    rng::AbstractRNG,
    state::Vector{Float64},
    θ::Dict;
    k::Integer,
    dt::Float64 = 1.0,
)
    β = θ[:R_0] / θ[:D_inf]
    ϵ = 1.0 / θ[:D_lat]
    ν = 1.0 / θ[:D_inf]
    τ = k / θ[:D_imm]
    α = θ[:α]

    ## Compartments are S = 1, E = 2, I = 3, T_j = 3 + j and L = k + 4. Every
    ## transition moves one individual from one compartment to another, so a
    ## pair of indices says all there is to say about its effect.
    transitions = [(1, 2), (2, 3), (3, 4)]      ## S → E, E → I, I → T_1
    for j in 1:(k - 1)
        push!(transitions, (3 + j, 4 + j))      ## T_j → T_(j+1)
    end
    push!(transitions, (3 + k, 1))              ## T_k → S, immunity wanes
    push!(transitions, (3 + k, k + 4))          ## T_k → L, long-term immunity

    function rates(s)
        S, E, I = s[1], s[2], s[3]
        T = view(s, 4:(3 + k))                  ## the k immunity stages
        N = sum(s)

        r = Vector{Float64}(undef, k + 4)
        r[1] = β * S * I / N                    ## S → E
        r[2] = ϵ * E                            ## E → I
        r[3] = ν * I                            ## I → T_1
        for j in 1:(k - 1)
            r[3 + j] = τ * T[j]                 ## T_j → T_(j+1)
        end
        r[k + 3] = (1 - α) * τ * T[k]           ## T_k → S
        r[k + 4] = α * τ * T[k]                 ## T_k → L
        return r
    end

    s = copy(state)

    ## Simulate up to `dt` time units
    t, daily_inc = 0.0, 0
    while t < dt
        r = rates(s)
        total_rate = sum(r)
        total_rate ≤ 0 && break

        ## Time to next event
        τ_wait = randexp(rng) / total_rate
        t + τ_wait > dt && break
        t += τ_wait

        ## Select which event occurs, with probability proportional to its rate
        cum, rnd, event = 0.0, rand(rng) * total_rate, 0
        for i in eachindex(r)
            cum += r[i]
            if rnd ≤ cum
                event = i
                break
            end
        end

        ## Move one individual along that transition
        from, to = transitions[event]
        s[from] -= 1
        s[to] += 1

        ## E → I transitions count as new cases
        event == 2 && (daily_inc += 1)
    end

    return s, daily_inc
end

"""
    gillespie_step!(rng, state, θ; k, dt = 1.0)
    gillespie_step!(state, θ; k, dt = 1.0)

In-place [`gillespie_step`](@ref): update `state` and return the incidence on
its own.

The two-argument form draws from the global random number generator; pass an
`rng` when the caller needs to control randomness, as the particle filter does.
"""
function gillespie_step!(
    rng::AbstractRNG,
    state::Vector{Float64},
    θ::Dict;
    k::Integer,
    dt::Float64 = 1.0,
)
    new_state, incidence = gillespie_step(rng, state, θ; k = k, dt = dt)
    for i in eachindex(state)
        state[i] = new_state[i]
    end
    return incidence
end

function gillespie_step!(state::Vector{Float64}, θ::Dict; k::Integer, dt::Float64 = 1.0)
    return gillespie_step!(Random.default_rng(), state, θ; k = k, dt = dt)
end
