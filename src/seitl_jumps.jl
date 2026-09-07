using JumpProcesses:
    DiscreteProblem, Direct, JumpProblem, MassActionJump, SSAStepper, remake, solve
using Random: AbstractRNG, default_rng

"""
    seitl_stoichiometry(k)

Build the reactant and net stoichiometry of the `k + 4` transitions of a SEITL
model with `k` stages of temporary immunity.

The state is `[S, E, I, T_1 ... T_k, L, Inc]`. `Inc` counts the E → I
transitions and takes part in no rate, so carrying it in the state costs
nothing and the daily incidence is read off the state like any compartment.

| Transition | Rate |
|------------|------|
| S → E | `β S I / N` |
| E → I | `ϵ E` |
| I → T_1 | `ν I` |
| T_j → T_{j+1} | `τ T_j` |
| T_k → S | `(1 - α) τ T_k` |
| T_k → L | `α τ T_k` |

Only `k` decides these, so they are built once per model and reused across
every simulated interval, leaving only the rate constants to compute.
"""
function seitl_stoichiometry(k::Integer)
    S, E, I, L, Inc = 1, 2, 3, k + 4, k + 5
    T(j) = 3 + j  ## T_1 ... T_k sit between I and L

    reactants = [[S => 1, I => 1], [E => 1], [I => 1]]
    net = [[S => -1, E => 1], [E => -1, I => 1, Inc => 1], [I => -1, T(1) => 1]]

    for j in 1:(k - 1)  ## progress through the stages of temporary immunity
        push!(reactants, [T(j) => 1])
        push!(net, [T(j) => -1, T(j + 1) => 1])
    end

    push!(reactants, [T(k) => 1])  ## immunity wanes
    push!(net, [T(k) => -1, S => 1])

    push!(reactants, [T(k) => 1])  ## immunity becomes long term
    push!(net, [T(k) => -1, L => 1])

    return reactants, net
end

"""
    seitl_rate_constants(θ, N, k)

The `k + 4` mass-action rate constants, in the order `seitl_stoichiometry`
gives the transitions.

Every transition conserves `N`, so the infection rate `β S I / N` is the
constant `β / N` times `S` times `I` and the whole model is mass action. The
`k` stages of temporary immunity each pass at rate `τ = k / D_imm`, so the
total time spent immune is Erlang with mean `D_imm` whatever `k` is.
"""
function seitl_rate_constants(θ::Dict, N::Real, k::Integer)
    β = θ[:R_0] / θ[:D_inf]
    ϵ = 1.0 / θ[:D_lat]
    ν = 1.0 / θ[:D_inf]
    τ = k / θ[:D_imm]
    α = θ[:α]

    rates = Vector{Float64}(undef, k + 4)
    rates[1], rates[2], rates[3] = β / N, ϵ, ν
    for j in 1:(k - 1)
        rates[3 + j] = τ
    end
    rates[k + 3] = (1 - α) * τ
    rates[k + 4] = α * τ
    return rates
end

"""
    seitl_transitions(θ, N, k)

The `MassActionJump` holding the `k + 4` transitions of a SEITL model with `k`
stages of temporary immunity, in a population of size `N`.
"""
function seitl_transitions(θ::Dict, N::Real, k::Integer)
    return MassActionJump(seitl_rate_constants(θ, N, k), seitl_stoichiometry(k)...)
end

"""
    seitl_jump_problem(θ, compartments; rng, aggregator)

Build the `JumpProblem` for the SEITL model whose compartments are
`compartments`, a vector `[S, E, I, T_1 ... T_k, L]`. The number of stages `k`
follows from its length.

`Direct` is Gillespie's direct method, which suits a system of this size.
`save_positions = (false, false)` stops the solver storing the state at every
event, since only the value at the end of the interval is wanted.
"""
function seitl_jump_problem(
    θ::Dict,
    compartments::AbstractVector{<:Real};
    rng::AbstractRNG = default_rng(),
    aggregator = Direct(),
)
    k = length(compartments) - 4
    k ≥ 1 || throw(ArgumentError("compartments must be [S, E, I, T_1 ... T_k, L]"))
    reactants, net = seitl_stoichiometry(k)
    rates = seitl_rate_constants(θ, sum(compartments), k)
    u0 = vcat(Float64.(compartments), 0.0)  ## the trailing slot counts incidence
    return JumpProblem(
        DiscreteProblem(u0, (0.0, 1.0)),
        aggregator,
        MassActionJump(rates, reactants, net);
        save_positions = (false, false),
        rng = rng,
    )
end

"""
    seitl_jump_step!(jump_problem, state, dt = 1.0)

Advance `state`, a vector `[S, E, I, T_1 ... T_k, L]`, by `dt` days and return
the incidence over that interval.

`state` is modified in place. The incidence slot starts each interval at zero,
so the value returned is the number of E → I transitions within `dt` rather
than a running total.
"""
function seitl_jump_step!(jump_problem, state::AbstractVector{<:Real}, dt::Real = 1.0)
    stepped = remake(jump_problem; u0 = vcat(state, 0.0), tspan = (0.0, Float64(dt)))
    final = solve(stepped, SSAStepper()).u[end]
    state .= @view final[1:(end - 1)]
    return final[end]
end

"""
    seitl_jump_trajectory(θ, compartments, times)

Simulate one trajectory over `times` and return a matrix whose rows are the
compartments at each time and a vector of the incidence between them.

The whole trajectory is one `solve`, with `saveat` set to `times`, so the
solver is entered once rather than once per day. The incidence slot
accumulates over the run, and differencing it gives the incidence within each
interval.
"""
function seitl_jump_trajectory(θ::Dict, compartments::AbstractVector{<:Real}, times)
    jump_problem = seitl_jump_problem(θ, compartments)
    stepped = remake(jump_problem; tspan = (Float64(first(times)), Float64(last(times))))
    solution = solve(stepped, SSAStepper(); saveat = times)
    states = reduce(hcat, solution.u)'
    incidence = [0.0; diff(states[:, end])]
    return states[:, 1:(end - 1)], incidence
end
