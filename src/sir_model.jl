using DifferentialEquations
using Distributions
using DataFrames

"""
    sir_ode!(du, u, p, t)

SIR model ordinary differential equations.

# States
- `u[1]` (S): Susceptible
- `u[2]` (I): Infectious
- `u[3]` (R): Recovered

# Parameters
- `p[1]` (R_0): Basic reproduction number
- `p[2]` (D_inf): Infectious period (days)
"""
function sir_ode!(du, u, p, t)
    S, I, R = u
    R_0, D_inf = p
    N = S + I + R

    β = R_0 / D_inf
    ν = 1 / D_inf

    du[1] = -β * S * I / N          # dS/dt
    du[2] = β * S * I / N - ν * I   # dI/dt
    du[3] = ν * I                   # dR/dt
end

"""
    SIR_BASE_PROBLEM

A template `ODEProblem` for `sir_ode!`, built once when the package loads.
`simulate_sir` calls `remake` on it rather than building a fresh problem on
every call, which matters because a likelihood asks for tens of thousands of
simulations. The placeholder state and parameters are never used: `remake`
replaces `u0`, `p` and `tspan` on every call.
"""
const SIR_BASE_PROBLEM = ODEProblem(sir_ode!, zeros(3), (0.0, 1.0), zeros(2))

"""
    simulate_sir(θ, init_state, times)

Simulate the deterministic SIR model.

# Arguments
- `θ`: Dict with keys :R_0, :D_inf
- `init_state`: Dict with keys :S, :I, :R
- `times`: Time points to return (e.g., 0.0:1.0:30.0)

# Returns
DataFrame with columns: time, S, I, R, Inc (daily incidence)

`Inc` is the flow out of S, so `Inc[i]` counts the infections between
`times[i - 1]` and `times[i]`. No such interval precedes the first time point,
so `Inc[1]` is zero and `times` needs a point before the first observation:
`Inc[i + 1]` is the incidence over the day the i-th observation counts.
"""
function simulate_sir(θ, init_state, times)
    u0 = [init_state[:S], init_state[:I], init_state[:R]]
    prob = remake(
        SIR_BASE_PROBLEM;
        u0 = u0,
        p = [θ[:R_0], θ[:D_inf]],
        tspan = (times[1], times[end]),
    )
    sol = solve(prob, Tsit5(), saveat = times)

    df = DataFrame(
        time = sol.t,
        S = [u[1] for u in sol.u],
        I = [u[2] for u in sol.u],
        R = [u[3] for u in sol.u],
    )

    # Daily incidence: new infections, the flow out of S
    df.Inc = [0.0; -diff(df.S)]

    return df
end
