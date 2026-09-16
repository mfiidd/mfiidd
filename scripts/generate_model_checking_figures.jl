# Generate the figures shown in sessions/slides/model_checking.qmd.
# Run with: julia --project=. scripts/generate_model_checking_figures.jl
#
# The SVGs under sessions/slides/images/ are OUTPUTS of this script, not
# sources. Re-running it overwrites every one of them. The seed below is what
# keeps the committed figures stable, so change it only if you mean to
# regenerate the lot and commit the new files.
#
# The deck is precomputed rather than executed because three of these four
# figures sit downstream of a NUTS fit, and CONTRIBUTING.md keeps anything
# that slow out of the render path.
#
# The SIR model, priors and initial state are copied from
# sessions/model_checking.qmd so the deck shows the practical's own figures.
# If the session's model changes, change it here and re-run.

using CSV
using DataFrames
using DifferentialEquations
using Distributions
using FlexiChains: @varname  ## refer to a model variable by name
using DrWatson
using Plots
using Random
using StatsBase
using Turing

ENV["GKSwstype"] = "100"  ## no display during rendering
Random.seed!(20260908)

const IMAGE_DIR = joinpath(@__DIR__, "..", "sessions", "slides", "images")

## Deck figures are projected, so they need larger type than a notebook plot.
default(
    legendfontsize = 11,
    guidefontsize = 12,
    tickfontsize = 10,
    titlefontsize = 14,
    grid = true,
    framestyle = :box,
    ## Without this the axis labels sit outside the SVG and are cut off.
    left_margin = 6Plots.mm,
    bottom_margin = 6Plots.mm,
)

# ---------------------------------------------------------------------------
# The session's SIR model
# ---------------------------------------------------------------------------

function sir_ode!(du, u, p, t)
    R_0, D_inf = p
    β = R_0 / D_inf
    ν = 1.0 / D_inf
    S, I, R, C = u
    N = S + I + R

    du[1] = -β * S * I / N
    du[2] = β * S * I / N - ν * I
    du[3] = ν * I
    du[4] = β * S * I / N  ## cumulative incidence
end

function simulate_sir(R_0, D_inf, init_state, times)
    times_vec = collect(times)
    u0 = [init_state[:S], init_state[:I], init_state[:R], 0.0]
    prob = ODEProblem(sir_ode!, u0, (times_vec[1], times_vec[end]), [R_0, D_inf])
    sol = solve(prob, Tsit5(), saveat = times_vec)

    ## Daily incidence is the difference between consecutive cumulative values,
    ## which is why times starts one day before the first observation.
    return diff(sol[4, :])
end

@model function sir_model(times, init_state, n_obs)
    R_0 ~ Uniform(0.5, 10.0)
    D_inf ~ Uniform(1.0, 14.0)
    ρ ~ Uniform(0.0, 1.0)

    inc = simulate_sir(R_0, D_inf, init_state, times)

    lambdas = [max(ρ * inc[i], 1e-10) for i in 1:n_obs]
    obs ~ arraydist(Poisson.(lambdas))
end

times = 0.0:1.0:59.0
init_state = Dict(:S => 279.0, :I => 2.0, :R => 3.0)
obs_times = collect(times)[2:end]

"""
    save_figure(plt, name)

Write `plt` to `sessions/slides/images/<name>`.

GR numbers the clip-path ids in its SVG from a counter that keeps running for
the length of the session, so the same picture picks up different ids
depending on how many plots came before it. Left alone, that makes every
re-run of this script show as a diff on all four files even when nothing about
the figures changed. Renumbering the ids in order of first appearance keeps the
committed SVGs byte-stable, so a real diff means a real change.
"""
function save_figure(plt, name)
    path = joinpath(IMAGE_DIR, name)
    savefig(plt, path)

    seen = Dict{String, String}()
    svg = replace(
        read(path, String),
        r"clip\d+" => m -> get!(seen, String(m), "clip" * lpad(length(seen), 4, '0')),
    )
    write(path, svg)

    return path
end

# ---------------------------------------------------------------------------
# 1. Prior predictive trajectories
# ---------------------------------------------------------------------------

println("1/4 prior predictive")

prior_samples =
    sample(sir_model(times, init_state, length(times) - 1), Prior(), 200, progress = false)

p_prior =
    plot(xlabel = "Day", ylabel = "Expected daily cases", legend = false, size = (900, 460))

for i in 1:200
    inc = simulate_sir(prior_samples[:R_0][i], prior_samples[:D_inf][i], init_state, times)
    plot!(p_prior, obs_times, prior_samples[:ρ][i] .* inc, alpha = 0.15, color = :steelblue)
end

save_figure(p_prior, "model_checking_prior_predictive.svg")

# ---------------------------------------------------------------------------
# 2. Posterior predictive envelope against the Tristan da Cunha data
# ---------------------------------------------------------------------------

println("2/4 fitting, then posterior predictive")

flu_tdc = CSV.read(datadir("flu_tdc_1971.csv"), DataFrame)

model = sir_model(times, init_state, length(flu_tdc.obs)) | (; obs = flu_tdc.obs)
chain = sample(model, NUTS(), 1000, progress = false)

predictions = predict(decondition(model), chain)
n_sims = 100
ppc_samples = [predictions[@varname(obs)][i, 1] for i in 1:n_sims]

p_ppc = plot(xlabel = "Day", ylabel = "Daily cases", size = (900, 460), legend = :topright)

for (j, sim) in enumerate(ppc_samples)
    plot!(
        p_ppc,
        flu_tdc.time,
        sim,
        alpha = 0.12,
        color = :steelblue,
        label = j == 1 ? "Posterior predictive" : "",
    )
end

scatter!(
    p_ppc,
    flu_tdc.time,
    flu_tdc.obs,
    color = :firebrick,
    markersize = 5,
    label = "Observed",
)

save_figure(p_ppc, "model_checking_posterior_predictive.svg")

# ---------------------------------------------------------------------------
# 3. Total cases, the summary statistic the model misses
# ---------------------------------------------------------------------------

println("3/4 total cases")

ppc_totals = [sum(sim) for sim in ppc_samples]
obs_total = sum(flu_tdc.obs)

p_total = histogram(
    ppc_totals,
    bins = 20,
    color = :steelblue,
    alpha = 0.75,
    xlabel = "Total cases",
    ylabel = "Count",
    size = (900, 460),
    label = "Posterior predictive",
    legend = :topleft,
)
vline!(p_total, [obs_total], color = :firebrick, linewidth = 3, label = "Observed")

save_figure(p_total, "model_checking_ppc_total.svg")

# ---------------------------------------------------------------------------
# 4. The four SBC rank histogram shapes
#
# Illustrative, and the deck says so. Running real SBC of the SIR model would
# mean hundreds of NUTS fits to draw a picture whose only job is to show what
# each shape looks like. Instead a conjugate Normal stands in, where the
# posterior is exact and can be corrupted in a controlled way. The ranks, the
# uniform band and the corruptions are all genuine; only the model is a
# stand-in.
# ---------------------------------------------------------------------------

println("4/4 SBC shapes")

const PRIOR_SD = 1.0    ## theta ~ Normal(0, PRIOR_SD)
const OBS_SD = 1.0      ## y | theta ~ Normal(theta, OBS_SD)
const N_SBC = 4000      ## SBC replicates
const L = 99            ## posterior draws per replicate, so ranks are 0:L

"""
    sbc_ranks(; sd_factor = 1.0, mean_shift = 0.0)

Ranks of the true value among draws from the conjugate Normal posterior.
`sd_factor` and `mean_shift` corrupt that posterior: 1.0 and 0.0 give correct
inference, below 1.0 makes it too narrow, above 1.0 too wide, and a non-zero
shift biases it.
"""
function sbc_ranks(; sd_factor = 1.0, mean_shift = 0.0)
    ranks = Vector{Int}(undef, N_SBC)

    for i in 1:N_SBC
        θ_true = rand(Normal(0.0, PRIOR_SD))
        y = rand(Normal(θ_true, OBS_SD))

        ## Exact posterior for a Normal prior and one Normal observation.
        precision = 1 / PRIOR_SD^2 + 1 / OBS_SD^2
        post_mean = (y / OBS_SD^2) / precision
        post_sd = sqrt(1 / precision)

        draws = rand(Normal(post_mean + mean_shift * post_sd, post_sd * sd_factor), L)
        ranks[i] = count(<(θ_true), draws)
    end

    return ranks
end

const N_BINS = 20

"""
    rank_panel(ranks, title)

One rank histogram with the band a uniform histogram should fall inside 99% of
the time, from the binomial distribution of counts per bin.
"""
function rank_panel(ranks, title, ymax)
    band = Binomial(N_SBC, 1 / N_BINS)
    lo, hi = quantile(band, 0.005), quantile(band, 0.995)

    p = plot(
        title = title,
        xlabel = "Rank",
        ylabel = "Count",
        legend = false,
        ylims = (0, ymax),
    )
    ## Shade the band first so the bars sit on top of it.
    plot!(
        p,
        [0, L],
        [lo, lo],
        fillrange = [hi, hi],
        color = :grey,
        alpha = 0.25,
        linewidth = 0,
    )
    histogram!(
        p,
        ranks,
        bins = range(0, L + 1, length = N_BINS + 1),
        color = :steelblue,
        alpha = 0.85,
    )
    return p
end

shapes = [
    ("Uniform — calibrated", sbc_ranks()),
    ("∪ — too narrow, overconfident", sbc_ranks(sd_factor = 0.5)),
    ("∩ — too wide, underconfident", sbc_ranks(sd_factor = 2.0)),
    ("Sloped — biased", sbc_ranks(mean_shift = 0.6)),
]

edges = range(0, L + 1, length = N_BINS + 1)
ymax = 1.1 * maximum(maximum(fit(Histogram, r, edges).weights) for (_, r) in shapes)

panels = [rank_panel(r, t, ymax) for (t, r) in shapes]

save_figure(
    plot(panels..., layout = (2, 2), size = (1000, 620)),
    "model_checking_sbc_shapes.svg",
)

println("\nWrote four SVGs to sessions/slides/images/")
