# Generate the figures shown in sessions/slides/observation_models.qmd.
# Run with: julia --project=. scripts/generate_observation_model_figures.jl
#
# The SVGs under sessions/slides/images/ are OUTPUTS of this script, not
# sources. Re-running it overwrites every one of them. The seed below is what
# keeps the committed figures stable, so change it only if you mean to
# regenerate the lot and commit the new files.
#
# The deck is precomputed rather than executed because two of these five
# figures sit downstream of a NUTS fit, and CONTRIBUTING.md keeps anything
# that slow out of the render path.
#
# The models, priors and initial state are the session's own, so the deck
# shows the figures the practical goes on to produce. If
# sessions/observation_models.qmd changes either model, change it here and
# re-run.

using CSV
using DataFrames
using Distributions
using DrWatson
using FlexiChains: @varname  ## refer to a model variable by name
using MFIIDD ## the course package, which holds the SIR implementation
using Plots
using Random
using StatsPlots
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

const POISSON_COLOUR = :steelblue
const NEGBIN_COLOUR = :darkorange
const DATA_COLOUR = :firebrick

"""
    save_figure(plt, name)

Write `plt` to `sessions/slides/images/<name>`.

GR numbers the clip-path ids in its SVG from a counter that keeps running for
the length of the session, so the same picture picks up different ids
depending on how many plots came before it. Left alone, that makes every
re-run of this script show as a diff on all five files even when nothing about
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

"""
    negbin(μ, φ)

The negative binomial with mean `μ` and variance `μ + μ^2 / φ`, in Julia's
`(r, p)` parameterisation. This is the one the session fits.
"""
negbin(μ, φ) = NegativeBinomial(φ, φ / (φ + μ))

# ---------------------------------------------------------------------------
# 1. What Poisson assumes: the spread is fixed by the mean
# ---------------------------------------------------------------------------

println("1/5 Poisson spread")

p_poisson = plot(
    xlabel = "Reported cases",
    ylabel = "Probability",
    size = (900, 460),
    legend = :topright,
)

for (λ, colour) in zip([5, 20, 50], [:steelblue, :seagreen, :purple])
    counts = 0:80
    plot!(
        p_poisson,
        counts,
        pdf.(Poisson(λ), counts),
        linewidth = 3,
        color = colour,
        label = "mean $λ, sd $(round(sqrt(λ), digits = 1))",
    )
end

save_figure(p_poisson, "observation_poisson_spread.svg")

# ---------------------------------------------------------------------------
# 2. The same mean, three spreads
# ---------------------------------------------------------------------------

println("2/5 overdispersion")

counts = 0:80
p_over = plot(
    counts,
    pdf.(Poisson(20), counts),
    linewidth = 3,
    color = POISSON_COLOUR,
    label = "Poisson, sd 4.5",
    xlabel = "Reported cases",
    ylabel = "Probability",
    size = (900, 460),
    legend = :topright,
)

for (φ, style) in zip([5.0, 1.0], [:solid, :dash])
    d = negbin(20.0, φ)
    plot!(
        p_over,
        counts,
        pdf.(d, counts),
        linewidth = 3,
        linestyle = style,
        color = NEGBIN_COLOUR,
        label = "NegBin φ = $(Int(φ)), sd $(round(std(d), digits = 1))",
    )
end

vline!(p_over, [20], color = :grey, linestyle = :dot, linewidth = 2, label = "Mean")

save_figure(p_over, "observation_overdispersion.svg")

# ---------------------------------------------------------------------------
# The session's two fits, which figures 3 and 4 both use
# ---------------------------------------------------------------------------

println("3/5 fitting both observation models")

flu_tdc = CSV.read(datadir("flu_tdc_1971.csv"), DataFrame)
times = [0.0; flu_tdc.time]

## The island as the ship lands, at the start of day 1. See CONTRIBUTING.md.
init_state = Dict(:S => 279.0, :I => 2.0, :R => 3.0)

@model function sir_poisson(times, init_state, n_obs)
    R_0 ~ Uniform(1.0, 20.0)
    D_inf ~ Uniform(1.0, 10.0)
    ρ ~ Uniform(0.1, 1.0)

    θ = Dict(:R_0 => R_0, :D_inf => D_inf)
    traj = simulate_sir(θ, init_state, times)

    lambdas = [max(ρ * traj.Inc[i + 1], 1e-10) for i in 1:n_obs]
    obs ~ arraydist(Poisson.(lambdas))

    return traj
end

@model function sir_negbin(times, init_state, n_obs)
    R_0 ~ Uniform(1.0, 20.0)
    D_inf ~ Uniform(1.0, 10.0)
    ρ ~ Uniform(0.1, 1.0)
    φ ~ Exponential(10.0)

    θ = Dict(:R_0 => R_0, :D_inf => D_inf)
    traj = simulate_sir(θ, init_state, times)

    mus = [max(ρ * traj.Inc[i + 1], 1e-10) for i in 1:n_obs]
    obs ~ arraydist([negbin(μ, φ) for μ in mus])

    return traj
end

n_obs = length(flu_tdc.obs)
model_poisson = sir_poisson(times, init_state, n_obs) | (; obs = flu_tdc.obs)
model_negbin = sir_negbin(times, init_state, n_obs) | (; obs = flu_tdc.obs)

chain_poisson = sample(model_poisson, NUTS(0.65), 1000, progress = false)
chain_negbin = sample(model_negbin, NUTS(0.65), 1000, progress = false)

"""
    predictive_band(model, chain)

Pointwise 2.5%, 50% and 97.5% quantiles of the posterior predictive
distribution of the observations.
"""
function predictive_band(model, chain)
    predictions = predict(decondition(model), chain)
    draws = predictions[@varname(obs)]
    per_day = [[draws[i, 1][t] for i in 1:size(draws, 1)] for t in 1:n_obs]

    return (
        lower = [quantile(day, 0.025) for day in per_day],
        median = [quantile(day, 0.5) for day in per_day],
        upper = [quantile(day, 0.975) for day in per_day],
    )
end

band_poisson = predictive_band(model_poisson, chain_poisson)
band_negbin = predictive_band(model_negbin, chain_negbin)

# ---------------------------------------------------------------------------
# 3. What each observation model says the data should look like
# ---------------------------------------------------------------------------

println("4/5 predictive intervals")

p_bands =
    plot(xlabel = "Day", ylabel = "Daily cases", size = (900, 460), legend = :topright)

plot!(
    p_bands,
    flu_tdc.time,
    band_negbin.median,
    ribbon = (
        band_negbin.median .- band_negbin.lower,
        band_negbin.upper .- band_negbin.median,
    ),
    linewidth = 3,
    color = NEGBIN_COLOUR,
    fillalpha = 0.25,
    label = "Negative binomial",
)

plot!(
    p_bands,
    flu_tdc.time,
    band_poisson.median,
    ribbon = (
        band_poisson.median .- band_poisson.lower,
        band_poisson.upper .- band_poisson.median,
    ),
    linewidth = 3,
    color = POISSON_COLOUR,
    fillalpha = 0.35,
    label = "Poisson",
)

scatter!(
    p_bands,
    flu_tdc.time,
    flu_tdc.obs,
    color = DATA_COLOUR,
    markersize = 5,
    label = "Observed",
)

save_figure(p_bands, "observation_predictive_bands.svg")

# ---------------------------------------------------------------------------
# 4. What that does to the posterior
# ---------------------------------------------------------------------------

println("5/5 posteriors and the distance functions")

p_post = density(
    vec(chain_poisson[:R_0]),
    linewidth = 3,
    color = POISSON_COLOUR,
    fill = (0, 0.2, POISSON_COLOUR),
    label = "Poisson",
    xlabel = "R₀",
    ylabel = "Posterior density",
    size = (900, 460),
    legend = :topright,
)
density!(
    p_post,
    vec(chain_negbin[:R_0]),
    linewidth = 3,
    color = NEGBIN_COLOUR,
    fill = (0, 0.2, NEGBIN_COLOUR),
    label = "Negative binomial",
)

save_figure(p_post, "observation_posteriors.svg")

# ---------------------------------------------------------------------------
# 5. The likelihood as a distance function
# ---------------------------------------------------------------------------

obs_val = 10
predicted = 1.0:0.25:25.0

p_dist = plot(
    predicted,
    [-logpdf(Poisson(μ), obs_val) for μ in predicted],
    linewidth = 3,
    color = POISSON_COLOUR,
    label = "Poisson",
    xlabel = "Predicted cases",
    ylabel = "Negative log-likelihood",
    size = (900, 460),
    legend = :topright,
)
plot!(
    p_dist,
    predicted,
    [-logpdf(negbin(μ, 5.0), obs_val) for μ in predicted],
    linewidth = 3,
    color = NEGBIN_COLOUR,
    label = "NegBin φ = 5",
)
plot!(
    p_dist,
    predicted,
    [-logpdf(Normal(μ, 3.0), obs_val) for μ in predicted],
    linewidth = 3,
    linestyle = :dash,
    color = :seagreen,
    label = "Normal σ = 3",
)
vline!(p_dist, [obs_val], color = DATA_COLOUR, linewidth = 2, label = "Observed 10")

save_figure(p_dist, "observation_distance.svg")

# ---------------------------------------------------------------------------
# The numbers the deck quotes. The interval coverage below appears on the
# "What each model expects the data to look like" slide, so change one and
# change the other.
# ---------------------------------------------------------------------------

for (name, chain) in (("Poisson", chain_poisson), ("NegBin", chain_negbin))
    for p in (:R_0, :D_inf, :ρ)
        draws = vec(chain[p])
        println(
            name,
            " ",
            p,
            ": mean ",
            round(mean(draws), digits = 2),
            ", sd ",
            round(std(draws), digits = 2),
        )
    end
end
println("NegBin φ: median ", round(median(vec(chain_negbin[:φ])), digits = 2))
println(
    "Days inside the Poisson 95% band: ",
    count(flu_tdc.obs .>= band_poisson.lower .&& flu_tdc.obs .<= band_poisson.upper),
    " of ",
    n_obs,
)
println(
    "Days inside the negative binomial 95% band: ",
    count(flu_tdc.obs .>= band_negbin.lower .&& flu_tdc.obs .<= band_negbin.upper),
    " of ",
    n_obs,
)

println("\nWrote five SVGs to sessions/slides/images/")
