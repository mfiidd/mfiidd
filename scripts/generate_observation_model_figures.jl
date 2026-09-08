# Generate the figures shown in sessions/slides/observation_models.qmd.
# Run with: julia --project=. scripts/generate_observation_model_figures.jl
#
# The SVGs under sessions/slides/images/ are OUTPUTS of this script, not
# sources. Re-running it overwrites them. The seed below is what keeps the
# committed figures stable, so change it only if you mean to regenerate them
# and commit the new files.
#
# The deck is precomputed rather than executed because the figure sits
# downstream of two NUTS fits, and CONTRIBUTING.md keeps anything that slow out
# of the render path.
#
# The two models are copied from sessions/observation_models.qmd so the deck
# shows the practical's own fits. The simulator is not copied: `simulate_sir`
# comes from the package, as it does in the session.

using CSV
using DataFrames
using Distributions
using DrWatson
using FlexiChains: @varname
using MFIIDD
using Plots
using Random
using Statistics
using Turing

ENV["GKSwstype"] = "100"  ## no display during rendering
Random.seed!(20260908)

const IMAGE_DIR = joinpath(@__DIR__, "..", "sessions", "slides", "images")
const N_DRAWS = 500

## Deck figures are projected, so they need larger type than a notebook plot.
default(
    legendfontsize = 11,
    guidefontsize = 12,
    tickfontsize = 10,
    titlefontsize = 14,
    grid = true,
    framestyle = :box,
    left_margin = 6Plots.mm,
    bottom_margin = 6Plots.mm,
)

# ---------------------------------------------------------------------------
# The session's two observation models
# ---------------------------------------------------------------------------

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
    obs ~ arraydist([NegativeBinomial(φ, φ / (φ + μ)) for μ in mus])

    return traj
end

# ---------------------------------------------------------------------------
# Fit both to the Tristan da Cunha data
# ---------------------------------------------------------------------------

flu_tdc = CSV.read(datadir("flu_tdc_1971.csv"), DataFrame)
times = [0.0; flu_tdc.time]
init_state = Dict(:S => 279.0, :I => 2.0, :R => 3.0)
n_obs = length(flu_tdc.obs)

println("Fitting the Poisson model ...")
chain_poisson = sample(
    sir_poisson(times, init_state, n_obs) | (; obs = flu_tdc.obs),
    NUTS(0.65),
    N_DRAWS,
    progress = false,
)

println("Fitting the negative binomial model ...")
chain_negbin = sample(
    sir_negbin(times, init_state, n_obs) | (; obs = flu_tdc.obs),
    NUTS(0.65),
    N_DRAWS,
    progress = false,
)

# ---------------------------------------------------------------------------
# Posterior predictive draws, on the observation scale
# ---------------------------------------------------------------------------

"""
    predictive(chain, draw_obs)

Draw one replicate observation series per posterior sample. `draw_obs` turns the
expected count for a day into a sampled count, which is the only thing the two
models do differently.
"""
function predictive(chain, draw_obs)
    R_0s, D_infs, ρs =
        vec(chain[@varname(R_0)]), vec(chain[@varname(D_inf)]), vec(chain[@varname(ρ)])
    reps = Matrix{Float64}(undef, length(R_0s), n_obs)
    for i in eachindex(R_0s)
        traj = simulate_sir(Dict(:R_0 => R_0s[i], :D_inf => D_infs[i]), init_state, times)
        for t in 1:n_obs
            μ = max(ρs[i] * traj.Inc[t + 1], 1e-10)
            reps[i, t] = draw_obs(μ, i)
        end
    end
    return reps
end

φs = vec(chain_negbin[@varname(φ)])
rep_poisson = predictive(chain_poisson, (μ, i) -> rand(Poisson(μ)))
rep_negbin =
    predictive(chain_negbin, (μ, i) -> rand(NegativeBinomial(φs[i], φs[i] / (φs[i] + μ))))

band(reps) = (
    lo = [quantile(view(reps, :, t), 0.025) for t in 1:n_obs],
    md = [quantile(view(reps, :, t), 0.5) for t in 1:n_obs],
    hi = [quantile(view(reps, :, t), 0.975) for t in 1:n_obs],
)

# ---------------------------------------------------------------------------
# The figure
# ---------------------------------------------------------------------------

function panel(reps, title_text, colour)
    b = band(reps)
    p = plot(
        title = title_text,
        xlabel = "Day",
        ylabel = "Reported cases",
        legend = :topright,
    )
    plot!(
        p,
        flu_tdc.time,
        b.md,
        ribbon = (b.md .- b.lo, b.hi .- b.md),
        color = colour,
        fillalpha = 0.25,
        linewidth = 2,
        label = "95% predictive",
    )
    scatter!(p, flu_tdc.time, flu_tdc.obs, color = :red, markersize = 3, label = "Observed")
    return p
end

ymax = maximum([maximum(band(rep_negbin).hi), maximum(flu_tdc.obs)]) * 1.05
p_pois = panel(rep_poisson, "Poisson", :steelblue)
p_nb = panel(rep_negbin, "Negative binomial", :seagreen)
plot!(p_pois, ylims = (0, ymax))
plot!(p_nb, ylims = (0, ymax))

fig = plot(
    p_pois,
    p_nb,
    layout = (1, 2),
    size = (1100, 420),
    bottom_margin = 8Plots.mm,
    left_margin = 8Plots.mm,
)
path = joinpath(IMAGE_DIR, "observation_models_poisson_vs_negbin.svg")
savefig(fig, path)
println("wrote ", path)

# The numbers the slide quotes, so they can be checked against the figure.
width(reps) = mean(band(reps).hi .- band(reps).lo)
inside(reps) = mean(band(reps).lo .<= flu_tdc.obs .<= band(reps).hi)
println(
    "mean 95% width  Poisson ",
    round(width(rep_poisson), digits = 1),
    "  negbin ",
    round(width(rep_negbin), digits = 1),
)
println(
    "data inside band Poisson ",
    round(100 * inside(rep_poisson), digits = 0),
    "%  negbin ",
    round(100 * inside(rep_negbin), digits = 0),
    "%",
)
println("φ posterior median ", round(median(φs), digits = 2))
