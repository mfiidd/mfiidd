# Generate the figures shown in sessions/slides/particle_filters.qmd.
# Run with: julia --project=. scripts/generate_particle_filter_figures.jl
#
# The SVGs under sessions/slides/images/ are OUTPUTS of this script, not
# sources. The seed below is what keeps the committed figures stable, so change
# it only if you mean to regenerate them and commit the new files.
#
# The deck does not execute Julia, so anything it shows has to be made here.
# The stepped particle_filter_*.svg frames on the algorithm slide are not made
# here: they are the originals from the Beamer version of this course, kept as
# drawn.
#
# The parameters and initial state are copied from sessions/particle_filters.qmd
# so the deck illustrates the practical's own model. If the session changes
# them, change them here and re-run.

using CSV
using DataFrames
using Distributions
using DrWatson
using MFIIDD
using Plots
using Printf
using Random

ENV["GKSwstype"] = "100"  ## no display during rendering
Random.seed!(20260908)

const IMAGE_DIR = joinpath(@__DIR__, "..", "sessions", "slides", "images")

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

"""
    save_figure(plt, name)

Write `plt` to `sessions/slides/images/<name>`, renumbering GR's clip-path ids
so the committed SVG is byte-stable across runs. See
scripts/generate_model_checking_figures.jl, which explains why.
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
# Why simulating blind does not work
#
# Draw whole trajectories from p(x | θ) and weight each by p(y | x, θ), which
# is the estimator on the deck's "Monte Carlo, first attempt" slide. The point
# of the figure is that the weights collapse onto a couple of particles, so the
# average over J of them is really an average over two.
# ---------------------------------------------------------------------------

flu_tdc = CSV.read(datadir("flu_tdc_1971.csv"), DataFrame)

## Copied from sessions/particle_filters.qmd
θ = Dict(:R_0 => 7.0, :D_lat => 1.0, :D_inf => 4.0, :α => 0.5, :D_imm => 10.0, :ρ => 0.65)
init_state = [279.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0]

const N_DAYS = length(flu_tdc.obs)
const J = 200

"""
    simulate_blind()

One SEIT4L trajectory, simulated forward with no reference to the data. This is
a draw from `p(x | θ)`.
"""
function simulate_blind()
    state = copy(init_state)
    return [gillespie_step_seit4l!(state, θ, 1.0) for _ in 1:N_DAYS]
end

"""
    log_weight(inc)

`log p(y | x, θ)` for one trajectory: the Poisson likelihood of the whole
observed series under the reported incidence that trajectory implies.
"""
function log_weight(inc)
    return sum(
        logpdf(Poisson(max(θ[:ρ] * inc[t], 1e-10)), flu_tdc.obs[t]) for t in 1:N_DAYS
    )
end

trajectories = [simulate_blind() for _ in 1:J]
log_weights = log_weight.(trajectories)

weights = exp.(log_weights .- maximum(log_weights))
weights ./= sum(weights)
ess = 1 / sum(abs2, weights)

# ---------------------------------------------------------------------------
# The build-up: one draw, then many
#
# The deck introduces the estimator a step at a time, so these two figures show
# the same 200 trajectories with progressively more painted on: one of them
# alone, then all of them, then all of them with the weights revealed. They
# reuse `trajectories` rather than drawing again, both so the three figures
# really are the same particles and so that adding them here does not consume
# random numbers and shift every figure below.
# ---------------------------------------------------------------------------

"""
    plot_data(; title)

An empty incidence-against-day panel with the observations on it, shared by the
three figures of the build-up so they line up when shown in sequence.
"""
function plot_data(; title)
    plt = plot(
        xlabel = "Day",
        ylabel = "Reported cases",
        title = title,
        legend = :topright,
        size = (900, 460),
        ylims = (0, 1.05 * maximum(flu_tdc.obs)),
    )
    scatter!(
        plt,
        flu_tdc.time,
        flu_tdc.obs,
        color = :firebrick,
        markersize = 4,
        label = "Observed",
    )
    return plt
end

## One draw. The first of the 200, not the best of them: a typical draw is what
## the slide is about.
p_one = plot_data(title = "One trajectory drawn from p(x | θ)")
plot!(
    p_one,
    flu_tdc.time,
    θ[:ρ] .* trajectories[1],
    color = :steelblue,
    linewidth = 3,
    label = @sprintf("log p(y | x, θ) = %.0f", log_weights[1]),
)
save_figure(p_one, "pf_one_draw.svg")
println("Wrote pf_one_draw.svg to sessions/slides/images/")

## The same 200 with none of them singled out. The next figure is this one with
## the weights revealed, so nothing here may hint at which two win.
p_many = plot_data(title = @sprintf("%d trajectories drawn from p(x | θ)", J))
for inc in trajectories
    plot!(p_many, flu_tdc.time, θ[:ρ] .* inc, alpha = 0.13, color = :steelblue, label = "")
end
plot!(p_many, [], [], alpha = 0.6, color = :steelblue, label = "$J trajectories")
save_figure(p_many, "pf_many_draws.svg")
println("Wrote pf_many_draws.svg to sessions/slides/images/")

order = sortperm(weights, rev = true)
top_two = order[1:2]
top_share = sum(weights[top_two])

p_traj = plot(
    xlabel = "Day",
    ylabel = "Reported cases",
    title = "200 trajectories drawn from p(x | θ)",
    legend = :topright,
)

## The 198 that contribute nothing, drawn first and faintly.
for (j, inc) in enumerate(trajectories)
    j in top_two && continue
    plot!(
        p_traj,
        flu_tdc.time,
        θ[:ρ] .* inc,
        alpha = 0.13,
        color = :steelblue,
        label = j == order[end] ? "The other 198" : "",
    )
end

## The two that carry the estimate. They track both waves where the rest of the
## cloud fans out, so it is clear enough why they won; the point is that only
## two of two hundred managed it.
for (rank, j) in enumerate(top_two)
    plot!(
        p_traj,
        flu_tdc.time,
        θ[:ρ] .* trajectories[j],
        color = :darkorange,
        linewidth = 3,
        label = rank == 1 ? @sprintf("The two carrying %.0f%%", 100 * top_share) : "",
    )
end

scatter!(
    p_traj,
    flu_tdc.time,
    flu_tdc.obs,
    color = :firebrick,
    markersize = 4,
    label = "Observed",
)

p_weights = bar(
    weights[order][1:40],
    color = :steelblue,
    linecolor = :steelblue,
    xlabel = "Trajectory, ordered by weight",
    ylabel = "Share of the total weight",
    title = @sprintf("Effective sample size: %.1f of %d", ess, J),
    legend = false,
)

save_figure(
    plot(p_traj, p_weights, layout = (1, 2), size = (1100, 440)),
    "pf_naive_monte_carlo.svg",
)

@printf(
    "log p(y|x,θ): best %.0f, worst %.0f; top weight %.2f; ESS %.1f of %d\n",
    maximum(log_weights),
    minimum(log_weights),
    maximum(weights),
    ess,
    J
)
println("Wrote pf_naive_monte_carlo.svg to sessions/slides/images/")

# ---------------------------------------------------------------------------
# Why resampling is the whole difference
#
# Run the same J particles twice: once propagating and weighting without ever
# resampling, which is exactly the naive estimator written sequentially, and
# once resampling at every observation. Track the effective sample size.
# ---------------------------------------------------------------------------

"""
    run_filter(; resample)

Propagate `J` particles day by day, weighting each by the observation. With
`resample = false` the weights simply accumulate, which is the naive estimator
of the first figure written out one day at a time. Returns the effective sample
size after each day.
"""
function run_filter(; resample::Bool)
    states = [copy(init_state) for _ in 1:J]
    log_w = zeros(J)
    ess = zeros(N_DAYS)

    for t in 1:N_DAYS
        for j in 1:J
            inc = gillespie_step_seit4l!(states[j], θ, 1.0)
            log_w[j] += logpdf(Poisson(max(θ[:ρ] * inc, 1e-10)), flu_tdc.obs[t])
        end

        w = exp.(log_w .- maximum(log_w))
        w ./= sum(w)
        ess[t] = 1 / sum(abs2, w)

        if resample
            keep = rand(Categorical(w), J)
            states = [copy(states[k]) for k in keep]
            fill!(log_w, 0.0)  ## resampling resets the weights to 1/J
        end
    end

    return ess
end

ess_plain = run_filter(resample = false)
ess_filter = run_filter(resample = true)

## The naive estimator on its own, scored on the first t days only. This is the
## deck's argument for going sequential at all: the fall is roughly a straight
## line on a log scale, so it is exponential in the number of days, and a bigger
## J cannot keep up with it. Drawn on the same axes as the comparison below so
## the two slides line up.
p_ess_naive = plot(
    flu_tdc.time,
    ess_plain,
    label = "",
    color = :firebrick,
    linewidth = 3,
    xlabel = "Days scored",
    ylabel = "Effective sample size",
    title = "Sampling whole trajectories",
    yscale = :log10,
    ylims = (0.8, 1.5J),
    legend = :topright,
    size = (900, 460),
)
hline!(p_ess_naive, [J], color = :grey, linestyle = :dash, label = "All $J trajectories")

save_figure(p_ess_naive, "pf_naive_ess_by_day.svg")
println("Wrote pf_naive_ess_by_day.svg to sessions/slides/images/")

p_ess = plot(
    flu_tdc.time,
    ess_plain,
    label = "No resampling",
    color = :firebrick,
    linewidth = 3,
    xlabel = "Day",
    ylabel = "Effective sample size",
    title = "With and without resampling",
    yscale = :log10,
    ylims = (0.8, 1.5J),
    legend = :right,
    size = (900, 460),
)
plot!(
    p_ess,
    flu_tdc.time,
    ess_filter,
    label = "Resampling",
    color = :steelblue,
    linewidth = 3,
)
hline!(p_ess, [J], color = :grey, linestyle = :dash, label = "All $J particles")

save_figure(p_ess, "pf_degeneracy.svg")

@printf(
    "ESS after 10 days: %.1f without resampling, %.1f with; at the end: %.2f vs %.1f\n",
    ess_plain[10],
    ess_filter[10],
    ess_plain[end],
    ess_filter[end]
)
println("Wrote pf_degeneracy.svg to sessions/slides/images/")
