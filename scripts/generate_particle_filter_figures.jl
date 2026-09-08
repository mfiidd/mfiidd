# Generate the figures shown in sessions/slides/particle_filters.qmd.
# Run with: julia --project=. scripts/generate_particle_filter_figures.jl
#
# The SVGs under sessions/slides/images/ are OUTPUTS of this script, not
# sources. The seed below is what keeps the committed figures stable, so change
# it only if you mean to regenerate them and commit the new files.
#
# The deck does not execute Julia, so anything it shows has to be made here.
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
    p_traj, flu_tdc.time, flu_tdc.obs, color = :firebrick, markersize = 4, label = "Observed"
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
    maximum(log_weights), minimum(log_weights), maximum(weights), ess, J
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

p_ess = plot(
    flu_tdc.time,
    ess_plain,
    label = "No resampling",
    color = :firebrick,
    linewidth = 3,
    xlabel = "Day",
    ylabel = "Effective number of particles",
    title = "What resampling buys",
    yscale = :log10,
    ylims = (0.8, 1.5J),
    legend = :right,
    size = (900, 460),
)
plot!(p_ess, flu_tdc.time, ess_filter, label = "Resampling", color = :steelblue, linewidth = 3)
hline!(p_ess, [J], color = :grey, linestyle = :dash, label = "All $J particles")

save_figure(p_ess, "pf_degeneracy.svg")

@printf(
    "ESS after 10 days: %.1f without resampling, %.1f with; at the end: %.2f vs %.1f\n",
    ess_plain[10], ess_filter[10], ess_plain[end], ess_filter[end]
)
println("Wrote pf_degeneracy.svg to sessions/slides/images/")

# ---------------------------------------------------------------------------
# The algorithm, walked across the whole series
#
# One frame per action: initialise, then propagate/weight/resample at each of
# the five observations. The archive deck (sbfnk/mfiidd.archive, Rmd/slides/
# smc.pdf) stepped through it this way and the point does not come across from
# a single cycle: the particles have to be seen advancing, and the survivors
# carrying the trajectory forward.
#
# A toy series rather than Tristan da Cunha, because five observations and five
# particles are legible on a projector and 59 are not.
# ---------------------------------------------------------------------------

Random.seed!(4)

const T_OBS = 5
const N_PART = 5
const TOY_DATA = [3.0, 5.2, 7.4, 5.6, 3.4]

## Particle positions: a random walk started spread around the first observation.
const STEP_SD = 1.9

toy_weight(x, y) = exp(-0.5 * ((x - y) / 1.3)^2)

"""
    run_toy_filter()

A five-particle bootstrap filter over `TOY_DATA`, recording what each frame
needs: where the particles sat before and after each move, their weights, and
which ones survived resampling.
"""
function run_toy_filter()
    before = Vector{Vector{Float64}}()   ## positions after propagating into t
    weights = Vector{Vector{Float64}}()  ## normalised weights at t
    after = Vector{Vector{Float64}}()    ## positions after resampling at t
    parents = Vector{Vector{Int}}()      ## which particle each survivor came from

    x = TOY_DATA[1] .+ range(-2.6, 2.6, length = N_PART)
    start = copy(x)

    for t in 1:T_OBS
        x = x .+ STEP_SD .* randn(N_PART) .+ (t == 1 ? 0.0 : TOY_DATA[t] - TOY_DATA[t-1])
        push!(before, copy(x))

        w = toy_weight.(x, TOY_DATA[t])
        w ./= sum(w)
        push!(weights, copy(w))

        keep = rand(Categorical(w), N_PART)
        push!(parents, copy(keep))
        x = x[keep]
        push!(after, copy(x))
    end

    return start, before, weights, after, parents
end

start_x, pos_before, pos_weights, pos_after, pos_parents = run_toy_filter()

"""
    toy_frame(stage, t; ...)

One frame of the walk. `stage` is `:initialise`, `:propagate`, `:weight` or
`:resample`, and `t` the observation it applies to.
"""
function toy_frame(stage::Symbol, t::Int)
    p = plot(
        xlabel = "Time",
        ylabel = "Incidence",
        legend = false,
        xlims = (-0.4, T_OBS + 0.5),
        ylims = (0.0, 11.0),
        xticks = 0:T_OBS,
        yticks = false,
        size = (1000, 580),
        titlefontsize = 21,
        guidefontsize = 15,
        tickfontsize = 13,
    )

    ## Everything already settled, drawn faintly so the eye follows the front.
    for s in 1:(t - 1)
        for j in 1:N_PART
            plot!(
                p, [s - 1, s], [s == 1 ? start_x[j] : pos_after[s-1][j], pos_before[s][j]],
                color = :grey, alpha = 0.35, linewidth = 1,
            )
        end
        scatter!(p, fill(s, N_PART), pos_after[s], color = :gold, markersize = 6,
                 markerstrokecolor = :black, alpha = 0.45)
    end

    scatter!(p, fill(0, N_PART), start_x, color = :gold, markersize = 6,
             markerstrokecolor = :black, alpha = stage == :initialise ? 1.0 : 0.45)

    if stage in (:propagate, :weight, :resample)
        from = t == 1 ? start_x : pos_after[t-1]
        for j in 1:N_PART
            plot!(p, [t - 1, t], [from[j], pos_before[t][j]],
                  color = :steelblue, alpha = 0.8, linewidth = 2)
        end
    end

    if stage == :propagate
        scatter!(p, fill(t, N_PART), pos_before[t], color = :gold, markersize = 9,
                 markerstrokecolor = :black)
    elseif stage == :weight
        scatter!(p, fill(t, N_PART), pos_before[t], color = :gold,
                 markersize = 4 .+ 30 .* pos_weights[t], markerstrokecolor = :black)
    elseif stage == :resample
        ## Survivors at full size; the ones no one chose struck through.
        ## A copied particle sits exactly on its parent, so nudge duplicates apart:
        ## "good ones copied" is the point of this frame and invisible otherwise.
        offsets = zeros(N_PART)
        for v in unique(pos_after[t])
            same = findall(==(v), pos_after[t])
            length(same) > 1 && (offsets[same] = range(-0.07, 0.07, length = length(same)))
        end
        scatter!(p, t .+ offsets, pos_after[t], color = :gold, markersize = 9,
                 markerstrokecolor = :black)
        ## The ones nobody chose, struck through on top so they are unmissable.
        died = setdiff(1:N_PART, unique(pos_parents[t]))
        scatter!(p, fill(t, length(died)), pos_before[t][died], color = :firebrick,
                 markersize = 11, markershape = :xcross, markerstrokewidth = 3,
                 markerstrokecolor = :firebrick)
    end

    ## The data last, so points sit on top.
    scatter!(p, 1:T_OBS, TOY_DATA, color = :firebrick, markershape = :diamond,
             markersize = 9, markerstrokecolor = :black)

    label = Dict(
        :initialise => "Initialise:  draw J particles from p(x₀ | θ), each of weight 1/J",
        :propagate => "Propagate:  simulate each particle to observation $t",
        :weight => "Weight:  by w = p(y | x, θ), drawn as size",
        :resample => "Resample:  in proportion to weight",
    )[stage]
    title!(p, label)

    return p
end

frames = [(:initialise, 1)]
for t in 1:T_OBS
    push!(frames, (:propagate, t), (:weight, t), (:resample, t))
end

for (i, (stage, t)) in enumerate(frames)
    save_figure(toy_frame(stage, t), @sprintf("pf_walk_%02d.svg", i))
end

println("Wrote $(length(frames)) pf_walk_*.svg frames to sessions/slides/images/")
