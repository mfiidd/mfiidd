# Generate the figures shown in sessions/slides/pmcmc.qmd.
# Run with: julia --project=. scripts/generate_pmcmc_figures.jl
#
# The SVGs under sessions/slides/images/ are OUTPUTS of this script, not
# sources. Re-running it overwrites every one of them. The seed below is what
# keeps the committed figures stable, so change it only if you mean to
# regenerate the lot and commit the new files.
#
# The deck is precomputed rather than executed because the last figure runs two
# PMMH chains, and CONTRIBUTING.md keeps anything that slow out of the render
# path.
#
# The model, priors and initial state come from scripts/pmmh_setup.jl, which is
# what generated the committed chains in data/. Including it rather than copying
# the six `~` statements keeps the deck's figures and the session's posteriors
# describing the same model.

include(joinpath(@__DIR__, "pmmh_setup.jl"))

using Plots
using Printf
using LinearAlgebra: Symmetric

ENV["GKSwstype"] = "100"  ## no display during rendering
Random.seed!(20260908)

const IMAGE_DIR = joinpath(@__DIR__, "..", "sessions", "slides", "images")

## The trace figure runs two PMMH chains with the committed chains' warmup
## budget, which takes the best part of an hour and wants the machine to itself.
## Name stages on the command line to run them separately:
##
##     julia --project=. scripts/generate_pmcmc_figures.jl trace
##
## With no argument every stage runs.
const STAGES = isempty(ARGS) ? ["noise", "tradeoff", "trace"] : ARGS

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

"""
    save_figure(plt, name)

Write `plt` to `sessions/slides/images/<name>`.

GR numbers the clip-path ids in its SVG from a counter that keeps running for
the length of the session, so the same picture picks up different ids depending
on how many plots came before it. Left alone, that makes every re-run of this
script show as a diff on every file even when nothing about the figures
changed. Renumbering the ids in order of first appearance keeps the committed
SVGs byte-stable, so a real diff means a real change.
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

const OBS = flu_observations()

## A representative θ: the posterior mean of the committed SEIT4L chain. The
## spread of the estimator depends on where in parameter space it is measured,
## and the posterior mean is where the chain actually spends its time.
const THETA = let df = CSV.read(datadir("pmcmc_seit4l_chain.csv"), DataFrame)
    Dict(p => mean(df[!, p]) for p in PARAMETERS)
end

"""
    log_lik_replicates(n_particles, n_reps)

Run the bootstrap filter `n_reps` times at `THETA` and return the log-likelihood
each run reported. Same θ, same data, different random numbers every time.
"""
function log_lik_replicates(n_particles, n_reps)
    return [run_particle_filter(THETA, OBS, n_particles) for _ in 1:n_reps]
end

"""
    reference_log_lik(n_particles, n_reps)

Estimate ``\\log p(y \\mid \\theta)`` by averaging the estimator on its own
scale before taking the log, which is the scale on which it is unbiased.
Averaging the log-likelihoods instead would return something systematically too
low, by Jensen's inequality, which is the whole reason the reference line has to
be computed this way.
"""
function reference_log_lik(n_particles, n_reps)
    ll = log_lik_replicates(n_particles, n_reps)
    m = maximum(ll)
    return m + log(mean(exp.(ll .- m)))
end

# ---------------------------------------------------------------------------
# 1. The estimator is noisy: same θ, different answer every run
# ---------------------------------------------------------------------------

if "noise" in STAGES

    ## 64 against 256: both land on a readable common axis, and the pair brackets
    ## the point where the estimator enters the band the next figure shades. Below
    ## about 32 particles the estimator is heavy-tailed enough that a single unlucky
    ## run sets the axis and the picture stops being about the bulk of the runs.
    const NOISE_PARTICLES = [64, 256]
    const N_REPS = 400

    reference = reference_log_lik(4096, 200)

    noise_replicates = Dict(J => log_lik_replicates(J, N_REPS) for J in NOISE_PARTICLES)

    ## The estimator has a long left tail: a run in which no particle stays near the
    ## data returns a log-likelihood far below the rest. Letting those few runs set
    ## the axis compresses the other 99% into a couple of pixels, so the window is
    ## set from the 1st percentile of the noisier sample and the runs that fall
    ## outside it are reported on the plot rather than quietly dropped.
    window_lo = minimum(quantile(noise_replicates[J], 0.01) for J in NOISE_PARTICLES)
    window_hi = maximum(maximum(noise_replicates[J]) for J in NOISE_PARTICLES) + 0.3
    edges = range(window_lo, window_hi, length = 45)
    n_outside = sum(count(<(window_lo), noise_replicates[J]) for J in NOISE_PARTICLES)

    p_noise = plot(
        xlabel = "Estimated log-likelihood at one θ",
        ylabel = "Runs",
        size = (900, 460),
        legend = :topleft,
        xlims = (window_lo, window_hi),
    )

    for (J, colour) in zip(NOISE_PARTICLES, (:firebrick, :steelblue))
        ll = noise_replicates[J]
        histogram!(
            p_noise,
            ll,
            bins = edges,
            alpha = 0.55,
            color = colour,
            linecolor = colour,
            label = @sprintf("%d particles (SD %.1f)", J, std(ll)),
        )
    end

    annotate!(
        p_noise,
        window_lo + 0.03 * (window_hi - window_lo),
        0.0,
        text("$n_outside runs fell below this axis", 10, :left, :bottom, :grey30),
    )

    vline!(
        p_noise,
        [reference],
        color = :black,
        linewidth = 3,
        linestyle = :dash,
        label = "log p(y | θ)",
    )

    save_figure(p_noise, "pmmh_likelihood_noise.svg")
end

# ---------------------------------------------------------------------------
# 2. The trade-off: noise falls with J, cost rises with J
# ---------------------------------------------------------------------------

if "tradeoff" in STAGES
    const TRADEOFF_PARTICLES = [8, 16, 32, 64, 128, 256, 512, 1024]
    const N_REPS_TRADEOFF = 150

    tradeoff_sd = Float64[]
    tradeoff_ms = Float64[]

    for J in TRADEOFF_PARTICLES
        t = @elapsed ll = log_lik_replicates(J, N_REPS_TRADEOFF)
        push!(tradeoff_sd, std(ll))
        push!(tradeoff_ms, 1000 * t / N_REPS_TRADEOFF)
    end

    ## Powers of two are the grid, but a projected slide wants the counts spelled
    ## out rather than 2^4 and 2^8.
    const TRADEOFF_TICKS = ([8, 32, 128, 512], ["8", "32", "128", "512"])

    p_sd = plot(
        TRADEOFF_PARTICLES,
        tradeoff_sd,
        xscale = :log2,
        xticks = TRADEOFF_TICKS,
        xlabel = "Particles",
        ylabel = "SD of log-likelihood estimate",
        marker = :circle,
        markersize = 6,
        linewidth = 3,
        color = :steelblue,
        label = "",
    )

    ## The band Pitt et al. and Doucet et al. recommend aiming for.
    hspan!(p_sd, [1.0, 3.0], color = :seagreen, alpha = 0.18, label = "Target: 1 to 3")

    ## Log on both axes, where cost linear in the particle count is a straight line.
    ## On a linear y axis the whole curve hugs zero until the last two points and
    ## the slide looks as though particles were free up to 256.
    p_cost = plot(
        TRADEOFF_PARTICLES,
        tradeoff_ms,
        xscale = :log2,
        yscale = :log10,
        xticks = TRADEOFF_TICKS,
        xlabel = "Particles",
        ylabel = "Time per filter run (ms)",
        marker = :circle,
        markersize = 6,
        linewidth = 3,
        color = :firebrick,
        label = "",
    )

    p_tradeoff = plot(p_sd, p_cost, layout = (1, 2), size = (1100, 460))
    save_figure(p_tradeoff, "pmmh_particle_tradeoff.svg")
end

# ---------------------------------------------------------------------------
# 3. What that noise does to the chain
# ---------------------------------------------------------------------------

## The saved chain already paid for the tuning. Its 50,000 warmup iterations
## bought an adapted proposal, and the posterior draws carry that information:
## the scaled posterior covariance is the proposal RAM was converging on. Seeding
## a fixed random walk with it means no warmup here at all, which is what makes
## this figure cheap enough to regenerate.
##
## Fixing the proposal also makes the comparison a clean experiment. Under RAM
## the two panels would adapt differently and the difference between them would
## confound the particle count with the adaptation. Here the proposal is
## identical in both, so the only thing that varies is the number of particles.
const TRACE_ITERATIONS = 8_000

"""
    seeded_proposal()

A random walk tuned from the committed SEIT4L chain: the posterior covariance on
the unconstrained scale Turing samples on, scaled by the usual `2.38^2 / d`.

The six transforms are the priors' own bijections — `log(x - a)` for the
truncated normals, `logit` for the two Beta parameters. The saved draws are on
the constrained scale, and a covariance taken there would describe the wrong
space.
"""
function seeded_proposal()
    df = CSV.read(datadir("pmcmc_seit4l_chain.csv"), DataFrame)
    u = hcat(
        log.(df.R_0 .- 1.0),
        log.(df.D_lat .- 0.5),
        log.(df.D_inf .- 0.5),
        log.(df.α ./ (1 .- df.α)),
        log.(df.D_imm .- 1.0),
        log.(df.ρ ./ (1 .- df.ρ)),
    )
    Σ = (2.38^2 / size(u, 2)) .* cov(u)
    return AdvancedMH.RandomWalkProposal(MvNormal(zeros(size(u, 2)), Symmetric(Σ)))
end

"""
    short_chain(n_particles)

Run PMMH at `n_particles` with the seeded proposal and return the draws of R_0
together with the proportion of iterations at which the chain moved. Short
enough to be a picture of the mixing rather than a fit: the committed chains in
data/ are what the session reads for inference, and this cannot replace them,
being far too short and never having adapted anything of its own.
"""
function short_chain(n_particles)
    chain = sample(
        pmmh_seit4l(OBS, n_particles),
        externalsampler(AdvancedMH.MetropolisHastings(seeded_proposal())),
        TRACE_ITERATIONS;
        check_model = false,
        progress = false,
    )
    return chain_frame(chain).R_0, acceptance_rate(chain)
end

if "trace" in STAGES
    trace_few, accept_few = short_chain(16)
    trace_enough, accept_enough = short_chain(256)

    ## The acceptance rate is the claim the picture is making. Printing it on each
    ## panel means a chance-flat stretch in the lower trace cannot make the figure
    ## say the opposite of the slide.
    p_few = plot(
        trace_few,
        ylabel = "R₀",
        title = @sprintf("16 particles — %.0f%% accepted", 100 * accept_few),
        color = :firebrick,
        linewidth = 1.5,
        label = "",
    )

    p_enough = plot(
        trace_enough,
        xlabel = "Iteration",
        ylabel = "R₀",
        title = @sprintf("256 particles — %.0f%% accepted", 100 * accept_enough),
        color = :steelblue,
        linewidth = 1.5,
        label = "",
    )

    ## A shared y axis, so the flat stretches in the top panel read as flat rather
    ## than as a chain exploring a narrower range.
    ylims = extrema(vcat(trace_few, trace_enough))
    p_sticky = plot(
        p_few,
        p_enough,
        layout = (2, 1),
        size = (1000, 620),
        ylims = ylims .+ (-0.1, 0.1) .* (ylims[2] - ylims[1]),
    )

    save_figure(p_sticky, "pmmh_sticky_trace.svg")
end

if "noise" in STAGES
    println("Reference log-likelihood: ", round(reference, digits = 2))
end
if "tradeoff" in STAGES
    for (J, sd, ms) in zip(TRADEOFF_PARTICLES, tradeoff_sd, tradeoff_ms)
        @printf("  J = %5d   SD %7.2f   %7.1f ms/run\n", J, sd, ms)
    end
end
if "trace" in STAGES
    @printf(
        "Acceptance: 16 particles %.1f%%, 256 particles %.1f%%\n",
        100 * accept_few,
        100 * accept_enough
    )
end
println("Written to ", IMAGE_DIR)
