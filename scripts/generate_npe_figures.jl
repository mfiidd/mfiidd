# Generate the figures shown in sessions/slides/neural_posterior_estimation.qmd.
# Run with: julia --project=. scripts/generate_npe_figures.jl
#
# The SVGs under sessions/slides/images/ are OUTPUTS of this script, not
# sources. Re-running it overwrites every one of them. The seeds below are what
# keep the committed figures stable, so change them only if you mean to
# regenerate the lot and commit the new files.
#
# The deck is precomputed rather than executed because every figure below sits
# downstream of a four-thousand-step flow training run, and CONTRIBUTING.md
# keeps anything that slow out of the render path.
#
# The simulator, priors, transform and flow are the session's own, copied here
# so the deck shows the figures the practical goes on to produce. If
# sessions/neural_posterior_estimation.qmd changes any of them, change it here
# and re-run.

using CSV
using DataFrames
using Distributions
using DrWatson
using LinearAlgebra
using Lux
using MFIIDD ## the course package, which holds the SEIT4L Gillespie stepper
using Optimisers
using Plots
using Random
using Statistics
using StatsPlots
using Zygote

ENV["GKSwstype"] = "100"  ## no display during rendering

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

const NPE_COLOUR = :seagreen
const PMMH_COLOUR = :steelblue
const DATA_COLOUR = :black
const RESIM_COLOUR = :darkorange

"""
    save_figure(plt, name)

Write `plt` to `sessions/slides/images/<name>`.

GR numbers the clip-path ids in its SVG from a counter that keeps running for
the length of the session, so the same picture picks up different ids depending
on how many plots came before it. Renumbering the ids in order of first
appearance keeps the committed SVGs byte-stable, so a real diff means a real
change. This is the same helper `generate_observation_model_figures.jl` uses.
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
# The session's own model, priors and flow
# ---------------------------------------------------------------------------

flu_tdc = CSV.read(datadir("flu_tdc_1971.csv"), DataFrame)
const YOBS = Vector{Float64}(flu_tdc.obs)
const T = length(YOBS)
const INIT = [279.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0]   ## S, E, I, T1-T4, L

const PNAMES = [:R_0, :D_lat, :D_inf, :α, :D_imm, :ρ]
const PRIORS = [
    truncated(Normal(3.0, 2.0), lower = 1.0),
    truncated(Normal(2.0, 1.0), lower = 0.5),
    truncated(Normal(3.0, 2.0), lower = 0.5),
    Beta(2, 2),
    truncated(Normal(15.0, 10.0), lower = 1.0),
    Beta(2, 2),
]
const D = length(PNAMES)

function simulate_obs(rng, θ)
    d = Dict{Symbol, Float64}(PNAMES[i] => Float64(θ[i]) for i in 1:D)
    state = copy(INIT)
    y = Vector{Float64}(undef, T)
    for t in 1:T
        state, inc = gillespie_step(rng, state, d, 1.0)
        y[t] = rand(rng, Poisson(max(θ[6] * inc, 1e-10)))
    end
    return y
end

const CLIP = 1e-6
unconstrain(θ) =
    [quantile(Normal(), clamp(cdf(PRIORS[i], θ[i]), CLIP, 1 - CLIP)) for i in 1:D]
constrain(u) =
    [quantile(PRIORS[i], clamp(cdf(Normal(), u[i]), CLIP, 1 - CLIP)) for i in 1:D]

pack(y, m) = vcat(Float32.(log1p.(y)) .* m, m)

function draw_mask(rng)
    m = ones(Float32, T)
    if rand(rng) < 0.5
        nobs = rand(rng, 8:T)
        m[randperm(rng, T)[1:(T - nobs)]] .= 0.0f0
    end
    return m
end

function build_training_set(n; seed = 1)
    rng = Xoshiro(seed)
    Θ = Matrix{Float32}(undef, D, n)
    Y = Matrix{Float32}(undef, 2T, n)
    for j in 1:n
        θ = [rand(rng, PRIORS[i]) for i in 1:D]
        Θ[:, j] = Float32.(unconstrain(θ))
        Y[:, j] = pack(simulate_obs(rng, θ), draw_mask(rng))
    end
    return Θ, Y
end

const FLOW = (couple = 8, hidden = 64, embed = 64)
const MASKS = [Float32[(i + k) % 2 for i in 1:D] for k in 1:FLOW.couple]

function build_flow(rng)
    embed = Chain(Dense(2T => FLOW.embed, gelu), Dense(FLOW.embed => FLOW.embed, gelu))
    nets = Tuple(
        Chain(
            Dense(D + FLOW.embed => FLOW.hidden, gelu),
            Dense(FLOW.hidden => FLOW.hidden, gelu),
            Dense(FLOW.hidden => 2D; init_weight = zeros32, init_bias = zeros32),
        ) for _ in 1:FLOW.couple
    )
    model = (embed = embed, nets = nets)
    return model, Lux.initialparameters(rng, model), Lux.initialstates(rng, model)
end

function coupling(model, ps, st, k, xm, h, mask)
    out, _ = model.nets[k](vcat(xm, h), ps.nets[k], st.nets[k])
    s = 2.0f0 .* tanh.(out[1:D, :]) .* (1 .- mask)
    t = out[(D + 1):(2D), :] .* (1 .- mask)
    return s, t
end

function flow_forward(model, ps, st, x, y)
    h, _ = model.embed(y, ps.embed, st.embed)
    z = x
    logdet = zeros(Float32, size(x, 2))
    for k in 1:FLOW.couple
        mask = MASKS[k]
        xm = z .* mask
        s, t = coupling(model, ps, st, k, xm, h, mask)
        z = xm .+ (1 .- mask) .* (z .* exp.(s) .+ t)
        logdet = logdet .+ vec(sum(s, dims = 1))
    end
    return z, logdet
end

function flow_inverse(model, ps, st, z, y)
    h, _ = model.embed(y, ps.embed, st.embed)
    x = z
    for k in FLOW.couple:-1:1
        mask = MASKS[k]
        xm = x .* mask
        s, t = coupling(model, ps, st, k, xm, h, mask)
        x = xm .+ (1 .- mask) .* ((x .- t) .* exp.(-s))
    end
    return x
end

function objective(model, ps, st, x, y)
    z, logdet = flow_forward(model, ps, st, x, y)
    return (sum(abs2, z) / 2 - sum(logdet)) / (size(x, 2) * D)
end

function train_streaming(model, ps, st, Xva, Yva; steps, batch, lr = 1.0f-3, val_every = 25)
    opt = Optimisers.setup(Optimisers.Adam(lr), ps)
    history = NamedTuple[]
    best = (loss = Inf, step = 0, ps = deepcopy(ps))
    for s in 1:steps
        Xb, Yb = build_training_set(batch; seed = 10_000 + s)
        loss, back = Zygote.pullback(p -> objective(model, p, st, Xb, Yb), ps)
        grads = back(one(loss))[1]
        opt, ps = Optimisers.update(opt, ps, grads)
        if s == steps ÷ 2
            Optimisers.adjust!(opt, lr / 2)
        elseif s == (3 * steps) ÷ 4
            Optimisers.adjust!(opt, lr / 4)
        end
        if s % val_every == 0
            vl = objective(model, ps, st, Xva, Yva)
            push!(history, (step = s, val = vl))
            isfinite(vl) &&
                vl < best.loss &&
                (best = (loss = vl, step = s, ps = deepcopy(ps)))
        end
    end
    return DataFrame(history), best
end

function npe_sample(model, ps, st, y, m, nsamp; seed = 7)
    Yc = repeat(pack(y, m), 1, nsamp)
    Zn = Float32.(randn(Xoshiro(seed), D, nsamp))
    U = flow_inverse(model, ps, st, Zn, Yc)
    return stack(constrain(Float64.(u)) for u in eachcol(U); dims = 1)
end

# ---------------------------------------------------------------------------
# 1. What the training set looks like
# ---------------------------------------------------------------------------

println("1/7 training pairs")

fig_rng = Xoshiro(20260909)
p_pairs = plot(layout = (2, 4), size = (1100, 480), legend = false, link = :all)
for k in 1:8
    θ = [rand(fig_rng, PRIORS[i]) for i in 1:D]
    plot!(p_pairs, 1:T, simulate_obs(fig_rng, θ), subplot = k, lw = 2, colour = NPE_COLOUR)
    plot!(
        p_pairs,
        subplot = k,
        title = "R₀ = $(round(θ[1], digits = 1))",
        titlefontsize = 11,
    )
end
plot!(
    p_pairs,
    plot_title = "Eight draws from the prior, simulated",
    plot_titlefontsize = 15,
)
save_figure(p_pairs, "npe_training_pairs.svg")

# ---------------------------------------------------------------------------
# Train the flow the deck reports on
# ---------------------------------------------------------------------------

println("training the flow (about three minutes)")

Xva, Yva = build_training_set(5_000; seed = 77)
model, ps, st = build_flow(Xoshiro(2024))
history, best = train_streaming(model, ps, st, Xva, Yva; steps = 4_000, batch = 128)
println("  best step $(best.step), loss $(round(best.loss, digits = 4))")

mask_full = ones(Float32, T)
post = npe_sample(model, best.ps, st, YOBS, mask_full, 10_000)
posterior = DataFrame(post, PNAMES)

# ---------------------------------------------------------------------------
# 2. One network, three datasets: what amortisation means
# ---------------------------------------------------------------------------

println("2/7 one network, three datasets")

demo_rng = Xoshiro(11)
demo = []
while length(demo) < 3
    θ = [rand(demo_rng, PRIORS[i]) for i in 1:D]
    θ[1] < 2.5 && continue          ## skip the draws that never take off
    y = simulate_obs(demo_rng, θ)
    sum(y) < 40 && continue
    push!(demo, (θ = θ, y = y))
end

demo_plots = []
for (k, d) in enumerate(demo)
    push!(
        demo_plots,
        plot(
            1:T,
            d.y,
            lw = 2,
            colour = :grey30,
            legend = false,
            title = "dataset $k",
            titlefontsize = 12,
            xlabel = "day",
            ylabel = k == 1 ? "cases" : "",
        ),
    )
end
for (k, d) in enumerate(demo)
    pk = npe_sample(model, best.ps, st, d.y, mask_full, 4_000; seed = 300 + k)
    pl = density(
        pk[:, 1],
        lw = 3,
        colour = NPE_COLOUR,
        fill = (0, 0.2, NPE_COLOUR),
        legend = false,
        xlabel = "R₀",
        ylabel = k == 1 ? "posterior" : "",
        titlefontsize = 12,
    )
    vline!(pl, [d.θ[1]], lw = 3, ls = :dash, colour = DATA_COLOUR)
    push!(demo_plots, pl)
end
save_figure(
    plot(
        demo_plots...,
        layout = (2, 3),
        size = (1100, 560),
        plot_title = "One trained network, three datasets (dashed: the truth)",
        plot_titlefontsize = 15,
    ),
    "npe_amortised.svg",
)

# ---------------------------------------------------------------------------
# 3. Training curve
# ---------------------------------------------------------------------------

println("3/7 training curve")

p_train = plot(
    history.step,
    history.val,
    lw = 3,
    colour = NPE_COLOUR,
    legend = false,
    xlabel = "gradient step",
    ylabel = "validation objective",
    title = "Every batch is freshly simulated, so it plateaus",
    size = (900, 460),
)
vline!(p_train, [best.step], ls = :dash, colour = :grey)
save_figure(p_train, "npe_training.svg")

# ---------------------------------------------------------------------------
# 4. NPE against the PMMH chain
# ---------------------------------------------------------------------------

println("4/7 NPE against PMMH")

pmmh = CSV.read(datadir("pmcmc_seit4l_chain.csv"), DataFrame)
plts = []
for p in PNAMES
    pl = density(
        pmmh[!, p],
        lw = 3,
        colour = PMMH_COLOUR,
        label = "PMMH",
        title = string(p),
        titlefontsize = 12,
    )
    density!(pl, posterior[!, p], lw = 3, ls = :dash, colour = NPE_COLOUR, label = "NPE")
    push!(plts, pl)
end
save_figure(plot(plts..., layout = (2, 3), size = (1100, 560)), "npe_vs_pmmh.svg")

# ---------------------------------------------------------------------------
# 5. Simulation-based calibration
# ---------------------------------------------------------------------------

println("5/7 calibration")

n_sbc, L_sbc = 200, 200
Xs, Ys = build_training_set(n_sbc; seed = 999)
ranks = zeros(Int, n_sbc, D)
for j in 1:n_sbc
    Yc = repeat(Ys[:, j:j], 1, L_sbc)
    Zj = Float32.(randn(Xoshiro(1000 + j), D, L_sbc))
    U = flow_inverse(model, best.ps, st, Zj, Yc)
    for i in 1:D
        ranks[j, i] = count(<(Xs[i, j]), U[i, :])
    end
end

grid = range(0, 1, length = 100)
## a rank takes L_sbc + 1 values, so under calibration P(u <= x) rises in steps
## of 1 / (L_sbc + 1) rather than along the diagonal
p0 = [(floor(Int, L_sbc * x) + 1) / (L_sbc + 1) for x in grid]
lo = [quantile(Binomial(n_sbc, p), 0.025) / n_sbc for p in p0]
hi = [quantile(Binomial(n_sbc, p), 0.975) / n_sbc for p in p0]

sbc_plots = []
for (i, p) in enumerate(PNAMES)
    u = ranks[:, i] ./ L_sbc
    ecdf_vals = [mean(u .<= x) for x in grid]
    pl = plot(
        grid,
        hi .- p0,
        fillrange = lo .- p0,
        fillalpha = 0.2,
        colour = :grey,
        lw = 0,
        legend = false,
        title = string(p),
        titlefontsize = 12,
    )
    plot!(pl, grid, ecdf_vals .- p0, lw = 3, colour = NPE_COLOUR)
    hline!(pl, [0], colour = :black, ls = :dot)
    push!(sbc_plots, pl)
end
save_figure(plot(sbc_plots..., layout = (2, 3), size = (1100, 560)), "npe_sbc.svg")

# ---------------------------------------------------------------------------
# 6. Re-simulated against data-conditioned trajectories
# ---------------------------------------------------------------------------

println("6/7 conditioned trajectories")

ppc_rng = Xoshiro(4242)
ppc = reduce(vcat, [simulate_obs(ppc_rng, post[j, :])' for j in 1:500])

Random.seed!(2)
filtered = map(1:200) do _
    r = posterior[rand(1:nrow(posterior)), :]
    θ = Dict(
        :R_0 => r.R_0,
        :D_lat => r.D_lat,
        :D_inf => r.D_inf,
        :α => r.α,
        :D_imm => r.D_imm,
        :ρ => r.ρ,
    )
    inc = filtered_incidence(θ, flu_tdc.obs, 256)
    [rand(Poisson(max(r.ρ * x, 1e-10))) for x in inc]
end
filt = reduce(hcat, filtered)'

function band!(p, M, label, colour)
    tt = 1:T
    lo_b = [quantile(M[:, t], 0.025) for t in tt]
    hi_b = [quantile(M[:, t], 0.975) for t in tt]
    md = [median(M[:, t]) for t in tt]
    plot!(
        p,
        tt,
        md,
        ribbon = (md .- lo_b, hi_b .- md),
        label = label,
        colour = colour,
        fillalpha = 0.2,
        lw = 3,
    )
end

p_cond = plot(
    xlabel = "day",
    ylabel = "reported cases",
    title = "The flow gives parameters; the path needs a filter",
    size = (1000, 500),
)
band!(p_cond, ppc, "re-simulated", RESIM_COLOUR)
band!(p_cond, filt, "filtered", NPE_COLOUR)
scatter!(p_cond, 1:T, YOBS, colour = DATA_COLOUR, ms = 4, label = "observed")
save_figure(p_cond, "npe_conditioned.svg")

# ---------------------------------------------------------------------------
# 7. The payoff: a posterior per surveillance schedule, for nothing
# ---------------------------------------------------------------------------

println("7/7 thinning the surveillance")

widths = DataFrame(nobs = Int[], param = String[], width = Float64[])
for nobs in [10, 20, 30, 40, 50, T], rep in 1:5
    m = zeros(Float32, T)
    m[randperm(Xoshiro(rep), T)[1:nobs]] .= 1.0f0
    p = npe_sample(model, best.ps, st, YOBS, m, 2_000; seed = 100 + rep)
    for i in 1:D
        push!(
            widths,
            (nobs, string(PNAMES[i]), quantile(p[:, i], 0.95) - quantile(p[:, i], 0.05)),
        )
    end
end

summary_widths = combine(groupby(widths, [:nobs, :param]), :width => mean => :w)

## D_imm is an order of magnitude wider than the rest, so on a shared axis it
## is the only curve you can read. Scale each parameter by its own width under
## the fully observed schedule instead: every curve then starts at 1 on the
## right and shows what thinning costs it.
p_thin = plot(
    xlabel = "number of observed days",
    ylabel = "90% width, relative to fully observed",
    title = "Thirty posteriors, no retraining",
    legend = :topright,
    size = (1000, 500),
)
for p in string.(PNAMES)
    srt = sort(summary_widths[summary_widths.param .== p, :], :nobs)
    plot!(
        p_thin,
        srt.nobs,
        srt.w ./ srt.w[end],
        lw = 3,
        marker = :circle,
        ms = 4,
        label = p,
    )
end
save_figure(p_thin, "npe_thinning.svg")

println("done: seven figures written to sessions/slides/images/")
