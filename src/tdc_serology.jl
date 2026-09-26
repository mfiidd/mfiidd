# =============================================================================
# Tristan da Cunha with antibody dynamics
# =============================================================================
#
# The outbreak the course has fitted since the model checking session, extended
# so that protection against reinfection depends on a participant's antibody
# rather than on time spent in a compartment. The universal differential
# equations session learns that dependence.

const TDC_N = 284.0                     ## islanders
const TDC_DAYS = 59                     ## days of the reported outbreak
const TDC_R0 = 9.0                      ## basic reproduction number
const TDC_REPORTING = 0.85              ## share of attacks that were dated

## Generation interval for influenza, mean about 3.2 days
const TDC_GENERATION = [0.0, 0.15, 0.35, 0.28, 0.14, 0.06, 0.02]

## Three kinds of antibody response, in the proportions a population might show:
## most islanders reach a protective titre, some reach an intermediate one, and a
## fifth mount little response at all
const RESPONSE_FRACTIONS = (0.62, 0.18, 0.20)
const ANTIBODY_PEAK = (8.0, 4.5, 2.0)   ## log2 titre reached by each kind
const ANTIBODY_RISE = 6.0               ## days over which the response builds
const EARLY_SCALE = 14.0                ## how long early protection holds
const SERO_SD = 0.5                     ## assay noise on a log2 titre

"""
    early_protection(tau)

Protection from the cellular response `tau` days after infection. It holds for
about a fortnight and then falls away, which leaves a window before the antibody
response has built.
"""
early_protection(tau) = exp(-(tau / EARLY_SCALE)^6)

"""
    antibody_level(class, tau, rise)

Log2 antibody titre `tau` days after infection for a responder of the given
class, numbered from the strongest. The titre climbs towards the class peak with
time constant `rise`.
"""
function antibody_level(class::Integer, tau, rise)
    tau <= 0 && return zero(typeof(rise + 0.0))
    return ANTIBODY_PEAK[class] * (1 - exp(-tau / rise))
end

"""
    true_protection(titre)

The protection curve used to simulate the data, and which the session pretends
not to know. It rises in two steps: partial protection appears at low titres,
and full protection arrives only above about 1:100. An intermediate responder
therefore sits on the shoulder, protected enough to delay a second attack and
not enough to prevent one.
"""
true_protection(titre) =
    0.5 / (1 + exp(-(titre - 3.0) / 0.4)) + 0.5 / (1 + exp(-(titre - 6.5) / 0.4))

"""
    outbreak(rise, protection; r0, seed_infections)

Run the island day by day. `protection(titre)` gives the protection a
participant has at that titre, and susceptibility to reinfection combines it
with the early response:

    susceptibility(tau) = (1 - early_protection(tau)) * (1 - protection(antibody))

Returns daily infections, the part of them that are second infections, and a
record of first infections by class and by day, which is who a serosurvey can
sample from.
"""
function outbreak(rise, protection; r0::Real = TDC_R0, seed_infections = (2.0, 2.0))
    T = typeof(protection(1.0) + rise + 0.0)
    days = TDC_DAYS + 1

    classes = eachindex(RESPONSE_FRACTIONS)
    naive = T[share * TDC_N for share in RESPONSE_FRACTIONS]
    cohort = [zeros(T, days) for _ in classes]  ## still on their first infection
    infected = [zeros(T, days) for _ in classes] ## everyone, by day of first infection
    infections = zeros(T, days)
    second = zeros(T, days)

    for (day, seeded) in enumerate(seed_infections)
        infections[day] = seeded
        for c in classes
            arriving = seeded * RESPONSE_FRACTIONS[c]
            cohort[c][day] += arriving
            infected[c][day] += arriving
            naive[c] -= arriving
        end
    end

    for t in (length(seed_infections) + 1):days
        force = zero(T)
        for (gap, weight) in enumerate(TDC_GENERATION)
            t - gap >= 1 && (force += weight * infections[t - gap])
        end
        hazard = r0 / TDC_N * force

        first_time = zero(T)
        again = zero(T)
        for c in classes
            new = min(hazard, 1.0) * naive[c]
            naive[c] -= new
            cohort[c][t] += new
            infected[c][t] += new
            first_time += new

            for d in 1:(t - 1)
                exposed = cohort[c][d]
                titre = antibody_level(c, t - d, rise)
                susceptibility = (1 - early_protection(t - d)) * (1 - protection(titre))
                reinfected = min(hazard * susceptibility, 1.0) * exposed
                cohort[c][d] -= reinfected
                again += reinfected
            end
        end
        infections[t] = first_time + again
        second[t] = again
    end

    return (infections = infections, second = second, cohort = cohort, infected = infected)
end

"""
    report_cases(infections; seed, reporting)

The dated case series: each infection is reported with probability `reporting`,
and the counts are Poisson.
"""
function report_cases(infections; seed::Integer = 4, reporting::Real = TDC_REPORTING)
    rng = Xoshiro(seed)
    return [rand(rng, Poisson(max(reporting * value, 0.0))) for value in infections]
end

"""
    draw_sera(infected, rise; n, bleed_days, seed)

Take blood from `n` islanders on each day in `bleed_days`, sampling in proportion
to the number first infected on each day. Each sample records the class, the day
that person was first infected, the day of the bleed, and the measured log2
titre. A bleed only happens after the person was infected.

Islanders who were infected twice are sampled alongside the rest, and their
titre is taken from their first infection. A second infection boosts them
towards the same class peak, so the approximation costs little here.
"""
function draw_sera(
    infected,
    rise;
    n::Integer = 80,
    bleed_days = (20, TDC_DAYS + 1),
    seed::Integer = 5,
)
    rng = Xoshiro(seed)
    pool = [
        (c, d, infected[c][d]) for c in eachindex(infected) for
        d in 1:(TDC_DAYS + 1) if infected[c][d] > 0.05
    ]
    weights = [entry[3] for entry in pool]
    picks = rand(rng, Categorical(weights ./ sum(weights)), n)

    samples = NTuple{4, Float64}[]
    for i in picks, bleed in bleed_days
        class, infected_on, _ = pool[i]
        tau = bleed - infected_on
        tau <= 0 && continue
        titre = antibody_level(class, tau, rise) + SERO_SD * randn(rng)
        push!(samples, (Float64(class), Float64(infected_on), Float64(bleed), titre))
    end
    return samples
end

"""
    case_loglik(infections, cases; reporting)

Poisson log likelihood of the dated case series.
"""
function case_loglik(infections, cases; reporting::Real = TDC_REPORTING)
    total = zero(eltype(infections))
    for (predicted, observed) in zip(infections, cases)
        rate = max(reporting * predicted, 1e-8)
        total += observed * log(rate) - rate
    end
    return total
end

"""
    sero_loglik(samples, rise)

Log likelihood of the measured titres under an antibody response that builds
with time constant `rise`.
"""
function sero_loglik(samples, rise)
    total = zero(typeof(rise + 0.0))
    for (class, infected_on, bleed, titre) in samples
        expected = antibody_level(Int(class), bleed - infected_on, rise)
        total -= (expected - titre)^2 / (2 * SERO_SD^2)
    end
    return total
end

"""
    protection_network()

The network standing in for the protection curve: one input, eight hidden units,
and a logistic output, which keeps protection between zero and one.
"""
protection_network() = Chain(Dense(1 => 8, tanh), Dense(8 => 1, sigmoid))

"""
    protection_weights(network; seed)

Starting weights in Float64, to match the rest of the model.
"""
function protection_weights(network; seed::Integer = 1)
    weights, state = Lux.setup(Xoshiro(seed), network)
    return ComponentArray{Float64}(weights), state
end

"""
    monotone(p)

Take the absolute value of every weight, which makes the network increasing in
its input. Protection cannot fall as antibody rises, and a network free to
learn that it does has a second optimum in which the strong responders are the
susceptible ones. The biology rules that out, so we build it into the
architecture.
"""
monotone(p) = ComponentArray(
    layer_1 = (weight = abs.(p.layer_1.weight), bias = p.layer_1.bias),
    layer_2 = (weight = abs.(p.layer_2.weight), bias = p.layer_2.bias),
)

"""
    protection_from(p, network, state)

The protection curve for weights `p`, as a scalar function of a log2 titre. The
monotone transform is applied once here, because a single epidemic evaluates the
curve a few thousand times.
"""
function protection_from(p, network, state)
    weights = monotone(p)
    return titre -> first(first(network([titre / 4], weights, state)))
end

"""
    network_protection(titre, p, network, state)

Evaluate the curve at one titre: a log2 titre in, a protection between zero and
one out, increasing in the titre.
"""
network_protection(titre, p, network, state) = protection_from(p, network, state)(titre)

"""
    train_protection(loss, p0; iterations, rates)

Minimise `loss` with Adam in stages, cutting the learning rate between them.
"""
function train_protection(
    loss,
    p0;
    iterations::Tuple = (600, 400),
    rates::Tuple = (0.02, 0.005),
)
    history = Float64[]
    callback = (state, l) -> (push!(history, l); false)
    opt_fn = OptimizationFunction((p, _) -> loss(p), Optimization.AutoForwardDiff())
    p = p0
    for (n, rate) in zip(iterations, rates)
        result = solve(
            OptimizationProblem(opt_fn, p),
            Optimisers.Adam(rate);
            maxiters = n,
            callback,
        )
        p = result.u
    end
    return p, history
end

"""
    fit_protection(loss, network; seeds, iterations, rates)

Fit from several starting points and keep the best.

Adam on this loss is sensitive to where it starts, so a single run can stop at a
curve a few log likelihood units short of the one the data support.
"""
function fit_protection(loss, network; seeds = 1:3, kwargs...)
    best_p = nothing
    best_loss = Inf
    best_history = Float64[]
    for seed in seeds
        start, _ = protection_weights(network; seed)
        p, history = train_protection(loss, start; kwargs...)
        value = loss(p)
        if value < best_loss
            best_p, best_loss, best_history = p, value, history
        end
    end
    return best_p, best_loss, best_history
end
