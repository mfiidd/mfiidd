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

## Two kinds of antibody response. Strong responders reach a titre that protects
## them; weak responders peak below any useful level
const WEAK_FRACTION = 0.33
const ANTIBODY_PEAK = (8.0, 3.5)        ## log2 titre reached, strong and weak
const ANTIBODY_RISE = 6.0               ## days over which the response builds
const EARLY_SCALE = 16.0                ## how long early protection holds
const SERO_SD = 0.5                     ## assay noise on a log2 titre

"""
    early_protection(tau)

Protection from the cellular response `tau` days after infection. It holds for
about a fortnight and then falls away, which leaves a window before the antibody
response has built.
"""
early_protection(tau) = exp(-(tau / EARLY_SCALE)^4)

"""
    antibody_level(class, tau, rise)

Log2 antibody titre `tau` days after infection, for a strong (`class = 1`) or
weak (`class = 2`) responder. The titre climbs towards the class peak with time
constant `rise`.
"""
function antibody_level(class::Integer, tau, rise)
    tau <= 0 && return zero(typeof(rise + 0.0))
    return ANTIBODY_PEAK[class] * (1 - exp(-tau / rise))
end

"""
    true_protection(titre)

The protection curve used to generate the data, and which the session pretends
not to know: protection switches on over a narrow range around a log2 titre of
5.5, so strong responders end up protected and weak responders do not.
"""
true_protection(titre) = 1 / (1 + exp(-(titre - 5.5) / 0.35))

"""
    outbreak(rise, protection; r0, seed_infections)

Run the island day by day. `protection(titre)` gives the protection a
participant has at that titre, and susceptibility to reinfection combines it
with the early response:

    susceptibility(tau) = (1 - early_protection(tau)) * (1 - protection(antibody))

Returns daily infections, the part of them that are second infections, and the
cohorts of once-infected people indexed by the day they were infected.
"""
function outbreak(rise, protection; r0::Real = TDC_R0, seed_infections = (2.0, 2.0))
    T = typeof(protection(1.0) + rise + 0.0)
    days = TDC_DAYS + 1

    naive = T[(1 - WEAK_FRACTION) * TDC_N, WEAK_FRACTION * TDC_N]
    cohort = [zeros(T, days), zeros(T, days)]   ## infected once, by day of infection
    infections = zeros(T, days)
    second = zeros(T, days)

    for (day, seeded) in enumerate(seed_infections)
        infections[day] = seeded
        for c in 1:2
            share = c == 1 ? 1 - WEAK_FRACTION : WEAK_FRACTION
            cohort[c][day] += seeded * share
            naive[c] -= seeded * share
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
        for c in 1:2
            new = min(hazard, 1.0) * naive[c]
            naive[c] -= new
            cohort[c][t] += new
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

    return (infections = infections, second = second, cohort = cohort)
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
    draw_sera(cohort, rise; n, bleed_days, seed)

Take blood from `n` islanders on each day in `bleed_days`, sampling people in
proportion to the size of the cohorts they belong to. Each sample records the
class, the day that person was infected, the day of the bleed, and the measured
log2 titre. A bleed only happens after the person was infected.
"""
function draw_sera(
    cohort,
    rise;
    n::Integer = 80,
    bleed_days = (20, TDC_DAYS + 1),
    seed::Integer = 5,
)
    rng = Xoshiro(seed)
    pool =
        [(c, d, cohort[c][d]) for c in 1:2 for d in 1:(TDC_DAYS + 1) if cohort[c][d] > 0.05]
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
    network_protection(titre, p, network, state)

Evaluate the network as a scalar function: a log2 titre in, a protection between
zero and one out.
"""
network_protection(titre, p, network, state) = first(first(network([titre / 4], p, state)))

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
