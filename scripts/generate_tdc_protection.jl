# Profile the protection curve against the assumed antibody rise time, with and
# without the serosurvey in the likelihood. Ten fits, a few minutes in total; the
# session loads the result from `data/tdc_protection_profile.csv`.
#
# Run with: julia --project=. scripts/generate_tdc_protection.jl

using MFIIDD
using CSV
using DataFrames
using DrWatson

run = outbreak(ANTIBODY_RISE, true_protection)
cases = report_cases(run.infections)
sera = draw_sera(run.infected, ANTIBODY_RISE)
## the same islanders bled only once, at the end of the outbreak
sera_end = draw_sera(run.infected, ANTIBODY_RISE; bleed_days = (TDC_DAYS + 1,))

network = protection_network()
p_init, network_state = protection_weights(network)

"""Negative log posterior of the case series, with whichever sera are included."""
function objective(p, rise, samples)
    total =
        -case_loglik(
            outbreak(rise, protection_from(p, network, network_state)).infections,
            cases,
        )
    samples === nothing || (total -= sero_loglik(samples, rise))
    return total
end

rows = DataFrame(
    rise = Float64[],
    data = String[],
    loss = Float64[],
    index = Int[],
    value = Float64[],
)

for rise in (3.0, 4.5, 6.0, 8.0, 11.0),
    (label, samples) in
    (("cases", nothing), ("serial sera", sera), ("final sera", sera_end))

    p, loss, _ = fit_protection(p -> objective(p, rise, samples), network)
    for (i, v) in enumerate(vec(p))
        push!(rows, (rise, label, loss, i, v))
    end
    println("rise ", rise, ", ", label, ": loss ", round(loss, digits = 1))
end

CSV.write(datadir("tdc_protection_profile.csv"), rows)
println("wrote ", datadir("tdc_protection_profile.csv"))

## An ensemble, for uncertainty on the learned curve: refit to resampled case
## counts, holding the rise time at the value the serial sera pick. The sera
## enter through that rise time; they say nothing about the protection curve
## directly, because a titre measurement does not record whether the person was
## reinfected
using Random, Distributions
ensemble = DataFrame(replicate = Int[], index = Int[], value = Float64[])
for replicate in 1:8
    rng = Xoshiro(200 + replicate)
    cases_b = [rand(rng, Poisson(max(Float64(c), 0.0))) for c in cases]
    loss_b(p) =
        -case_loglik(
            outbreak(ANTIBODY_RISE, protection_from(p, network, network_state)).infections,
            cases_b,
        )
    p, _, _ = fit_protection(loss_b, network)
    for (i, v) in enumerate(vec(p))
        push!(ensemble, (replicate, i, v))
    end
    println("ensemble member ", replicate, ": loss ", round(loss_b(p), digits = 1))
end
CSV.write(datadir("tdc_protection_ensemble.csv"), ensemble)
println("wrote ", datadir("tdc_protection_ensemble.csv"))
