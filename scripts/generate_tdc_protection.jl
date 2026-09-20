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
sera = draw_sera(run.cohort, ANTIBODY_RISE)
## the same islanders bled only once, at the end of the outbreak
sera_end = draw_sera(run.cohort, ANTIBODY_RISE; bleed_days = (TDC_DAYS + 1,))

network = protection_network()
p_init, network_state = protection_weights(network)

"""Negative log posterior of the case series, with whichever sera are included."""
function objective(p, rise, samples)
    protection(titre) = network_protection(titre, p, network, network_state)
    total = -case_loglik(outbreak(rise, protection).infections, cases)
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

    p, _ = train_protection(p -> objective(p, rise, samples), p_init)
    loss = objective(p, rise, samples)
    for (i, v) in enumerate(vec(p))
        push!(rows, (rise, label, loss, i, v))
    end
    println("rise ", rise, ", ", label, ": loss ", round(loss, digits = 1))
end

CSV.write(datadir("tdc_protection_profile.csv"), rows)
println("wrote ", datadir("tdc_protection_profile.csv"))
