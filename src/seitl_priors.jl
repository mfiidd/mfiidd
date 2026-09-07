using Distributions: Beta, Normal, truncated
using Turing: @model

"""
    SEITL_PRIORS

Weakly informative priors on the six SEITL and SEIT4L parameters, in the order
`[R_0, D_lat, D_inf, α, D_imm, ρ]`.

Each is centred on what the influenza literature reports, and its spread sets
how far the parameter may travel before the data have to argue for it. The
central 95% of each is `R_0` 1.2 to 7.1, `D_lat` 0.7 to 4 days, `D_inf` 0.7 to
7 days and `D_imm` 2 to 35 days. The truncation at 1 on `R_0` is a mechanical
bound, because below 1 there is no epidemic to fit. `Beta(2, 2)` covers the
whole unit interval while pulling gently away from 0 and 1.

Turing models take these through [`seitl_priors`](@ref). Code that draws from
the priors or inverts their CDFs, rather than writing `~` statements, takes the
distributions directly as `collect(values(SEITL_PRIORS))`.

!!! warning "Changing these invalidates the committed chains"
    `data/pmcmc_seitl_chain.csv` and `data/pmcmc_seit4l_chain.csv` were
    generated under these priors. Changing any of them means re-running
    `scripts/generate_pmcmc_seitl.jl` and `scripts/generate_pmcmc_seit4l.jl`
    and committing the new output, which takes around fifteen hours apiece.
"""
const SEITL_PRIORS = (
    R_0 = truncated(Normal(3.0, 2.0), lower = 1.0),
    D_lat = truncated(Normal(2.0, 1.0), lower = 0.5),
    D_inf = truncated(Normal(3.0, 2.0), lower = 0.5),
    α = Beta(2, 2),
    D_imm = truncated(Normal(15.0, 10.0), lower = 1.0),
    ρ = Beta(2, 2),
)

"""
    seitl_priors()

The [`SEITL_PRIORS`](@ref) as a Turing submodel, returning the six sampled
parameters as a named tuple.

Use it from a model with the two-argument `to_submodel`, which suppresses the
name prefixing that would otherwise turn `R_0` into `priors.R_0` in the chain:

```julia
priors ~ to_submodel(seitl_priors(), false)
```

The parameters then appear in the chain under their own names, so `chain[:R_0]`
works exactly as it does when the `~` statements are written out in the model.
"""
@model function seitl_priors()
    R_0 ~ SEITL_PRIORS.R_0
    D_lat ~ SEITL_PRIORS.D_lat
    D_inf ~ SEITL_PRIORS.D_inf
    α ~ SEITL_PRIORS.α
    D_imm ~ SEITL_PRIORS.D_imm
    ρ ~ SEITL_PRIORS.ρ
    return (; R_0, D_lat, D_inf, α, D_imm, ρ)
end
