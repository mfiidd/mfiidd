using Distributions: Beta, Normal, truncated
using Turing: @model, DynamicPPL

"""
    seitl_priors()

Weakly informative priors on the six SEITL and SEIT4L parameters, as a Turing
submodel returning them as a named tuple.

Each is centred on what the influenza literature reports, and its spread sets
how far the parameter may travel before the data have to argue for it. The
central 95% of each is `R_0` 1.2 to 7.1, `D_lat` 0.7 to 4 days, `D_inf` 0.7 to
7 days and `D_imm` 2 to 35 days. The truncation at 1 on `R_0` is a mechanical
bound, because below 1 there is no epidemic to fit. `Beta(2, 2)` covers the
whole unit interval while pulling gently away from 0 and 1.

Use it from a model with the two-argument `to_submodel`, which suppresses the
name prefixing that would otherwise turn `R_0` into `priors.R_0` in the chain:

```julia
priors ~ to_submodel(seitl_priors(), false)
```

The parameters then appear in the chain under their own names, so `chain[:R_0]`
works exactly as it does when the `~` statements are written out in the model.
Code that draws from the priors or inverts their CDFs, rather than writing `~`
statements, takes the distributions through [`seitl_prior_distributions`](@ref).

!!! warning "Changing these invalidates the committed chains"
    `data/pmcmc_seitl_chain.csv` and `data/pmcmc_seit4l_chain.csv` were
    generated under these priors. Changing any of them means re-running
    `scripts/generate_pmcmc_seitl.jl` and `scripts/generate_pmcmc_seit4l.jl`
    and committing the new output, which takes around fifteen hours apiece.
"""
@model function seitl_priors()
    R_0 ~ truncated(Normal(3.0, 2.0), lower = 1.0)
    D_lat ~ truncated(Normal(2.0, 1.0), lower = 0.5)
    D_inf ~ truncated(Normal(3.0, 2.0), lower = 0.5)
    α ~ Beta(2, 2)
    D_imm ~ truncated(Normal(15.0, 10.0), lower = 1.0)
    ρ ~ Beta(2, 2)
    return (; R_0, D_lat, D_inf, α, D_imm, ρ)
end

"""
    SEITL_PARAMETERS

The SEITL parameter names, in the order the sessions expect them.
"""
const SEITL_PARAMETERS = (:R_0, :D_lat, :D_inf, :α, :D_imm, :ρ)

"""
    seitl_prior_distributions()

The distributions [`seitl_priors`](@ref) samples from, as a vector ordered by
[`SEITL_PARAMETERS`](@ref).

The submodel is the definition; this reads the distributions back out of it, so
there is no second copy of the numbers to drift. Use it where the priors are
needed as objects to `rand`, `cdf` or `quantile` over rather than as `~`
statements.
"""
function seitl_prior_distributions()
    priors = DynamicPPL.extract_priors(seitl_priors())
    ## Look each parameter up by name rather than trusting iteration order
    by_name = Dict(Symbol(string(k)) => v for (k, v) in pairs(priors))
    return [by_name[name] for name in SEITL_PARAMETERS]
end
