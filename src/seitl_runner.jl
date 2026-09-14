using Random: default_rng
using GeneralisedFilters: GeneralisedFilters, BF, DenseAncestorCallback, get_ancestry
using ForwardDiff: value
using SSMProblems: StateSpaceModel

"""
    seitl_ssm(θ, init_state)

Assemble the state-space model the two runners below both need.

The number of temporary immunity stages is whatever `init_state` says, so this
serves SEITL (`[S, E, I, T, L]`) and SEIT4L (`[S, E, I, T1, T2, T3, T4, L]`)
alike without being told which.

`init_state` is coerced to `Float64` so callers may pass an integer vector, and
ForwardDiff duals are stripped from `θ` because the bootstrap filter is not
differentiable and the component types declare concrete `Float64` fields.
"""
function seitl_ssm(θ, init_state)
    init_f64 = collect(Float64.(init_state))
    length(init_f64) >= 5 || throw(
        ArgumentError(
            "init_state must be [S, E, I, T_1 ... T_k, L] with at least one " *
            "temporary immunity stage; got $(length(init_f64)) compartments",
        ),
    )
    θ_f64 = Dict{Symbol, Float64}(k => value(v) for (k, v) in θ)
    return StateSpaceModel(
        SEITLInitial(init_f64),
        SEITLDynamics(θ_f64, init_f64),
        PoissonObservation(θ_f64[:ρ]),
    )
end

"""
    run_particle_filter(θ, obs, n_particles; init_state)

Run the bootstrap particle filter and return the estimated log-likelihood.

# Arguments
- `θ`: parameter dictionary (`:R_0`, `:D_lat`, `:D_inf`, `:α`, `:D_imm`, `:ρ`)
- `obs`: vector of observed daily incidence
- `n_particles`: number of particles
- `init_state`: initial compartments; its length selects SEITL or SEIT4L
"""
function run_particle_filter(
    θ,
    obs,
    n_particles;
    init_state = [279.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
)
    model = seitl_ssm(θ, init_state)
    _, log_lik = GeneralisedFilters.filter(default_rng(), model, BF(n_particles), obs)
    return log_lik
end

"""
    filtered_incidence(θ, obs, n_particles; init_state)

Run the bootstrap filter and return one draw from the smoothing distribution of
daily incidence, that is a latent path conditioned on `obs`.

`DenseAncestorCallback` records the particles and their ancestor indices at
every step, so a particle drawn from the final weights can be traced back to
give its whole path. The last element of the state is the daily incidence, so
reading that element off the path gives the trajectory directly.

# Returns
- `Vector{Float64}` of length `length(obs)`, the filtered daily incidence
"""
function filtered_incidence(
    θ,
    obs,
    n_particles;
    init_state = [279.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
)
    model = seitl_ssm(θ, init_state)
    callback = DenseAncestorCallback(nothing)
    final, _ =
        GeneralisedFilters.filter(default_rng(), model, BF(n_particles), obs; callback)

    ## draw one particle in proportion to its final weight, then follow its
    ## ancestry back to the start
    log_w = getfield.(final.particles, :log_w)
    w = exp.(log_w .- maximum(log_w))
    u = rand() * sum(w)
    idx = findfirst(>=(u), cumsum(w))
    path = get_ancestry(callback.container, idx)

    return [path[t][end] for t in 1:length(obs)]
end
