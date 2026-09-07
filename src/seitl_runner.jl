using Random: default_rng
using GeneralisedFilters: GeneralisedFilters, BF, DenseAncestorCallback, get_ancestry
using ForwardDiff: value
using SSMProblems: StateSpaceModel

"""
    run_particle_filter(θ, obs, n_particles; init_state)

Run a bootstrap particle filter for SEITL or SEIT4L and return the estimated
log-likelihood.

Which model is filtered follows from `init_state`: five compartments
[S, E, I, T, L] give SEITL, eight [S, E, I, T1, T2, T3, T4, L] give SEIT4L. The
components, the filter and the observation process are the same either way.

# Arguments
- `θ`: Parameter dictionary (must include :R_0, :D_lat, :D_inf, :α, :D_imm, :ρ)
- `obs`: Vector of observed daily incidence
- `n_particles`: Number of particles
- `init_state`: Initial compartment vector, defaulting to the SEIT4L one

# Returns
- `log_likelihood`: Estimated log-likelihood
"""
function run_particle_filter(
    θ,
    obs,
    n_particles;
    init_state = [279.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
)
    # Coerce init_state to Float64 so callers can pass an integer vector
    # (e.g. [279, 0, 2, ...]) without hitting SEITLInitial's Vector{Float64}
    # signature.
    init_state_f64 = collect(Float64.(init_state))
    length(init_state_f64) ≥ 5 ||
        throw(ArgumentError("init_state must be [S, E, I, T_1 ... T_k, L]"))

    # Strip ForwardDiff Duals before constructing the SSM components: the
    # bootstrap filter is non-differentiable, and SEITLDynamics /
    # PoissonObservation declare concrete Float64 fields. Mirrors the pattern
    # used in sessions/pmcmc.qmd.
    θ_f64 = Dict{Symbol, Float64}(k => value(v) for (k, v) in θ)

    # Define SSM components
    initial = SEITLInitial(init_state_f64)
    dynamics = SEITLDynamics(θ_f64)
    observation = PoissonObservation(θ_f64[:ρ])

    # Create state-space model
    model = StateSpaceModel(initial, dynamics, observation)

    # Run bootstrap particle filter
    rng = default_rng()
    algo = BF(n_particles)  # Bootstrap Filter
    _, log_lik = GeneralisedFilters.filter(rng, model, algo, obs)

    return log_lik
end

"""
    run_particle_filter_seitl(θ, obs, n_particles; init_state)

Filter the five compartment SEITL model. A thin wrapper over
[`run_particle_filter`](@ref) with the SEITL initial state as its default.
"""
function run_particle_filter_seitl(
    θ,
    obs,
    n_particles;
    init_state = [279.0, 0.0, 2.0, 3.0, 0.0],
)
    return run_particle_filter(θ, obs, n_particles; init_state)
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
    θ_f64 = Dict{Symbol, Float64}(k => value(v) for (k, v) in θ)
    model = StateSpaceModel(
        SEITLInitial(collect(Float64.(init_state))),
        SEITLDynamics(θ_f64),
        PoissonObservation(θ_f64[:ρ]),
    )

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
