"""
$(TYPEDEF)

Parameter struct to build an acyclic REN with behavioural
constraints encoded in Q, S, R matrices
"""
mutable struct GeneralRENParams{T} <: AbstractRENParams{T}
    nl                          # Sector-bounded nonlinearity
    nu::Int
    nx::Int
    nv::Int
    ny::Int
    direct::DirectParams{T}
    output::OutputLayer{T}
    αbar::T
    Q::Matrix{T}
    S::Matrix{T}
    R::Matrix{T}
end

"""
    GeneralRENParams(nu, nx, nv, ny; ...)

Main constructor for `GeneralRENParams`.
ᾱ ∈ (0,1] is the upper bound on contraction rate.
"""
function GeneralRENParams{T}(
    nu::Int, nx::Int, nv::Int, ny::Int,
    Q = nothing, S = nothing, R = nothing;
    init = :random,
    nl = Flux.relu, 
    ϵ = T(1e-6), 
    αbar = T(1),
    bx_scale = T(0), 
    bv_scale = T(1), 
    polar_param = true,
    rng = Random.GLOBAL_RNG
) where T

    # IQC params
    (Q === nothing) && (Q = zeros(T, ny, ny))
    (S === nothing) && (S = zeros(T, nu, ny))
    (R === nothing) && (R = zeros(T, nu, nu))

    # Check conditions on Q
    if !isposdef(-Q)
        Q = Q .- ϵ*I
        if ~isposdef(-Q)
            error("Q must be negative semi-definite for this construction.")
        end
    end

    # Direct (implicit) params
    direct_ps = DirectParams{T}(
        nu, nx, nv, ny; 
        init=init, ϵ=ϵ, bx_scale=bx_scale, bv_scale=bv_scale, 
        polar_param=polar_param, D22_free=false, rng=rng
    )

    # Output layer
    output_ps = OutputLayer{T}(nu, nx, nv, ny; D22_trainable=false, rng=rng)

    return GeneralRENParams{T}(nl, nu, nx, nv, ny, direct_ps, output_ps, αbar, Q, S, R)

end

"""
    Flux.trainable(m::GeneralRENParams)

Define trainable parameters for `ContractingRENParams` type
Filter empty ones (handy when nx=0)
"""
Flux.trainable(m::GeneralRENParams) = filter(
    p -> length(p) !=0, 
    (Flux.trainable(m.direct)..., Flux.trainable(m.output)...)
)

"""
    Flux.gpu(m::GeneralRENParams{T}) where T

Add GPU compatibility for `GeneralRENParams` type
"""
function Flux.gpu(m::GeneralRENParams{T}) where T
    direct_ps = Flux.gpu(m.direct)
    output_ps = Flux.gpo(m.output)
    return GeneralRENParams{T}(
        m.nl, m.nu, m.nx, m.nv, m.ny, direct_ps, output_ps, m.αbar, m.Q, m.S, m.R
    )
end

"""
    Flux.cpu(m::GeneralRENParams{T}) where T

Add CPU compatibility for `GeneralRENParams` type
"""
function Flux.cpu(m::GeneralRENParams{T}) where T
    direct_ps = Flux.cpu(m.direct)
    output_ps = Flux.cpo(m.output)
    return GeneralRENParams{T}(
        m.nl, m.nu, m.nx, m.nv, m.ny, direct_ps, output_ps, m.αbar, m.Q, m.S, m.R
    )
end


"""
    direct_to_explicit(ps::GeneralRENParams)

Convert direct REN parameterisation to explicit parameterisation
using behavioural constraints encoded in Q, S, R
"""
function direct_to_explicit(ps::GeneralRENParams)

    # TODO: Fill this in

    return nothing
end