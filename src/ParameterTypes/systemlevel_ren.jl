mutable struct SystemlevelRENParams{T} <: AbstractRENParams{T}
    nl::Function                          # Sector-bounded nonlinearity
    nu::T
    nx::Int
    nv::Int
    ny::T
    direct::DirectRENParams{T}
    αbar::T
    A::AbstractArray{T}
    B::AbstractArray{T}
    y::Vector{T}
end


function SystemlevelRENParams{T}(
    nx::Int, nv::Int,
    A::AbstractArray{T}, B::AbstractArray{T};
    nl::Function = relu, 
    αbar::T = T(1),
    init = :random,
    polar_param::Bool = true,
    bx_scale::T = T(0), 
    bv_scale::T = T(1), 
    ϵ::T = T(1e-12), 
    rng::AbstractRNG = Random.GLOBAL_RNG
) where T
    
    nu = size(A,1)
    ny = size(A,1)+size(B,2)
    y = zeros(nx*size(A,1)+nx*size(B,2)+nv*size(B,2)+size(A,1)*size(B,2)+size(A,1)+size(B,2))

    # Direct (implicit) params
    direct_ps = DirectRENParams{T}(
        nu, nx, nv, ny; 
        init, ϵ, bx_scale, bv_scale, polar_param, 
        D22_free=true, rng
    )

    return SystemlevelRENParams{T}(nl, nu, nx, nv, ny, direct_ps, αbar, A, B, y)

end

@functor SystemlevelRENParams 
function trainable(m::SystemlevelRENParams)
    ps = [m.direct.ρ, m.direct.X, m.direct.Y1, m.direct.B2, m.direct.D12, m.direct.bx, m.direct.bv]
    !(m.direct.polar_param) && popfirst!(ps)
    # return filter(p -> length(p) !=0, ps)
    (direct = ps, y = m.y)
end

function explicit_to_H(ps::SystemlevelRENParams, explicit::ExplicitRENParams, return_h::Bool=false)

    # System sizes
    nx = ps.nx
    nv = ps.nv
    
    nX = size(ps.A, 1)
    nU = size(ps.B, 2)

    A = explicit.A
    B1 = explicit.B1
    B2 = explicit.B2
    bx = explicit.bx

    # proj = B1'*pinv(B1*B1')*B1

    ℍ1 = hcat(kron(A',Matrix(I,nX,nX))-kron(Matrix(I,nx,nx),ps.A), -kron(Matrix(I,nx,nx),ps.B),
        zeros(nx*nX,nv*nU+nX*nU+nX+nU))
    ℍ2 = hcat(kron(B1',Matrix(I,nX,nX)), zeros(nv*nX,nx*nU), -kron(Matrix(I,nv,nv),ps.B),
        zeros(nv*nX,nX*nU+nX+nU))
    ℍ3 = hcat(kron(B2',Matrix(I,nX,nX)), zeros(nX*nX,nx*nU+nv*nU), -kron(Matrix(I,nX,nX),(ps.B)),
        zeros(nX*nX,nX+nU))
    ℍ4 = hcat(kron(bx',Matrix(I,nX,nX)), zeros(nX,nx*nU+nv*nU+nX*nU), I-ps.A, -ps.B)

    ℍ = vcat(ℍ1,ℍ2,ℍ3,ℍ4)

    𝕗 = vcat(zeros(nx*nX+nv*nX),vec(ps.A),zeros(nX))

    𝕘 = pinv(ℍ)*𝕗+(I-pinv(ℍ)*ℍ)*ps.y
    # recover explicit parameters
    C2 = vcat(reshape(𝕘[1:nx*nX],nX,nx),reshape(𝕘[nx*nX+1:nx*nX+nx*nU],nU,nx))
    # C2 = vcat(C2x,reshape(𝕘[nx*nX+1:nx*nX+nx*nU],nU,nx))
    D21 = vcat(zeros(nX,nv), reshape(𝕘[nx*nX+nx*nU+1:nx*nX+nx*nU+nv*nU],nU,nv))
    # D21 = vcat(zeros(nX,nv), zeros(nU,nv))
    D22 = vcat(Matrix(I,nX,nX), reshape(𝕘[nx*nX+nx*nU+nv*nU+1:nx*nX+nx*nU+nv*nU+nX*nU],nU,nX))
    by = 𝕘[nx*nX+nx*nU+nv*nU+nX*nU+1:end]
    
    !return_h && (return C2, D21, D22, by)
    return ℍ, 𝕗, 𝕘
end

function direct_to_explicit(ps::SystemlevelRENParams{T}) where T

   #  from contracting ren
    ϵ = ps.direct.ϵ
    ρ = ps.direct.ρ[1]
    X = ps.direct.X
    polar_param = ps.direct.polar_param
    H = x_to_h(X, ϵ, polar_param, ρ)
    
    explicit_params = hmatrix_to_explicit(ps, H)

    A = explicit_params.A
    B1 = explicit_params.B1
    B2 = explicit_params.B2

    C1 = explicit_params.C1
    D11 = explicit_params.D11
    D12 = explicit_params.D12

    bx = explicit_params.bx
    bv = explicit_params.bv
    
    # system level constraints
    C2, D21, D22, by = explicit_to_H(ps, explicit_params)

    return ExplicitRENParams{T}(A, B1, B2, C1, C2, D11, D12, D21, D22, bx, bv, by)
end