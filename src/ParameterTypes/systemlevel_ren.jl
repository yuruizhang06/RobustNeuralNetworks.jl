mutable struct SystemlevelRENParams{T} <: AbstractRENParams{T}
    nl                          # Sector-bounded nonlinearity
    nu::T
    nx::Int
    nv::Int
    ny::T
    direct::DirectParams{T}
    αbar::T
    A::AbstractArray{T}
    B::AbstractArray{T}
    y::Vector{T}
end


function SystemlevelRENParams{T}(
    nx::Int, nv::Int,
    A::AbstractArray{T}, B::AbstractArray{T};
    nl = Flux.relu, 
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
    y = glorot_normal(nx*size(A,1)+nx*size(B,2)+nv*size(B,2)+size(A,1)*size(B,2)+size(A,1)+size(B,2); T=T, rng=rng)

    # Direct (implicit) params
    direct_ps = DirectParams{T}(
        nu, nx, nv, ny; 
        init=init, ϵ=ϵ, bx_scale=bx_scale, bv_scale=bv_scale, 
        polar_param=polar_param, D22_free=true, rng=rng
    )

    return SystemlevelRENParams{T}(nl, nu, nx, nv, ny, direct_ps, αbar, A, B, y)

end

function systemlevel_trainable(L::DirectParams, y::Vector)
    ps = [L.ρ, L.X, L.Y1, L.B2, L.D12, L.bx, L.bv, y]
    !(L.polar_param) && popfirst!(ps)
    return filter(p -> length(p) !=0, ps)
end

Flux.trainable(m::SystemlevelRENParams) = systemlevel_trainable(m.direct, m.y)

function Flux.gpu(m::SystemlevelRENParams{T}) where T
    # TODO: Test and complete this
    direct_ps = Flux.gpu(m.direct)
    return SystemlevelRENParams{T}(
        m.nl, m.nu, m.nx, m.nv, m.ny, direct_ps, m.αbar, m.ν
    )
end

function Flux.cpu(m::SystemlevelRENParams{T}) where T
    # TODO: Test and complete this
    direct_ps = Flux.cpu(m.direct)
    return SystemlevelRENParams{T}(
        m.nl, m.nu, m.nx, m.nv, m.ny, direct_ps, m.αbar, m.ν
    )
end

function direct_to_explicit(ps::SystemlevelRENParams{T}, return_h=false) where T

    # System sizes
    nu = ps.nu
    nx = ps.nx
    ny = ps.ny
    nv = ps.nv

    nX = size(ps.A, 1)
    nU = size(ps.B, 2)

   #  from contracting ren
    ϵ = ps.direct.ϵ
    ρ = ps.direct.ρ
    X = ps.direct.X
    H = ps.direct.polar_param ? exp(ρ[1])*(X'*X + ϵ*I) / norm(X)^2 : X'*X + ϵ*I
    
    expilict_params = hmatrix_to_explicit(ps, H)

    A = expilict_params.A
    B1 = expilict_params.B1
    B2 = expilict_params.B2

    C1 = expilict_params.C1
    D11 = expilict_params.D11
    D12 = expilict_params.D12

    bx = expilict_params.bx
    bv = expilict_params.bv

    
    # system level constraints
    # ℍ = zeros((nx+nv+nX+1)*nX,(nx*nX+nx*nU+nv*nU+nX*nU+nX+nU))
    # ℍ[1:nx*nX,1:nx*nX] = kron(A',Matrix(I,nX,nX))-kron(Matrix(I,nx,nx),ps.A)
    # ℍ[nx*nX+1:nx*nX+nv*nX,1:nx*nX] = kron(B1',Matrix(I,nX,nX))
    # ℍ[nx*nX+nv*nX+1:nx*nX+nv*nX+nX*nX,1:nx*nX] = kron(B2',Matrix(I,nX,nX))
    # ℍ[nx*nX+nv*nX+nX*nX+1:end,1:nx*nX] = kron(ps.direct.bx',Matrix(I,nX,nX))

    # ℍ[1:nx*nX,nx*nX+1:nx*nX+nx*nU] = -kron(Matrix(I,nx,nx),ps.B)
    # ℍ[nx*nX+1:nx*nX+nv*nX,nx*nX+nx*nU+1:nx*nX+nx*nU+nv*nU] = -kron(Matrix(I,nv,nv),ps.B)
    # ℍ[nx*nX+nv*nX+1:nx*nX+nv*nX+nX*nX,nx*nX+nx*nU+nv*nU+1:nx*nX+nx*nU+nv*nU+nX*nU] = -kron(Matrix(I,nX,nX),ps.B)
    # ℍ[nx*nX+nv*nX+nX*nX+1:end,nx*nX+nx*nU+nv*nU+nX*nU+1:nx*nX+nx*nU+nv*nU+nX*nU+nX] = I-ps.A
    # ℍ[nx*nX+nv*nX+nX*nX+1:end,nx*nX+nx*nU+nv*nU+nX*nU+nX+1:end] = -ps.B

    ℍ1 = hcat(kron(A',Matrix(I,nX,nX))-kron(Matrix(I,nx,nx),ps.A), -kron(Matrix(I,nx,nx),ps.B),
        zeros(nx*nX,nv*nU+nX*nU+nX+nU))
    ℍ2 = hcat(kron(B1',Matrix(I,nX,nX)), zeros(nv*nX,nx*nU), -kron(Matrix(I,nv,nv),ps.B),
        zeros(nv*nX,nX*nU+nX+nU))
    ℍ3 = hcat(kron(B2',Matrix(I,nX,nX)), zeros(nX*nX,nx*nU+nv*nU), -kron(Matrix(I,nX,nX),ps.B),
        zeros(nX*nX,nX+nU))
    ℍ4 = hcat(kron(ps.direct.bx',Matrix(I,nX,nX)), zeros(nX,nx*nU+nv*nU+nX*nU), I-ps.A, -ps.B)

    ℍ = vcat(ℍ1,ℍ2,ℍ3,ℍ4)

    𝕗 = vcat(zeros(nx*nX+nv*nX),vec(ps.A),zeros(nX))
    
    𝕘 = pinv(ℍ)*𝕗+(I-pinv(ℍ)*ℍ)*ps.y

    # recover explicit parameters
    C2 = vcat(reshape(𝕘[1:nx*nX],nX,nx),reshape(𝕘[nx*nX+1:nx*nX+nx*nU],nU,nx))
    D21 = vcat(zeros(nX,nv), reshape(𝕘[nx*nX+nx*nU+1:nx*nX+nx*nU+nv*nU],nU,nv))
    D22 = vcat(Matrix(I,nX,nX), reshape(𝕘[nx*nX+nx*nU+nv*nU+1:nx*nX+nx*nU+nv*nU+nX*nU],nU,nX))
    by = 𝕘[nx*nX+nx*nU+nv*nU+nX*nU+1:end]

    # println(norm(ℍ*𝕘-𝕗))
    # println(reshape(𝕘[1:nx*nX],nX,nx)*B2-ps.A-ps.B*reshape(𝕘[nx*nX+nx*nU+nv*nU+1:nx*nX+nx*nU+nv*nU+nX*nU],nU,nX))
    
    !return_h && (return ExplicitParams{T}(A, B1, B2, C1, C2, D11, D12, D21, D22, bx, bv, by))
    return ℍ, 𝕗, 𝕘 
end