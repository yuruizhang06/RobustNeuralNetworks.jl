using Distributions
using Random
# using LinearAlgebra
using MatrixEquations

# utility
glorot_normal(n, m; rng=Random.GLOBAL_RNG) = randn(rng, n, m) / sqrt(n + m)
glorot_normal(n; rng=Random.GLOBAL_RNG) = randn(rng, n) / sqrt(n)

# mutable struct lti 
#     nx
#     nu 
#     A 
#     B 
# end
mutable struct lti
    A
    B
    C
    σw::AbstractFloat               # (To add Guassian white process noise)
    σv::AbstractFloat               # (To add Guassian white measurement noise)
    nx::Int64
    nu::Int64
    ny::Int64
    max_steps::Int64
    x0_lims::AbstractVector
end

function lti(
    A, B, C;
    σw = 0.0, σv = 0.0,
    max_steps=200, 
    x0_lims=[]
)
    nx, nu, ny = size(A,1), size(B, 2), size(C,1)
    isempty(x0_lims) && (x0_lims = ones(typeof(A[1,1]), nx))
    lti(A, B, C, σw, σv, nx, nu, ny, max_steps, x0_lims)
end


function lqr(G::lti, L)
    Q = diagm(0 => L[1:G.nx]) 
    R = diagm(0 => L[G.nx+1:G.nx+G.nu])
    X, E, K, Z = ared(G.A, G.B, R, Q, zeros(G.nx, G.nu))
    return K
end

function wgen(G::lti, batches, T, x0_lims, w_sigma; rng = StableRNG(0))
    x0 = [x0_lims .* randn(rng, G.nx, batches)]
    wt = [w_sigma .* randn(rng, G.nx, batches) for t in 1:T-1]
    return vcat(x0, wt)
end

function rollout(G::lti, K, w)

    batch = size(w[1], 2)
    X1 = (zeros(G.nx, batch), zeros(G.nu, batch))

    function f(X_1, t) 
        x_1, u_1 = X_1 
        xt = G(x_1, u_1, w[t])
        ut = -K*xt
        Xt = (xt, ut)
        zt = vcat(xt, ut)
        return Xt, zt
    end

    md = Flux.Recur(f,X1)
    z = md.(1:length(w))

    return z
end

function linearised_cartpole(;dt=0.08, max_steps=50, σw=0.005, σv=0.001)

    δ, mp, l, mc, g = (dt, 0.2, 0.5, 1.0, 9.81)
    Ac = [0 1 0 0; 0 0 -mp*g/mc 0; 0 0 0 1; 0 0 g*(mc+mp)/(l*mc) 0]
    Bc = reshape([0; 1/mc; 0; -1/mc],4,1)
    Ag = Matrix(I,4,4)+δ*Ac
    Bg = δ*Bc
    Cg = [1.0 0 0 0; 0 0 1.0 0]
    x0_lims = [0.5, 0.2, 0.5, 0.2]/2

    G = lti(
        Ag, Bg, Cg; 
        x0_lims = x0_lims, 
        σw = σw, σv = σv, 
        max_steps = max_steps
    )

    return G

end

(G::lti)(xt, ut, wt) = G.A * xt + G.B * ut + wt 