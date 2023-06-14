cd(@__DIR__)
using Pkg
Pkg.activate("../../")

using Revise
using Flux
using Flux.Optimise:update!
using Flux.Optimise:ADAM
using Zygote
using LinearAlgebra
using Random
using StableRNGs
using Plots

using RobustNeuralNetworks


includet("./rollout_and_projection.jl")

rng = StableRNG(0)
A = [1.5 0.5 1; 0 1 2; 2 3 1]
B = [1; 0; 1.1]
C = [1 0 0]
L = [5, 5 ,1, 1]
# A = [1.5 0.5 2 3; 3 0.7 4 1; 3 6 4 2; 7 6 0 1]
# B = [1; 0.0; 1.1; 0.5]
# C = [1 0 0 0]
# L = [10, 5, 5, 1, 1]
G = lti(A, B, C)
nx = G.nx
nu = G.nu
x0_lims = ones(nx,1)
w_sigma = 0.0*ones(nx,1)
_cost(zt) = mean(sum(L .* zt.^2; dims=1))
cost(z) = mean(_cost.(z))

tbatch = 100
tsim = 50

K = lqr(G, L)
# vbatch = 200
# vsim = 40
wv = wgen(G, tbatch, tsim, x0_lims, w_sigma; rng=rng)
zb = rollout(G,K,wv)
Jb = cost(zb)

nqx, nqv, batches, Epoch, η = (40, 10, 100, 1000, 1E-3)
Q = SystemlevelRENParams{Float64}(nqx, nqv, G.A, G.B; polar_param = :false, init = :cholesky)


zr1, _, _ = rollout(G, Q, wv)
Jr1 = cost(zr1)
zv1, _, _ = validation(G, Q, wv)
println(norm(zr1-zv1))
Jvs = [Jr1]

opt = ADAM()
optimizer = Flux.Optimiser(ADAM(),ExpDecay(η, 0.5, 400, 1e-7, 1))
ps = Flux.params(Q)

for epoch in 1:Epoch
    # optimization
    wt = wgen(G,tbatch,tsim,G.x0_lims,w_sigma;rng=rng)

    function loss()
        z_, ψx_, ψu_ = validation(G, Q, wt)
        return cost(z_)
    end
    
    J, back = Zygote.pullback(loss, ps)
    ∇J = back(one(J)) 
    update!(opt, ps, ∇J)  
    # println(Q.direct.X)
    # validation
    zr_, ψx1_, ψu1_ = rollout(G, Q, wt)
    Jr = cost(zr_)
    zv_, ψx2_, ψu2_ = validation(G, Q, wt)
    Jv = cost(zv_)

    # # checking sls constraint
    # local ψx = ψx2_
    # local ψu = ψu2_
    # # cosine distance and norm
    # diff = []
    # cos_dis = []
    # for i in 1:tsim-1
    #     ψxn = A*ψx[(i-1)*nx+1:nx*i,:]+ B*ψu[i,:]' + wt[i]
    #     diff = append!(diff,ψxn-ψx[nx*i+1:nx*(i+1),:])  
    #     cos_dis= append!(cos_dis, dot(ψxn, ψx[nx*i+1:nx*(i+1),:]) 
    #         / (norm(ψxn)* norm(ψx[nx*i+1:nx*(i+1),:])))
    # end
    # cosinedis = mean(cos_dis)
    # meandiff = mean(diff)
    # println("Cosine distance: $cosinedis, Mean: $meandiff")

    push!(Jvs, Jv)
    println("Epoch: $epoch, Jr: $Jr, Jv: $Jv, J0: $Jb")

end

# Forward simulation
sim = 150
ws = wgen(G, 1, sim, x0_lims, w_sigma; rng=rng)

zs1, ψxs1, ψus1 = rollout(G, Q, ws)
ψxr = reshape(ψxs1, (nx, sim))
ψur = reshape(ψus1, (nu, sim))

zs2, ψxs2, ψus2 = validation(G, Q, ws)
ψxv = reshape(ψxs2, (nx, sim))
ψuv = reshape(ψus2, (nu, sim))
# println(norm(zs1-zs2))

plt1 = plot()
plt2 = plot()
for i in 1:nx
    plot!(plt1, ψxr[i,:], label="ψx$i")
    plot!(plt2, ψxv[i,:], label="ψx_r$i")
end
for i in 1:nu
    plot!(plt1, ψur[i,:], label="ψu$i")
    plot!(plt2, ψuv[i,:], label="ψu_r$i")
end
plt3 = plot(Jvs, label="Jv")
display(plt1)
display(plt2)
display(plt3)