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


includet("./utils.jl")
includet("./rollout_and_projection.jl")

rng = StableRNG(0)
vbatch = 200
vsim = 40
A = [1.5 0.5 1 2; 0 1 2 3; 2 3 1 3; 1 2 3 1]
B = [1; 0; 1; 0.3]
C = [1 0 0 0]
L = [10, 5, 1 ,5, 1]
# A = [1.5 0.5 1; 0 1 2; 2 3 1]
# B = [1; 0.0; 1]
# C = [1 0 0]
# L = [10, 5, 1]
G = lti(A, B, C)
nx = G.nx
nu = G.nu
x0_lims = ones(nx,1)
w_sigma = 0.0*ones(nx,1)
_cost(zt) = mean(sum(L .* zt.^2; dims=1))
cost(z) = mean(_cost.(z))


K = lqr(G, L)
wv = wgen(G, vbatch, vsim, x0_lims, w_sigma; rng=rng)
zb = rollout(G,K,wv)
Jb = cost(zb)

nqx, nqv, batches, Epoch, η = (30, 60, 80, 200, 1E-3)
Q = SystemlevelRENParams{Float64}(nqx, nqv, G.A, G.B; init = :cholesky)
zv1 = rollout(G, Q, wv)
Jv1 = cost(zv1)

opt = ADAM()
optimizer = Flux.Optimiser(ADAM(),ExpDecay(η))
ps = Flux.params(Q)
Q.direct.bx=0*Q.direct.bx 
Q.direct.bv=0*Q.direct.bv
Q.direct.by=0*Q.direct.by
tbatch = 100
tsim = 50
Jvs = [Jv1]

for epoch in 1:Epoch
    # optimization
    wt = wgen(G,tbatch,tsim,G.x0_lims,w_sigma;rng=rng)

    function loss()
        zt = rollout(G, Q, wt)
        return cost(zt)
    end

    J, back = Zygote.pullback(loss, ps)
    ∇J = back(one(J)) 
    update!(opt, ps, ∇J)  

    # validation with lqr
    zv= validation(G, Q, wt)
    Jv = cost(zv)

    # # checking sls constraint
    # zt, ψxs, ψus = validation(G, Q, wt)
    # ψx = ψxs[2:end,:]
    # ψu = ψus[2:end,:]
    # # cosine distance and norm
    # diff = []
    # cos_dis = []
    # for i in 1:tsim-1
    #     ψxn = A*ψx[2i-1:2i,:]+ B*ψu[i,:]' + wt[i]
    #     diff = append!(diff,ψxn-ψx[2i+1:2i+2,:])  
    #     cos_dis= append!(cos_dis, dot(ψxn, ψx[2i+1:2i+2,:]) / (norm(ψxn)* norm(ψx[2i+1:2i+2,:])))
    # end

    # cosinedis = mean(cos_dis)
    # normdiff = norm(diff)
    # println("Cosine distance: $cosinedis, Norm: $normdiff")
    # global Jvs =[Jvs..., Jv]
    push!(Jvs, Jv)
    println("Epoch: $epoch, Jt: $J, Jr: $Jv, J0: $Jb")

end

# Forward simulation
x0 = zeros(nqx,1)
ws = wgen(G, 1, 200, x0_lims, w_sigma; rng=rng)
function simulate(model::SystemlevelRENParams, w, x0)
    model_e = REN(model)
    eval_cell = (x, u) -> model_e(x, u)
    recurrent = Flux.Recur(eval_cell, x0)
    output = [recurrent(input) for input in w]
    return output
end
output = simulate(Q, ws, x0)
ψx = []
ψu = []
for i in 1:lastindex(output)
    push!(ψx, output[i][1:nx])
    push!(ψu, output[i][nx+1:end])
end
ψx = reduce(hcat, ψx)
ψu = reduce(hcat, ψu)

plt = plot()
for i in 1:nx
    plot!(plt, ψx[i,:], label="ψx$i")
end
for i in 1:nu
    plot!(plt, ψu[i,:], label="ψu$i")
end
display(plt)