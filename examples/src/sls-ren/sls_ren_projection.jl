cd(@__DIR__)
using Pkg
Pkg.activate("../../")

using Revise
using Flux
using Flux.Optimise:update!
using Flux.Optimise:ADAM
using Zygote
using Random
using StableRNGs
using LinearAlgebra
using Plots
using MatrixEquations
using ControlSystems

using RobustNeuralNetworks

includet("./utils.jl")
includet("./rollout_and_projection.jl")

# initialization
rng = StableRNG(0)
vbatch = 200
vsim = 40
A = [1.5 0.5 1; 0 1 2; 2 3 1]
B = [1; 0; 1]
C = [1 0 0 ]
L = [10, 5, 1 ,5]
G = lti(A, B, C)
nx = G.nx
nu = G.nu
x0_lims = ones(nx,1)
w_sigma = 0.0*ones(nx,1)
_cost(zt) = mean(sum(L .* zt.^2; dims=1))
cost(z) = mean(_cost.(z))

# lqr for comparison
K = lqr(G, L)
wv = wgen(G, vbatch, vsim, x0_lims, w_sigma; rng=rng)
zb = rollout(G,K,wv)
Jb = cost(zb)

nqx, nqv, batches, Epoch, η = (20, 50, 80, 200, 1E-3)
nqu = nx 
nqy = nx+nu 
Q = ContractingRENParams{Float64}(nqu, nqx, nqv, nqy; init = :cholesky)
proj!(G, Q)
zv1 = rollout(G, Q, wv)
Jv1 = cost(zv1)
opt = ADAM(η)
optimizer = Flux.Optimiser(ADAM(η))
ps = Flux.params(Q)
Q.direct.bx=0*Q.direct.bx 
Q.direct.bv=0*Q.direct.bv
Q.direct.by=0*Q.direct.by
tbatch = 100
tsim = 50
Jvs = [Jv1]

for epoch in 1:Epoch
    # optimization
    wt = wgen(G,tbatch,tsim,x0_lims,w_sigma;rng=rng)
    
    function loss()
        zt = rollout(G, Q, wt)
        return cost(zt)
    end

    J, back = Zygote.pullback(loss, ps)
    ∇J = back(one(J)) 
    update!(optimizer, ps, ∇J)  
    proj!(G, Q)

    # validation with lqr
    zv, ψxs, ψus = validation(G, Q, wt)
    Jv = cost(zv)

    # checking sls constraint
    # zt, ψxs, ψus = validation(G, Q, wt)
    local ψx = ψxs[2:end,:]
    local ψu = ψus[2:end,:]
    # cosine distance and norm
    diff = []
    cos_dis = []
    for i in 1:tsim-1
        ψxn = A*ψx[(i-1)*nx+1:nx*i,:]+ B*ψu[i,:]' + wt[i]
        diff = append!(diff,ψxn-ψx[nx*i+1:nx*(i+1),:])  
        cos_dis= append!(cos_dis, dot(ψxn, ψx[nx*i+1:nx*(i+1),:]) 
            / (norm(ψxn)* norm(ψx[nx*i+1:nx*(i+1),:])))
    end
    cosinedis = mean(cos_dis)
    meandiff = mean(diff)
    println("Cosine distance: $cosinedis, Mean: $meandiff")

    push!(Jvs, Jv)
    println("Epoch: $epoch, Jt: $J, Jr: $Jv, J0: $Jb")

end

# Forward simulation
x0 = zeros(nqx,1)
ws = wgen(G, 1, 200, x0_lims, w_sigma; rng=rng)
function simulate(model::ContractingRENParams, w, x0)
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


#model = Dict("Q" => Q, "Qe" => Qe)
# data = Dict(
#     "x0_lims" => x0_lims,
#     "w_sigma" => w_sigma,
#     "wv"  => wv,
#     "zv"  => zv,
#     "Jvs" => Jvs,
#     "Jb"  => Jb,
#     "zb"  => zb,
# )

# bson("./results/SLS-LDS-model.bson",model)
# bson("./results/SLS-LDS-data.bson",data)