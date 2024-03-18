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
using BSON
using MAT

using RobustNeuralNetworks

includet("./rollout_and_projection.jl")

dir = "./"

test_data = BSON.load(string(dir, "SLS-directparam-lc2-data.bson"))

wv = get(test_data, "wv", -1)
ws = get(test_data, "ws", -1)

rng = StableRNG(0)
# A = [1.5 0.5 1; 0 1 2; 0 0 1]
# B = [0; 1; 1.1]
# C = [1 0 0]
# L = [1, 1 ,5, 1]
# A = [1.5 0.5 2 3; 3 0.7 4 1; 3 6 4 2; 1 2 1 1]
# B = [1; 0.1; 1.1; 0.5]
# C = [1 0 0 0]
# L = [1, 1, 1, 1, 1]
# A = [1 2.1 3 4 3; 3 4 2 1 2; 2 3 1 2 1; 4 3 2 1 2; 2 3 4 5 6]
# B = [0; 1.1; 1; 0; 1]
# C = [1, 0, 0, 0, 0]
# L = [1, 1, 1, 1, 1, 1]
# A = [1.5 0.5 2 3; 3 0.7 4 1; 3 6 4 2; 1 2 1 1]
# B = [1 2; 0.0 1; 1.1 0; 0.5 1]
# C = [1 0 0 0]
# L = [1, 1, 1, 1, 1, 1]
# G = lti(A, B, C)
G= linearised_cartpole()
L = [1, 1 ,1, 1, 1]
nx = G.nx
nu = G.nu
x0_lims = 2*ones(nx,1)
w_sigma = .00*ones(nx,1)

# test for nonquadratic cost
# ub = 3
xb = 5
px = 10000
# pu = 300
# _u(u) = pu*max(abs(u) - ub, 0)
_x(x) = px*max(abs(x) - xb, 0)
# _cu(zt) = mean(_u.(zt[nx+1,:]))
_cx(zt) = mean(_x.(zt[2,:]))

_cost(zt) = mean(sum(L .* zt.^2; dims=1))
cost(z::AbstractVector) = mean(_cost.(z))+ mean(_cx.(z))

#+ mean(_cu.(z))
# _cost(zt) = mean(sum(L .* zt.^2; dims=1))
# cost(z) = mean(_cost.(z))

tbatch = 100
tsim = 150

K = lqr(G, L)
# wv = wgen(G, tbatch, tsim, x0_lims, w_sigma; rng=rng)
zb = rollout(G,K,wv)
Jb = cost(zb)
Jbk = mean(_cost.(zb))

nqx, nqv, batches, Epoch, η = (20, 30, tbatch, 1000, 1E-4)
# left = nqv*G.nx + G.nx*G.nx
# right = nqx*G.nu + nqv*G.nu + G.nx*G.nu + G.nu
# if left>=right
#     println("The number of parameters is not enough!")
#     # stop_here()
# end
Q = SystemlevelRENParams{Float64}(nqx, nqv, G.A, G.B)
# Q.direct.X = Q.direct.X+0.01I

Qe = REN(Q)
H, f, g = explicit_to_H(Q, Qe.explicit, true)
# if rank(H) != rank(hcat(H,f))
#     println("The rank of H is not equal to the rank of [H,f]")
#     # stop_here()
# end
zr1, _, _ = rollout(G, Q, wv)
Jr1 = cost(zr1)
zv1, _, _ = validation(G, Q, wv)
Jv1 = cost(zv1)
println("Jr1: $Jr1, Jv1: $Jv1")
Jvs = [Jv1]
# stop_here()
opt = ADAM()
optimizer = Flux.Optimiser(ADAM())
# ExpDecay(η, 0.5, 1500, 1e-7, 1)
ps = Flux.params(Q)

for epoch in 1:Epoch
    # optimization
    wt = wgen(G,tbatch,tsim,G.x0_lims,w_sigma;rng=rng)
    # wt = step_gen(G, 1, sim, x0_lims, 0.1*randn(rng, sim+1), rng = StableRNG(0))
    # println(wt[1])

    function loss()
        z1_, ψx_, ψu_ = validation(G, Q, wt)
        # z2_, ψx_, ψu_ = validation(G, Q, wt)
        return cost(z1_)
    end
    
    J, back = Zygote.pullback(loss, ps)
    ∇J = back(one(J))
    # println(J)
    # stop_here()
    update!(opt, ps, ∇J)  
    # validation
    zr_, ψx1_, ψu1_ = rollout(G, Q, wv)
    Jr = cost(zr_)
    zv_, ψx2_, ψu2_ = validation(G, Q, wv)
    Jv = cost(zv_)
    Jk = mean(_cost.(zv_))
    # # stop_here()
    # Qe = REN(Q)
    # H, f, g = explicit_to_H(Q, Qe.explicit, true)
    # println(rank(H))
    # println(rank(hcat(H,f)))
    # # println(norm(H*g-f))
    # println(size(H))
    # # println(cond(H))

    # normdiff1, cosdis1 = cost_diff(ψx1_, ψu1_, wt, tsim, G)
    # normdiff2, cosdis2 = cost_diff(ψx2_, ψu2_, wt, tsim, G)
    # normdiff3, cosdis3 = cost_diff(ψx1_, ψu2_, wt, tsim, G)
    # normdiff4, cosdis4 = cost_diff(ψx2_, ψu1_, wt, tsim, G)
    # println("Cosine distance: $cosdis1, Norm: $normdiff1")
    # println("Cosine distance: $cosdis2, Norm: $normdiff2")
    # println("Cosine distance: $cosdis3, Norm: $normdiff3")
    # println("Cosine distance: $cosdis4, Norm: $normdiff4")

    push!(Jvs, Jv)
    println("Epoch: $epoch, Jr: $Jr, Jv: $Jv,Jk: $Jk J0: $Jb, J0k: $Jbk")

end

# Forward simulation
# Qe = REN(Q)
sim = 150
# ws = wgen(G, 1, sim, x0_lims, w_sigma; rng=rng)
# ws = step_gen(G, 1, sim, x0_lims, 0.5*randn(rng, sim+1), rng = StableRNG(0))
# ws_ = reduce(hcat, ws)
zs1, ψxs1, ψus1 = rollout(G, Q, ws)
ψxr = reshape(ψxs1, (nx, sim))
ψur = reshape(ψus1, (nu, sim))
Jss1 = cost(zs1)
println("Jss1: $Jss1")
zs2, ψxs2, ψus2 = validation(G, Q, ws)
Jss = cost(zs2)
println("Jss: $Jss")
ψxv = reshape(ψxs2, (nx, sim))
ψuv = reshape(ψus2, (nu, sim))

xr_=zeros(nx,1)
ur_=zeros(nu,1)
# ws_=zeros(nx,1)
zs3 = rollout(G,K,ws)
Js3 = cost(zs3)
println("Js3: $Js3")
for i in 1:sim
    global xr_ = hcat(xr_, zs3[i][1:nx])
    global ur_ = hcat(ur_, zs3[i][nx+1:nx+nu])
    # global ws_ = hcat(ws_, ws[i])
end
xr = xr_[:,2:end]
ur = ur_[:,2:end]
# ws = ws_[:,2:end]

plt1 = plot()
plt2 = plot()
# plt3 = plot()
plt5 = plot()
plt6 = plot()
plt7 = plot()
plt8 = plot()
for i in 1:nx
    plot!(plt1, ψxr[i,:], label="ψx$i")
    plot!(plt2, ψxv[i,:], label="ψx_r$i")
    # plot!(plt1, ws_[i,:], label="w$i")
    # plot!(plt2, ws_[i,:], label="w$i")
    # plot!(plt3, ψxr[i,:]-ψxv[i,:], label="diffx$i")
    plot!(plt5,xr[i,:], label="x$i")
    plot!(plt7, ψxr[i,:], label="ψx$i", color=:red)
    plot!(plt7,xr[i,:], label="x$i", color=:blue)
    plot!(plt7, ws_[i,:], label="w$i", color=:green)
end

for i in 1:nu
    # plot!(plt1, ψur[i,:], label="ψu$i")
    plot!(plt6, ψur[i,:], label="ψu$i")
    # plot!(plt2, ψuv[i,:], label="ψu_r$i")
    # plot!(plt3, ψur[i,:]-ψuv[i,:], label="diffu$i")
    # plot!(plt5,ur[i,:], label="u$i")
    plot!(plt8, ψur[i,:], label="ψu$i", color=:red)
    plot!(plt8,ur[i,:], label="u$i", color=:blue)
end
# plot!(plt1, xb*ones(sim), color=:black)
# plot!(plt1, -xb*ones(sim), color=:black)
# plot!(plt5, xb*ones(sim), color=:black)
# plot!(plt5, -xb*ones(sim), color=:black)
# plot!(plt6, ub*ones(sim), color=:black)
# plot!(plt6, -ub*ones(sim), color=:black)
# plot!(plt8, ub*ones(sim), color=:black)
# plot!(plt8, -ub*ones(sim), color=:black)
plt4 = plot(log.(Jvs), label="Jv")
display(plt1)
# display(plt2)
# display(plt3)
display(plt5)
display(plt6)
# display(plt7)
display(plt8)
display(plt4)



# data = Dict(
#     "Jvs" => Jvs,
#     "Jb" => Jb,
#     "xr"  => xr,
#     "ur"  => ur,
#     "ws"  => ws,
#     "ψx"  => ψxr,
#     "ψu"  => ψur,
#     "ub" => ub,
#     "x0_lims" => x0_lims,
# )
# dir = "./"
# name = "sl-ren"
# bson(string(dir, name,  "-eta-",η,"-n-",Epoch, ".bson"),data)
model = Dict("Q" => Q)
data = Dict(
    "x0_lims" => x0_lims,
    "w_sigma" => w_sigma,
    "wv"  => wv,
    # "zv"  => zv,
    "Jvs" => Jvs,
    "Jb"  => Jb,
    "ws" => ws,
    # "zb"  => zb,
)

bson("./SLS-directparam-lc2-model.bson",model)
bson("./SLS-directparam-lc2-data.bson",data)

file = matopen("./task3_directparam.mat", "w")
write(file, "Jv", Jvs)
write(file, "x_lqr", xr)
write(file, "u_lqr", ur)
write(file, "x_sls", ψxv)
write(file, "u_sls", ψuv)

close(file)