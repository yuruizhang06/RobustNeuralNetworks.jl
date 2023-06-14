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
using ControlSystems

using RobustNeuralNetworks

# include("../test_utils.jl")
includet("./utils.jl")
includet("./rollout_and_projection.jl")
"""
Test system level constraints
"""
batches = 1
nx, nv = 100, 20
# T = 100

# A = [1 2.1 3; 3 2 7;5 10 1]
# B = [0; 1.1; 1]
# C = [1, 0, 0]
# L = [10, 5, 1, 1]

A = [1 2.1 3 4 3; 3 4 2 1 2; 2 3 1 2 1; 4 3 2 1 2; 2 3 4 5 6]
B = [0; 1.1; 1; 0; 1]
C = [1, 0, 0, 0, 0]
L = [10, 5, 5, 5, 1, 1]

G =lti(A,B,C)
println(rank(ctrb(A, B)))
# Test constructors
ren_ps = SystemlevelRENParams{Float64}(nx, nv, A, B; polar_param = :false, init = :cholesky)
# ren_ps.direct.X = 0.5*ren_ps.direct.X


ren = REN(ren_ps)
# ren.explicit.A = 2*ren.explicit.A
# ren_ps.direct.bx=0*ren_ps.direct.bx 
# ren_ps.direct.bv=0*ren_ps.direct.bv
# ren_ps.direct.by=0*ren_ps.direct.by

# Generate random noise
sim = 30
x0_lims = ones(size(A,1),1)
w_sigma = 1.0*ones(size(A,1),1)
ws = wgen(G, batches, sim, x0_lims, w_sigma)

w_1 = zeros(size(A,1), batches)
# w0 = randn(size(A,1), batches)
# w1 = randn(size(A,1), batches)
# w2 = randn(size(A,1), batches)
w0 = ws[1]
w1 = ws[2]
w2 = ws[3]

# Initialize system state and controls
X_1 = zeros(size(A,1), batches)
U_1 = zeros(size(B,2), batches)

# Initialize REN state
h_1 = zeros(nx, batches)
wh_1= zeros(size(A,1), batches)
h0, ψ_1 = ren(h_1, w_1)

# t = 0
X0 = A*X_1 + B*U_1 + w0
h1, ψ0 = ren(h0, w0)
# stop_here()
ψx0 = ψ0[1:size(A,1),:]
ψu0 = ψ0[size(A,1)+1:end,:]
# println(ψx0)
hr0, ψr_1 = ren(h_1, wh_1)
wh0 = X0 - ren.explicit.C2[1:size(A,1),:]*hr0 .- ren.explicit.by[1:size(A,1),:]
hr1, ψr0 = ren(hr0, wh0)
ψur0 = ψr0[size(A,1)+1:end,:]
ψxr0 = ψr0[1:size(A,1),:]
# println(ψxr0)
# t = 1
X1 = A*X0 + B*ψur0 +w1
h2, ψ1 = ren(h1, w1)
ψx1 = ψ1[1:size(A,1),:]
ψu1 = ψ1[size(A,1)+1:end,:]
# println(ψx1)
# hr1, ψr0 = ren(hr0, wh0)
wh1 = X1 - ren.explicit.C2[1:size(A,1),:]*hr1 .- ren.explicit.by[1:size(A,1),:]
hr2, ψr1 = ren(hr1, wh1)
ψur1 = ψr1[size(A,1)+1:end,:]
ψxr1 = ψr1[1:size(A,1),:]
# println(ψxr1)
# t = 2
X2 = A*X1 + B*ψur1 +w2
h3, ψ2 = ren(h2, w2)
ψx2 = ψ2[1:size(A,1),:]
ψu2 = ψ2[size(A,1)+1:end,:]
# println(ψx2)
# hr2, ψr1 = ren(hr1, wh1)
wh2 = X2 - ren.explicit.C2[1:size(A,1),:]*hr2 .- ren.explicit.by[1:size(A,1),:]
hr3, ψr2 = ren(hr2, wh2)
ψur2 = ψr2[size(A,1)+1:end,:]
ψxr2 = ψr2[1:size(A,1),:]
# println(ψxr2)
# Validation for the system level constraints
diff1 = ψx2 - A*ψx1 - B*ψu1
diff2 = ψxr2 - X2 
diff3 = ψx2 - X2
diff4 = ψxr2 - A*ψxr1 - B*ψur1
# println(norm(diff1))
# println(norm(diff2))
# println(norm(diff3))
# println(norm(diff4))

_cost(zt) = mean(sum(L .* zt.^2; dims=1))
cost(z) = mean(_cost.(z))

z, ψx, ψu = rollout(G, ren_ps, ws)
ψx_1 = reshape(ψx[:,1], (G.nx, sim))
ψu_1 = reshape(ψu[:,1], (G.nu, sim))
J = cost(z)
zv, ψxr, ψur= validation(G, ren_ps, ws)
ψx_2 = reshape(ψxr[:,1], (G.nx, sim))
ψu_2 = reshape(ψur[:,1], (G.nu, sim))
Jv = cost(zv)
println(J-Jv)
# println(ψxs)
# println(ψus)
# println(ψxr)
# println(ψur)
# println(norm(ψx_1-ψx_2))
# Xt = G.A*ψxr[1:size(A,1),:] + G.B*ψur[1] + ws[2]
plt1 = plot()
plt2 = plot()
for i in 1:G.nx
    plot!(plt1, ψx_1[i,:], label="ψx$i")
    plot!(plt2, ψx_2[i,:], label="ψx_r$i")
    # println(norm(ψxr[i,:]-ψx[i,:]))
end
for i in 1:G.nu
    plot!(plt1, ψu_1[i,:], label="ψu$i")
    plot!(plt2, ψu_2[i,:], label="ψu_r$i")
    # println(norm(ψur[i,:]-ψu[i,:]))
end
display(plt1)
display(plt2)
