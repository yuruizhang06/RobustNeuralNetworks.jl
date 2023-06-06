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
batches = 100
nx, nv = 10, 20
T = 100

A = [1 2.1 5; 3 4 7;5 6 1]
B = [0; 1.1; 1]
C = [1, 0, 0]
G =lti(A,B,C)
L = [10, 5, 1, 1]
println(rank(ctrb(A, B)))

# A = [1 2.1; 3 4]
# B = [0; 1.1]
# C = [1, 0]
# G =lti(A,B,C)
# L = [10, 1, 1]

# Test constructors
ren_ps = SystemlevelRENParams{Float64}(nx, nv, A, B)
ren = REN(ren_ps)

# Generate random noise
w_1 = zeros(size(A,1), batches)
w0 = randn(size(A,1), batches)
w1 = randn(size(A,1), batches)
w2 = randn(size(A,1), batches)

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

hr0, ψr_1 = ren(h_1, wh_1)
wh0 = X0 - ren.explicit.C2[1:size(A,1),:]*hr0 .- ren.explicit.by[1:size(A,1),:]
hr1, ψr0 = ren(hr0, wh0)
ψur0 = ψr0[size(A,1)+1:end,:]
ψxr0 = ψr0[1:size(A,1),:]

# t = 1
X1 = A*X0 + B*ψu0 +w1
h2, ψ1 = ren(h1, w1)
ψx1 = ψ1[1:size(A,1),:]
ψu1 = ψ1[size(A,1)+1:end,:]

# hr1, ψr0 = ren(hr0, wh0)
wh1 = X1 - ren.explicit.C2[1:size(A,1),:]*hr1 .- ren.explicit.by[1:size(A,1),:]
hr2, ψr1 = ren(hr1, wh1)
ψur1 = ψr1[size(A,1)+1:end,:]
ψxr1 = ψr1[1:size(A,1),:]

# t = 2
X2 = A*X1 + B*ψu1 +w2
h3, ψ2 = ren(h2, w2)
ψx2 = ψ2[1:size(A,1),:]
ψu2 = ψ2[size(A,1)+1:end,:]

# hr2, ψr1 = ren(hr1, wh1)
wh2 = X2 - ren.explicit.C2[1:size(A,1),:]*hr2 .- ren.explicit.by[1:size(A,1),:]
hr3, ψr2 = ren(hr2, wh2)
ψur2 = ψr2[size(A,1)+1:end,:]
ψxr2 = ψr2[1:size(A,1),:]

# Validation for the system level constraints
diff1 = ψx2 - A*ψx1 - B*ψu1
diff2 = ψxr2 - X2 
println(norm(diff2))

sim = 50

_cost(zt) = mean(sum(L .* zt.^2; dims=1))
cost(z) = mean(_cost.(z))
x0_lims = ones(size(A,1),1)
w_sigma = 1.0*ones(size(A,1),1)
ws = wgen(G, 1, sim, x0_lims, w_sigma)
z = rollout(G, ren_ps, ws)
J = cost(z)
zv, ψxs, ψus= validation(G, ren_ps, ws)
Jv = cost(zv)
println(J-Jv)
