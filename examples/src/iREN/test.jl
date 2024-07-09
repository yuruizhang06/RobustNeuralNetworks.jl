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
using BSON
using RowEchelon

using RobustNeuralNetworks

# include("../test_utils.jl")
includet("./utils.jl")
includet("./rollout_and_projection.jl")
# includet("./sls_ren_linearizaztion.jl")
"""
Test system level constraints
"""
batches = 1
nx, nv = 6,8
# T = 100

# A = [1.5 0.5; 0 1]
# B = [1.1; .3]
# C = [1 0]
# L = [1 ,1, 1]

# A = [1.5 0.5 2 1; 3 0.7 4 3; 3 6 2 1; 1 1 2 5]
# B = [1 0.5; 0.0 1; 1.1 2;0.7 1]
# C = [1 0 0 0]
# L = [1, 1, 1, 1, 1]

A = [1 2.1 3 4 3 1 ; 3 4 2 1 2 2 ; 2 3 1 2 11 1; 4 3 2 1 2 1; 2 3 4 5 6 2; 4 3 5 2 1 1]
B = [0; 1.1; 1 ; 0.11; 1; 1.3]
C = [1 0 0 0 0 0]
L = [1, 5, 5, 5, 1, 1,1]
G =lti(A,B,C)
sim = 150
x0_lims = ones(size(A,1),1)
w_sigma = 0.0*ones(size(A,1),1)
ws = wgen(G, batches, sim, x0_lims, w_sigma)

# sys = ss(A,B,C,[0 0])
# ctrb(sys)
# println(rank(ctrb(sys)))

# println(rank(ctrb(A, B)))
# Test constructors
# u =solve_lqr(G, L, sim, x0_lims, 5)
# test = step_gen(G, batches, sim, x0_lims, randn(16), rng = StableRNG(0))
# stop_here()
ren_ps = SystemlevelRENParams{Float64}(nx, nv, A, B)

# left = nv*G.nx + G.nx*G.nx
# right = nx*G.nu + nv*G.nu + G.nx*G.nu + G.nu
# if left>=right
#     println("The number of parameters is not enough!")
#     # stop_here()
# end
# ren_ps.direct.B2[1:G.nx, 1:G.nx] = G.A
# println(ren_ps.direct.bx)
ren = REN(ren_ps)
# println(ren.explicit.bx)

# ren.explicit.A = 2*ren.explicit.A
# ren_ps.direct.bx=0*ren_ps.direct.bx 
# ren_ps.direct.bv=0*ren_ps.direct.bv
# ren_ps.direct.by=0*ren_ps.direct.by
# stop_here()
# ren.explicit.B1 = [randn(nx,G.nu)*G.B' zeros(nx, nv-G.nx)]
# ren.explicit.B1 = zeros(nx,nv)
# ren.explicit.C1 = zeros(nv, nx)
# ren.explicit.D11 = zeros(nv, nv)
# ren.explicit.D12 = zeros(nv, G.nx+G.nu)
# ren.explicit.D21 = zeros(G.nx+G.nu, nv)
H, f, g, 𝔸, 𝕏 = explicit_to_H(ren_ps, ren.explicit, true)

println(rank(𝔸))
println(rank(hcat(𝔸,A)))
# # rref_aug = rref(hcat(H,f))
# println(size(nullspace(hcat(kron(ren.explicit.B1',Matrix(I,G.nx,G.nx)), -kron(Matrix(I,nv,nv),G.B)))))
# println(size(nullspace(vcat(hcat(kron(ren.explicit.B1',Matrix(I,G.nx,G.nx)), zeros(nv*G.nx,nx*G.nu), -kron(Matrix(I,nv,nv),G.B)),
#     hcat(kron(ren.explicit.A',Matrix(I,G.nx,G.nx))-kron(Matrix(I,nx,nx),G.A), -kron(Matrix(I,nx,nx),G.B), zeros(nx*G.nx,nv*G.nu))))))
c2x = ren.explicit.C2[1:G.nx,:]
c2u = ren.explicit.C2[G.nx+1:end,:]
d21u = ren.explicit.D21[G.nx+1:end,:]
d22u = ren.explicit.D22[G.nx+1:end,:]
𝔸 = ren.explicit.A
𝔹1 = ren.explicit.B1
𝔹2 = ren.explicit.B2
# println(size(𝔸))
# println(norm(𝔸*𝕏-A))

stop_here()
# count = 0
# for i in 1:size(H, 2)
#     if rank(H[:, 1:i]) < i-count
#         # rank_indices = i
#         global count += 1
#         println(i)
#     end
# end
# stop_here()
# count2 = 0
# for i in 1:size(H, 1)
#     if rank(hcat(H,f)[1:i, :]) < i-count2
#         # rank_indices = i
#         global count2 += 1
#         println(i)
#     end
# end

# println(rank(kron(ren.explicit.B2', Matrix(I, G.nx, G.nx))))  
# println(rank(hcat(H,f)[1:30,:]))
# println(rank(H[1:30,:]))
# Generate random noise



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

hr0, ψr_1 = ren(h_1, wh_1)
wh0 = X0 - ren.explicit.C2[1:size(A,1),:]*hr0 .- ren.explicit.by[1:size(A,1),:]
hr1, ψr0 = ren(hr0, wh0)
ψur0 = ψr0[size(A,1)+1:end,:]
ψxr0 = ψr0[1:size(A,1),:]

h1, ψ0 = ren(h0, wh0)
# stop_here()
ψx0 = ψ0[1:size(A,1),:]
ψu0 = ψ0[size(A,1)+1:end,:]
# println(ψx0)

# println(ψxr0)
# t = 1
X1 = A*X0 + B*ψur0 + w1
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
X2 = A*X1 + B*ψur1 + w2
h3, ψ2 = ren(h2, w2)
ψx2 = ψ2[1:size(A,1),:]
ψu2 = ψ2[size(A,1)+1:end,:]
# println(ψx2)
# hr2, ψr1 = ren(hr1, wh1)
wh2 = X2 - ren.explicit.C2[1:size(A,1),:]*hr2 .- ren.explicit.by[1:size(A,1),:]
# println(norm(wh2-w2))
hr3, ψr2 = ren(hr2, wh2)
ψur2 = ψr2[size(A,1)+1:end,:]
ψxr2 = ψr2[1:size(A,1),:]
# println(ψxr2)
# Validation for the system level constraints
diff1 = ψx2 - A*ψx1 - B*ψu1 - w2
diff2 = ψxr2 - X2 
diff3 = ψx2 - X2
diff4 = ψxr2 - A*ψxr1 - B*ψur1 - w2
diff5 = ψxr2 - A*X1 - B*ψu1 - w2
diff6 = ψx2 - A*X1 - B*ψur1 - w2
diff7 = X2 - A*X1 - B*ψur1 - w2
diff8 = ψxr0- ψx0 
println(norm(diff1))
println(norm(diff2))
println(norm(diff3))
println(norm(diff4))
println(norm(diff5))
println(norm(diff6))
println(norm(diff7))
println(norm(diff8))
# println(H[30,:])

_cost(zt) = mean(sum(L .* zt.^2; dims=1))
cost(z) = mean(_cost.(z))

z, ψx, ψu = rollout(G, ren_ps, ws)
ψx_1 = reshape(ψx[:,1], (G.nx, sim))
ψu_1 = reshape(ψu[:,1], (G.nu, sim))
J_ = cost(z)
zv, ψxr, ψur= validation(G, ren_ps, ws)
ψx_2 = reshape(ψxr[:,1], (G.nx, sim))
ψu_2 = reshape(ψur[:,1], (G.nu, sim))
Jv_ = cost(zv)
# println(J_-Jv_)
# println(J_)
# println(ψxs)
# println(ψus)
# println(ψxr)
# println(ψur)
# println(norm(ψu_1-ψu_2))
# Xt = G.A*ψxr[1:size(A,1),:] + G.B*ψur[1] + ws[2]
plt1 = plot()
plt2 = plot()
plt3 = plot()
for i in 1:G.nx
    plot!(plt1, ψx_1[i,:], label="ψx$i")
    plot!(plt2, ψx_2[i,:], label="ψx_r$i")
    plot!(plt3, ψx_1[i,:]-ψx_2[i,:], label="diffx$i")
    # println(norm(ψxr[i,:]-ψx[i,:]))
end
for i in 1:G.nu
    plot!(plt1, ψu_1[i,:], label="ψu$i")
    plot!(plt2, ψu_2[i,:], label="ψu_r$i")
    plot!(plt3, ψu_2[i,:]-ψu_2[i,:], label="diffu$i")
    # println(norm(ψur[i,:]-ψu[i,:]))
end
display(plt1)
display(plt2)
display(plt3)
