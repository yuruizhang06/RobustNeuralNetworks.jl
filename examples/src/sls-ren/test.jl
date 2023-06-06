using LinearAlgebra
using Random
using RobustNeuralNetworks
using Test

# include("../test_utils.jl")

"""
Test system level constraints
"""
batches = 100
nx, nv = 5, 10
T = 100

A = [1 2.1; 3 4]
B = [0; 1.1]

# Test constructors
ren_ps = SystemlevelRENParams{Float64}(nx, nv, A, B)
ren = REN(ren_ps)

# Generate random noise
w_1 = zeros(size(A,1), batches)
w0 = randn(size(A,1), batches)
w1 = randn(size(A,1), batches)
w2= zeros(size(A,1), batches)

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
ψx0 = ψ0[1:size(A,1),:]
ψu0 = ψ0[size(A,1)+1:end,:]

hr0, ψr_1 = ren(h_1, wh_1)
wh0 = X0 - ren.explicit.C2[1:size(A,1),:]*hr0 .- ren.explicit.by[1:size(A,1),:]
hnr0, ψnr0 = ren(hr0, wh0)
ψur0 = ψnr0[size(A,1)+1:end,:]
ψxr0 = ψnr0[1:size(A,1),:]

# t = 1
X1 = A*X0 + B*ψu0 +w1
h2, ψ1 = ren(h1, w1)
ψx1 = ψ1[1:size(A,1),:]
ψu1 = ψ1[size(A,1)+1:end,:]

hr1, ψr0 = ren(hnr0, wh0)
wh1 = X1 - ren.explicit.C2[1:size(A,1),:]*hr1 .- ren.explicit.by[1:size(A,1),:]
hnr1, ψnr1 = ren(hr1, wh1)
ψur1 = ψnr1[size(A,1)+1:end,:]
ψxr1 = ψnr1[1:size(A,1),:]

# t = 2
X2 = A*X1 + B*ψu1 +w2
h3, ψ2 = ren(h2, w2)
ψx2 = ψ2[1:size(A,1),:]
ψu2 = ψ2[size(A,1)+1:end,:]

hr2, ψr1 = ren(hnr1, wh0)
wh2 = X2 - ren.explicit.C2[1:size(A,1),:]*hr2 .- ren.explicit.by[1:size(A,1),:]
hnr2, ψnr2 = ren(hr2, wh2)
ψur2 = ψnr2[size(A,1)+1:end,:]
ψxr2 = ψnr2[1:size(A,1),:]

# Validation for the system level constraints
diff1 = ψx2 - A*ψx1 - B*ψu1
diff2 = ψxr2 - X2 
