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
w0 = randn(size(A,1), batches)

# Initialize system state anad controls
X0 = rand(size(A,1), batches)
U0 = zeros(size(B,2), batches)

# Initialize REN state
h0 = randn(nx, batches)

# Forward simulation
# Test system level synthesis
X1 = A*X0 + B*U0 + w0
h1, v0 = ren(h0, w0)

# Controller realization
w1 = X1 - v0[1:size(A,1),:]
hn1, v1 = ren(h1, w1)
ψx1 = v1[1:size(A,1),:]
ψu1 = v1[size(A,1)+1:end,:]

X2 = A*X1 + B*ψu1 + w1
w2 = X2 - v1[1:size(A,1),:]
h2, v2 = ren(hn1, w2)
ψx2 = v2[1:size(A,1),:]
ψu2 = v2[size(A,1)+1:end,:]

# Validation for the system level constraints
diff = ψx2 - A*ψx1 - B*ψu1 - w2
@test all(norm(diff) <= 1e-6)