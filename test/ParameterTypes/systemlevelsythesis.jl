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
X0 = zeros(size(A,1), batches)
U0 = zeros(size(B,2), batches)

# Initialize REN state
h0 = randn(nx, batches)

# Test system level synthesis
X1 = A*X0 + B*U0 + w0
h1, v0 = ren(h0, w0)

# Controller realization
w1 = X1 - v0[1:size(A,1),:]
h2, v1 = ren(h1, w1)
u1 = v1[size(A,1)+1:end,:]

# Validation for the system level constraints
ψx= v1[1:size(A,1),:]
diff = X1 - ψx
