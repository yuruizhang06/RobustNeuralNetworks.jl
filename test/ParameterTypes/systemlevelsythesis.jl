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

# Initialize system state and controls
X_1 = zeros(size(A,1), batches)
U_1 = zeros(size(B,2), batches)

# Initialize REN state
h_1 = zeros(nx, batches)
wh_1= zeros(size(A,1), batches)

# Forward simulation
# At t=0
X0 = A*X_1 + B*U_1 + w0
h0, v_1 = ren(h_1, wh_1)
wh0 = X0 - v_1[1:size(A,1),:]
h1, v0 = ren(h0, wh0)
ψx0 = v0[1:size(A,1),:]
ψu0 = v0[size(A,1)+1:end,:]

# At t=1
X1 = A*X0 + B*ψu0
wh1 = X1 - ψx0
h2, v1 = ren(h1, wh1)
ψx1 = v1[1:size(A,1),:]
ψu1 = v1[size(A,1)+1:end,:]

# At t=2
X2 = A*X1 + B*ψu1
wh2 = X2 - ψx1
h3, v2 = ren(h2, wh2)
ψx2 = v2[1:size(A,1),:]
ψu2 = v2[size(A,1)+1:end,:]

# Validation for the system level constraints
diff = ψx2 - A*ψx1 - B*ψu1 - wh2
# @test all(norm(diff) <= 1e-6)