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

A = [1 2; 3 4]
B = [0; 1]

# Test constructors
ren_ps = SystemlevelRENParams{Float64}(nx, nv, A, B)
ren = REN(ren_ps)


# Test system level constraints

# Test pseudo inverse
