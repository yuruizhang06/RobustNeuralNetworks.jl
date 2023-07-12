# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

using Flux
using Random
using RobustNeuralNetworks
using Test

# include("../test_utils.jl")

"""
Test that backpropagation runs and parameters change
"""
batches = 10
nu, nx, nv, ny, γ = 4, 5, 0, 2, 10
ren_ps = LipschitzRENParams{Float64}(nu, nx, nv, ny, γ)
model = DiffREN(ren_ps)

# Dummy data
us = randn(nu, batches)
ys = randn(ny, batches)
data = [(us, ys)]

# Dummy loss function just for testing
function loss(m, u, y)
    x0 = init_states(m, size(u,2))
    x1, y1 = m(x0, u)
    return Flux.mse(y1, y) + sum(x1.^2)
end

# Debug batch updates
opt_state = Flux.setup(Adam(0.01), model)
gs = Flux.gradient(loss, model, us, ys)
println()

# # Check if parameters change after a Flux update
# ps1 = deepcopy(Flux.params(model))
# opt_state = Flux.setup(Adam(0.01), model)
# Flux.train!(loss, model, data, opt_state)
# ps2 = Flux.params(model)

# @test !any(ps1 .≈ ps2)
