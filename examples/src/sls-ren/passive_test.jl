using Pkg
Pkg.activate("./")

using Revise
using Flux
using Flux.Optimise:update!
using Flux.Optimise:ADAM
using Zygote
using Random
using StableRNGs
using LinearAlgebra
using Plots
using MatrixEquations
using ControlSystems
using Distributions

using RobustNeuralNetworks

Q = PassiveRENParams{Float64}(10, 10, 10, 10, 1/3, 1/3)
Qe = REN(Q)


Random.seed!(123)
rng = StableRNG(0)

mean = 0
std = 1

dist = Normal(mean, std)

n_steps = 100
points_per_step = 100

data = [rand(dist, points_per_step) for _ in 1:n_steps]

steps_to_plot = 1:100

x_values = repeat(steps_to_plot, inner=points_per_step)
y_values = vcat(data[steps_to_plot]...)

scatter(x_values, y_values, title="Scatter Plot of Selected Steps", xlabel="Step", ylabel="Value", legend=false, markersize=3)

# for i in 1:n_steps
#     x1 = 2*randn(rng, 10)
#     # Keep track of outputs
#     y1 = zeros(n_steps, 100)

#     # Simulate and return outputs
#     x1, ya = Qe(x1, vec(data[i]))
#     y1[i,:] = ya
# end
x1 = 2*randn(rng, 10)
x1, ya = Qe(x1, randn(rng, 10))