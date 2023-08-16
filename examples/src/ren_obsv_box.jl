# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

cd(@__DIR__)
using Pkg
Pkg.activate("..")

using CairoMakie
using CUDA
using Flux
using Printf
using Random
using RobustNeuralNetworks
using Statistics

rng = MersenneTwister(0)
dev = gpu
T = Float32

# TODO: Currently getting NaNs in gradient
# The loss function does not give consistent values for a given input
# In fact, the REN does not give consistent next states for a given input!
# This is all fine on the CPU, but kills it on the GPU

#####################################################################
# Problem setup

# System parameters
m = 1                   # Mass (kg)
k = 5                   # Spring constant (N/m)
μ = 0.5                 # Viscous damping coefficient (kg/m)
nx = 2                  # Number of states

# Continuous and discrete dynamics and measurements
_visc(v::Matrix) = μ * v .* abs.(v)
f(x::Matrix,u::Matrix) = [x[2:2,:]; (u[1:1,:] - k*x[1:1,:] - _visc(x[2:2,:]))/m]
fd(x,u) = x + dt*f(x,u)
gd(x::Matrix) = x[1:1,:]

# Generate training data
dt = T(0.01)               # Time-step (s)
Tmax = 10               # Simulation horizon
ts = 1:Int(Tmax/dt)     # Time array indices

batches = 200
u  = fill(zeros(T, 1, batches), length(ts)-1)
X  = fill(zeros(T, 1, batches), length(ts))
X[1] = (2*rand(rng, T, nx, batches) .- 1) / 2

for t in ts[1:end-1]
    X[t+1] = fd(X[t],u[t])
end

Xt = X[1:end-1]
Xn = X[2:end]
y = gd.(Xt)

# Store data for training
observer_data = [[ut; yt] for (ut,yt) in zip(u, y)]
indx = shuffle(rng, 1:length(observer_data))
data = zip(Xn[indx], Xt[indx], observer_data[indx]) |> dev


#####################################################################
# Train a model

# Define a REN model for the observer
nv = 20
nu = size(observer_data[1], 1)
ny = nx
model_ps = ContractingRENParams{T}(nu, nx, nv, ny; output_map=false, rng)
# model = REN(model_ps) |> dev
model = DiffREN(model_ps) |> dev

# Loss function: one step ahead error (average over time)
function loss(model, xn, xt, inputs)
    xpred = model(xt, inputs)[1]
    return mean(sum((xn - xpred).^2; dims=1))
end


# TODO: Testing with GPU
xn, xt, inputs = data.is[1][1], data.is[2][1], data.is[3][1]
# train_loss, ∇J = Flux.withgradient(loss, model, xn, xt, inputs)

function test_me()
    x0 = model(xt, inputs)[1]
    all_good = true
    for _ in 1:1000
        xpred = model(xt, inputs)[1]
        !(xpred ≈ x0) && (all_good = false)
        x0 = xpred
    end
    return all_good
end

println("Evaluates correctly? ", test_me())


# # Train the model
# function train_observer!(model, data; epochs=50, lr=1e-3, min_lr=1e-6)

#     opt_state = Flux.setup(Adam(lr), model)
#     mean_loss = [T(1e5)]
#     for epoch in 1:epochs

#         batch_loss = []
#         for (xn, xt, inputs) in data
#             train_loss, ∇J = Flux.withgradient(loss, model, xn, xt, inputs)
#             Flux.update!(opt_state, model, ∇J[1])
#             push!(batch_loss, train_loss)
#             println(train_loss)
#         end
#         @printf "Epoch: %d, Lr: %.1g, Loss: %.4g\n" epoch lr mean(batch_loss)

#         # Drop learning rate if mean loss is stuck or growing
#         push!(mean_loss, mean(batch_loss))
#         if (mean_loss[end] >= mean_loss[end-1]) && !(lr < min_lr || lr ≈ min_lr)
#             lr = 0.1lr
#             Flux.adjust!(opt_state, lr)
#         end
#     end
#     return mean_loss
# end
# tloss = train_observer!(model, data)


# #####################################################################
# # Generate test data

# # Generate test data (a bunch of initial conditions)
# batches   = 50
# ts_test   = 1:Int(20/dt)
# u_test    = fill(zeros(T, 1, batches), length(ts_test))
# x_test    = fill(zeros(T, nx,batches), length(ts_test))
# x_test[1] = (2*rand(rng, T, nx, batches) .- 1) / 5

# for t in ts_test[1:end-1]
#     x_test[t+1] = fd(x_test[t], u_test[t])
# end
# observer_inputs = [[u;y] for (u,y) in zip(u_test, gd.(x_test))]


# #######################################################################
# # Simulate observer error

# # Simulate the model through time
# function simulate(model::AbstractREN, x0, u)
#     recurrent = Flux.Recur(model, x0)
#     output = recurrent.(u)
#     return output
# end
# x0hat = init_states(model, batches)
# xhat = simulate(model, x0hat |> dev, observer_inputs |> dev)

# # Plot results
# function plot_results(x, x̂, ts)

#     # Observer error
#     Δx = x .- x̂

#     ts = ts.*dt
#     _get_vec(x, i) = reduce(vcat, [xt[i:i,:] for xt in x])
#     q   = _get_vec(x,1)
#     q̂   = _get_vec(x̂,1)
#     qd  = _get_vec(x,2)
#     q̂d  = _get_vec(x̂,2)
#     Δq  = _get_vec(Δx,1)
#     Δqd = _get_vec(Δx,2)

#     fig = Figure(resolution = (600, 400))
#     ga = fig[1,1] = GridLayout()

#     ax1 = Axis(ga[1,1], xlabel="Time (s)", ylabel="Position (m)", title="States")
#     ax2 = Axis(ga[1,2], xlabel="Time (s)", ylabel="Position (m)", title="Observer Error")
#     ax3 = Axis(ga[2,1], xlabel="Time (s)", ylabel="Velocity (m/s)")
#     ax4 = Axis(ga[2,2], xlabel="Time (s)", ylabel="Velocity (m/s)")
#     axs = [ax1, ax2, ax3, ax4]

#     for k in axes(q,2)
#         lines!(ax1, ts,  q[:,k],  linewidth=0.5,  color=:grey)
#         lines!(ax1, ts,  q̂[:,k],  linewidth=0.25, color=:red)
#         lines!(ax2, ts, Δq[:,k],  linewidth=0.5,  color=:grey)
#         lines!(ax3, ts,  qd[:,k], linewidth=0.5,  color=:grey)
#         lines!(ax3, ts,  q̂d[:,k], linewidth=0.25, color=:red)
#         lines!(ax4, ts, Δqd[:,k], linewidth=0.5,  color=:grey)
#     end

#     qmin, qmax = minimum(minimum.((q,q̂))), maximum(maximum.((q,q̂)))
#     qdmin, qdmax = minimum(minimum.((qd,q̂d))), maximum(maximum.((qd,q̂d)))
#     ylims!(ax1, qmin, qmax)
#     ylims!(ax2, qmin, qmax)
#     ylims!(ax3, qdmin, qdmax)
#     ylims!(ax4, qdmin, qdmax)
#     xlims!.(axs, ts[1], ts[end])
#     display(fig)
#     return fig
# end
# fig = plot_results(x_test, xhat |> cpu, ts_test)
# save("../results/ren-obsv/ren_box_obsv.svg", fig)
