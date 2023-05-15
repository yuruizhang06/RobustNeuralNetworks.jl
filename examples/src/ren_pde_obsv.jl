using Revise
# using BenchmarkTools
using Distributions
using Flux
using Flux.Optimise:update!
using Flux.Optimise
using Zygote
using Random
using LinearAlgebra
using Formatting
using Plots
# using DifferentialEquations

using RobustNeuralNetworks

# Observer design experiment - start with linear system

nv = 200
n = 51
m = 1
p = 1


# Generate data from finite difference approximation of heat equation
function reaction_diffusion_equation(;L=10.0, steps=5, nx=51, c=1.0, sigma=0.1, process_noise=0.0, measurement_noise=0.0)
    dx = L / (nx - 1)
    dt = sigma * dx^2

    xs = range(0.0, length=nx, stop=L)

    function f(u0, d)
        u = copy(u0)
        un = copy(u0)
        for t in 1:steps
            u = copy(un) 
            # FD approximation of heat equation
            f_local(v) = v[2:end - 1, :] .* (1 .- v[2:end - 1, :]) .* ( v[2:end - 1, :] .- 0.5)
            laplacian(v) = (v[1:end - 2, :] + v[3:end, :] - 2v[2:end - 1, :]) / dx^2
            
            # Euler step for time
            un[2:end - 1, :] = u[2:end - 1, :] + dt * (laplacian(u) + f_local(u) / 2 ) +
                                    process_noise*randn(size(u[2:end - 1, :]))

            # Boundary condition
            un[1:1, :]   = d;
            un[end:end, :] = d;
        end
        return u
    end

    function g(u, d)
        return [d .+ measurement_noise*randn(1, size(d, 2));
                u[end ÷ 2:end ÷ 2, :] .+ measurement_noise*randn(1, size(u, 2))]
    end
    return f, g
end

f, g = reaction_diffusion_equation()

nPoints = 100000
X = zeros(n, nPoints)
U = zeros(m, nPoints)
for t in 1:nPoints - 1
    X[:, t + 1:t + 1] = f(X[:, t:t], U[:, t:t])
    
    # Calculate next u
    u_next = U[1,t] .+ 0.05f0 * randn(Float64)
    if u_next > 1
        u_next = 1
    elseif u_next < 0
        u_next = 0
    end
    U[:,t + 1] .= u_next
end
xt = X[:, 1:end - 1]
xn = X[:, 2:end]
y = g(X, U)

input_data = [U; y][:, 1:end - 1]  # inputs to observer
batchsize = 20

data = Flux.Data.DataLoader((xn, xt, input_data), batchsize=batchsize, shuffle=true)

# Model parameters
nx = n
nu = size(input_data, 1)
ny = nx

# Constuction REN
model = ContractingRENParams{Float64}(nu, nx, nv, ny; is_output = false)

# function contracting_trainable_(L::DirectRENParams)
#     ps = [L.ρ, L.X, L.Y1, L.B2, L.D12, L.bx, L.bv]
#     !(L.polar_param) && popfirst!(ps)
#     return filter(p -> length(p) !=0, ps)
# end
# model.direct.C2 = Matrix(1.0I, nx, nx)
# model.direct.D21 = zeros(nx,nv)
# model.direct.by = zeros(nx)

# Flux.trainable(m::ContractingRENParams) = contracting_trainable_(m.direct)

function train_observer!(model, data, opt; Epochs=200, regularizer=nothing, solve_tol=1E-5, min_lr=1E-7)
    θ = Flux.trainable(model)
    ps = Flux.Params(θ)
    # model_e = REN(model)
    mean_loss = [1E5]
    loss_std = []
    for epoch in 1:Epochs
        batch_loss = []
        for (xni, xi, ui) in data
            model_e = REN(model)
            function calc_loss()
                xpred = model_e(xi, ui)[1]
                return mean(norm(xpred[:, i] - xni[:, i]).^2 for i in 1:size(xi, 2))
            end

            train_loss, back = Zygote.pullback(calc_loss, ps)

            # Calculate gradients and update loss
            ∇J = back(one(train_loss))
            update!(opt, ps, ∇J)
        
            push!(batch_loss, train_loss)
            printfmt("Epoch: {1:2d}\tTraining loss: {2:1.4E} \t lr={3:1.1E}\n", epoch, train_loss, opt.eta)
        end

        # Print stats through epoch
        println("------------------------------------------------------------------------")
        printfmt("Epoch: {1:2d} \t mean loss: {2:1.4E}\t std: {3:1.4E}\n", epoch, mean(batch_loss), std(batch_loss))
        println("------------------------------------------------------------------------")
        push!(mean_loss, mean(batch_loss))
        push!(loss_std, std(batch_loss))

        # Check for decrease in loss.
        if mean_loss[end] >= mean_loss[end - 1]
            println("Reducing Learning rate")
            opt.eta *= 0.1
            if opt.eta <= min_lr  # terminate optim.
                return mean_loss, loss_std
            end
        end
    end
    return mean_loss, loss_std
end

opt = Flux.Optimise.ADAM(1E-3)
tloss, loss_std = train_observer!(model, data, opt; Epochs=200, min_lr=1E-7)

# Test observer
T = 1000
time = 1:T

u = ones(Float64, m, length(time)) / 2
x = ones(Float64, n, length(time))

for t in 1:T - 1
    x[:, t + 1] = f(x[:, t:t], u[t:t])
    
    # Calculate next u
    u_next = u[t] + 0.05f0 * (randn(Float64))
    if u_next > 1
        u_next = 1
    elseif u_next < 0
        u_next = 0
    end
    u[t + 1] = u_next
end
y = [g(x[:, t:t], u[t]) for t in time]

batches = 1
observer_inputs = [repeat([ui; yi], outer=(1, batches)) for (ui, yi) in zip(u, y)]
# println(typeof(observer_inputs),size(observer_inputs))

# Foward simulation
function simulate(model_e::REN, x0, u)
    eval_cell = (x, u) -> model_e(x, u)
    recurrent = Flux.Recur(eval_cell, x0)
    output = [recurrent(input) for input in u]
    return output
end

x0 = zeros(nx, batches)
model_e = REN(model)
xhat = simulate(model_e, x0, observer_inputs)

p1 = heatmap(x, color=:cividis, aspect_ratio=1);

Xhat = reduce(hcat, xhat)
p2 = heatmap(Xhat[:, 1:batches:end], color=:cividis, aspect_ratio=1);
p3 = heatmap(abs.(x - Xhat[:, 1:batches:end]), color=:cividis, aspect_ratio=1);

p = plot(p1, p2, p3; layout=(3, 1))
# savefig(p,"pde_observer.png")