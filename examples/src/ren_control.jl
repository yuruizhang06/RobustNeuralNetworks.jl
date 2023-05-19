cd(@__DIR__)
using Pkg
Pkg.activate("../")

using CairoMakie
using ControlSystems
using Convex
using LinearAlgebra
using Random
using RobustNeuralNetworks

do_plot = false
rng = MersenneTwister(42)

# System parameters and poles: λ = ρ*exp(± im ϕ)
ρ = 0.8
ϕ = 0.2π
λ = ρ .* [cos(ϕ) + sin(ϕ)*im, cos(ϕ) - sin(ϕ)*im] #exp.(im*ϕ.*[1,-1])

# Construct discrete-time system with gain 0.3, sampling time 1.0s
k = 0.3
Ts = 1.0
sys = zpk([], λ, k, Ts)

# Closed-loop system components
sim_sys(u::AbstractMatrix) = lsim(sys, u, 1:size(u,2))[1]
T0(u) = sim_sys(u)
T1(u) = sim_sys(u)
T2(u) = -sim_sys(u)

# Sample disturbances
function sample_disturbance(amplitude=10, samples=30, hold=50)
    d = 2 * amplitude * (rand(rng, 1, samples) .- 0.5)
    return kron(d, ones(1, hold))
end
d = sample_disturbance()
batches = size(d, 2)

# Set up a contracting REN whose outputs are yt = [xt; wt; ut]
nu = 1
nx, nv = 10, 20
ny = nx + nv + nu
ren_ps = ContractingRENParams{Float64}(nu, nx, nv, nx; rng=rng)
model  = REN(ren_ps)

model.explicit.C2  = Matrix{Float64}([I(nx); zeros(nv, nx); zeros(nu, nx)])
model.explicit.D21 = Matrix{Float64}([zeros(nx, nv); I(nv); zeros(nu, nv)])
model.explicit.D22 = Matrix{Float64}([zeros(nx, nu); zeros(nv, nu); I(nu)])
model.explicit.by  = zeros(ny)

# Echo-state network params θ = [C2, D21, D22, by]
θ = Convex.Variable(1, nx+nv+nu+1)

# Echo-state components (add ones for bias vector)
function Qᵢ(u)
    x0 = init_states(model, batches)
    _, y = model(x0, u)
    return [y; ones(1,batches)]
end

# Complete the closed-loop response and control inputs 
# z = T₀ + ∑ θᵢ*T₁(Qᵢ(T₂(d)))
# u = ∑ θᵢ*Qᵢ(T₂(d))
z0 = T0(d)
ỹ  = T2(d)
ũ  = Qᵢ(ỹ)
z1 = reduce(vcat, T1(ũ') for ũ in eachrow(ũ))
z  = z0 + θ * z1
u  = θ * ũ

# Optimize the closed-loop response
J = norm(z, 1)


# Plot just to check
if do_plot
    f = lines(vec(d))
    lines!(vec(z))
    display(f)
end