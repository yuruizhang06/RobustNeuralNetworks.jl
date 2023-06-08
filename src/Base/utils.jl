# This file is a part of RobustNeuralNetworks.jl. License is MIT: https://github.com/acfr/RobustNeuralNetworks.jl/blob/main/LICENSE 

"""
    glorot_normal(n::Int, m::Int; T=Float64, rng=Random.GLOBAL_RNG)

Generate matrices or vectors from the Glorot normal distribution.
"""
glorot_normal(n::Int, m::Int; T=Float64, rng=Random.GLOBAL_RNG) = 
    convert.(T, randn(rng, n, m) / sqrt(n + m))
glorot_normal(n::Int; T=Float64, rng=Random.GLOBAL_RNG) = 
    convert.(T, randn(rng, n) / sqrt(n))
