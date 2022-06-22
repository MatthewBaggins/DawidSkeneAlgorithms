#TODO: module ?
using CategoricalArrays
using Clustering
using LinearAlgebra
using Random: AbstractRNG, GLOBAL_RNG, randperm
using RDatasets
using Statistics
using StatsBase

include("utilities.jl")

function compute_r(o::Vector{Float64}, μ::Matrix{Float64})::Int
    mapslices(μᵢ -> norm(μᵢ - o), μ; dims=2)[:] |> argmin
end

# assign data points to clusters
function e_step(observations::Matrix{Float64}, μ::Matrix{Float64})::Vector{Int}
    mapslices(o -> compute_r(o, μ), observations; dims=2)[:]
end

# find new cluster centers
function m_step(observations::Matrix{Float64}, r::Vector{Int}, μ::Matrix{Float64}, k::Int)::Matrix{Float64}
    r_inds = [findall(equals(rᵢ), r) for rᵢ in 1:k]
    new_μ_vecs = [
        !isempty(inds) ? mean(observations[inds, :], dims=1)[:] : μ[i, :] 
        for (i, inds) in enumerate(r_inds)
            ]
    new_μ_matrix = permutedims(hcat(new_μ_vecs...))
    return new_μ_matrix
end

# Pick `k` (number of clusters) vectors from the observations as initial cluster centers
function init_μ(rng::AbstractRNG, observations::Matrix{Float64}, k::Int)::Matrix{Float64}
    n = size(observations)[1] # number of observations
    rand_ids = randperm(rng, n)[1:k]
    return observations[rand_ids, :]
end

# k - number of clusters
function em(observations::Matrix{Float64}; k::Int=3, n_steps::Int=10)::Tuple{Vector, Vector}
    # Step 1
    # `k` `d`-dimensional vectors - prototypes for the clusters
    μ = init_μ(GLOBAL_RNG, observations, k)
    # class assignments / predcitions
    r_history = []
    μ_history = []
    # Next steps
    for step in 1:n_steps
        # @show step
        r = e_step(observations, μ)
        μ = m_step(observations, r, μ, k)
        push!(r_history, r)
        push!(μ_history, μ)
    end

    return μ_history, r_history
end
