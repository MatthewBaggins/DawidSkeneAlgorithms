####################################
#             Imports              # 
####################################

using CategoricalArrays
using Clustering
using Distributions
using FillArrays
using LinearAlgebra
using Random: AbstractRNG, GLOBAL_RNG, randperm
using RDatasets
using Statistics
using StatsBase

include("utilities.jl")

####################################
#             Abstract             #
####################################

abstract type AbstractMixtureModel end

struct KMeans <: AbstractMixtureModel end

struct GMM <: AbstractMixtureModel end

####################################
#             KMeans               #
####################################

θ_KMeans = Matrix{Float64}
r_KMeans = Vector{Int}

function init_θ(
    ::KMeans, 
    rng::AbstractRNG, 
    x::Matrix{Float64}, 
    k::Int
)::θ_KMeans

    n = size(x)[1]
    rand_ids = randperm(rng, n)[1:k]
    μ = x[rand_ids, :]
    θ = μ
    return θ
end

function e_step(alg::KMeans, x::Matrix{Float64}, θ::θ_KMeans)::r_KMeans
    μ = θ
    mapslices(xᵢ -> compute_r(alg, xᵢ, μ), x; dims = 2)[:]
end

function compute_r(::KMeans, xᵢ::Vector{Float64}, μ::Matrix{Float64})::Int
    mapslices(μᵢ -> norm(μᵢ - xᵢ), μ; dims = 2)[:] |> argmin
end

function m_step(
    ::KMeans,
    x::Matrix{Float64},
    r::Vector{Int},
    θ::θ_KMeans
)::θ_KMeans

    μ = θ
    k = size(μ)[1]
    r_inds = [findall(equals(rᵢ), r) for rᵢ in 1:k]
    new_μ_vecs = [
        !isempty(inds) ? mean(x[inds, :], dims = 1)[:] : μ[i, :] 
        for (i, inds) in enumerate(r_inds)
            ]
    new_μ_matrix = permutedims(hcat(new_μ_vecs...))
    θ = new_μ_matrix
    return θ
end

####################################
#             GMM                  #
####################################

θ_GMM = Tuple{Vector{Float64}, Vector{MvNormal}} # Π, MvNormal
r_GMM = Matrix{Float64}

function init_θ(
    ::GMM, 
    rng::AbstractRNG, 
    x::Matrix{Float64}, 
    k::Int
)::θ_GMM

    N, D = size(x)
    μ_inds = randperm(rng, N)[1:k]
    Π = ones(k) / k
    𝓝 = [MvNormal(x[μ_ind, :], I(D))  for μ_ind in μ_inds]
    θ = (Π, 𝓝)
    return θ
end

function e_step(alg::GMM, x::Matrix{Float64}, θ::θ_GMM)::r_GMM
    Π, 𝓝 = θ
    K = length(Π)
    r = mapslices(xᵢ -> compute_r(alg, xᵢ, Π, 𝓝, K), x; dims = 2)
    return r
end

function compute_r(
    ::GMM,
    xᵢ::Vector{Float64}, 
    Π::Vector{Float64}, 
    𝓝::Vector{MvNormal}, 
    K::Int
)::Vector{Float64}

    r = [Π[k] * pdf(𝓝[k], xᵢ) for k in 1:K]
    return r /= sum(r)
end

function m_step(
    ::GMM, 
    x::Matrix{Float64}, 
    r::Matrix{Float64}, 
    θ::θ_GMM
)::θ_GMM

    Π, _ = θ
    K = length(Π)
    N, D = size(x)
    Nₖ = [sum([r[n, k] for n in 1:N]) for k in 1:K]
    new_μ = compute_new_μ(x, r, N, D, K, Nₖ)
    new_Σ = compute_new_Σ(x, new_μ, r, N, K, Nₖ)
    new_Π = compute_new_Π(N, Nₖ)
    new_𝓝 = [MvNormal(new_μ[k, :], new_Σ[k]) for k in 1:K]
    new_θ = (new_Π, new_𝓝)
    return new_θ
end

#TODO: functionalize
function compute_new_μ(
    x::Matrix{Float64}, 
    r::Matrix{Float64}, 
    N::Int, 
    D::Int, 
    K::Int, 
    Nₖ::Vector{Float64}
)::Matrix{Float64}

    new_μ = zeros(K, D)
    for k in 1:K
        new_μₖ = [r[n, k] * x[n, :] for n in 1:N]
        new_μ[k, :] = sum(permutedims(hcat(new_μₖ...)), dims=1) / Nₖ[k]
    end
    return new_μ
end

function compute_new_Σ(
    x::Matrix{Float64}, 
    new_μ::Matrix{Float64}, 
    r::Matrix{Float64}, 
    N::Int, 
    K::Int, 
    Nₖ::Vector{Float64}
)::Vector{Matrix{Float64}}

    new_Σ::Vector{Matrix{Float64}} = []
    for k in 1:K
        new_Σₖ = Hermitian(sum([r[n, k] * (x[n, :] - new_μ[k, :]) * transpose(x[n, :] - new_μ[k, :])  #TODO: tranpose with '
                                for n in 1:N]) / Nₖ[k])
        push!(new_Σ, new_Σₖ)
    end
    return new_Σ

end

function compute_new_Π(N::Int, Nₖ::Vector{Float64})::Vector{Float64}
    [Nₖ[k] / N for k in 1:length(Nₖ)]
end

####################################
#             Generic EM           #
####################################

function em(
    alg::AbstractMixtureModel, 
    x::Matrix{Float64};
    k::Int=3,
    n_steps::Int=10
)::Tuple{Vector, Vector}

    θ = init_θ(alg, GLOBAL_RNG, x, k)
    r_history = []
    θ_history = []
    for step in 1:n_steps
        r = e_step(alg, x, θ)
        θ = m_step(alg, x, r, θ)
        push!(r_history, r)
        push!(θ_history, θ)
    end

    return θ_history, r_history
end
