####################################
#             Imports              # 
####################################

using CategoricalArrays
using Clustering
using Distributions
using FillArrays
using LinearAlgebra
using Pipe
using Random: AbstractRNG, GLOBAL_RNG, randperm
using RDatasets
using Statistics
using StatsBase
using StatsBase: sample

include("utilities.jl")

####################################
#             Abstract             #
####################################

abstract type AbstractEMAlgorithm end

# Mixture model algorithms

abstract type AbstractMixtureModel <: AbstractEMAlgorithm end
AMM = AbstractMixtureModel

struct KMeans <: AMM end

struct GMM <: AMM end

# Dawid Skene algorithms

abstract type AbstractDawidSkene <: AbstractEMAlgorithm end
ADS = AbstractDawidSkene

struct FastDawidSkene <: ADS end
FDS = FastDawidSkene

struct DawidSkene <: ADS end
DS = DawidSkene

struct HybridDawidSkene <: ADS end
HDS = HybridDawidSkene

# Other

struct MajorityVoting <: AbstractEMAlgorithm end
MV = MajorityVoting

####################################
#             KMeans               #
####################################

θ_KMeans = Matrix{<:Real}
r_KMeans = Vector{Int}

function init_θ(
    ::KMeans, 
    rng::AbstractRNG, 
    x::Matrix{<:Real}, 
    k::Int
)::θ_KMeans

    n = size(x)[1]
    rand_ids = randperm(rng, n)[1:k]
    μ = x[rand_ids, :]
    θ = μ
    return θ
end

function e_step(alg::KMeans, x::Matrix{<:Real}, θ::θ_KMeans)::r_KMeans
    μ = θ
    mapslices(xᵢ -> compute_r(alg, xᵢ, μ), x; dims = 2)[:]
end

function compute_r(::KMeans, xᵢ::Vector{<:Real}, μ::Matrix{<:Real})::Int
    mapslices(μᵢ -> norm(μᵢ - xᵢ), μ; dims = 2)[:] |> argmin
end

function m_step(
    ::KMeans,
    x::Matrix{<:Real},
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

θ_GMM = Tuple{Vector{<:Real}, Vector{MvNormal}} # Π, MvNormal
r_GMM = Matrix{<:Real}

function init_θ(
    ::GMM, 
    rng::AbstractRNG, 
    x::Matrix{<:Real}, 
    k::Int
    )::θ_GMM
    
    N, D = size(x)
    μ_inds = randperm(rng, N)[1:k]
    Π = ones(k) / k
    𝓝 = [MvNormal(x[μ_ind, :], I(D))  for μ_ind in μ_inds]
    θ = (Π, 𝓝)
    return θ
end

function e_step(alg::GMM, x::Matrix{<:Real}, θ::θ_GMM)::r_GMM
    Π, 𝓝 = θ
    K = length(Π)
    r = mapslices(xᵢ -> compute_r(alg, xᵢ, Π, 𝓝, K), x; dims = 2)
    return r
end

function compute_r(
    ::GMM,
    xᵢ::Vector{<:Real}, 
    Π::Vector{<:Real}, 
    𝓝::Vector{MvNormal}, 
    K::Int
    )::Vector{<:Real}
    
    r = [Π[k] * pdf(𝓝[k], xᵢ) for k in 1:K]
    
    if sum(r) > 0
        return r / sum(r)
    end
    return r
end

function m_step(
    ::GMM, 
    x::Matrix{<:Real}, 
    r::Matrix{<:Real}, 
    θ::θ_GMM
    )::θ_GMM

    Π, _ = θ
    K = length(Π)
    N, D = size(x)
    Nₖ = [sum([r[n, k] for n in 1:N]) for k in 1:K]
    new_μ = compute_new_μ(x, r, N, D, K, Nₖ)
    new_Σ = compute_new_Σ(x, new_μ, r, N, K, Nₖ)
    new_Π = compute_new_Π(N, Nₖ)
    new_𝓝 = [MvNormal(new_μ[k, :], new_Σ[k] + I * 1e-7) for k in 1:K]
    new_θ = (new_Π, new_𝓝)
    return new_θ
end

#TODO: functionalize
function compute_new_μ(
    x::Matrix{<:Real}, 
    r::Matrix{<:Real}, 
    N::Int, 
    D::Int, 
    K::Int, 
    Nₖ::Vector{<:Real}
    )::Matrix{<:Real}

    new_μ = zeros(K, D)
    for k in 1:K
        new_μₖ = [r[n, k] * x[n, :] for n in 1:N]
        new_μ[k, :] = sum(permutedims(hcat(new_μₖ...)), dims=1) / Nₖ[k]
    end
    return new_μ
end

function compute_new_Σ(
    x::Matrix{<:Real}, 
    new_μ::Matrix{<:Real}, 
    r::Matrix{<:Real}, 
    N::Int, 
    K::Int, 
    Nₖ::Vector{<:Real}
    )::Vector{Matrix{<:Real}}
    
    new_Σ = []
    for k in 1:K
        new_Σₖ = Hermitian(sum([r[n, k] * (x[n, :] - new_μ[k, :]) * transpose(x[n, :] - new_μ[k, :])  #TODO: tranpose with '
        for n in 1:N]) / Nₖ[k])
        push!(new_Σ, convert(Matrix{Float64}, new_Σₖ))
    end
    return new_Σ
    
end

function compute_new_Π(N::Int, Nₖ::Vector{<:Real})::Vector{<:Real}
    [Nₖ[k] / N for k in 1:length(Nₖ)]
end

####################################
#             Dawid-Skene          #
####################################

function calc_likelihood(::ADS, counts, class_marginals, error_rates)
    nPatients, nObservers, nClasses = size(counts)
    log_L = 0.0
    
    for i in 1:nPatients
        patient_likelihood = 0.0
        for j in 1:nClasses
            class_prior = class_marginals[j]
            patient_class_likelihood = prod(error_rates[:, j, :] .^ counts[i, :, :])
            patient_class_posterior = class_prior * patient_class_likelihood
            patient_likelihood += patient_class_posterior
        end 
        temp = log_L + log(patient_likelihood)
        if isnan(temp) || isnothing(temp) || isinf(temp)
            error("Invalid temp value: $temp")
        end
        log_L = temp
    end
    return log_L
end

function initialize_question_classes(::FDS, counts::Array{<:Real, 3})
    nQuestions, nParticipants, nClasses = size(counts)
    response_sums = reshape(sum(counts, dims = 2), (nQuestions, nClasses))
    # display(response_sums)
    question_classes = zeros(nQuestions, nClasses)
    # if mode == "FDS" ||
    for p in 1:nQuestions
        maxval = maximum(response_sums[p, :])
        maxinds = argwhere(response_sums[p, :], ==(maxval))
        question_classes[p, sample(maxinds, 1)[1]] = 1 #TODO: add RNG
    end
    # else 
    return question_classes
end

function m_step(
    ::FDS,
    counts, 
    question_classes
)
    nQuestions, nParticipants, nClasses = size(counts)
    class_marginals = sum(question_classes, dims = 1) ./ nQuestions
    error_rates = zeros(nParticipants, nClasses, nClasses)
    for k in 1:nParticipants
        for j in 1:nClasses
            for l in 1:nClasses
                error_rates[k, j, l] = question_classes[:, j]' * counts[:, k, l]
            end
            sum_over_responses = sum(error_rates[k, j, :])
            if sum_over_responses > 0
                error_rates[k, j, :] = error_rates[k, j, :] / sum_over_responses
            end
        end
    end
    return class_marginals, error_rates
end

function e_step(
    ::FDS,
    counts,
    class_marginals, 
    error_rates
)
    nQuestions, nParticipants, nClasses = size(counts)
    question_classes = zeros(nQuestions, nClasses)
    final_classes = zeros(nQuestions, nClasses)
    for i in 1:nQuestions
        for j in 1:nClasses
            estimate = class_marginals[j] * prod(error_rates[:, j, :] .^ counts[i, :, :])
            question_classes[i, j] = estimate
        end
        # if mode ...
        maxval = maximum(question_classes[i, :])
        maxinds = argwhere(question_classes[i, :], ==(maxval))
        final_classes[i, sample(maxinds, 1)[1]] = 1
    end
    return final_classes
end

function em(
    alg::ADS,
    counts::AbstractArray{<:Real, 3}
    # args: algorithm and verbose #TODO: implement
    ;
    tol = .0001,
    CM_tol = .005,
    max_iter = 100,
    verbose::Bool=true
)

    # Initialize
    question_classes = initialize_question_classes(alg, counts)
    nIter = 0
    converged = false
    old_class_marginals = nothing
    old_error_rates = nothing
    # total_time = 0
    log_L = nothing
    while !converged
        nIter += 1
        class_marginals, error_rates = m_step(alg, counts, question_classes)
        question_classes = e_step(alg, counts, class_marginals, error_rates)
        log_L = calc_likelihood(alg, counts, class_marginals, error_rates)
        
        # Check for convergence
        if old_class_marginals ≠ nothing
            class_marginals_diff = sum(abs.(class_marginals - old_class_marginals))
            error_rates_diff = sum(abs.(error_rates - old_error_rates))
            # if verbose 
            if class_marginals_diff < tol || nIter >= max_iter
                converged = true
            end #elif mode == "H"
        end # else if verbose
        old_class_marginals = class_marginals
        old_error_rates = error_rates
        if verbose
            @show nIter
            @show log_L
        end
    end
    result = @pipe argmax(question_classes, dims = 2)[:] |> map(x -> x[2], _)
end



####################################
#             Generic EM           #
####################################

function em(
    alg::AbstractMixtureModel, 
    x::Matrix{<:Real};
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
