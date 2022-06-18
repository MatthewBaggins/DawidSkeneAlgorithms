using LinearAlgebra
using Random
using RDatasets
using Statistics
using StatsBase

# x
iris = dataset("datasets", "iris")

function as_vecs(m::Matrix{T})::Vector{Vector{T}} where T<:Real
    [m[i, :] for i in 1:size(m)[1]]
end

function compute_r(o::Vector{Float64}, μ::Vector{Vector{Float64}})::Int64
    map(μᵢ -> norm(μᵢ - o), μ) |> argmin
end

# assign data points to clusters
function e_step(observations::Vector{Vector{Float64}}, μ::Vector{Vector{Float64}})::Vector{Int64}
    map(o -> compute_r(o, μ), observations) # can this function be easily curried?
end

# find new cluster centers
function m_step(observations::Vector{Vector{Float64}}, r::Vector{Int64})#::Vector{Vector{Float64}}
    r_inds = [findall(rⱼ -> rⱼ == rᵢ, r) for rᵢ in minimum(r):maximum(r)]
    μ = [mean(observations[inds]) for inds in r_inds]
    return μ
end

# Pick `k` (number of clusters) vectors from the observations as initial cluster centers
function init_μ(observations::Vector{Vector{Float64}}, k::Integer)::Vector{Vector{Float64}}
    Random.seed!(42)
    n = length(observations) # number of observations
    rand_ids = randperm(n)[1:k]
    return observations[rand_ids]
end


# k - number of clusters
function em(observations::Vector{Vector{Float64}}, k::Integer=3, n_steps::Integer=10)
    # Step 1
    # `k` `d`-dimensional vectors - prototypes for the clusters
    μ = init_μ(observations, k)
    # class assignments / predcitions
    r_history = []
    μ_history = []
    # Next steps
    for _ in 1:n_steps
        r = e_step(observations, μ)
        μ = m_step(observations, r)
        push!(r_history, r)
        push!(μ_history, μ)
    end

    return μ_history, r_history
end

function compare(preds::Vector, labels::Vector)#::Float64
    pred_val2inds = Dict(
        pred_val => findall(p -> p == pred_val, preds) 
        for pred_val in minimum(preds):maximum(preds))
    pred_val2labels = Dict(
        pred_val => labels[inds] 
        for (pred_val, inds) in pred_val2inds)
    pred_val2label_counts = Dict(
        pred_val => countmap(labels) 
        for (pred_val, labels) in pred_val2labels)
    
    display(pred_val2inds)
    println()
    display(pred_val2labels)
    println()
    display(pred_val2label_counts)
end

function main()
    observations = select(iris, Not(:Species)) |> Matrix |> as_vecs
    y = iris.Species |> Vector
    μ_history, r_history = em(observations)
    preds = r_history[end]
    compare(preds, y)
end

main()
