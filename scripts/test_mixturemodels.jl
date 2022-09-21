using Revise

using Clustering
using StatsBase
using ExpectationMaximization
import ExpectationMaximization: em
using Test

include("../src/load_datasets.jl")
# include("evaluate.jl")

# Clustering datasets
datasets = [load_iris(), load_beans(), load_stars()]

"""
A "wrapper method" to simplify calling `em` on `DatasetClustering` structs. 
"""
function em(
    alg::ExpectationMaximization.AMM,
    dataset::DatasetClustering;
    n_steps::Int=10
)
    K = length(unique(dataset.y))
    μ_history, r_history = ExpectationMaximization.em(alg, dataset.x;
        k=K, n_steps=n_steps)
    pred = tocategorical(r_history[end])
end

"""
Run `em` `n_runs` times on the `dataset` and average their MI scores.
"""
function em_avg(
    alg::ExpectationMaximization.AMM,
    dataset::DatasetClustering;
    n_runs::Int=20
)::Float64
    preds = [em(alg, dataset) for _ in 1:n_runs]
    mis = [round(mutualinfo(pred, dataset.y); digits=3) for pred in preds]
    return mean(mis)
end

function main()
    for dataset in datasets
        println("=== Dataset: $(dataset.name) ===\n")
        for alg in CLUSTERING_ALGORITHMS
            println("Algorithm: $alg")
            mean_mi = em_avg(alg, dataset)
            println("Mean mutual information ≈ $mean_mi\n")
        end
    end
end

# main()
