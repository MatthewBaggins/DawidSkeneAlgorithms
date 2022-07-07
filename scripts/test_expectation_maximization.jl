using Revise

using Clustering
using ExpectationMaximization
using Test

include("load_datasets.jl")
include("evaluate.jl")

function em(alg::ExpectationMaximization.AMM, dataset::Dataset; n_steps::Int = 10)
    K = length(unique(dataset.y))
    μ_history, r_history = ExpectationMaximization.em(alg, dataset.x; k = K, n_steps = n_steps)
end

datasets = [load_iris(), load_beans(), load_stars()]

function main()
    # Setup and data
    algs = [KMeans(), GMM()]
    # iris_x, iris_y = load_iris()
    for dataset in datasets
        println("Dataset: $(dataset.name)")
        for alg in algs
            # Algorithm
            μ_history, r_history = em(alg, dataset)
            pred = tocategorical(r_history[end])
            # Evaluation
            mi = round(mutualinfo(pred, dataset.y); digits=3)
            # @test mi > .5
            println("alg: $alg:\tMI ≈ $mi")
        end
    end
end

# main()
