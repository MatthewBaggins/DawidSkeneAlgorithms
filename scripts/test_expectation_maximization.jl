using Clustering
using ExpectationMaximization
using Test

include("load_datasets.jl")
include("evaluate.jl")

function main()
    # Setup and data
    algs = [KMeans(), GMM()]
    iris_x, iris_y = load_iris()
    for alg in algs    
        # Algorithm
        μ_history, r_history = em(alg, iris_x, n_steps=100)
        iris_pred = tocategorical(r_history[end])
        # Evaluation
        mi = round(mutualinfo(iris_pred, iris_y), digits=3)
        @test mi > .5
        println("alg: $alg:\tMI ≈ $mi")
        # evaluate(iris_pred, iris_y)
    end
end

main()
