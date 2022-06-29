using DataFrames
using RDatasets

include("../src/ExpectationMaximization/src/em_kmeans_and_gmm.jl")

function main()
    # Setup and data
    alg = KMeans()
    iris = dataset("datasets", "iris")
    x = select(iris, Not(:Species)) |> Matrix
    y_labels = iris.Species |> Vector{String}
    
    # Algorithm
    μ_history, r_history = em(alg, x, n_steps=100)
    y_preds = r_history[end]
    
    # Evaluation
    evaluate(tocategorical(y_preds), tocategorical(y_labels))
end

function evaluate(y_preds::Vector{Int}, y_labels::Vector{Int})
    println("Mutual information: $(round(mutualinfo(y_preds, y_labels), digits = 3))")
    println("Counts matrix [ predicted × true labels ]")
    countmat = diagreshufflematrix(counts(y_preds, y_labels))
    display(countmat)
end

main()
