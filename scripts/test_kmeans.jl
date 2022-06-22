using DataFrames
using RDatasets

include("../src/kmeans.jl")

function main()
    iris = dataset("datasets", "iris")
    observations = select(iris, Not(:Species)) |> Matrix
    y_labels = iris.Species |> Vector{String}
    μ_history, r_history = em(observations)
    y_preds = r_history[end]
    
    y2cat, y_categorical = labels2int(y_labels, y_preds)
    display(counts(y_preds, y_categorical))
    
    @show randindex(y_preds, y_categorical)
    @show Clustering.varinfo(y_preds, y_categorical)
    @show vmeasure(y_preds, y_categorical)
    @show mutualinfo(y_preds, y_categorical)
    print()
end

main()

