using Revise
using BenchmarkTools

using DawidSkeneAlgorithms

function test_mixture_models()
    for dataset in load_clustering_datasets()
        println("=== Dataset: $(dataset.name) ===\n")
        for alg in CLUSTERING_ALGORITHMS
            println("Algorithm: $alg\nTime:")
            avg_mi = @btime em_avg($alg, $dataset) seconds = 5
            println("Average mutual information ≈ $avg_mi\n")
        end
    end
end

main = test_mixture_models
