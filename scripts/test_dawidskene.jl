using Revise
using BenchmarkTools

using DawidSkeneAlgorithms

function test_dawidskene()
    for dataset in load_voting_datasets()
        println("=== Dataset: $(dataset.name) ===\n")
        for alg in VOTING_ALGORITHMS
            println("Algorithm: $alg\nTime:")
            avg_nll, avg_mi = @btime em_avg($alg, $dataset) seconds = 5
            # result, negloglik = @btime em($alg, $(dataset.x)) seconds = 5
            println("Average negative log-likelihood: $(avg_nll))\n")
            println("Average mutual information ≈ $avg_mi\n")
        end
    end
end

main = test_dawidskene
