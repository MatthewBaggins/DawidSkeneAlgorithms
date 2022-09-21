using Revise

using BenchmarkTools

using ExpectationMaximization

include("../src/load_datasets.jl")

# Voting datasets
datasets = [load_adult2(), load_rte()]

function main()
    for dataset in datasets
        println("=== Dataset: $(dataset.name) ===\n")
        for alg in VOTING_ALGORITHMS
            println("Algorithm: $alg\nTime:")
            result, negloglik = @btime em($alg, $(dataset.crowd_counts)) seconds = 5
            println("Negative log-likelihood: \t$(round(negloglik; digits=2))\n")
        end
    end
end

# main()
