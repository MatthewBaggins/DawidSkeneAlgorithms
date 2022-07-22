using Revise

using BenchmarkTools
# using Test

using ExpectationMaximization
import ExpectationMaximization: em

include("../src/load_datasets.jl")

datasets = [load_adult2(), load_rte()]

function main()
    # Setup and data
    for dataset in datasets
        for alg in VOTING_ALGORITHMS[1:end-1] # no MV (for now)
            println("Dataset:\t$(dataset.name)")
            println("Algorithm:\t$alg")
            println("Time:")
            # result, negloglik = em(alg, dataset.crowd_counts)
            result, negloglik = @btime em($alg, $(dataset.crowd_counts)) seconds=5
            println("Negative log-likelihood: $negloglik")
        end
    end
end

# main()
