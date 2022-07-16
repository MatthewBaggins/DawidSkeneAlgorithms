using Revise

using Clustering
using ExpectationMaximization
import ExpectationMaximization: em
using Test

include("load_datasets.jl")

datasets = [load_adult2(), load_rte(), load_toy()]

function main()
    # Setup and data
    for dataset in datasets
        for alg  in VOTING_ALGORITHMS[1:end-1]
            println("Dataset: $(dataset.name)")
            println("Algorithm: $alg")
            em(alg, dataset.crowd_counts)
            println()
        end
    end
end

# main()
