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
        println("\n=== Dataset: $(dataset.name) ===")
        for alg in VOTING_ALGORITHMS
            println("Alg: $alg")
            result, negloglik = @btime em($alg, $(dataset.crowd_counts)) seconds=5
            println("-log(p):\t$(round(negloglik; digits=2))\n")
        end
    end
end

# main()
