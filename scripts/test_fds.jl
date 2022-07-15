
using Revise

using Clustering
using ExpectationMaximization
import ExpectationMaximization: em
using Test

include("load_datasets.jl")

datasets = [load_adult2(), load_rte()]

function main()
    # Setup and data
    for dataset in datasets
        println("Dataset: $(dataset.name)")
        em(FDS(), dataset.crowd_counts)
    end
end

# main()
