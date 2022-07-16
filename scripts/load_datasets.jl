using CSV
using DataFrames
using ExpectationMaximization: tocategorical
using JLD
using RDatasets

# DatasetClustering

struct DatasetClustering
    name::String
    x::Matrix
    y::Vector{Int}
end

function load_iris()::DatasetClustering
    iris = dataset("datasets", "iris")
    x = select(iris, Not(:Species)) |> Matrix
    y = iris.Species |> Vector{String} |> tocategorical
    return DatasetClustering("iris", x, y)
end

function load_beans()::DatasetClustering
    beans = CSV.read("data/beans.csv", DataFrame)
    x = select(beans, Not(:Class)) |> Matrix
    y = beans.Class |> Vector{String} |> tocategorical
    return DatasetClustering("beans", x, y)
end

function load_stars()::DatasetClustering
    stars = CSV.read("data/stars.csv", DataFrame)
    x = select(stars, Not(5, 6, 7)) |> Matrix
    y = stars.Type
    return DatasetClustering("stars", x, y)
end

# DatasetVoting

struct DatasetVoting
    name::String
    crowd_counts::AbstractArray{<:Real, 3}
    gold::DataFrame #TODO: 1. change to something more sensible; 2. use in eval; 3. document conversion
end

function load_adult2()::DatasetVoting
    DatasetVoting(
        "adult2",
        load("data/adult2_dataset/crowd_counts.jld")["crowd_counts"],
        CSV.read("data/adult2_dataset/gold.csv", DataFrame, header=false)
    )
end

function load_rte()::DatasetVoting
    DatasetVoting(
        "rte",
        load("data/rte_dataset/crowd_counts.jld")["crowd_counts"],
        CSV.read("data/rte_dataset/gold.csv", DataFrame, header=false)
    )
end

function load_toy()::DatasetVoting
    DatasetVoting(
        "toy",
        load("data/toy_dataset/crowd_counts.jld")["crowd_counts"],
        CSV.read("data/toy_dataset/gold.csv", DataFrame, header=false)
    )
end