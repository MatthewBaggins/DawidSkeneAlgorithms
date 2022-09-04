# Abstract

abstract type AbstractDataset end

function show(io::IO, d::AbstractDataset)
    println(io,
        """$(typeof(d)) $(d.name)
            Shapes:
            x: $(size(d.x))
            y: $(size(d.y))""")
end

# Clustering

struct ClusteringDataset <: AbstractDataset
    name::String
    x::Matrix
    y::Vector{Int}
end

function load_iris()::ClusteringDataset
    iris = dataset("datasets", "iris")
    ClusteringDataset(
        "iris",
        select(iris, Not(:Species)) |> Matrix,
        iris.Species |> Vector{String} |> tocategorical)
end

function load_beans()::ClusteringDataset
    beans = CSV.read("data/beans.csv", DataFrame)
    ClusteringDataset(
        "beans",
        select(beans, Not(:Class)) |> Matrix,
        beans.Class |> Vector{String} |> tocategorical)
end

function load_stars()::ClusteringDataset
    stars = CSV.read("data/stars.csv", DataFrame)
    ClusteringDataset(
        "stars",
        select(stars, Not(5, 6, 7)) |> Matrix,
        stars.Type)
end

load_clustering_datasets() = [load_iris(), load_beans(), load_stars()]

# Voting

struct VotingDataset <: AbstractDataset
    name::String
    x::AbstractArray{<:Real,3} # [n_questions x n_annotators x n_options ]
    y::Vector{Int} # Categorical vector
end

function load_adult2()::VotingDataset
    VotingDataset(
        "Adult2",
        load("data/adult2_dataset/crowd_counts.jld")["crowd_counts"],
        CSV.read("data/adult2_dataset/gold.csv", DataFrame, header=false)[!, 2] |> Vector |> tocategorical
    )
end

function load_rte()::VotingDataset
    VotingDataset(
        "RTE",
        load("data/rte_dataset/crowd_counts.jld")["crowd_counts"],
        CSV.read("data/rte_dataset/gold.csv", DataFrame, header=false)[!, 2]
    )
end

function load_toy()::VotingDataset
    VotingDataset(
        "toy",
        load("data/toy_dataset/crowd_counts.jld")["crowd_counts"],
        CSV.read("data/toy_dataset/gold.csv", DataFrame, header=false)[!, 2] |> tocategorical
    )
end

load_voting_datasets() = [load_adult2(), load_rte()]
