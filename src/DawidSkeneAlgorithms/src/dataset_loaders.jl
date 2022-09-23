# Abstract

abstract type AbstractDataset end

const DATA_PATH = rsplit(pathof(DawidSkeneAlgorithms), '/'; limit=3)[1] * "/data"
const DATA_CLUSTERING_PATH = DATA_PATH * "/clustering"
const DATA_VOTING_PATH = DATA_PATH * "/voting"

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
    beans = CSV.read("$DATA_CLUSTERING_PATH/beans.csv", DataFrame)
    ClusteringDataset(
        "beans",
        select(beans, Not(:Class)) |> Matrix,
        beans.Class |> Vector{String} |> tocategorical)
end

function load_stars()::ClusteringDataset
    stars = CSV.read("$DATA_CLUSTERING_PATH/stars.csv", DataFrame)
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
    load("$DATA_VOTING_PATH/adult2.jld")["adult2"]
end

function load_rte()::VotingDataset
    load("$DATA_VOTING_PATH/rte.jld")["rte"]
end


load_voting_datasets() = [load_adult2(), load_rte()]
