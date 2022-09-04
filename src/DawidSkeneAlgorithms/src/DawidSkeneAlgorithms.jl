module DawidSkeneAlgorithms

# Imports & utilities

using CategoricalArrays
using CSV
using Clustering
using DataFrames
using Distributions
using FillArrays
using JLD
using LinearAlgebra
using Pipe
using Random: AbstractRNG, GLOBAL_RNG, randperm
using RDatasets
using Statistics
using StatsBase
using StatsBase: sample

include("utilities.jl")

# Abstract

abstract type AbstractAlgorithm end

abstract type AbstractMixtureModel <: AbstractAlgorithm end
const AMM = AbstractMixtureModel

#= We have another intermedaite abstract type 
    because there is one voting algorithm 
    that is not a Dawid-Skene variant, namely MajorityVoting =#

abstract type AbstractVotingAlgorithm <: AbstractAlgorithm end
const AVA = AbstractVotingAlgorithm

abstract type AbstractDawidSkene <: AbstractVotingAlgorithm end
const ADS = AbstractDawidSkene

# Custom string representation

function Base.show(io::IO, alg::AbstractAlgorithm)
    alg_name = split("$(typeof(alg))", ".")[end]
    print(io, alg_name)
end

# Algorithms

include("mixturemodels.jl")
export AMM, KMeans, GMM, RandomClustering, CLUSTERING_ALGORITHMS
include("dawidskene.jl")
export AVA, ADS, FDS, DS, HDS, MV, VOTING_ALGORITHMS

# Dataset loaders

include("dataset_loaders.jl")
export
    ClusteringDataset, load_iris, load_beans, load_stars, load_clustering_datasets,
    VotingDataset, load_adult2, load_rte, load_voting_datasets

# EM wrappers

include("em_wrappers.jl")
export em, em_avg # Export EM and averaging wrappers

end # module
