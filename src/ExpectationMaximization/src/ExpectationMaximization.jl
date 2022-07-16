module ExpectationMaximization

####################################
#        Imports & utilities       #
####################################

using CategoricalArrays
using Clustering
using Distributions
using FillArrays
using LinearAlgebra
using Pipe
using Random: AbstractRNG, GLOBAL_RNG, randperm
using RDatasets
using Statistics
using StatsBase
using StatsBase: sample

include("utilities.jl")

# Abstract

abstract type AbstractEMAlgorithm end

abstract type AbstractMixtureModel <: AbstractEMAlgorithm end
const AMM = AbstractMixtureModel

abstract type AbstractDawidSkene <: AbstractEMAlgorithm end
const ADS = AbstractDawidSkene

# Algorithms

include("mixturemodels.jl")
include("dawidskene.jl")

export AMM, KMeans, GMM, em, diagreshufflematrix, tocategorical, ADS, FDS, CLUSTERING_ALGORITHMS, VOTING_ALGORITHMS

end # module
