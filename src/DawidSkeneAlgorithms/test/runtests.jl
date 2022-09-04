using Revise

using Test
using BenchmarkTools
using DawidSkeneAlgorithms
# Exporting full names for greater clarity of comparison in "Voting Algorithms" testset
import DawidSkeneAlgorithms:
    FastDawidSkene, DawidSkene, HybridDawidSkene, MajorityVoting

const DatasetName = String


function evaluate_clustering_algorithms()
    println("Evaluating clustering algorithms...")
    Dict{DatasetName,Dict{AMM,Float64}}(
        dataset.name => Dict(
            alg => em_avg(alg, dataset; log=true)
            for alg in CLUSTERING_ALGORITHMS
        )
        for dataset in load_clustering_datasets()
    )
end

@testset "Clustering Algorithms" begin
    #= Minimum mutual information that we expect from an algorithm.
        The threshold is that low for "beans" only because `GMM` deals 
        surprisingly poorly with that dataset. =#
    dataset_min_mi = Dict("iris" => 0.7, "beans" => 0.016, "stars" => 0.3)
    for (dname, dscores) in evaluate_clustering_algorithms()
        #= We expect both `KMeans` and `GMM` to perform better than random
            but also above some threshold. =#
        @test dscores[RandomClustering()] < dataset_min_mi[dname] < dscores[KMeans()]
        @test dscores[RandomClustering()] < dataset_min_mi[dname] < dscores[GMM()]
    end
end

function evaluate_voting_algorithms()
    println("Evaluating voting algorithms...")
    Dict{DatasetName,Dict{AVA,Tuple{Float64,Float64}}}(
        dataset.name => Dict(
            alg => em_avg(alg, dataset; log=true, return_type=:both)
            for alg in VOTING_ALGORITHMS
        )
        for dataset in load_voting_datasets()
    )
end

@testset "Voting Algorithms" begin
    for (dname, scores) in evaluate_voting_algorithms()

        # Negative log-likelihood scores
        nll = Dict(alg => alg_scores[1] for (alg, alg_scores) in scores)
        #= This is the ordering of mean neg-log-likelihoods 
            that we expect from these voting algorithms.
            However, it seems that FDS sometimes doesn't outperform MajorityVoting.
        =#
        @test (nll[DawidSkene()]
               < nll[HybridDawidSkene()]
               < nll[FastDawidSkene()])
        @test (nll[DawidSkene()]
               < nll[HybridDawidSkene()]
               < nll[MajorityVoting()])

        # Mutual information scores
        mi = Dict(alg => alg_scores[2] for (alg, alg_scores) in scores)
        #= DS and HDS usually have very similar mutual information scores.
            Their difference in negative log-likelihood scores seems more robust.
        =#
        @test (mi[DawidSkene()]
               > mi[FastDawidSkene()]
               > mi[MajorityVoting()])
        @test (mi[HybridDawidSkene()]
               > mi[FastDawidSkene()]
               > mi[MajorityVoting()])
    end
end
