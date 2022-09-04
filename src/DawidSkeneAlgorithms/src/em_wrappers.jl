"""
A "wrapper method" to simplify calling `em` on `DatasetClustering` structs. 
"""
function em(
    alg::AMM,
    dataset::ClusteringDataset;
    n_steps::Int=10
)
    K = length(unique(dataset.y))
    μ_history, r_history = DawidSkeneAlgorithms.em(alg, dataset.x;
        k=K, n_steps=n_steps)
    pred = tocategorical(r_history[end])
end

em_avg_log(dataset, alg, log) = log && println("Dataset: $(dataset.name)\t|\tAlgorithm: $alg")

"""
Run `em` `n_runs` times on the ClusteringDataset 
and return average mutual information scores.
"""
function em_avg(
    alg::AMM,
    dataset::ClusteringDataset;
    n_runs::Int=30,
    log::Bool=false
)::Float64
    em_avg_log(dataset, alg, log)
    results = [em(alg, dataset) for _ in 1:n_runs]
    mis = [round(mutualinfo(pred, dataset.y); digits=3) for pred in results]
    return round(mean(mis); digits=3)
end

"""
Run `em` `n_runs` times on the VotingDataset
and return average negative log-likelihoods.
"""
function em_avg(
    alg::AVA,
    dataset::VotingDataset;
    n_runs::Int=30,
    log::Bool=false,
    return_type::Symbol=:both
    # whether to return average neg-log-likelihoods, mutual information, or both
)::Union{Float64,Tuple{Float64,Float64}}

    # Log (optionally) and run
    em_avg_log(dataset, alg, log)
    em_results = [em(alg, dataset.x) for _ in 1:n_runs]

    # Negative log likelihoods
    nlls = last.(em_results)

    # Mutual information scores
    mis = [mutualinfo(r, dataset.y) for r in first.(em_results)]

    # Compute average
    avg_nll = round(mean(nlls); digits=3)
    avg_mi = round(mean(mis); digits=3)

    if return_type === :both
        return avg_nll, avg_mi
    elseif return_type === :nll
        return avg_nll
    elseif return_type === :mi
        return avg_mi
    end
end
