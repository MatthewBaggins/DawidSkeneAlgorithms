####################################
#             Types                #
####################################

struct FastDawidSkene <: ADS end
const FDS = FastDawidSkene

struct DawidSkene <: ADS end
const DS = DawidSkene

struct HybridDawidSkene <: ADS end
const HDS = HybridDawidSkene
struct HDS_phase2 <: ADS end

struct MajorityVoting <: AbstractEMAlgorithm end
const MV = MajorityVoting

const VOTING_ALGORITHMS = [FDS(), DS(), HDS(), MV()]

####################################
#            Utilities             #
####################################

include("dawidskene_utilities.jl")

####################################
#             M-Step               #
####################################

function m_step(
    ::AbstractEMAlgorithm,
    counts::AbstractArray{<:Real,3},
    class_assignments::AbstractArray{<:Real,2}
)::Tuple{AbstractArray{<:Real,2},AbstractArray{<:Real,3}} # class_marginals, error_rates

    n_questions, n_annotators, n_classes = size(counts)
    class_marginals = sum(class_assignments, dims=1) ./ n_questions
    error_rates = zeros(n_annotators, n_classes, n_classes)
    for k in 1:n_annotators
        for j in 1:n_classes
            for l in 1:n_classes
                error_rates[k, j, l] = class_assignments[:, j]' * counts[:, k, l]
            end
            sum_over_responses = sum(error_rates[k, j, :])
            if sum_over_responses > 0
                error_rates[k, j, :] = error_rates[k, j, :] / sum_over_responses
            end
        end
    end
    return class_marginals, error_rates
end

####################################
#             E-Step               #
####################################

# Common core

function e_step(
    alg::ADS,
    counts::AbstractArray{<:Real,3},
    class_marginals::AbstractArray{<:Real,2},
    error_rates::AbstractArray{<:Real,3}
)::AbstractArray{<:Real,2} # class_assignments or final_class_assignments

    n_questions, n_participants, n_classes = size(counts)
    class_assignments = zeros(n_questions, n_classes)
    final_class_assignments = zeros(n_questions, n_classes)
    for i in 1:n_questions
        for j in 1:n_classes
            estimate = class_marginals[j] * prod(error_rates[:, j, :] .^ counts[i, :, :])
            class_assignments[i, j] = estimate
        end
        _e_step_estimate_classes!(alg, i, class_assignments, final_class_assignments)
    end

    if typeof(alg) ∈ [DS, HDS]
        return class_assignments
    else # FDS / HDS_phase2
        return final_class_assignments
    end
end

# Class estimation -- the only part that differs between algorithms

function _e_step_estimate_classes!(
    ::Union{FDS,HDS_phase2},
    i::Int,
    class_assignments::AbstractArray{<:Real,2},
    final_class_assignments::AbstractArray{<:Real,2}
)::Nothing

    maxval = maximum(class_assignments[i, :])
    maxinds = argwhere(class_assignments[i, :], ==(maxval))
    final_class_assignments[i, sample(maxinds, 1)[1]] = 1
    return
end

function _e_step_estimate_classes!(
    ::Union{DS,HDS},
    i::Int,
    class_assignments::AbstractArray{<:Real,2},
    final_class_assignments::AbstractArray{<:Real,2}
)::Nothing

    class_assignments_sum = sum(class_assignments[i, :])
    if class_assignments_sum > 0
        class_assignments[i, :] = class_assignments[i, :] / class_assignments_sum
    end
    return
end

####################################
#               EM                 #
####################################

@enum Verbosity SILENT NORMAL VERBOSE

function em(
    alg::ADS,
    counts::AbstractArray{<:Real,3};
    tol=0.0001,
    CM_tol=0.005,
    max_iter=100,
    verbosity::Verbosity=SILENT
)
    # History of class assignments
    class_assignments_history = []

    # Initialize
    class_assignments = initialize_class_assignments(alg, counts)

    nIter = 0
    converged = false
    old_class_marginals = nothing
    old_error_rates = nothing
    negloglik = nothing

    while !converged
        nIter += 1
        class_marginals, error_rates = m_step(alg, counts, class_assignments)
        class_assignments = e_step(alg, counts, class_marginals, error_rates)
        negloglik = calculate_negloglikelihood(counts, class_marginals, error_rates)

        # Check for convergence
        if old_class_marginals ≠ nothing
            class_marginals_diff = sum(abs.(class_marginals - old_class_marginals))
            error_rates_diff = sum(abs.(error_rates - old_error_rates))
            if class_marginals_diff < tol || nIter >= max_iter
                converged = true
            elseif alg isa HDS && class_marginals_diff ≤ CM_tol
                alg = HDS_phase2()
            end
        end

        old_class_marginals = class_marginals
        old_error_rates = error_rates

        if verbosity == VERBOSE
            @show nIter
            @show negloglik
        end

        # Update history
        push!(class_assignments_history, class_assignments)
    end

    verbosity != SILENT && @show negloglik
    result = map(
        x -> x[2],
        argmax(class_assignments, dims=2)[:])
    return result, negloglik
end


function em(
    alg::MV,
    counts::AbstractArray{<:Real,3};
    verbose::Verbosity=SILENT,
    kwargs...
)
    class_assignments = initialize_class_assignments(alg, counts)
    result = argmax(class_assignments, dims=2)
    class_marginals, error_rates = m_step(alg, counts, class_assignments)
    negloglik = calculate_negloglikelihood(counts, class_marginals, error_rates)
    verbose ≠ SILENT && @show result
    return result, negloglik
end
