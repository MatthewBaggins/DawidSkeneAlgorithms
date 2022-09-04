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

struct MajorityVoting <: AVA end
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
    ::AVA,
    counts::AbstractArray{<:Real,3},
    class_assignments::AbstractArray{<:Real,2}
)::Tuple{AbstractArray{<:Real,2},AbstractArray{<:Real,3}} # class_marginals, error_rates

    n_questions, n_annotators, n_options = size(counts)
    class_marginals = sum(class_assignments; dims=1) ./ n_questions
    error_rates = zeros(n_annotators, n_options, n_options)
    for a in 1:n_annotators
        for o1 in 1:n_options
            for o2 in 1:n_options
                error_rates[a, o1, o2] = class_assignments[:, o1]' * counts[:, a, o2]
            end
            sum_over_responses = sum(error_rates[a, o1, :])
            if sum_over_responses > 0
                error_rates[a, o1, :] = error_rates[a, o1, :] / sum_over_responses
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

    n_questions, n_annotators, n_options = size(counts)
    class_assignments = zeros(n_questions, n_options)
    final_class_assignments = zeros(n_questions, n_options)
    for q in 1:n_questions
        for o in 1:n_options
            estimate = class_marginals[o] * prod(error_rates[:, o, :] .^ counts[q, :, :])
            class_assignments[q, o] = estimate
        end
        _e_step_estimate_classes!(alg, q, class_assignments, final_class_assignments)
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
)
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
)
    class_assignments_sum = sum(class_assignments[i, :])
    if class_assignments_sum > 0
        class_assignments[i, :] = class_assignments[i, :] / class_assignments_sum
    end
    return
end

####################################
#               EM                 #
####################################

function em(
    alg::ADS,
    counts::AbstractArray{<:Real,3};
    tol=0.0001,
    CM_tol=0.005,
    max_iter=100,
    verbose::Bool=false
)
    # History of class assignments
    class_assignments_history = []

    # Initialize
    class_assignments = initialize_class_assignments(alg, counts)

    n_iter = 0
    converged = false
    old_class_marginals = nothing
    old_error_rates = nothing
    negloglik = nothing

    while !converged
        n_iter += 1
        class_marginals, error_rates = m_step(alg, counts, class_assignments)
        class_assignments = e_step(alg, counts, class_marginals, error_rates)
        negloglik = calculate_negloglikelihood(counts, class_marginals, error_rates)

        # Check for convergence
        if old_class_marginals ≠ nothing
            class_marginals_diff = sum(abs.(class_marginals - old_class_marginals))
            error_rates_diff = sum(abs.(error_rates - old_error_rates))
            if class_marginals_diff < tol || n_iter >= max_iter
                converged = true
            elseif alg isa HDS && class_marginals_diff ≤ CM_tol
                alg = HDS_phase2()
            end
        end

        old_class_marginals = class_marginals
        old_error_rates = error_rates

        if verbose
            @show n_iter
            @show negloglik
        end

        # Update history
        push!(class_assignments_history, class_assignments)
    end

    verbose && @show negloglik
    result = map(
        x -> x[2],
        argmax(class_assignments, dims=2)[:])
    return result, negloglik
end

"""
Slurped `kwargs...` are meant to prevent errors when `em` is called with `MV`
but also other kwargs that are relevant for the `ADS` method but not for this one.
(i.e. `tol`, `CM_tol`, and `max_iter`)
"""
function em(
    alg::MV,
    counts::AbstractArray{<:Real,3};
    verbose::Bool=false,
    kwargs...
)
    class_assignments = initialize_class_assignments(alg, counts)
    result = second.(argmax(class_assignments; dims=2))[:]
    class_marginals, error_rates = m_step(alg, counts, class_assignments)
    negloglik = calculate_negloglikelihood(counts, class_marginals, error_rates)
    verbose && @show result
    return result, negloglik
end
