####################################
#             Types                #
####################################

struct FastDawidSkene <: ADS end
const FDS = FastDawidSkene
Base.show(io::IO, ::FDS) = print(io, "FastDawidSkene")

struct DawidSkene <: ADS end
const DS = DawidSkene
Base.show(io::IO, ::DS) = print(io, "DawidSkene")

struct HybridDawidSkene <: ADS end
const HDS = HybridDawidSkene
Base.show(io::IO, ::HDS) = print(io, "HybridDawidSkene")

struct MajorityVoting <: AbstractEMAlgorithm end
const MV = MajorityVoting
Base.show(io::IO, ::MV) = print(io, "MajorityVoting")

const VOTING_ALGORITHMS = [FDS(), DS(), HDS(), MV()]

####################################
#            Utilities             #
####################################

include("dawidskene_utilities.jl")

##############################################
#      M-Step (same for all variations)      #
##############################################

function m_step(
    ::ADS,
    counts, 
    question_classes
)
    nQuestions, nParticipants, nClasses = size(counts)
    class_marginals = sum(question_classes, dims = 1) ./ nQuestions
    error_rates = zeros(nParticipants, nClasses, nClasses)
    for k in 1:nParticipants
        for j in 1:nClasses
            for l in 1:nClasses
                error_rates[k, j, l] = question_classes[:, j]' * counts[:, k, l]
            end
            sum_over_responses = sum(error_rates[k, j, :])
            if sum_over_responses > 0
                error_rates[k, j, :] = error_rates[k, j, :] / sum_over_responses
            end
        end
    end
    return class_marginals, error_rates
end

#####################################################
#         E-Steps (depend on alg variation)         #
#####################################################

# Fast Dawid-Skene

function e_step(
    ::FDS,
    counts,
    class_marginals, 
    error_rates
)
    nQuestions, nParticipants, nClasses = size(counts)
    question_classes = zeros(nQuestions, nClasses)
    final_classes = zeros(nQuestions, nClasses)
    for i in 1:nQuestions
        for j in 1:nClasses
            estimate = class_marginals[j] * prod(error_rates[:, j, :] .^ counts[i, :, :])
            question_classes[i, j] = estimate
        end
        # if mode ...
        maxval = maximum(question_classes[i, :])
        maxinds = argwhere(question_classes[i, :], ==(maxval))
        final_classes[i, sample(maxinds, 1)[1]] = 1
    end
    return final_classes
end

# Normal Dawid-Skene
# Hybrid Dawid-Skene

function e_step(
    ::Union{DS, HDS},
    counts,
    class_marginals, 
    error_rates
)
    nQuestions, nParticipants, nClasses = size(counts)
    question_classes = zeros(nQuestions, nClasses)
    final_classes = zeros(nQuestions, nClasses)
    for i in 1:nQuestions
        for j in 1:nClasses
            estimate = class_marginals[j] * prod(error_rates[:, j, :] .^ counts[i, :, :])
            question_classes[i, j] = estimate
        end
        question_sum = sum(question_classes[i, :])
        if question_sum > 0
            question_classes[i, :] .= sum(question_classes[i, :]) / question_sum
        end
    end
    return question_classes
end

####################################
#               EM                 #
####################################

@enum Verbosity SILENT NORMAL VERBOSE

function em(
    alg::ADS,
    counts::AbstractArray{<:Real, 3};
    tol = .0001,
    CM_tol = .005,
    max_iter = 100,
    verbosity::Verbosity = NORMAL
)
    # Initialize
    question_classes = initialize_question_classes(alg, counts)
    nIter = 0
    converged = false
    old_class_marginals = nothing
    old_error_rates = nothing
    # total_time = 0
    log_L = nothing
    while !converged
        nIter += 1
        class_marginals, error_rates = m_step(alg, counts, question_classes)
        question_classes = e_step(alg, counts, class_marginals, error_rates)
        log_L = calc_likelihood(alg, counts, class_marginals, error_rates)
        
        # Check for convergence
        if old_class_marginals ≠ nothing
            class_marginals_diff = sum(abs.(class_marginals - old_class_marginals))
            error_rates_diff = sum(abs.(error_rates - old_error_rates))
            # if verbose 
            if class_marginals_diff < tol || nIter >= max_iter
                converged = true
            end #elif mode == "H"
        end # else if verbose
        old_class_marginals = class_marginals
        old_error_rates = error_rates
        if verbosity == VERBOSE
            @show nIter
            @show log_L
        end
    end
    verbosity == NORMAL && @show log_L
    result = @pipe argmax(question_classes, dims = 2)[:] |> map(x -> x[2], _)
end


function em(
    alg::MV,
    counts::AbstractArray{<:Real, 3};
    tol = .0001,
    CM_tol = .005,
    max_iter = 100,
    verbose::Bool = true
)
    question_classes = initialize_question_classes(alg, counts)
    return argmax(question_classes, dims = 2)
end

