function calculate_negloglikelihood(counts, class_marginals, error_rates)
    n_questions, n_annotators, n_options = size(counts)
    loglikelihood = 0.0

    for q in 1:n_questions
        question_likelihood = 0.0
        for o in 1:n_options
            class_prior = class_marginals[o]
            question_option_likelihood = prod(error_rates[:, o, :] .^ counts[q, :, :])
            question_option_posterior = class_prior * question_option_likelihood
            question_likelihood += question_option_posterior
        end
        temp = loglikelihood + log(question_likelihood)
        if isnan(temp) || isnothing(temp) || isinf(temp)
            error("Invalid temp value: $temp")
        end
        loglikelihood = temp
    end
    return -loglikelihood
end

function initialize_class_assignments(
    ::Union{FDS,MV},
    counts::AbstractArray{<:Real,3}
)::AbstractArray{<:Real,2}

    n_questions, n_annotators, n_options = size(counts)
    response_sums = reshape(sum(counts, dims=2), (n_questions, n_options))
    class_assignments = zeros(n_questions, n_options)
    for q in 1:n_questions
        maxval = maximum(response_sums[q, :])
        maxinds = argwhere(response_sums[q, :], ==(maxval))
        class_assignments[q, sample(maxinds, 1)[1]] = 1
    end
    return class_assignments
end

function initialize_class_assignments(
    ::Union{DS,HDS},
    counts::AbstractArray{<:Real,3}
)::AbstractArray{<:Real,2}

    n_questions, n_annotators, n_options = size(counts)
    response_sums = reshape(sum(counts, dims=2), (n_questions, n_options))
    class_assignments = zeros(n_questions, n_options)
    for q in 1:n_questions
        class_assignments[q, :] = response_sums[q, :] ./ sum(response_sums[q, :])
    end
    return class_assignments
end
