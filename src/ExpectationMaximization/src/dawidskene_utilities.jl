function calc_likelihood(::ADS, counts, class_marginals, error_rates)
    nPatients, nObservers, nClasses = size(counts)
    log_L = 0.0
    
    for i in 1:nPatients
        patient_likelihood = 0.0
        for j in 1:nClasses
            class_prior = class_marginals[j]
            patient_class_likelihood = prod(error_rates[:, j, :] .^ counts[i, :, :])
            patient_class_posterior = class_prior * patient_class_likelihood
            patient_likelihood += patient_class_posterior
        end 
        temp = log_L + log(patient_likelihood)
        if isnan(temp) || isnothing(temp) || isinf(temp)
            error("Invalid temp value: $temp")
        end
        log_L = temp
    end
    return log_L
end

function initialize_question_classes(
    ::Union{FDS, MV},
    counts::Array{<:Real, 3}
)
    nQuestions, nParticipants, nClasses = size(counts)
    response_sums = reshape(sum(counts, dims = 2), (nQuestions, nClasses))    
    question_classes = zeros(nQuestions, nClasses)
    for p in 1:nQuestions
        maxval = maximum(response_sums[p, :])
        maxinds = argwhere(response_sums[p, :], ==(maxval))
        question_classes[p, sample(maxinds, 1)[1]] = 1 #TODO: add RNG
    end
    return question_classes
end

function initialize_question_classes(
    ::AbstractEMAlgorithm,
    counts::Array{<:Real, 3}
)
    nQuestions, nParticipants, nClasses = size(counts)
    response_sums = reshape(sum(counts, dims = 2), (nQuestions, nClasses))
    question_classes = zeros(nQuestions, nClasses)
    for p in 1:nQuestions
        question_classes[p, :] = response_sums[p, :] ./ sum(response_sums[p, :])
    end
    return question_classes
end


