function calculate_negloglikelihood(counts, class_marginals, error_rates)
    n_patients, n_observers, n_classes = size(counts)
    loglikelihood = 0.0
    
    for i in 1:n_patients
        patient_likelihood = 0.0
        for j in 1:n_classes
            class_prior = class_marginals[j]
            patient_class_likelihood = prod(error_rates[:, j, :] .^ counts[i, :, :])
            patient_class_posterior = class_prior * patient_class_likelihood
            patient_likelihood += patient_class_posterior
        end 
        temp = loglikelihood + log(patient_likelihood)
        if isnan(temp) || isnothing(temp) || isinf(temp)
            error("Invalid temp value: $temp")
        end
        loglikelihood = temp
    end
    return -loglikelihood
end


function initialize_class_assignments(
    ::Union{FDS, MV},
    counts::AbstractArray{<:Real, 3}
)::AbstractArray{<:Real, 2}

    n_questions, n_participants, n_classes = size(counts)
    response_sums = reshape(sum(counts, dims = 2), (n_questions, n_classes))    
    class_assignments = zeros(n_questions, n_classes)
    for p in 1:n_questions
        maxval = maximum(response_sums[p, :])
        maxinds = argwhere(response_sums[p, :], ==(maxval))
        class_assignments[p, sample(maxinds, 1)[1]] = 1 #TODO: add RNG
    end
    return class_assignments
end

function initialize_class_assignments(
    ::Union{DS, HDS},
    counts::AbstractArray{<:Real, 3}
)::AbstractArray{<:Real, 2}

    n_questions, n_participants, n_classes = size(counts)
    response_sums = reshape(sum(counts, dims = 2), (n_questions, n_classes))
    class_assignments = zeros(n_questions, n_classes)
    for p in 1:n_questions
        class_assignments[p, :] = response_sums[p, :] ./ sum(response_sums[p, :])
    end
    return class_assignments
end


