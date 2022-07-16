function tocategorical(x::Vector)::Vector{Int}
    x2cat = Dict(val => cat for (cat, val) in enumerate(unique(x)))
    return [x2cat[val] for val in x]
end

function tocategorical(x::Matrix{<:AbstractFloat})::Vector{Int}
    mapslices(argmax, x, dims = 2)[:] 
end

tocategorical(x::Vector{<:Integer}) = x

function diagreshufflematrix(m::Matrix{T})::Matrix{T} where T <: Real
    @assert issquarematrix(m)
    D, _ = size(m)
    maxs = [argmax(m[:, d]) for d in 1:D]
    if length(unique(maxs)) ≠ D
        return m
    end
    new_m = permutedims(hcat([m[i, :] for i in maxs]...))
    return new_m
end

issquarematrix(m::Matrix) = ==(size(m)...)

function curry(f::Function, x)::Function
    (xs...) -> f(x, xs...)
end

equals(x) = curry(==, x)

function argwhere(xs, condition)
    [i for (i, x) in enumerate(xs) if condition(x)]
end

function convert_responses_to_counts(path::String)::AbstractArray{<:Real, 3}
    responses = CSV.read(path, DataFrame)
    participants = responses[!, 1] |> unique |> Vector
    questions = responses[!, 2] |> unique |> Vector
    classes = responses[!, 3] |> unique |> Vector
    
    counts = zeros(length(questions), length(participants), length(classes))
    for (q_i, q) in enumerate(questions)
        for (p_i, p) in enumerate(participants)
            for (c_i, c) in enumerate(classes)
                counts[q_i, p_i, c_i], _ = filter(
                    r -> (r[1] == p && r[2] == q && r[3] == c), 
                    responses
                    ) |> size
            end
        end
    end
    return counts
end
