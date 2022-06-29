function tocategorical(x::Vector)::Vector{Int}
    x2cat = Dict(val => cat for (cat, val) in enumerate(unique(x)))
    return [x2cat[val] for val in x]
end

function tocategorical(x::Matrix{Float64})::Vector{Int}
    mapslices(argmax, x, dims = 2)[:] 
end

tocategorical(x::Vector{Int}) = x

function diagreshufflematrix(m::Matrix{T})::Matrix{T} where T <: Real
    @assert issquarematrix(m)
    D, _ = size(m)
    maxs = [argmax(m[:, d]) for d in 1:D]
    if length(unique(maxs)) != D
        return m
    end
    new_m = permutedims(hcat([m[i, :] for i in maxs]...))
    return new_m
end

issquarematrix(m::Matrix) = ==(size(m)...)
