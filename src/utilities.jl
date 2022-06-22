function partition_matrix_by_rows(x::Matrix, partition_size::Int, n_partitions::Int)
    [x[(i*partition_size+1):((i+1)*partition_size), :] for i in 0:(n_partitions-2)]
end

function as_vecs(m::Matrix{T})::Vector{Vector{T}} where T<:Any
    [m[i, :] for i in 1:size(m)[1]]
end

function as_vecs(df::DataFrame)::Vector{Vector}
    df |> Matrix |> as_vecs
end

function curry(f::Function, x)::Function
    (xs...) -> f(x, xs...)
end

function equals(x)::Function
    curry(==, x)
end

function vals2inds(v::Vector{T})::Dict{T, Vector{Int}} where {T <: Any}
    Dict(
        vᵢ => findall(equals(vᵢ), v) 
        for vᵢ in sort(unique(v)))
end

function dict2tuplevec(d::Dict{K, V}; reverse::Bool=false)::Vector{Tuple{K, V}} where {K <: Any, V <: Any}
    [reverse ? (v, k) : (k, v) for (k, v) in d]
end

function take_max_val_key(d_o::Dict)::Dict #{Ko, Ki} where {Ko <: Any, Ki <: Any}
    Dict(ko => maximum(dict2tuplevec(d_i, reverse=true))[2]
        for (ko, d_i) in d_o)
end

function labels2int(x::Vector{T})::Tuple{Dict{T, Int}, Vector{Int}} where T <: Any
    x_vals = unique(x)
    x_val2int = Dict(v => i for (i, v) in enumerate(x_vals))
    x_categorical = map(v -> x_val2int[v], x)
    return x_val2int, x_categorical
end

function labels2int(x::Vector, ref::Vector)::Tuple{Dict{Any, Int}, Vector{Int}}
    x_vals2inds = vals2inds(x)
    x_vals2ref_counts = Dict(x_val => countmap(ref[inds])
                            for (x_val, inds) in x_vals2inds)
    x_val2int = take_max_val_key(x_vals2ref_counts)
    x_categorical = map(v -> x_val2int[v], x)
    return x_val2int, x_categorical
end
