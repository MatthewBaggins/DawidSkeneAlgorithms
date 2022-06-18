function partition_matrix_by_rows(x::Matrix, partition_size::Integer, n_partitions::Integer)
    partitions = [x[(i*partition_size+1):((i+1)*partition_size), :] for i in 0:(n_partitions-2)]
    return partitions
end
