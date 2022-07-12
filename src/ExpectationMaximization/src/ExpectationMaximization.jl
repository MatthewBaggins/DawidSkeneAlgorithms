module ExpectationMaximization

include("expectationmaximization.jl")

# test() = "test2"
# test(x::Int) = "test: $x"
# test(x::AbstractFloat) = "test: $(x*2)"
# test(x::Number) = "test number: $(x * 3)"

export AMM, KMeans, GMM, em, diagreshufflematrix, tocategorical, ADS, FDS
export test


end # module
