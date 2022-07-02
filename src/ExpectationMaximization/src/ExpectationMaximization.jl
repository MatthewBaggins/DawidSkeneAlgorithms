module ExpectationMaximization

include("expectationmaximization.jl")

greet() = print("Hello World!")

export KMeans, GMM, em, diagreshufflematrix, tocategorical

end # module
