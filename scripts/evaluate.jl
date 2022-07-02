using Clustering: mutualinfo
using ExpectationMaximization: diagreshufflematrix

function evaluate(y_pred::Vector{Int}, y_true::Vector{Int})
    println("Mutual information: $(round(mutualinfo(y_pred, y_true), digits = 3))")
    println("Counts matrix [ predicted × true labels ]")
    countmat = diagreshufflematrix(counts(y_pred, y_true))
    display(countmat)
end
