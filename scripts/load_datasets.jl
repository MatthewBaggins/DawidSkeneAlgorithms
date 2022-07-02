using ExpectationMaximization: tocategorical
using DataFrames
using RDatasets

function load_iris()
    iris = dataset("datasets", "iris")
    x = select(iris, Not(:Species)) |> Matrix
    y_str = iris.Species |> Vector{String}
    y_cat = tocategorical(y_str)
    return x, y_cat
end
