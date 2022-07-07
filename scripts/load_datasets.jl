using CSV
using DataFrames
using ExpectationMaximization: tocategorical
using RDatasets

struct Dataset
    name::String
    x::Matrix
    y::Vector{Int}
end

function load_iris()
    iris = dataset("datasets", "iris")
    x = select(iris, Not(:Species)) |> Matrix
    y = iris.Species |> Vector{String} |> tocategorical
    return Dataset("iris", x, y)
end

function load_beans()
    beans = CSV.read("data/beans.csv", DataFrame)
    x = select(beans, Not(:Class)) |> Matrix
    y = beans.Class |> Vector{String} |> tocategorical
    return Dataset("beans", x, y)
end

function load_stars()
    stars = CSV.read("data/stars.csv", DataFrame)
    x = select(stars, Not(5, 6, 7)) |> Matrix
    y = stars.Type
    return Dataset("stars", x, y)
end