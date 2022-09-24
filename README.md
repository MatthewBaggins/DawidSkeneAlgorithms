# Fast Dawid-Skene algorithm in Julia for [Turing.jl](https://turing.ml/) ([Google Summer of Code 2022](https://summerofcode.withgoogle.com/))

This repository contains the work done for a project of implementing in Julia an [expectation maximization](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm) version of the Fast Dawid-Skene algorithm ([paper](https://arxiv.org/abs/1803.02781)) for voting aggregation. The core of the algorithm (in [dawidskene.jl](src/DawidSkeneAlgorithms/src/dawidskene.jl)) is mostly a translation of the [Python implementation](https://github.com/sukrutrao/Fast-Dawid-Skene) published by the authors of that paper.

[mixturemodels.jl](src/DawidSkeneAlgorithms/src/mixturemodels.jl) contains EM implementations of two clustering algorithms which was done as an exercise.

The work was done during the summer of 2022 under Kai Xu's mentorship. (Thanks!)

## Project scope

[Link to the project proposal](https://summerofcode.withgoogle.com/proposals/details/lWF4IAFX).

I managed to translate the Python code into Julia and tested it at RTE and Adult2 datasets (achieving almost the same scores as those reported in the paper). I didn't succeed at integrating it with Turing by using the `@model` macro.

However, `@model` can be used to implement expectation-maximization algorithms as can be seen in the [`em-gmm`](./notebooks/em-gmm.ipynb) notebook. The other notebook ([em-fds](./notebooks/em-fds.ipynb)) is a record of my attempt at implementing EM-FDS with `@model`, details problems I've encountered and should be a good place to pick up if anybody would like to.

## Scripts

I suggest running them using Julia REPL, e.g. [in VSCode](https://www.julia-vscode.org/docs/dev/userguide/runningcode/). Alternatively, you can run them from the console (but the former solution is preferrable).

```bash
julia scripts/test_dawidskene.jl
julia scripts/test_mixturemodels.jl
```

- [test_dawidskene.jl](scripts/test_dawidskene.jl) - runs the three variants of the algorithm (Fast Dawid-Skene, normal Dawid-Skene, and Hybrid Dawid-Skene) as well as majority voting algorithm, comparing their time performance (using `@btime` from [BenchmarkTools](https://github.com/JuliaCI/BenchmarkTools.jl)) and result (average negative log-likelihood and mutual information).
  - One can see that time performance and neg-log-likelihoods are similar to those published in the [paper](http://sentic.net/wisdom2018sinha.pdf) (tables 1 and 2; see also the [showcase notebook](./notebooks/showcase.ipynb)).
- [test_mixturemodels.jl](scripts/test_mixturemodels.jl) - runs and compares EM implementations of two clustering algorithms, [k-means](https://en.wikipedia.org/wiki/K-means_clustering) and [gaussian mixture models](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model) (also with `@btime`).

## Tests

- [runtests.jl](src/DawidSkeneAlgorithms/test/runtests.jl) - uses the standard library [`Test`](https://docs.julialang.org/en/v1/stdlib/Test/) module to validate that the algorithms work correctly.

## Jupyter notebooks

- [showcase.ipynb](./notebooks/showcase.ipynb) - explains this implementation of EM-FDS step-by-step.
- [em-gmm.ipynb](./notebooks/em-gmm.ipynb) - implements the [gaussian mixture model](https://en.wikipedia.org/wiki/Mixture_model#Gaussian_mixture_model) with Turing's `@model` macro for defining statistical models.
- [em-fds.ipynb](./notebooks/em-fds.ipynb) - details my attempt at implementing EM-FDS with `@model`.
