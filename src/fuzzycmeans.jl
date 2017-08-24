# Fuzzy C means algorithm

## Interface

immutable FuzzyCMeansResult{T<:AbstractFloat} <: ClusteringResult
    centers::Matrix{T}          # cluster centers (d x C)
    weights::Matrix{T}          # assigned weights (n x C)
    iterations::Int             # number of elasped iterations
    converged::Bool             # wheather the procedure converged
end

## Utility functions

function update_weights!(weights, data, centers, fuzziness, dist_metric)
    pow = 2.0/(fuzziness-1)
    nrows, ncols = size(weights)
    dists = pairwise(dist_metric, data, centers)
    for i in 1:nrows
        for j in 1:ncols
            den = 0.0
            for k in 1:ncols
                den += (dists[i,j]/dists[i,k])^pow
            end
            weights[i,j] = 1.0/den
        end
    end
end

function update_centers!(centers, data, weights, fuzziness)
    nrows, ncols = size(weights)
    for j in 1:ncols
        num = zeros(Float64, size(data[:,1]))
        den = 0.0
        for i in 1:nrows
            δm = weights[i,j]^fuzziness
            num += δm * data[:,i]
            den += δm
        end
        centers[:,j] = num/den
    end
end

const _fcmeans_default_maxiter = 100
const _fcmeans_default_tol = 1.0e-3
const _fcmeans_default_display = :none

function fuzzy_cmeans{T<:Real}(
    data::Matrix{T},
    C::Int,
    fuzziness::Real;
    maxiter::Int = _fcmeans_default_maxiter,
    tol::Real = _fcmeans_default_tol,
    dist_metric::Metric = Euclidean(),
    display::Symbol = _fcmeans_default_display
    )

    nrows, ncols = size(data)
    2 <= C < ncols || error("C must have 2 <= C < n")

    _fuzzy_cmeans(data, C, fuzziness, maxiter, tol, dist_metric, display_level(display))

end

## Core implementation

function _fuzzy_cmeans{T<:Real}(
    data::Matrix{T},                                # data matrix
    C::Int,                                         # total number of classes
    fuzziness::Real,                                # fuzziness
    maxiter::Int,                                   # maximum number of iterations
    tol::Real,                                      # tolerance
    dist_metric::Metric,                            # metric to calculate distance
    displevel::Int                                  # the level of display
    )

    nrows, ncols = size(data)

    # Initialize weights randomly
    weights = rand(Float64, ncols, C)
    weights ./= sum(weights, 2)

    centers = fill(0.0, nrows, C)
    prev_centers = identity.(centers)

    δ = Inf
    iter = 0

    if displevel >= 2
        prog = ProgressThresh(tol, "Converging:")
    end

    while iter < maxiter && δ > tol
        update_centers!(centers, data, weights, fuzziness)
        update_weights!(weights, data, centers, fuzziness, dist_metric)
        δ = maximum(colwise(dist_metric, prev_centers, centers))
        copy!(prev_centers, centers)
        iter += 1
        if displevel >= 2
            ProgressMeter.update!(prog, δ, showvalues = [(:Iteration, iter)])
        end
    end

    if displevel >= 1
        if δ <= tol
            println("Fuzzy C-means converged with $iter iterations (δ = $δ)")
        else
            println("Fuzzy C-means terminated without convergence after $t iterations (δ = $δ)")
        end
    end

    FuzzyCMeansResult(centers, weights, iter, δ <= tol)
end
