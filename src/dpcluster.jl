# Density peak clustering
#
#   Reference:
#       Clustering by fast search and find of density peaks.
#       Alex Rodriguez and Alessandro Laio
#       Science, Vol. 344, Issue 6191, pp. 1492-1496, 2014.
#       DOI: 10.1126/science.1242072
#

#### Interface

mutable struct DPClusterResult <: ClusteringResult
    centers::Vector{Int}           # indices of cluster centers
    assignments::Vector{Int}       # cluster assignments for each point
    counts::Vector{Int}            # number of points assigned to each cluster
    densities::Vector{Int}         # local densities (ρ) for each point
    mindists::Vector{<: Real}      # minimum dists between each point and a point of higher density (δ)
    nearestneighbors::Vector{Int}  # indices of nearest points with higher densities
    distcut::Real                  # radius of neighborhood to consider density
    ρcut::Real                     # density threshold to be considered a center
    δcut::Real                     # distance to nearest denser point threshold
    iters::Integer                 # number of iterations taken to assign all points
end

function _isdistmatrix(D::DenseMatrix{T}) where T <: Real
    m, n = size(D)
    (m == n) && issymmetric(D) && all(D .≥ 0)
end

# distance matrix version
# distance matrix version
function dpcluster(D::DenseMatrix{T},
                   distcut::Real,
                   ρcut::Real,
                   δcut::Real;
                   maxiter::Integer=1000,
                   displevel::Symbol=:none) where T <: Real
    # check arguments
    _isdistmatrix(D) || error("D must be a distance matrix")
    distcut > 0 || error("distance cutoff must be positive")
    ρcut > 0 || error("ρ cutoff must be positive")
    δcut > 0 || error("δ cutoff must be positive")
    maxiter > 0 || error("max number of iterations must be positive")
    displevel in [:none, :final, :iter] || error("bad value for display level: $displevel")

    # call the core implementation
    _dpcluster(LinearAlgebra.Symmetric(D), distcut, ρcut, δcut, maxiter, displevel)
end

function ρ(D::LinearAlgebra.Symmetric, distcut::Real)
    densities = sum(D .< distcut, dims=1)[:]
end

function δ(D::LinearAlgebra.Symmetric, densities::Array{Int64, 1})
    function _nn(D::LinearAlgebra.Symmetric, L::BitArray{2}, i::Integer)
        denserpoints = L[:, i] # all neighbors of higher density
        if any(denserpoints)
            findall(denserpoints)[argmin(D[i, denserpoints])] # nearest of these
        else # no point of higher density; choose furthest point away
            argmax(D[i, :])
        end
    end

    # islarger[i, j] == true ⟺ densities[i] > densities[j]
    islarger = densities .> densities'
    nearestneighbors = [_nn(D, islarger, i) for i = 1:size(D, 1)]
    mindists = D[CartesianIndex.(1:length(nearestneighbors), nearestneighbors)]

    mindists, nearestneighbors
end

function _dpcluster(D::LinearAlgebra.Symmetric,
                    distcut::Real,
                    ρcut::Real,
                    δcut::Real,
                    maxiter::Integer=1000,
                    displevel::Symbol=:none)

    # compute ⁠ρ
    densities = ρ(D, distcut)
    # compute δ
    mindists, nearestneighbors = δ(D, densities)

    # find cluster centers
    centers = findall((densities .> ρcut) .& (mindists .> δcut))
    assignments = zeros(Int64, size(D, 1))

    if isempty(centers) # failed; adjust parameters
        if displevel != :none
            println("No cluster centers found; adjust your parameters?")
        end
        return DPClusterResult(centers,          # empty
                               assignments,      # assignments (zeros)
                               centers,          # counts (empty)
                               densities,        # useful data
                               mindists,         # useful data
                               nearestneighbors, # useful data
                               distcut,          # parameter (to adjust?)
                               ρcut,             # parameter (to adjust?)
                               δcut,             # parameter (to adjust?)
                               0)                # no iterations
    end

    # assign clusters
    assignments[centers] = 1:length(centers)
    iters = 0
    for i = 1:maxiter
        unassigned = (assignments .== 0) # points which have not been assigned yet
        hasneighbor = assignments[nearestneighbors[unassigned]] .> 0 # have nearest neighbors with assignments
        toassign = findall(unassigned)[hasneighbor] # indices of these points in global space
        nniter = nearestneighbors[toassign] # nearest neighbors for these points
        assignments[toassign] = assignments[nniter] # assign these points the cluster of their nearest neighbor

        if displevel == :iter
            println("$i/$maxiter: $(length(toassign)) points assigned to clusters")
        end

        iters = i

        if all(assignments .> 0) # converged
            break
        end
    end

    if any(assignments .== 0) && displevel != :none # failed to converge
        println("Not all points were assigned to clusters")
    elseif displevel != :none # converged and verbose
        println("All points assigned to clusters in $iters iterations")
    end

    assigncounts = [sum(assignments .== i) for i in 1:length(centers)]

    DPClusterResult(centers,          # centers
                    assignments,      # assignments
                    assigncounts,     # counts
                    densities,        # useful data
                    mindists,         # useful data
                    nearestneighbors, # useful data
                    distcut,          # parameter
                    ρcut,             # parameter
                    δcut,             # parameter
                    iters)            # iters
end
