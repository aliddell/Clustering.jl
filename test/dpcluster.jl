using Test
using Clustering
using Distances

@testset "dpcluster() (density-peak clustering)" begin
    Random.seed!(34568)

    X1 = randn(2, 300) .+ [0., 5.]
    X2 = randn(2, 250) .+ [-5., 0.]
    X3 = randn(2, 200) .+ [5., 0.]
    X = hcat(X1, X2, X3)

    D = pairwise(Euclidean(), X)
    distcut = 1
    ρcut = 70
    δcut = 6

    res = dpcluster(D, distcut, ρcut, δcut; displevel=:none)
    @test isa(res, DPClusterResult)
    @test nclusters(res) == 3
    @test all(0 .< assignments(res) .< 4)
    @test [sum(assignments(res) .== i) for i=1:3] == counts(res)
    @test counts(res) == [300 250 200]

    # test check for dist matrix
    @test_throws(dpcluster(D + rand(Float64, size(D)), distcut, ρcut, δcut))

    # 11 too large a cutoff for δ
    res2 = dpcluster(D, distcut, ρcut, 11; displevel=:none)
    @test nclusters(res2) == 0
    @test all(assignments(res2) .== 0)
    @test isempty(counts(res2))
end
