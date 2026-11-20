using Test
include(joinpath(@__DIR__, "..", "setup_paths.jl"))
using .GFMPerfSetupPaths

@testset "setup source directory identity" begin
    mktempdir() do root
        physical = mkpath(joinpath(root, "physical checkout"))
        harness = mkpath(joinpath(physical, "perf"))
        alias = joinpath(root, "logical checkout")
        symlink(physical, alias)
        other = mkpath(joinpath(root, "different checkout"))
        mkpath(joinpath(other, "perf"))
        @test same_source_path(alias, physical)
        @test same_source_path(physical, alias)
        @test same_source_path(joinpath(alias, "perf"), harness)
        @test same_source_path(joinpath(alias, "perf", ".."), physical)
        @test !same_source_path(other, physical)
        @test !same_source_path(joinpath(other, "perf"), harness)
        @test !same_source_path(joinpath(root, "old missing checkout"), physical)
        @test !same_source_path(nothing, physical)
        @test !same_source_path(physical, nothing)
    end
end
