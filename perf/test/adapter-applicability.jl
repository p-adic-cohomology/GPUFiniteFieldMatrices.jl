using Test
using Revise
using BenchmarkTools
using JSON3
using CUDA
using GPUFiniteFieldMatrices
using GFMPerf

println("GPUFiniteFieldMatrices ", Base.pkgversion(GPUFiniteFieldMatrices),
    " at ", pathof(GPUFiniteFieldMatrices))

@testset "CPU constructor and storage adapter applicability" begin
    for (T, D) in ((Float32, 1), (Float32, 2), (Float64, 1), (Float64, 2))
        profile = GFMPerf._adapter_profile(T, D)
        cmtype = GFMPerf._cumod_type(T, D)
        sample = Array{T,D}(undef, ntuple(_ -> 2, D)...)
        @test profile.storage_field == :data
        @test :data in fieldnames(cmtype)
        if profile.constructor_style == :positional_modulus
            @test applicable(cmtype, sample, 101)
        elseif profile.constructor_style == :keyword_modulus
            @test applicable(cmtype, sample; modulus = 101)
        else
            @test false
        end
        println("  ", T, " rank ", D, ": ", profile)
    end
end
