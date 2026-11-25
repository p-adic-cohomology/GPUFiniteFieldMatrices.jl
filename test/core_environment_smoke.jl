using Test
using GPUFiniteFieldMatrices

@testset "Minimal core environment has no optional GPU vendors" begin
    G = GPUFiniteFieldMatrices
    @info "Minimal core environment" julia=VERSION source=realpath(pathof(G)) ka=Base.pkgversion(
        G.KernelAbstractions,
    ) ki=Base.pkgversion(G.KernelInterface)
    @test Base.pkgversion(G.KernelAbstractions) == v"0.10.8"
    @test Base.pkgversion(G.KernelInterface) == v"0.4.7"
    @test Base.find_package("CUDA") === nothing
    @test Base.find_package("Metal") === nothing
    @test Base.get_extension(G, :GPUFiniteFieldMatricesCUDAExt) === nothing
    @test Base.get_extension(G, :GPUFiniteFieldMatricesMetalExt) === nothing
    @test realpath(pathof(G)) ==
          realpath(joinpath(@__DIR__, "..", "src", "GPUFiniteFieldMatrices.jl"))
    @test_throws ArgumentError G.default_backend()
    @test_throws MethodError G._check_backend_symbol(G.KernelAbstractions.CPU())
end
