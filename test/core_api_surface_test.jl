@testset "Core API and host Karatsuba contracts" begin
    G = GPUFiniteFieldMatrices
    public_names = (
        :GPUModArray,
        :GPUModMatrix,
        :GPUModVector,
        :CuModArray,
        :CuModMatrix,
        :CuModVector,
        :MtlModArray,
        :MtlModMatrix,
        :MtlModVector,
        :KaratsubaArray,
        :KaratsubaMatrix,
        :KaratsubaVector,
        :to_karatsuba,
        :from_karatsuba,
        :modulus,
        :low_modulus,
        :high_modulus,
        :low_part,
        :high_part,
        :pluq,
        :inverse_batched,
        :default_backend,
        :set_default_backend!,
        :with_backend,
    )
    @test all(name -> Base.isexported(G, name), public_names)
    @test realpath(pathof(G)) ==
          realpath(joinpath(@__DIR__, "..", "src", "GPUFiniteFieldMatrices.jl"))
    @test G._check_backend_symbol(:metal) === :metal
    @test_throws ArgumentError G._check_backend_symbol(:cpu)

    low = Int[2 4; 6 8]
    high = Int[1 3; 5 7]
    packed =
        G.KaratsubaMatrix(low, high; low_modulus = 10, high_modulus = 10, modulus = 100)
    @test G.low_part(packed) === low
    @test G.high_part(packed) === high
    @test G.modulus(packed) == 100
    @test BackendTestSupport.exact_modular_reference(BigInt[2^24+1], 7) == BigInt[2]
    @test BackendTestSupport.exact_modular_product([1 2], reshape([3, 4], 2, 1), 7) ==
          reshape(BigInt[4], 1, 1)
end
