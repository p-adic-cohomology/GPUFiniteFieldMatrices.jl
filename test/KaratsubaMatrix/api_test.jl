function test_karatsuba_public_api_values()
    @testset "Karatsuba public API values" begin
        mod = GPUFiniteFieldMatrices
        modulus = 100
        a_values = Float64[20 30; 40 50]
        b_values = Float64[20 30; 40 50]
        a = mod.CuModMatrix{Float64}(a_values, modulus)
        b = mod.CuModMatrix{Float64}(b_values, modulus)

        explicit = mod.to_karatsuba(a, modulus)
        automatic = mod.to_karatsuba(a)
        @test Array(mod.from_karatsuba(Float64, explicit)) == a_values
        @test Array(mod.from_karatsuba(Float64, automatic)) == a_values

        typed_explicit = mod.to_karatsuba(Float64, a, modulus)
        typed_automatic = mod.to_karatsuba(Float64, a)
        @test Array(mod.from_karatsuba(Float64, typed_explicit)) == a_values
        @test Array(mod.from_karatsuba(Float64, typed_automatic)) == a_values

        different_source_values = Float32[20 30; 40 50]
        different_source = mod.CuModMatrix{Float32}(different_source_values, modulus)
        typed_different_source = mod.to_karatsuba(Float64, different_source, modulus)
        @test eltype(typed_different_source.data1) == Float32
        typed_different_roundtrip = mod.from_karatsuba(Float64, typed_different_source)
        @test eltype(typed_different_roundtrip) == Float32
        @test Array(typed_different_roundtrip) == different_source_values

        ak = mod.to_karatsuba(a, modulus)
        bk = mod.to_karatsuba(b, modulus)
        ck = mod.to_karatsuba(
            mod.CuModMatrix{Float64}(zeros(Float64, 2, 2), modulus),
            modulus,
        )
        foreach(mod.initialize_plan!, (ak, bk, ck))

        ak_copy = copy(ak)
        @test ak_copy.plan !== nothing
        @test ak_copy.plan !== ak.plan

        LinearAlgebra.mul!(ck, ak_copy, bk)
        @test Array(ck) == Base.mod.(a_values * b_values, modulus^2)
    end
end
