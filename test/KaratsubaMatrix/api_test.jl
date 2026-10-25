function test_karatsuba_public_api_values()
    @testset "Karatsuba public API values" begin
        mod = GPUFiniteFieldMatrices
        modulus = 100
        a_values = Float64[20 30; 40 50]
        b_values = Float64[2 1; 0 3]
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

        rect_a_values = Float64[1 2 3; 4 5 6]
        rect_b_values = Float64[2 1 0 3; 1 2 3 4; 0 1 2 3]
        rect_a = mod.CuModMatrix{Float64}(rect_a_values, modulus)
        rect_b = mod.CuModMatrix{Float64}(rect_b_values, modulus)
        rect_ak = mod.to_karatsuba(rect_a, modulus)
        rect_bk = mod.to_karatsuba(rect_b, modulus)
        rect_ck = mod.to_karatsuba(
            mod.CuModMatrix{Float64}(zeros(Float64, 2, 4), modulus),
            modulus,
        )
        foreach(mod.initialize_plan!, (rect_ak, rect_bk, rect_ck))

        LinearAlgebra.mul!(rect_ck, rect_ak, rect_bk)
        @test size(rect_ck) == (2, 4)
        @test Array(rect_ck) == Base.mod.(rect_a_values * rect_b_values, modulus^2)
    end
end
