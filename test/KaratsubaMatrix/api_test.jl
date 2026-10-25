function test_karatsuba_public_api_values()
    @testset "Karatsuba public API values" begin
        mod = GPUFiniteFieldMatrices
        modulus = 100
        a_values = Float64[20 30; 40 50]
        b_values = Float64[2 1; 0 3]
        a = mod.CuModMatrix{Float64}(a_values, modulus)
        b = mod.CuModMatrix{Float64}(b_values, modulus)

        @testset "Copied converted operand preserves M" begin
            # Match the existing 32-thread launch geometry without kernel edits.
            copy_size = 32
            copy_a_values = Matrix{Float64}(LinearAlgebra.I, copy_size, copy_size)
            copy_b_values = Matrix{Float64}(LinearAlgebra.I, copy_size, copy_size)
            copy_a_values[1:2, 1:2] .= a_values
            copy_b_values[1:2, 1:2] .= b_values
            converted_ak =
                mod.to_karatsuba(mod.CuModMatrix{Float64}(copy_a_values, modulus), modulus)
            converted_bk =
                mod.to_karatsuba(mod.CuModMatrix{Float64}(copy_b_values, modulus), modulus)
            converted_ck = mod.to_karatsuba(
                mod.CuModMatrix{Float64}(zeros(Float64, copy_size, copy_size), modulus),
                modulus,
            )
            foreach(mod.initialize_plan!, (converted_ak, converted_bk, converted_ck))
            converted_copy = copy(converted_ak)
            @test converted_copy.M ==
                  converted_ak.M ==
                  converted_bk.M ==
                  converted_ck.M ==
                  modulus
            @test converted_copy.N1 == converted_ak.N1
            @test converted_copy.N2 == converted_ak.N2
            @test converted_copy.data1 !== converted_ak.data1
            @test converted_copy.data2 !== converted_ak.data2
            @test converted_copy.data1.data !== converted_ak.data1.data
            @test converted_copy.data2.data !== converted_ak.data2.data
            @test converted_copy.plan !== nothing
            @test converted_copy.plan !== converted_ak.plan
            @test converted_copy.plan.data !== converted_ak.plan.data
            copied_data1 = Array(converted_copy.data1)
            copied_data2 = Array(converted_copy.data2)
            fill!(converted_ak.data1.data, 0)
            fill!(converted_ak.data2.data, 0)
            fill!(converted_ak.plan.data, -1)
            @test Array(converted_copy.data1) == copied_data1
            @test Array(converted_copy.data2) == copied_data2
            @test Array(converted_copy.plan) == zeros(Float64, copy_size, copy_size)
            @test copy_a_values * copy_b_values != copy_b_values * copy_a_values
            # Existing split arithmetic reduces modulo N1*N2, independently of
            # the stored-M compatibility field repaired by this change.
            converted_expected = Base.mod.(copy_a_values * copy_b_values, modulus^2)

            LinearAlgebra.mul!(converted_ck, converted_copy, converted_bk)
            @test Array(converted_ck) == converted_expected
            @test Array(converted_ak.plan) == fill(-1.0, copy_size, copy_size)
        end

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

        # Retain initialized-copy coverage for a coherent split representation.
        ak = mod.KaratsubaMatrix(
            a,
            mod.CuModMatrix{Float64}(zeros(Float64, 2, 2), modulus),
            modulus,
            modulus,
            modulus^2,
        )
        bk = mod.KaratsubaMatrix(
            b,
            mod.CuModMatrix{Float64}(zeros(Float64, 2, 2), modulus),
            modulus,
            modulus,
            modulus^2,
        )
        ck = mod.KaratsubaMatrix(
            mod.CuModMatrix{Float64}(zeros(Float64, 2, 2), modulus),
            mod.CuModMatrix{Float64}(zeros(Float64, 2, 2), modulus),
            modulus,
            modulus,
            modulus^2,
        )
        foreach(mod.initialize_plan!, (ak, bk, ck))

        ak_copy = copy(ak)
        @test ak_copy.plan !== nothing
        @test ak_copy.plan !== ak.plan
        @test ak_copy.M == ak.M

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
