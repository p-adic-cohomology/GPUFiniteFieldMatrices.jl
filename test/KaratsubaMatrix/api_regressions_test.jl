function test_karatsuba_api_regressions()
    @testset "Karatsuba required GPU API regressions" begin
        mod = GPUFiniteFieldMatrices
        q = 100
        a_values = Float64[20 30; 40 50]
        b_values = Float64[2 1; 0 3]
        a = mod.CuModMatrix{Float64}(a_values; modulus = q)
        @test mod.modulus(a) == q
        @test size(a) == size(a_values)

        # Keep the public logical-size wrapper contract over padded CUDA storage.
        backing = CUDA.CuArray(zeros(Float64, 5, 6))
        wrapped_backing = mod.CuModMatrix{Float64}(backing; modulus = q, new_size = (2, 3))
        @test Base.parent(wrapped_backing) === backing
        @test size(wrapped_backing) == (2, 3)
        @test size(Base.parent(wrapped_backing)) == (5, 6)
        @test mod.modulus(wrapped_backing) == q

        explicit = mod.to_karatsuba(a; modulus = q)
        automatic = mod.to_karatsuba(a)
        @test Array(mod.from_karatsuba(Float64, explicit)) == a_values
        @test Array(mod.from_karatsuba(Float64, automatic)) == a_values

        typed_explicit = mod.to_karatsuba(Float64, a; modulus = q)
        typed_automatic = mod.to_karatsuba(Float64, a)
        @test Array(mod.from_karatsuba(Float64, typed_explicit)) == a_values
        @test Array(mod.from_karatsuba(Float64, typed_automatic)) == a_values

        different_source_values = Float32[20 30; 40 50]
        different_source = mod.CuModMatrix{Float32}(different_source_values; modulus = q)
        typed_different_source = mod.to_karatsuba(Float64, different_source; modulus = q)
        @test low_part(typed_different_source) isa mod.CuModMatrix{Float64}
        @test high_part(typed_different_source) isa mod.CuModMatrix{Float64}
        @test parent(low_part(typed_different_source)) !== parent(different_source)
        @test parent(high_part(typed_different_source)) !== parent(different_source)
        typed_different_roundtrip = mod.from_karatsuba(Float64, typed_different_source)
        @test typed_different_roundtrip isa CUDA.CuArray{Float64,2}
        @test Array(typed_different_roundtrip) == Float64.(different_source_values)

        gpu_low = CUDA.CuArray(Int64[1 0])
        gpu_high = CUDA.CuArray(Int64[37 0])
        gpu_wrapped = mod.KaratsubaMatrix(
            gpu_low,
            gpu_high;
            low_modulus = 2,
            high_modulus = 7,
            modulus = 19,
        )
        gpu_reconstructed = @inferred mod.from_karatsuba(Int64, gpu_wrapped)
        @test gpu_reconstructed isa CUDA.CuArray{Int64,2}
        @test Array(gpu_reconstructed) == Int64[75 0]

        gpu_overflow = mod.KaratsubaVector(
            CUDA.CuArray(Int64[0]),
            CUDA.CuArray(Int64[typemax(Int64)]);
            low_modulus = 2,
            high_modulus = 7,
            modulus = 19,
        )
        @test_throws InexactError mod.from_karatsuba(Int64, gpu_overflow)

        gpu_float_exact = mod.KaratsubaVector(
            CUDA.CuArray(Float64[0]),
            CUDA.CuArray(Float64[2^52]);
            low_modulus = 2,
            high_modulus = 7,
            modulus = 19,
        )
        gpu_float_result = mod.from_karatsuba(Float64, gpu_float_exact)
        @test gpu_float_result isa CUDA.CuArray{Float64,1}
        @test Array(gpu_float_result) == Float64[2^53]

        gpu_float_inexact = mod.KaratsubaVector(
            CUDA.CuArray(Float64[1]),
            CUDA.CuArray(Float64[2^53]);
            low_modulus = 2,
            high_modulus = 7,
            modulus = 19,
        )
        @test_throws InexactError mod.from_karatsuba(Float64, gpu_float_inexact)

        gpu_component_inexact = mod.KaratsubaVector(
            CUDA.CuArray(Float64[0]),
            CUDA.CuArray(Float64[2^24+1]);
            low_modulus = 2,
            high_modulus = 7,
            modulus = 19,
        )
        @test_throws InexactError mod.from_karatsuba(Float32, gpu_component_inexact)

        # Legacy copy recomputes M as N1*N2. Keep the square fixture coherent
        # with that established behavior while checking an independent oracle.
        zero_component() = mod.CuModMatrix{Float64}(zeros(Float64, 2, 2); modulus = q)
        ak = mod.KaratsubaMatrix(
            a,
            zero_component();
            low_modulus = q,
            high_modulus = q,
            modulus = q^2,
        )
        bk = mod.KaratsubaMatrix(
            mod.CuModMatrix{Float64}(b_values; modulus = q),
            zero_component();
            low_modulus = q,
            high_modulus = q,
            modulus = q^2,
        )
        ck = mod.KaratsubaMatrix(
            zero_component(),
            zero_component();
            low_modulus = q,
            high_modulus = q,
            modulus = q^2,
        )
        @test (@inferred mod.initialize_plan!(ak)) === ak
        foreach(mod.initialize_plan!, (bk, ck))

        ak_copy = copy(ak)
        @test ak_copy.plan !== nothing
        @test ak_copy.plan !== ak.plan
        @test parent(ak_copy.plan) !== parent(ak.plan)
        square_result = LinearAlgebra.mul!(ck, ak_copy, bk)
        @test square_result === ck
        square_expected = Float64.(mod.(BigInt.(a_values) * BigInt.(b_values), BigInt(q)^2))
        @test Array(ck) == square_expected

        rect_a_values = Float64[1 2 3; 4 5 6]
        rect_b_values = Float64[2 1 0 3; 1 2 3 4; 0 1 2 3]
        rect_a = mod.CuModMatrix{Float64}(rect_a_values; modulus = q)
        rect_b = mod.CuModMatrix{Float64}(rect_b_values; modulus = q)
        rect_ak = mod.to_karatsuba(rect_a; modulus = q)
        rect_bk = mod.to_karatsuba(rect_b; modulus = q)
        rect_output = mod.CuModMatrix{Float64}(zeros(Float64, 2, 4); modulus = q)
        rect_ck = mod.to_karatsuba(rect_output; modulus = q)
        foreach(mod.initialize_plan!, (rect_ak, rect_bk, rect_ck))

        rectangular_result = LinearAlgebra.mul!(rect_ck, rect_ak, rect_bk)
        @test rectangular_result === rect_ck
        @test size(rect_ck) == (2, 4)
        rectangular_expected =
            Float64.(mod.(BigInt.(rect_a_values) * BigInt.(rect_b_values), BigInt(q)^2))
        @test Array(rect_ck) == rectangular_expected
    end
end
