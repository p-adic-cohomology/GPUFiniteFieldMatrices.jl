function test_karatsuba_public_api_values()
    @testset "Karatsuba public API values" begin
        mod = GPUFiniteFieldMatrices
        modulus = 100
        a_values = Float64[20 30; 40 50]
        b_values = Float64[2 1; 0 3]
        a = mod.CuModMatrix{Float64}(a_values; modulus = modulus)
        b = mod.CuModMatrix{Float64}(b_values; modulus = modulus)
        @test mod.modulus(a) == modulus
        @test size(a) == size(a_values)
        backing = CUDA.CuArray(zeros(Float64, 5, 6))
        wrapped_backing =
            mod.CuModMatrix{Float64}(backing; modulus = modulus, new_size = (2, 3))
        @test Base.parent(wrapped_backing) === backing
        @test size(wrapped_backing) == (2, 3)
        @test size(Base.parent(wrapped_backing)) == (5, 6)
        @test mod.modulus(wrapped_backing) == modulus

        explicit = mod.to_karatsuba(a; modulus = modulus)
        automatic = mod.to_karatsuba(a)
        @test Array(mod.from_karatsuba(Float64, explicit)) == a_values
        @test Array(mod.from_karatsuba(Float64, automatic)) == a_values

        typed_explicit = mod.to_karatsuba(Float64, a; modulus = modulus)
        typed_automatic = mod.to_karatsuba(Float64, a)
        @test Array(mod.from_karatsuba(Float64, typed_explicit)) == a_values
        @test Array(mod.from_karatsuba(Float64, typed_automatic)) == a_values

        different_source_values = Float32[20 30; 40 50]
        different_source =
            mod.CuModMatrix{Float32}(different_source_values; modulus = modulus)
        typed_different_source =
            mod.to_karatsuba(Float64, different_source; modulus = modulus)
        @test eltype(low_part(typed_different_source)) == Float64
        @test low_part(typed_different_source) isa mod.CuModMatrix{Float64}
        @test parent(low_part(typed_different_source)) !== parent(different_source)
        typed_different_roundtrip = mod.from_karatsuba(Float64, typed_different_source)
        @test eltype(typed_different_roundtrip) == Float64
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
        @test mod.low_part(gpu_wrapped) === gpu_low
        @test mod.high_part(gpu_wrapped) === gpu_high
        @test (
            mod.low_modulus(gpu_wrapped),
            mod.high_modulus(gpu_wrapped),
            mod.modulus(gpu_wrapped),
        ) == (2, 7, 19)
        @test (@inferred mod.initialize_plan!(gpu_wrapped)) === gpu_wrapped
        @test gpu_wrapped.plan isa CUDA.CuArray{Int64,2}
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

        # Keep the initialized-copy product in a coherent split representation.
        # Legacy copying sets M to N1*N2, so all operands here use that M.
        ak = mod.KaratsubaMatrix(
            a,
            mod.CuModMatrix{Float64}(zeros(Float64, 2, 2); modulus = modulus),
            ;
            low_modulus = modulus,
            high_modulus = modulus,
            modulus = modulus^2,
        )
        bk = mod.KaratsubaMatrix(
            b,
            mod.CuModMatrix{Float64}(zeros(Float64, 2, 2); modulus = modulus),
            ;
            low_modulus = modulus,
            high_modulus = modulus,
            modulus = modulus^2,
        )
        ck = mod.KaratsubaMatrix(
            mod.CuModMatrix{Float64}(zeros(Float64, 2, 2); modulus = modulus),
            mod.CuModMatrix{Float64}(zeros(Float64, 2, 2); modulus = modulus),
            ;
            low_modulus = modulus,
            high_modulus = modulus,
            modulus = modulus^2,
        )
        @test (@inferred mod.initialize_plan!(ak)) === ak
        foreach(mod.initialize_plan!, (bk, ck))
        @test ak.plan isa mod.CuModMatrix{Float64}

        ak_copy = copy(ak)
        @test ak_copy.plan !== nothing
        @test ak_copy.plan !== ak.plan

        LinearAlgebra.mul!(ck, ak_copy, bk)
        @test Array(ck) == Base.mod.(a_values * b_values, modulus^2)

        rect_a_values = Float64[1 2 3; 4 5 6]
        rect_b_values = Float64[2 1 0 3; 1 2 3 4; 0 1 2 3]
        rect_a = mod.CuModMatrix{Float64}(rect_a_values; modulus = modulus)
        rect_b = mod.CuModMatrix{Float64}(rect_b_values; modulus = modulus)
        rect_ak = mod.to_karatsuba(rect_a; modulus = modulus)
        rect_bk = mod.to_karatsuba(rect_b; modulus = modulus)
        rect_ck = mod.to_karatsuba(
            mod.CuModMatrix{Float64}(zeros(Float64, 2, 4); modulus = modulus),
            ;
            modulus = modulus,
        )
        foreach(mod.initialize_plan!, (rect_ak, rect_bk, rect_ck))

        LinearAlgebra.mul!(rect_ck, rect_ak, rect_bk)
        @test size(rect_ck) == (2, 4)
        @test Array(rect_ck) == Base.mod.(rect_a_values * rect_b_values, modulus^2)
    end
end
