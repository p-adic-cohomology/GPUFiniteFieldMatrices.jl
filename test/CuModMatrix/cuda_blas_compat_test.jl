@testset "CPU-safe cuBLAS scalar cache inference" begin
    for T in (Float16, Float32, Float64)
        cache_name =
            T === Float16 ? :cublas_scalars_f16 :
            T === Float32 ? :cublas_scalars_f32 : :cublas_scalars_f64
        cache_f = getfield(GPUFiniteFieldMatrices, cache_name)
        cu_ref_type = only(Base.return_types(CUDA.CuRef, Tuple{T}))
        scalar_pair_type = Tuple{cu_ref_type,cu_ref_type}

        @test isconcretetype(cu_ref_type)
        @test only(Base.return_types(cache_f, Tuple{})) === scalar_pair_type
        @test only(
            Base.return_types(GPUFiniteFieldMatrices._stripe_gemm_scalars, Tuple{Type{T}}),
        ) === scalar_pair_type
        @test only(
            Base.return_types(GPUFiniteFieldMatrices._stripe_gemv_scalars, Tuple{Type{T}}),
        ) === scalar_pair_type

        CU = CUDA.CuArray{T,2,CUDA.DeviceMemory}
        run_signature = Tuple{CU,CU,CU,Int,Int,Nothing}
        run_code = only(Base.code_typed(GPUFiniteFieldMatrices._stripe_run!, run_signature))
        run_ir = sprint(show, MIME"text/plain"(), run_code.first)

        @test run_code.second === CU
        @test occursin("CuRefValue", run_ir)
        @test occursin("gemm!", run_ir)
        @test !any(
            line -> occursin("dynamic (", line) && occursin("gemm!", line),
            split(run_ir, Char(10)),
        )
    end
end

function test_cuda_blas_compatibility()
    @testset "task-local cuBLAS GEMM scalar cache" begin
        for T in (Float16, Float32, Float64)
            zero_ptr, one_ptr = GPUFiniteFieldMatrices._stripe_gemm_scalars(T)
            zero_ptr_again, one_ptr_again = GPUFiniteFieldMatrices._stripe_gemm_scalars(T)

            @test eltype(zero_ptr) === T
            @test eltype(one_ptr) === T
            @test zero_ptr[] === zero(T)
            @test one_ptr[] === one(T)
            @test zero_ptr_again === zero_ptr
            @test one_ptr_again === one_ptr
            @test zero_ptr !== one_ptr

            if T !== Float16
                gemv_zero, gemv_one = GPUFiniteFieldMatrices._stripe_gemv_scalars(T)
                @test gemv_zero === zero_ptr
                @test gemv_one === one_ptr
            end
        end
    end

    N = 127
    A_data = [1 4 7 10 13; 2 5 8 11 14; 3 6 9 12 15]
    B_data = [2 7; 3 8; 4 9; 5 10; 6 11]
    matrix_expected = mod.(A_data * B_data, N)
    vector_A_data = [1 4 7; 2 5 8; 3 6 9]
    vector_x_data = [3, 5, 7]
    vector_expected = mod.(vector_A_data * vector_x_data, N)

    for T in (Float32, Float64), stripe_width in (8, 5, 2, 1)
        A = GPUFiniteFieldMatrices.CuModMatrix{T}(A_data, N)
        B = GPUFiniteFieldMatrices.CuModMatrix{T}(B_data, N)
        C = GPUFiniteFieldMatrices.zeros(T, size(A_data, 1), size(B_data, 2), N)

        CUDA.@sync GPUFiniteFieldMatrices.stripe_mul!(C, A, B; max_ops = stripe_width)
        @test Array(C) == T.(matrix_expected)

        C_single_stripe =
            GPUFiniteFieldMatrices.zeros(T, size(A_data, 1), size(B_data, 2), N)
        CUDA.@sync GPUFiniteFieldMatrices.stripe_mul!(
            C_single_stripe,
            A,
            B;
            max_ops = 1,
            input_bound = N,
        )
        @test Array(C_single_stripe) == T.(matrix_expected)

        vector_A = GPUFiniteFieldMatrices.CuModMatrix{T}(vector_A_data, N)
        x = GPUFiniteFieldMatrices.CuModVector{T}(vector_x_data, N)
        z = GPUFiniteFieldMatrices.zeros(T, size(vector_A_data, 1), N)

        CUDA.@sync GPUFiniteFieldMatrices.stripe_mul!(
            z,
            vector_A,
            x;
            max_ops = stripe_width,
        )
        @test Array(z) == T.(vector_expected)

        z_recomputed = GPUFiniteFieldMatrices.zeros(T, size(vector_A_data, 1), N)
        safe_ops = GPUFiniteFieldMatrices.find_max_stripe_ops(T, N)
        @test_throws ArgumentError GPUFiniteFieldMatrices.stripe_mul!(
            z_recomputed,
            vector_A,
            x;
            max_ops = safe_ops + 1,
            input_bound = N,
        )
        CUDA.@sync GPUFiniteFieldMatrices.stripe_mul!(
            z_recomputed,
            vector_A,
            x;
            max_ops = safe_ops + 1,
            input_bound = N,
            respect_max_ops = false,
        )
        @test Array(z_recomputed) == T.(vector_expected)

        if T === Float32
            @test_throws ArgumentError GPUFiniteFieldMatrices.stripe_mul!(
                z_recomputed,
                vector_A,
                x;
                max_ops = 1,
                input_bound = 100_000,
            )
        end

        z_override = GPUFiniteFieldMatrices.zeros(T, size(vector_A_data, 1), N)
        override_modulus = 31
        CUDA.@sync LinearAlgebra.mul!(
            z_override,
            vector_A,
            x;
            input_bound = N,
            modulus_override = override_modulus,
        )
        @test Array(z_override) == T.(mod.(vector_A_data * vector_x_data, override_modulus))
        @test z_override.N == N

        # A different stored destination modulus does not replace the working
        # reduction modulus, and the allocating adapter records that modulus.
        C_different_modulus = GPUFiniteFieldMatrices.zeros(T, 3, 2, 19)
        CUDA.@sync GPUFiniteFieldMatrices.stripe_mul!(
            C_different_modulus,
            A,
            B;
            modulus_override = 31,
            input_bound = N,
        )
        @test Array(C_different_modulus) == T.(mod.(A_data * B_data, 31))
        @test C_different_modulus.N == 19
        allocated_override = GPUFiniteFieldMatrices.stripe_mul(A, B; modulus_override = 31)
        @test Array(allocated_override) == T.(mod.(A_data * B_data, 31))
        @test allocated_override.N == 31
        @test allocated_override.data !== A.data

        z_different_modulus = GPUFiniteFieldMatrices.zeros(T, 3, 19)
        CUDA.@sync GPUFiniteFieldMatrices.stripe_mul!(
            z_different_modulus,
            vector_A,
            x;
            modulus_override = 31,
        )
        @test Array(z_different_modulus) == T.(mod.(vector_A_data * vector_x_data, 31))
        @test z_different_modulus.N == 19
        allocated_vector_override =
            GPUFiniteFieldMatrices.stripe_mul(vector_A, x; modulus_override = 31)
        @test Array(allocated_vector_override) ==
              T.(mod.(vector_A_data * vector_x_data, 31))
        @test allocated_vector_override.N == 31
    end

    # Float16 uses the default 16-bit GEMM scale type. Small exact inputs keep
    # the test within Float16's integer precision while exercising one and
    # multiple stripes through the cached scalar references.
    if CUDA.capability(CUDA.device()) >= v"5.3"
        half_modulus = 7
        half_A_data = [1 2 3 4 5; 2 3 4 5 6]
        half_B_data = [1 2; 2 3; 3 4; 4 5; 5 6]
        half_expected =
            Float16.(mod.(BigInt.(half_A_data) * BigInt.(half_B_data), half_modulus))
        half_A = GPUFiniteFieldMatrices.CuModMatrix{Float16}(half_A_data, half_modulus)
        half_B = GPUFiniteFieldMatrices.CuModMatrix{Float16}(half_B_data, half_modulus)
        half_A_before = Array(half_A)
        half_B_before = Array(half_B)

        for stripe_width in (5, 2)
            half_C = GPUFiniteFieldMatrices.zeros(Float16, 2, 2, half_modulus)
            CUDA.@sync GPUFiniteFieldMatrices.stripe_mul!(
                half_C,
                half_A,
                half_B;
                max_ops = stripe_width,
            )
            @test Array(half_C) == half_expected
        end

        @test Array(half_A) == half_A_before
        @test Array(half_B) == half_B_before
    end

    # Exercise the computed Float32 stripe budget at the exclusive input bound.
    precision_bound = 2048
    precision_max_ops = GPUFiniteFieldMatrices.find_max_stripe_ops(Float32, precision_bound)
    @test precision_max_ops == 3

    precision_inner_dim = 2 * precision_max_ops + 1
    precision_A_values = fill(precision_bound - 1, 2, precision_inner_dim)
    precision_x_values = fill(precision_bound - 1, precision_inner_dim)
    precision_A =
        GPUFiniteFieldMatrices.CuModMatrix{Float32}(precision_A_values, precision_bound)
    precision_x =
        GPUFiniteFieldMatrices.CuModVector{Float32}(precision_x_values, precision_bound)
    precision_z = GPUFiniteFieldMatrices.zeros(Float32, 2, precision_bound)

    CUDA.@sync GPUFiniteFieldMatrices.stripe_mul!(
        precision_z,
        precision_A,
        precision_x;
        input_bound = precision_bound,
    )
    # Keep the oracle exact, independent of Float32 accumulation.
    precision_expected = Float32.(
        mod.(BigInt.(precision_A_values) * BigInt.(precision_x_values), precision_bound),
    )
    @test Array(precision_z) == precision_expected

    # Exercise matrix-matrix stripe reductions at the computed Float32 budget.
    precision_B_values = fill(precision_bound - 1, precision_inner_dim, 2)
    precision_B =
        GPUFiniteFieldMatrices.CuModMatrix{Float32}(precision_B_values, precision_bound)
    precision_C = GPUFiniteFieldMatrices.zeros(Float32, 2, 2, precision_bound)

    CUDA.@sync GPUFiniteFieldMatrices.stripe_mul!(
        precision_C,
        precision_A,
        precision_B;
        max_ops = precision_max_ops,
    )
    precision_matrix_expected = Float32.(
        mod.(BigInt.(precision_A_values) * BigInt.(precision_B_values), precision_bound),
    )
    @test Array(precision_C) == precision_matrix_expected

    # The matrix single-stripe compatibility route is retained only while its
    # complete inner dimension fits the exact-integer bound.
    @test_throws ArgumentError GPUFiniteFieldMatrices.stripe_mul!(
        precision_C,
        precision_A,
        precision_B;
        input_bound = precision_bound,
    )

    # Nested GPU views preserve parent offsets/strides and leave unrelated
    # destination storage unchanged. The k=7 product exercises multiple
    # stripes with a one-contribution tail and uses an independent BigInt oracle.
    A_parent = CUDA.zeros(Float32, 12, 14)
    A_outer = @view A_parent[2:11, 3:12]
    A_nested = @view A_outer[2:4, 2:8]
    A_host = [1 2 3 4 5 6 7; 8 9 10 11 12 13 14; 15 16 1 2 3 4 5]
    A_nested .= Float32.(A_host)

    B_parent = CUDA.zeros(Float32, 14, 9)
    B_outer = @view B_parent[3:12, 1:8]
    B_nested = @view B_outer[2:8, 3:4]
    B_host = [2 4; 6 8; 10 12; 14 16; 1 3; 5 7; 9 11]
    B_nested .= Float32.(B_host)

    C_parent = CUDA.fill(Float32(77), 8, 7)
    C_outer = @view C_parent[2:7, 2:6]
    C_nested = @view C_outer[2:4, 2:3]
    expected_nested = Float32.(mod.(BigInt.(A_host) * BigInt.(B_host), 211))
    A_parent_before = Array(A_parent)
    B_parent_before = Array(B_parent)
    CUDA.@sync GPUFiniteFieldMatrices._stripe_mul!(
        C_nested,
        @view(A_outer[2:4, 2:8]),
        @view(B_outer[2:8, 3:4]);
        modulus = 211,
        input_bound = 17,
        max_ops = 2,
    )
    expected_C_parent = fill(Float32(77), 8, 7)
    expected_C_parent[3:5, 3:4] .= expected_nested
    @test Array(C_parent) == expected_C_parent
    @test Array(A_parent) == A_parent_before
    @test Array(B_parent) == B_parent_before

    alias_buffer = CUDA.zeros(Float32, 2, 2)
    @test_throws ArgumentError GPUFiniteFieldMatrices._stripe_mul!(
        alias_buffer,
        alias_buffer,
        alias_buffer;
        modulus = 101,
    )

    # Exercise the batched GEMV path with initialized plans and an independent
    # CPU modular product. The inputs are small and deterministic.
    n = 32
    N1 = 11
    N2 = 13
    modulus = N1 * N2
    matrix_indices = collect(0:(n*n-1))
    vector_indices = collect(0:(n-1))
    A1_data = reshape(mod.(matrix_indices, N1), n, n)
    A2_data = reshape(mod.(3 .* matrix_indices .+ 1, N2), n, n)
    x1_data = mod.(vector_indices .+ 1, N1)
    x2_data = mod.(3 .* vector_indices .+ 1, N2)

    A1 = GPUFiniteFieldMatrices.CuModMatrix{Float64}(A1_data, N1)
    A2 = GPUFiniteFieldMatrices.CuModMatrix{Float64}(A2_data, N2)
    x1 = GPUFiniteFieldMatrices.CuModVector{Float64}(x1_data, N1)
    x2 = GPUFiniteFieldMatrices.CuModVector{Float64}(x2_data, N2)
    A = GPUFiniteFieldMatrices.KaratsubaMatrix(A1, A2, N1, N2, modulus)
    x = GPUFiniteFieldMatrices.KaratsubaVector(x1, x2, N1, N2, modulus)
    y = GPUFiniteFieldMatrices.KaratsubaZeros(Float64, n, N1, N2, modulus, true)

    GPUFiniteFieldMatrices.initialize_plan!(A)
    GPUFiniteFieldMatrices.initialize_plan!(x)
    GPUFiniteFieldMatrices.initialize_plan!(y)

    expected = mod.((A1_data + N1 .* A2_data) * (x1_data + N1 .* x2_data), modulus)
    CUDA.@sync GPUFiniteFieldMatrices.karatsuba_gemv!(y, A, x)
    @test Array(y) == expected

end
