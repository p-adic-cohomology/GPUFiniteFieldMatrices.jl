function test_cuda_blas_compatibility()
    N = 127
    A_data = [1 4 7 10 13; 2 5 8 11 14; 3 6 9 12 15]
    B_data = [2 7; 3 8; 4 9; 5 10; 6 11]
    matrix_expected = mod.(BigInt.(A_data) * BigInt.(B_data), N)
    vector_A_data = A_data
    vector_x_data = [3, 5, 7, 9, 11]
    vector_expected = mod.(BigInt.(vector_A_data) * BigInt.(vector_x_data), N)

    for T in (Float32, Float64), stripe_width in (8, 5, 2, 1)
        A = GPUFiniteFieldMatrices.CuModMatrix{T}(A_data; modulus = N)
        B = GPUFiniteFieldMatrices.CuModMatrix{T}(B_data; modulus = N)
        C = GPUFiniteFieldMatrices.zeros(
            GPUFiniteFieldMatrices.CuModMatrix{T},
            size(A_data, 1),
            size(B_data, 2);
            modulus = N,
        )

        CUDA.@sync GPUFiniteFieldMatrices.stripe_mul!(C, A, B; max_ops = stripe_width)
        @test Array(C) == T.(matrix_expected)

        C_single_stripe = GPUFiniteFieldMatrices.zeros(
            GPUFiniteFieldMatrices.CuModMatrix{T},
            size(A_data, 1),
            size(B_data, 2);
            modulus = N,
        )
        CUDA.@sync GPUFiniteFieldMatrices.stripe_mul!(
            C_single_stripe,
            A,
            B;
            max_ops = 1,
            input_bound = N,
        )
        @test Array(C_single_stripe) == T.(matrix_expected)

        vector_A = GPUFiniteFieldMatrices.CuModMatrix{T}(vector_A_data; modulus = N)
        x = GPUFiniteFieldMatrices.CuModVector{T}(vector_x_data; modulus = N)
        z = GPUFiniteFieldMatrices.zeros(
            GPUFiniteFieldMatrices.CuModVector{T},
            size(vector_A_data, 1);
            modulus = N,
        )

        CUDA.@sync GPUFiniteFieldMatrices.stripe_mul!(
            z,
            vector_A,
            x;
            max_ops = stripe_width,
        )
        @test Array(z) == T.(vector_expected)

        z_recomputed = GPUFiniteFieldMatrices.zeros(
            GPUFiniteFieldMatrices.CuModVector{T},
            size(vector_A_data, 1);
            modulus = N,
        )
        CUDA.@sync GPUFiniteFieldMatrices.stripe_mul!(
            z_recomputed,
            vector_A,
            x;
            max_ops = 1,
            input_bound = N,
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

        z_override = GPUFiniteFieldMatrices.zeros(
            GPUFiniteFieldMatrices.CuModVector{T},
            size(vector_A_data, 1);
            modulus = N,
        )
        override_modulus = 31
        CUDA.@sync LinearAlgebra.mul!(
            z_override,
            vector_A,
            x;
            input_bound = N,
            modulus_override = override_modulus,
        )
        @test Array(z_override) == T.(mod.(vector_A_data * vector_x_data, override_modulus))
        @test GPUFiniteFieldMatrices.modulus(z_override) == N

        z_override_different = GPUFiniteFieldMatrices.zeros(
            GPUFiniteFieldMatrices.CuModVector{T},
            size(vector_A_data, 1);
            modulus = 19,
        )
        CUDA.@sync LinearAlgebra.mul!(
            z_override_different,
            vector_A,
            x;
            modulus_override = override_modulus,
        )
        @test Array(z_override_different) ==
              T.(mod.(BigInt.(vector_A_data) * BigInt.(vector_x_data), override_modulus))
        @test GPUFiniteFieldMatrices.modulus(z_override_different) == 19

        # A different stored destination modulus does not replace the working
        # reduction modulus, and the allocating adapter records that modulus.
        C_different_modulus = GPUFiniteFieldMatrices.zeros(
            GPUFiniteFieldMatrices.CuModMatrix{T},
            3,
            2;
            modulus = 19,
        )
        CUDA.@sync GPUFiniteFieldMatrices.stripe_mul!(
            C_different_modulus,
            A,
            B;
            modulus_override = 31,
            input_bound = N,
        )
        @test Array(C_different_modulus) == T.(mod.(A_data * B_data, 31))
        @test GPUFiniteFieldMatrices.modulus(C_different_modulus) == 19
        allocated_override = GPUFiniteFieldMatrices.stripe_mul(A, B; modulus_override = 31)
        @test Array(allocated_override) == T.(mod.(A_data * B_data, 31))
        @test GPUFiniteFieldMatrices.modulus(allocated_override) == 31
        @test parent(allocated_override) !== parent(A)

        z_different_modulus = GPUFiniteFieldMatrices.zeros(
            GPUFiniteFieldMatrices.CuModVector{T},
            3;
            modulus = 19,
        )
        CUDA.@sync GPUFiniteFieldMatrices.stripe_mul!(
            z_different_modulus,
            vector_A,
            x;
            modulus_override = 31,
        )
        @test Array(z_different_modulus) == T.(mod.(vector_A_data * vector_x_data, 31))
        @test GPUFiniteFieldMatrices.modulus(z_different_modulus) == 19
        allocated_vector_override =
            GPUFiniteFieldMatrices.stripe_mul(vector_A, x; modulus_override = 31)
        @test Array(allocated_vector_override) ==
              T.(mod.(vector_A_data * vector_x_data, 31))
        @test GPUFiniteFieldMatrices.modulus(allocated_vector_override) == 31

        z_reject = GPUFiniteFieldMatrices.zeros(
            GPUFiniteFieldMatrices.CuModVector{T},
            size(vector_A_data, 1);
            modulus = N,
        )
        parent(z_reject)[1:length(z_reject)] .= T.([17, 19, 23])
        z_before_reject = Array(z_reject)
        matrix_before_reject = Array(C)
        vector_matrix_before_reject = Array(vector_A)
        vector_before_reject = Array(x)

        x_bad_modulus =
            GPUFiniteFieldMatrices.CuModVector{T}(vector_x_data; modulus = N + 1)
        x_bad_modulus_before = Array(x_bad_modulus)
        @test_throws GPUFiniteFieldMatrices.CuModArrayModulusMismatchException GPUFiniteFieldMatrices.stripe_mul!(
            z_reject,
            vector_A,
            x_bad_modulus,
        )
        @test Array(z_reject) == z_before_reject
        @test Array(vector_A) == vector_matrix_before_reject
        @test Array(x_bad_modulus) == x_bad_modulus_before

        x_bad_length =
            GPUFiniteFieldMatrices.CuModVector{T}(vector_x_data[1:(end-1)]; modulus = N)
        x_bad_length_before = Array(x_bad_length)
        @test_throws GPUFiniteFieldMatrices.CuModArraySizeMismatchException GPUFiniteFieldMatrices.stripe_mul!(
            z_reject,
            vector_A,
            x_bad_length,
        )
        @test Array(z_reject) == z_before_reject
        @test Array(x_bad_length) == x_bad_length_before

        z_bad_length = GPUFiniteFieldMatrices.zeros(
            GPUFiniteFieldMatrices.CuModVector{T},
            size(vector_A_data, 1) - 1;
            modulus = N,
        )
        parent(z_bad_length)[1:length(z_bad_length)] .= T.([17, 19])
        z_bad_length_before = Array(z_bad_length)
        @test_throws GPUFiniteFieldMatrices.CuModArraySizeMismatchException GPUFiniteFieldMatrices.stripe_mul!(
            z_bad_length,
            vector_A,
            x,
        )
        @test Array(z_bad_length) == z_bad_length_before
        @test Array(vector_A) == vector_matrix_before_reject
        @test Array(x) == vector_before_reject

        z_bad_modulus = GPUFiniteFieldMatrices.zeros(
            GPUFiniteFieldMatrices.CuModVector{T},
            size(vector_A_data, 1);
            modulus = N + 1,
        )
        parent(z_bad_modulus)[1:length(z_bad_modulus)] .= T.([17, 19, 23])
        z_bad_modulus_before = Array(z_bad_modulus)
        @test_throws GPUFiniteFieldMatrices.CuModArrayModulusMismatchException GPUFiniteFieldMatrices.stripe_mul!(
            z_bad_modulus,
            vector_A,
            x,
        )
        @test Array(z_bad_modulus) == z_bad_modulus_before

        elementwise_destination = GPUFiniteFieldMatrices.zeros(
            T,
            size(vector_A_data, 1),
            size(vector_A_data, 2),
            N,
        )
        scalar_destination = GPUFiniteFieldMatrices.zeros(
            T,
            size(vector_A_data, 1),
            size(vector_A_data, 2),
            N,
        )
        elementwise_before_reject = Array(elementwise_destination)
        scalar_before_reject = Array(scalar_destination)
        for invalid_override in (0, -1)
            @test_throws ArgumentError GPUFiniteFieldMatrices.stripe_mul!(
                C,
                A,
                B;
                modulus_override = invalid_override,
            )
            @test Array(C) == matrix_before_reject
            @test_throws ArgumentError GPUFiniteFieldMatrices.stripe_mul(
                A,
                B;
                modulus_override = invalid_override,
            )
            @test_throws ArgumentError GPUFiniteFieldMatrices.stripe_mul!(
                z_reject,
                vector_A,
                x;
                modulus_override = invalid_override,
            )
            @test Array(z_reject) == z_before_reject
            @test_throws ArgumentError GPUFiniteFieldMatrices.stripe_mul(
                vector_A,
                x;
                modulus_override = invalid_override,
            )
            @test_throws ArgumentError LinearAlgebra.mul!(
                z_reject,
                vector_A,
                x_bad_modulus;
                modulus_override = invalid_override,
            )
            @test Array(z_reject) == z_before_reject
            @test_throws ArgumentError elementwise_multiply!(
                elementwise_destination,
                vector_A,
                vector_A;
                modulus_override = invalid_override,
            )
            @test Array(elementwise_destination) == elementwise_before_reject
            @test_throws ArgumentError GPUFiniteFieldMatrices.mul_elementwise!(
                elementwise_destination,
                vector_A,
                vector_A;
                modulus_override = invalid_override,
            )
            @test Array(elementwise_destination) == elementwise_before_reject
            @test_throws ArgumentError LinearAlgebra.mul!(
                scalar_destination,
                vector_A,
                2;
                modulus_override = invalid_override,
            )
            @test Array(scalar_destination) == scalar_before_reject
        end

        for invalid_bound in (1, 0, -1)
            @test_throws ArgumentError GPUFiniteFieldMatrices.stripe_mul!(
                C,
                A,
                B;
                input_bound = invalid_bound,
            )
            @test Array(C) == matrix_before_reject
            @test_throws ArgumentError GPUFiniteFieldMatrices.stripe_mul(
                A,
                B;
                input_bound = invalid_bound,
            )
            @test_throws ArgumentError LinearAlgebra.mul!(
                C,
                A,
                B;
                input_bound = invalid_bound,
            )
            @test Array(C) == matrix_before_reject
            @test_throws ArgumentError GPUFiniteFieldMatrices.stripe_mul!(
                z_reject,
                vector_A,
                x;
                input_bound = invalid_bound,
            )
            @test Array(z_reject) == z_before_reject
            @test_throws ArgumentError GPUFiniteFieldMatrices.stripe_mul(
                vector_A,
                x;
                input_bound = invalid_bound,
            )
            @test_throws ArgumentError LinearAlgebra.mul!(
                z_reject,
                vector_A,
                x;
                input_bound = invalid_bound,
            )
            @test Array(z_reject) == z_before_reject
        end

        @test Array(A) == T.(A_data)
        @test Array(B) == T.(B_data)
        @test Array(vector_A) == vector_matrix_before_reject
        @test Array(x) == vector_before_reject
    end

    # Exercise the computed Float32 stripe budget at the exclusive input bound.
    precision_bound = 2048
    precision_max_ops = GPUFiniteFieldMatrices.find_max_stripe_ops(Float32, precision_bound)
    @test precision_max_ops == 3

    precision_inner_dim = 2 * precision_max_ops + 1
    precision_A_values = fill(precision_bound - 1, 2, precision_inner_dim)
    precision_x_values = fill(precision_bound - 1, precision_inner_dim)
    precision_A = GPUFiniteFieldMatrices.CuModMatrix{Float32}(
        precision_A_values;
        modulus = precision_bound,
    )
    precision_x = GPUFiniteFieldMatrices.CuModVector{Float32}(
        precision_x_values;
        modulus = precision_bound,
    )
    precision_z = GPUFiniteFieldMatrices.zeros(
        GPUFiniteFieldMatrices.CuModVector{Float32},
        2;
        modulus = precision_bound,
    )

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
    precision_B = GPUFiniteFieldMatrices.CuModMatrix{Float32}(
        precision_B_values;
        modulus = precision_bound,
    )
    precision_C = GPUFiniteFieldMatrices.zeros(
        GPUFiniteFieldMatrices.CuModMatrix{Float32},
        2,
        2;
        modulus = precision_bound,
    )

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

    A1 = GPUFiniteFieldMatrices.CuModMatrix{Float64}(A1_data; modulus = N1)
    A2 = GPUFiniteFieldMatrices.CuModMatrix{Float64}(A2_data; modulus = N2)
    x1 = GPUFiniteFieldMatrices.CuModVector{Float64}(x1_data; modulus = N1)
    x2 = GPUFiniteFieldMatrices.CuModVector{Float64}(x2_data; modulus = N2)
    A = GPUFiniteFieldMatrices.KaratsubaMatrix(
        A1,
        A2;
        low_modulus = N1,
        high_modulus = N2,
        modulus = modulus,
    )
    x = GPUFiniteFieldMatrices.KaratsubaVector(
        x1,
        x2;
        low_modulus = N1,
        high_modulus = N2,
        modulus = modulus,
    )
    y = GPUFiniteFieldMatrices.zeros(
        GPUFiniteFieldMatrices.KaratsubaVector{Float64},
        n;
        low_modulus = N1,
        high_modulus = N2,
        modulus = modulus,
    )

    GPUFiniteFieldMatrices.initialize_plan!(A)
    GPUFiniteFieldMatrices.initialize_plan!(x)
    GPUFiniteFieldMatrices.initialize_plan!(y)

    expected = mod.((A1_data + N1 .* A2_data) * (x1_data + N1 .* x2_data), modulus)
    CUDA.@sync GPUFiniteFieldMatrices.karatsuba_gemv!(y, A, x)
    @test Array(y) == expected

end
