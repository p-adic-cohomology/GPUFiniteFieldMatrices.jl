function test_cuda_blas_compatibility()
    N = 127
    A_data = [1 4 7 10 13; 2 5 8 11 14; 3 6 9 12 15]
    B_data = [2 7; 3 8; 4 9; 5 10; 6 11]
    matrix_expected = mod.(A_data * B_data, N)
    vector_A_data = [1 4 7; 2 5 8; 3 6 9]
    vector_x_data = [3, 5, 7]
    vector_expected = mod.(vector_A_data * vector_x_data, N)

    for T in (Float32, Float64), stripe_width in (8, 2)
        A = GPUFiniteFieldMatrices.CuModMatrix(A_data, N, elem_type = T)
        B = GPUFiniteFieldMatrices.CuModMatrix(B_data, N, elem_type = T)
        C = GPUFiniteFieldMatrices.zeros(T, size(A_data, 1), size(B_data, 2), N)

        CUDA.@sync GPUFiniteFieldMatrices.stripe_mul!(C, A, B; M = stripe_width)
        @test Array(C) == T.(matrix_expected)

        vector_A = GPUFiniteFieldMatrices.CuModMatrix(vector_A_data, N, elem_type = T)
        x = GPUFiniteFieldMatrices.CuModVector(vector_x_data, N, elem_type = T)
        z = GPUFiniteFieldMatrices.zeros(T, size(vector_A_data, 1), N)

        CUDA.@sync GPUFiniteFieldMatrices.stripe_mul!(z, vector_A, x; M = stripe_width)
        @test Array(z) == T.(vector_expected)
    end

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
    CUDA.@sync GPUFiniteFieldMatrices.KMatMul_gemv!(y, A, x)
    @test Array(y) == expected
end
