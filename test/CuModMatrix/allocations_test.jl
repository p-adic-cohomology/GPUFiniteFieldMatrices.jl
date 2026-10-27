
function test_allocations()

    println("Begin testing allocations.")

    n = 3003
    A_data = Base.rand(0:10, n, n)
    B_data = Base.rand(0:10, n, n)
    x_data = Base.rand(0:10, n)

    A = CuModMatrix(A_data, 11)
    B = CuModMatrix(B_data, 11)
    x = CuModVector(x_data, 11)

    C = GPUFiniteFieldMatrices.zeros(Float32, n, n, 11)
    z = GPUFiniteFieldMatrices.zeros(Float32, n, 11)

    #Note: BenchmarkTools doesn't seem to support tracking gpu allocations
    #as of May 2025

    result = CUDA.@timed GPUFiniteFieldMatrices.add!(C, A, B)
    @test result[:gpu_bytes] == 0

    result = CUDA.@timed GPUFiniteFieldMatrices.sub!(C, A, B)
    @test result[:gpu_bytes] == 0

    result = CUDA.@timed GPUFiniteFieldMatrices.neg!(C, A)
    @test result[:gpu_bytes] == 0

    result = CUDA.@timed scalar_add!(C, A, 2)
    @test result[:gpu_bytes] == 0

    result = CUDA.@timed scalar_sub!(C, A, 2)
    @test result[:gpu_bytes] == 0

    result = CUDA.@timed mul!(C, A, 2)
    @test result[:gpu_bytes] == 0

    result = CUDA.@timed elementwise_multiply!(C, A, B)
    @test result[:gpu_bytes] == 0

    result = CUDA.@timed GPUFiniteFieldMatrices.copy!(C, A)
    @test result[:gpu_bytes] == 0

    result = CUDA.@timed mod_elements!(C; modulus_override = 3)
    @test result[:gpu_bytes] == 0

    result = CUDA.@timed mul!(C, A, B)
    @test result[:gpu_bytes] < 20

    result = CUDA.@timed mul!(z, A, x)
    @test result[:gpu_bytes] < 20

    # A mutating multi-stripe product reuses its destination and does not
    # allocate a GPU result buffer for each stripe.
    A_small = CuModMatrix{Float32}([1 2 3 4; 5 6 7 8], 11)
    B_small = CuModMatrix{Float32}([1 2; 3 4; 5 6; 7 8], 11)
    C_small = GPUFiniteFieldMatrices.zeros(Float32, 2, 2, 11)
    GPUFiniteFieldMatrices.stripe_mul!(C_small, A_small, B_small; max_ops = 1)
    result = CUDA.@timed GPUFiniteFieldMatrices.stripe_mul!(
        C_small,
        A_small,
        B_small;
        max_ops = 1,
    )
    @test result[:gpu_bytes] < 20

    A_fl_data = convert.(Float64, A_data)
    trash = CUDA.@timed CuModMatrix{Float64}(A_fl_data, 11) # prime thie jitter
    result = CUDA.@timed CuModMatrix{Float64}(A_fl_data, 11)
    @test result[:cpu_bytes] < 1_000_000 # 1 megabyte

    CUDA.@sync println("Done testing allocations.")
end
