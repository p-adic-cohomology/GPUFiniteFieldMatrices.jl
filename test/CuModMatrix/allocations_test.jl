
function test_allocations()

    println("Begin testing allocations.")

    n = 3003
    A_data = Base.rand(0:10, n, n)
    B_data = Base.rand(0:10, n, n)
    x_data = Base.rand(0:10, n)

    A = CuModMatrix(A_data; modulus = 11)
    B = CuModMatrix(B_data; modulus = 11)
    x = CuModVector(x_data; modulus = 11)

    C = GPUFiniteFieldMatrices.zeros(
        GPUFiniteFieldMatrices.CuModMatrix{Float32},
        n,
        n;
        modulus = 11,
    )
    z = GPUFiniteFieldMatrices.zeros(
        GPUFiniteFieldMatrices.CuModVector{Float32},
        n;
        modulus = 11,
    )

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

    # CUDA's native mul! uses two device scalars per GEMM. Allow their bytes
    # while requiring each stripe to reuse the destination buffer.
    A_small = CuModMatrix{Float32}([1 2 3 4; 5 6 7 8]; modulus = 11)
    B_small = CuModMatrix{Float32}([1 2; 3 4; 5 6; 7 8]; modulus = 11)
    C_small = GPUFiniteFieldMatrices.zeros(
        GPUFiniteFieldMatrices.CuModMatrix{Float32},
        2,
        2;
        modulus = 11,
    )
    GPUFiniteFieldMatrices.stripe_mul!(C_small, A_small, B_small; max_ops = 1)
    result = CUDA.@timed GPUFiniteFieldMatrices.stripe_mul!(
        C_small,
        A_small,
        B_small;
        max_ops = 1,
    )
    scalar_bytes = 2 * size(A_small, 2) * sizeof(eltype(C_small))
    @test result[:gpu_bytes] <= scalar_bytes

    A_fl_data = convert.(Float64, A_data)
    trash = CUDA.@timed CuModMatrix{Float64}(A_fl_data; modulus = 11) # prime thie jitter
    result = CUDA.@timed CuModMatrix{Float64}(A_fl_data; modulus = 11)
    @test result[:cpu_bytes] < 1_000_000 # 1 megabyte

    CUDA.@sync println("Done testing allocations.")
end
