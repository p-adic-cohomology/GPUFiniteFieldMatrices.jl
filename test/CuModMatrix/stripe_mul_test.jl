function _stripe_budget_bigint_matmul(
    A::AbstractMatrix{<:Integer},
    B::AbstractMatrix{<:Integer},
    modulus::Integer,
)
    result = zeros(Int, size(A, 1), size(B, 2))
    q = BigInt(modulus)
    for j in axes(B, 2), i in axes(A, 1)
        acc = BigInt(0)
        for k in axes(A, 2)
            acc += BigInt(A[i, k]) * BigInt(B[k, j])
        end
        result[i, j] = Int(mod(acc, q))
    end
    return result
end

function _stripe_budget_bigint_matvec(
    A::AbstractMatrix{<:Integer},
    x::AbstractVector{<:Integer},
    modulus::Integer,
)
    result = zeros(Int, size(A, 1))
    q = BigInt(modulus)
    for i in axes(A, 1)
        acc = BigInt(0)
        for k in axes(A, 2)
            acc += BigInt(A[i, k]) * BigInt(x[k])
        end
        result[i] = Int(mod(acc, q))
    end
    return result
end

function test_stripe_budget_semantics()
    input_bound = 2_048
    reduction_modulus = 5_000_000
    max_ops = GPUFiniteFieldMatrices.find_max_stripe_ops(
        Float32,
        input_bound;
        modulus = reduction_modulus,
    )
    @test max_ops == 2

    # Two inner products fit one stripe; four are an exact multiple of the
    # default budget; five exercise a final tail stripe. Inputs use stored
    # modulus R while the result is reduced by independent working modulus q.
    for (inner, explicit_max_ops) in ((2, nothing), (4, nothing), (5, nothing), (4, 1))
        A_data = [mod(37i + 17k, input_bound) for i = 1:2, k = 1:inner]
        B_data = [mod(23k + 11j, input_bound) for k = 1:inner, j = 1:3]
        A = CuModMatrix{Float32}(A_data; modulus = input_bound)
        B = CuModMatrix{Float32}(B_data; modulus = input_bound)
        C = GPUFiniteFieldMatrices.zeros(
            GPUFiniteFieldMatrices.CuModMatrix{Float32},
            2,
            3;
            modulus = reduction_modulus,
        )

        GPUFiniteFieldMatrices.stripe_mul!(
            C,
            A,
            B;
            modulus_override = reduction_modulus,
            input_bound,
            max_ops = explicit_max_ops,
        )

        @test Array(C) == _stripe_budget_bigint_matmul(A_data, B_data, reduction_modulus)
        @test A.N == input_bound
        @test B.N == input_bound
    end

    inner = 5
    A_data = [mod(31i + 13k, input_bound) for i = 1:2, k = 1:inner]
    x_data = [mod(19k + 5, input_bound) for k = 1:inner]
    A = CuModMatrix{Float32}(A_data; modulus = input_bound)
    x = CuModVector{Float32}(x_data; modulus = input_bound)
    z = GPUFiniteFieldMatrices.zeros(
        GPUFiniteFieldMatrices.CuModVector{Float32},
        2;
        modulus = reduction_modulus,
    )
    GPUFiniteFieldMatrices.stripe_mul!(
        z,
        A,
        x;
        modulus_override = reduction_modulus,
        input_bound,
    )
    @test Array(z) == _stripe_budget_bigint_matvec(A_data, x_data, reduction_modulus)

    unsafe_C = GPUFiniteFieldMatrices.zeros(
        GPUFiniteFieldMatrices.CuModMatrix{Float32},
        2,
        3;
        modulus = reduction_modulus,
    )
    fill!(unsafe_C.data, 7)
    original_C = Array(unsafe_C)
    A_data = [mod(37i + 17k, input_bound) for i = 1:2, k = 1:4]
    B_data = [mod(23k + 11j, input_bound) for k = 1:4, j = 1:3]
    A = CuModMatrix{Float32}(A_data; modulus = input_bound)
    B = CuModMatrix{Float32}(B_data; modulus = input_bound)
    @test_throws ArgumentError GPUFiniteFieldMatrices.stripe_mul!(
        unsafe_C,
        A,
        B;
        modulus_override = reduction_modulus,
        input_bound,
        max_ops = max_ops + 1,
    )
    @test Array(unsafe_C) == original_C

    unsafe_z = GPUFiniteFieldMatrices.zeros(
        GPUFiniteFieldMatrices.CuModVector{Float32},
        2;
        modulus = reduction_modulus,
    )
    fill!(unsafe_z.data, 7)
    original_z = Array(unsafe_z)
    x_data = [mod(19k + 5, input_bound) for k = 1:4]
    x = CuModVector{Float32}(x_data; modulus = input_bound)
    @test_throws ArgumentError GPUFiniteFieldMatrices.stripe_mul!(
        unsafe_z,
        A,
        x;
        modulus_override = reduction_modulus,
        input_bound,
        max_ops = max_ops + 1,
    )
    @test Array(unsafe_z) == original_z

    # Empty inner dimensions produce zero outputs without entering cuBLAS.
    A_empty_data = zeros(Int, 2, 0)
    B_empty_data = zeros(Int, 0, 3)
    A_empty = CuModMatrix{Float32}(A_empty_data; modulus = input_bound)
    B_empty = CuModMatrix{Float32}(B_empty_data; modulus = input_bound)
    C_empty = GPUFiniteFieldMatrices.zeros(
        GPUFiniteFieldMatrices.CuModMatrix{Float32},
        2,
        3;
        modulus = reduction_modulus,
    )
    GPUFiniteFieldMatrices.stripe_mul!(
        C_empty,
        A_empty,
        B_empty;
        modulus_override = reduction_modulus,
        input_bound,
    )
    @test Array(C_empty) == zeros(Int, 2, 3)

    x_empty = CuModVector{Float32}(Int[]; modulus = input_bound)
    z_empty = GPUFiniteFieldMatrices.zeros(
        GPUFiniteFieldMatrices.CuModVector{Float32},
        2;
        modulus = reduction_modulus,
    )
    GPUFiniteFieldMatrices.stripe_mul!(
        z_empty,
        A_empty,
        x_empty;
        modulus_override = reduction_modulus,
        input_bound,
    )
    @test Array(z_empty) == zeros(Int, 2)
end

function test_stripe_mul()
    test_stripe_budget_semantics()

    A_data = Base.rand(1:100, 100, 100)
    B_data = Base.rand(1:100, 100, 100)
    A = CuModMatrix{Float32}(A_data; modulus = 2^11)
    B = CuModMatrix{Float32}(B_data; modulus = 2^11)
    C = GPUFiniteFieldMatrices.zeros(
        GPUFiniteFieldMatrices.CuModMatrix{Float32},
        100,
        100;
        modulus = 2^11,
    )

    GPUFiniteFieldMatrices.stripe_mul!(C, A, B)

    C_cpu = A_data * B_data .% 2^11

    @test convert.(Float32, C_cpu) == Array(C)

    A_data = Base.rand(1:100, 100, 100)
    B_data = Base.rand(1:100, 100, 100)
    A_data[5, 5] = 11^3 - 5
    B_data[95, 95] = 11^3 - 23

    A = CuModMatrix{Float32}(A_data; modulus = 11^3)
    B = CuModMatrix{Float32}(B_data; modulus = 11^3)
    C = GPUFiniteFieldMatrices.zeros(
        GPUFiniteFieldMatrices.CuModMatrix{Float32},
        100,
        100;
        modulus = 11^3,
    )

    GPUFiniteFieldMatrices.stripe_mul!(C, A, B)

    C_cpu = A_data * B_data .% 11^3

    @test C_cpu == Array(C)

    println("Starting bigger example")
    s = 3003
    # bigger example
    println("Allocating memory...")
    A_data = Base.rand(1:(11^3), s, s)
    B_data = Base.rand(1:(11^3), s, s)
    A_data[5, 5] = 11^3 - 5
    B_data[95, 95] = 11^3 - 23

    A = CuModMatrix{Float32}(A_data; modulus = 11^3)
    B = CuModMatrix{Float32}(B_data; modulus = 11^3)
    C = GPUFiniteFieldMatrices.zeros(
        GPUFiniteFieldMatrices.CuModMatrix{Float32},
        s,
        s;
        modulus = 11^3,
    )

    println("Multiplying on the GPU...")
    CUDA.@time GPUFiniteFieldMatrices.stripe_mul!(C, A, B)
    CUDA.@sync println("Done with GPU")

    println("Multiplying on the CPU...")
    C_cpu = A_data * B_data .% 11^3

    @test C_cpu == Array(C)
    println("Done testing stripe mat-mat mul")

    println("Testing matrix-vector multiplication")

    A_data = Base.rand(1:100, 100, 100)
    A_data[5, 5] = 11^3 - 5
    A_data[95, 95] = 11^3 - 23
    x_data = Base.rand(1:100, 100)

    A = CuModMatrix{Float32}(A_data; modulus = 11^3)
    x = CuModVector{Float32}(x_data; modulus = 11^3)
    z = GPUFiniteFieldMatrices.zeros(
        GPUFiniteFieldMatrices.CuModVector{Float32},
        100;
        modulus = 11^3,
    )

    GPUFiniteFieldMatrices.stripe_mul!(z, A, x)

    z_cpu = A_data * x_data .% 11^3

    @test z_cpu == Array(z)

    println("Starting bigger mat-vec-mul example")

    s = 3003
    # bigger example
    println("Allocating memory...")
    A_data = Base.rand(1:(7^4), s, s)
    x_data = Base.rand(1:(7^4), s)
    A_data[5, 5] = 7^4 - 5
    A_data[95, 95] = 7^4 - 23

    A = CuModMatrix{Float32}(A_data; modulus = 7^4)
    x = CuModVector{Float32}(x_data; modulus = 7^4)
    z = GPUFiniteFieldMatrices.zeros(
        GPUFiniteFieldMatrices.CuModVector{Float32},
        s;
        modulus = 7^4,
    )

    println("Multiplying on the GPU...")
    CUDA.@time GPUFiniteFieldMatrices.stripe_mul!(z, A, x)
    CUDA.@sync println("Done with GPU")

    println("Multiplying on the CPU...")
    z_cpu = A_data * x_data .% 7^4

    @test z_cpu == Array(z)
    println("Done testing stripe mat-vec-mul")
end
