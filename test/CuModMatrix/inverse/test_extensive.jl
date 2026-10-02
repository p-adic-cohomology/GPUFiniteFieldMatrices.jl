using Random

function _random_invertible_matrix(n::Int, N::Int; max_tries::Int = 40)
    for _ = 1:max_tries
        Ahost = rand(0:(N-1), n, n)
        A = CuModMatrix(Ahost, N)
        if is_invertible_new(A)
            return A
        end
    end
    error("failed to sample invertible matrix of size $n modulo $N")
end

function _id_matrix(::Type{T}, n::Int) where {T}
    M = zeros(T, n, n)
    for i = 1:n
        M[i, i] = one(T)
    end
    return M
end

function _strategy_invertible_host(n::Int, N::Int, ::Type{T} = Int) where {T}
    A = Matrix{T}(I, n, n)
    for j = 2:n
        A[1, j] = T(mod(3j + 5, N))
    end
    return A
end

function _assert_inverse_identity(A::CuModMatrix, X::CuModMatrix)
    n = size(A, 1)
    got = mod.(round.(Int, Array(A * X)), A.N)
    want = _id_matrix(Int, n)
    @test got == want
end

function test_identity_range()
    N = 101
    for n = 1:64
        Ih = _id_matrix(Int, n)
        Icu = CuModMatrix(Ih, N)
        F = pluq_new(Icu)
        @test F.rank == n
        @test pluq_check_identity(F, Icu)
        Iinv = inverse_new(Icu)
        @test Array(Iinv) == Ih
    end
end

function test_square_inverse_strategies()
    N = 101
    for strat in (:augmented, :pluq)
        for n in (4, 16, 32, 33, 64, 129)
            A = CuModMatrix(_strategy_invertible_host(n, N), N)
            X = inverse_new(A, options = PLUQOptions(inverse_strategy = strat))
            _assert_inverse_identity(A, X)
        end
    end

    A = CuModMatrix(_strategy_invertible_host(33, N), N)
    X = inverse_pluq_new(A)
    _assert_inverse_identity(A, X)
end

function test_random_invertible_batch()
    N = 101
    for n in (2, 3, 4, 5, 7, 8, 12, 16, 24, 31, 32, 33, 48, 64)
        A = _random_invertible_matrix(n, N)
        F = pluq_new(A)
        @test F.rank == n
        @test pluq_check_identity(F, A)
        Ainv = inverse_new(A)
        left = mod.(Array(A * Ainv), N)
        right = _id_matrix(eltype(left), n)
        @test left == right
    end
end

function test_dense_random_blocked_inverse()
    N = 101
    rng = Random.MersenneTwister(9096)
    Ahost = Float32.(rand(rng, 0:(N-1), 96, 96))
    A = CuModMatrix(Ahost, N; elem_type = Float32)
    options = PLUQOptions(blocksize = 64, basecase = 32)
    F = pluq_new(A, options = options)
    @test F.rank == 96
    @test pluq_check_identity(F, A)
    _assert_inverse_identity(A, inverse_new(A, options = options))
end

function test_random_singular_batch()
    N = 101
    for n in (2, 3, 4, 5, 8, 16, 32, 33, 64)
        Ahost = rand(0:(N-1), n, n)
        if n > 1
            Ahost[2, :] = Ahost[1, :]
        end
        A = CuModMatrix(Ahost, N)
        @test !is_invertible_new(A)
        @test_throws GPUFiniteFieldMatrices.InverseNotDefinedException inverse_new(
            A,
            options = PLUQOptions(inverse_strategy = :augmented),
        )
        @test_throws GPUFiniteFieldMatrices.InverseNotDefinedException inverse_new(
            A,
            options = PLUQOptions(inverse_strategy = :pluq),
        )
    end
end

function test_rank_deficient_blocked_pluq()
    N = 101
    n = 64
    Ahost = _strategy_invertible_host(n, N)
    Ahost[2, :] = Ahost[1, :]
    A = CuModMatrix(Ahost, N)
    F = pluq_new(A)
    @test F.rank < n
    @test_throws GPUFiniteFieldMatrices.InverseNotDefinedException inverse_new(
        A,
        options = PLUQOptions(inverse_strategy = :pluq),
    )
end

function test_modulus_contracts()
    A = CuModMatrix([1 0; 0 1], 9)
    @test_throws GPUFiniteFieldMatrices.CuModMatrixModulusNotPrimeException inverse_new(
        A,
        options = PLUQOptions(check_prime = true),
    )
    @test inverse_new(A, options = PLUQOptions(check_prime = false)) isa CuModMatrix

    bigN = Int(typemax(Int32)) + 2
    Abig = CuModMatrix([1.0 0.0; 0.0 1.0], bigN; mod = false, elem_type = Float64)
    @test_throws GPUFiniteFieldMatrices.InverseOverflowError inverse_new(Abig)

    mersenne = 2_147_483_647
    H = Float64[mersenne - 1 0; 0 mersenne - 1]
    Alarge = CuModMatrix(H, mersenne; elem_type = Float64)
    @test_throws GPUFiniteFieldMatrices.InverseOverflowError inverse_new(
        Alarge,
        options = PLUQOptions(inverse_strategy = :pluq, trsm_mode = :warp),
    )
    Xaug = inverse_new(Alarge, options = PLUQOptions(inverse_strategy = :augmented))
    @test Array(Xaug)[1:2, 1:2] == H
end

function test_modulus_diversity()
    for N in (2, 101)
        A = CuModMatrix([1 1; 0 1], N)
        for strategy in (:augmented, :pluq)
            X = inverse_new(
                A,
                options = PLUQOptions(inverse_strategy = strategy, check_prime = true),
            )
            @test mod.(round.(Int, Array(A * X)), N) == Matrix{Int}(I, 2, 2)
        end
    end
end

function test_api_rejections()
    wide = CuModMatrix([1 0 0; 0 1 0], 101)
    tall = CuModMatrix([1 0; 0 1; 0 0], 101)
    @test_throws GPUFiniteFieldMatrices.CuModMatrixNotSquareException inverse_new(wide)
    @test_throws GPUFiniteFieldMatrices.CuModArraySizeMismatchException right_inverse_new(
        tall,
    )
    @test_throws GPUFiniteFieldMatrices.CuModArraySizeMismatchException left_inverse_new(
        wide,
    )
end

function test_noninvolutory_permutation_inverse()
    # Exercises permutation cycles longer than two; using p/q directly instead
    # of their inverse maps returns a permutation matrix rather than identity.
    H = [
        0 0 0 0 0 0 0 -3 0 0;
        0 0 0 0 0 0 0 0 -3 0;
        0 1 0 0 0 0 0 0 0 0;
        0 0 1 0 0 0 0 0 0 0;
        0 -2 0 0 0 0 0 0 0 -3;
        0 0 -2 0 2 0 0 0 0 0;
        0 0 0 1 0 2 0 0 0 0;
        0 -3 0 -2 0 0 0 -1 0 0;
        0 0 -3 0 0 0 2 0 -1 0;
        1 0 0 -3 0 0 0 0 0 -1;
    ]
    A = CuModMatrix(H, 7)
    X = inverse_new(A, options = PLUQOptions(inverse_strategy = :pluq))
    @test mod.(round.(Int, Array(A * X)), 7) == Matrix{Int}(I, 10, 10)
end

function test_padding_sensitive_sizes()
    N = 101
    for n in (1, 2, 3, 31, 32, 33, 63, 64)
        A = _random_invertible_matrix(n, N)
        F = pluq_new(A)
        @test pluq_check_identity(F, A)
    end
end
