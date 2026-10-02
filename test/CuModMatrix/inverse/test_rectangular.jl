function _rect_id(::Type{T}, n::Int) where {T}
    M = zeros(T, n, n)
    for i = 1:n
        M[i, i] = one(T)
    end
    return M
end

function _random_full_row_rank(m::Int, n::Int, p::Int; max_tries::Int = 64)
    for _ = 1:max_tries
        Ahost = rand(0:(p-1), m, n)
        A = CuModMatrix(Ahost, p)
        try
            X = right_inverse_new(A)
            return A, X
        catch
        end
    end
    error("failed to sample full-row-rank matrix of size $(m)x$(n)")
end

function _random_full_col_rank(m::Int, n::Int, p::Int; max_tries::Int = 64)
    for _ = 1:max_tries
        Ahost = rand(0:(p-1), m, n)
        A = CuModMatrix(Ahost, p)
        try
            X = left_inverse_new(A)
            return A, X
        catch
        end
    end
    error("failed to sample full-col-rank matrix of size $(m)x$(n)")
end

function test_rectangular_inverses()
    p = 101
    for (m, n) in ((2, 3), (3, 5), (8, 16), (16, 33))
        A, X = _random_full_row_rank(m, n, p)
        AX = mod.(Array(A * X), p)
        @test AX == _rect_id(eltype(AX), m)
    end
    for (m, n) in ((3, 2), (5, 3), (16, 8), (33, 16))
        A, X = _random_full_col_rank(m, n, p)
        XA = mod.(Array(X * A), p)
        @test XA == _rect_id(eltype(XA), n)
    end
end

function test_rectangular_rank_and_failures()
    p = 101
    Awide = CuModMatrix([1 2 3 4; 2 4 6 8], p)
    Fwide = pluq_new(Awide)
    @test Fwide.rank == 1
    @test_throws GPUFiniteFieldMatrices.InverseNotDefinedException right_inverse_new(Awide)

    Atall = CuModMatrix([1 2; 2 4; 3 6], p)
    Ftall = pluq_new(Atall)
    @test Ftall.rank == 1
    @test_throws GPUFiniteFieldMatrices.InverseNotDefinedException left_inverse_new(Atall)

    # A full-row-rank matrix may have an all-zero leading column.  The blocked
    # device panel stops there and `pluq_new` must restart from its saved input
    # with complete pivoting rather than treating the matrix as rank-deficient.
    Ashuffled = CuModMatrix([0 1 0 2; 0 0 1 3], p)
    Fshuffled = pluq_new(Ashuffled)
    @test Fshuffled.rank == 2
    Xshuffled = right_inverse_new(Ashuffled)
    @test mod.(Array(Ashuffled * Xshuffled), p) == _rect_id(eltype(Ashuffled.data), 2)
end
