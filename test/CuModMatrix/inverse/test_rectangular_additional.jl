function _extra_id(::Type{T}, n::Int) where {T}
    M = zeros(T, n, n)
    for i = 1:n
        M[i, i] = one(T)
    end
    return M
end

function _extra_apply_col_perm(A::Matrix{T}, perm::Vector{Int}) where {T}
    out = similar(A)
    for j in eachindex(perm)
        out[:, j] = A[:, perm[j]]
    end
    return out
end

function _extra_apply_row_perm(A::Matrix{T}, perm::Vector{Int}) where {T}
    out = similar(A)
    for i in eachindex(perm)
        out[i, :] = A[perm[i], :]
    end
    return out
end

function _extra_full_row_rank_host(m::Int, n::Int, p::Int, ::Type{T}, rng) where {T}
    A = Matrix{T}(rand(rng, 0:(p-1), m, n))
    Ipart = _extra_id(T, m)
    for j = 1:m
        A[:, j] .= Ipart[:, j]
    end
    return _extra_apply_col_perm(A, randperm(rng, n))
end

function _extra_full_col_rank_host(m::Int, n::Int, p::Int, ::Type{T}, rng) where {T}
    A = Matrix{T}(rand(rng, 0:(p-1), m, n))
    Ipart = _extra_id(T, n)
    for i = 1:n
        A[i, :] .= Ipart[i, :]
    end
    return _extra_apply_row_perm(A, randperm(rng, m))
end

function _extra_invertible_square_host(n::Int, p::Int, ::Type{T}, rng) where {T}
    A = _extra_id(T, n)
    for j = 2:min(n, 17)
        A[1, j] = T(rand(rng, 0:(p-1)))
    end
    return A
end

function test_rectangular_additional_sizes()
    rng = Random.MersenneTwister(97)
    row_cases = ((17, 37), (31, 64), (48, 96), (65, 129))
    col_cases = ((37, 17), (64, 31), (96, 48), (129, 65))
    square_cases = (17, 33, 65)
    for p in (7, 11, 101)
        for T in (Float32, Float64)
            for (m, n) in row_cases
                Ahost = _extra_full_row_rank_host(m, n, p, T, rng)
                A = CuModMatrix(Ahost, p; elem_type = T)
                X = right_inverse_new(A, options = PLUQOptions(autotune = true))
                AX = mod.(Array(A * X), p)
                @test AX == _extra_id(Int, m)
            end
            for (m, n) in col_cases
                Ahost = _extra_full_col_rank_host(m, n, p, T, rng)
                A = CuModMatrix(Ahost, p; elem_type = T)
                X = left_inverse_new(A, options = PLUQOptions(autotune = true))
                XA = mod.(Array(X * A), p)
                @test XA == _extra_id(Int, n)
            end
            for n in square_cases
                Ahost = _extra_invertible_square_host(n, p, T, rng)
                A = CuModMatrix(Ahost, p; elem_type = T)
                Xaug = inverse_new(
                    A,
                    options = PLUQOptions(autotune = true, inverse_strategy = :augmented),
                )
                Iaug = mod.(Array(A * Xaug), p)
                @test Iaug == _extra_id(Int, n)
            end
        end
    end
end

# The augmented elimination kernel must read each multiplier before any thread
# clears the pivot column.  A dense leading block makes that dependency present
# in every elimination step, unlike identity-leading construction helpers.
function test_dense_rectangular_right_inverse()
    rng = Random.MersenneTwister(20260915)
    m, n, p = 96, 192, 101
    Ahost = zeros(Float32, m, n)
    Ahost[:, 1:m] .= Float32.(rand(rng, 0:(p-1), m, m))
    for i = 1:m
        Ahost[i, m+i] = 1.0f0
    end
    A = CuModMatrix(Ahost, p; elem_type = Float32)
    X = right_inverse_new(A, options = PLUQOptions(autotune = true, check_prime = true))
    @test mod.(Array(A * X), p) == _extra_id(Int, m)
end

function test_right_inverse_singular_leading_block_fallback()
    p = 101
    # The leading 2×2 block is singular, but columns 2 and 3 form I₂.
    A = CuModMatrix(Float32[0 1 0; 0 0 1], p; elem_type = Float32)
    X = right_inverse_new(A, options = PLUQOptions(autotune = true, check_prime = true))
    @test mod.(Array(A * X), p) == _extra_id(Int, 2)
end
