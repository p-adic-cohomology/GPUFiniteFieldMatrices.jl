using Random

struct Phase0Regime
    name::Symbol
    square_n::Int
    rect_m::Int
    rect_n::Int
    run_inverse::Bool
    long_only::Bool
end

const _PHASE0_REGIMES = [
    Phase0Regime(:tiny, 16, 12, 20, true, false),
    Phase0Regime(:small, 96, 72, 128, true, false),
    Phase0Regime(:medium, 384, 256, 640, false, false),
    Phase0Regime(:fivekish, 5008, 640, 5032, false, true),
]

const _PHASE0_CASES = [
    (:square, Float32, 7),
    (:square, Float32, 11),
    (:square, Float64, 7),
    (:square, Float64, 11),
    (:rectangular, Float32, 7),
    (:rectangular, Float32, 11),
    (:rectangular, Float64, 7),
    (:rectangular, Float64, 11),
]

function _phase0_long_enabled()
    return get(ENV, "GPUFFM_PHASE0_LONG", "0") == "1"
end

function _phase0_id(::Type{T}, n::Int) where {T}
    M = zeros(T, n, n)
    for i = 1:n
        M[i, i] = one(T)
    end
    return M
end

function _phase0_apply_col_perm(A::Matrix{T}, perm::Vector{Int}) where {T}
    out = similar(A)
    for j in eachindex(perm)
        out[:, j] = A[:, perm[j]]
    end
    return out
end

function _phase0_apply_row_perm(A::Matrix{T}, perm::Vector{Int}) where {T}
    out = similar(A)
    for i in eachindex(perm)
        out[i, :] = A[perm[i], :]
    end
    return out
end

function _phase0_full_row_rank_host(m::Int, n::Int, p::Int, ::Type{T}, rng) where {T}
    A = Matrix{T}(rand(rng, 0:(p-1), m, n))
    Ipart = _phase0_id(T, m)
    for j = 1:m
        A[:, j] .= Ipart[:, j]
    end
    perm = randperm(rng, n)
    return _phase0_apply_col_perm(A, perm)
end

function _phase0_full_col_rank_host(m::Int, n::Int, p::Int, ::Type{T}, rng) where {T}
    A = Matrix{T}(rand(rng, 0:(p-1), m, n))
    Ipart = _phase0_id(T, n)
    for i = 1:n
        A[i, :] .= Ipart[i, :]
    end
    perm = randperm(rng, m)
    return _phase0_apply_row_perm(A, perm)
end

function _phase0_square_host(
    n::Int,
    p::Int,
    ::Type{T},
    rng;
    force_invertible::Bool = true,
) where {T}
    if force_invertible
        A = _phase0_id(T, n)
        for j = 2:min(n, 9)
            A[1, j] = T(rand(rng, 0:(p-1)))
        end
        return A
    end
    return Matrix{T}(rand(rng, 0:(p-1), n, n))
end

function _phase0_mod_ok(A::CuModMatrix)
    H = Array(A)
    return all(x -> 0 <= round(Int, x) < A.N, H)
end

function _phase0_kernel_square_checks(A::CuModMatrix)
    n = size(A, 1)
    p = GPUFiniteFieldMatrices.pluq_init_perm(n)
    q = GPUFiniteFieldMatrices.pluq_init_perm(n)
    Awork = copy(A.data)
    kend = min(16, n)
    rank1 = GPUFiniteFieldMatrices.pluq_basecase_gpu!(Awork, A.N, p, q, 1, kend, n)
    @test rank1 >= 1
    @test length(p) == n
    @test length(q) == n
    if kend < n
        GPUFiniteFieldMatrices.pluq_trsm_left_lower_unit_gpu!(Awork, A.N, 1, kend, n)
        GPUFiniteFieldMatrices.pluq_trsm_right_upper_gpu!(Awork, A.N, 1, kend, n)
        GPUFiniteFieldMatrices.pluq_schur_update_gpu!(Awork, A.N, 1, kend, n)
    end
    Achecked = CuModMatrix(Awork, A.N; new_size = (n, n))
    @test _phase0_mod_ok(Achecked)
    if n <= 128
        popts = PLUQOptions(blocksize = kend, basecase = kend)
        p2, q2, r2 = GPUFiniteFieldMatrices.pluq_blocked_gpu!(copy(A.data), A.N, popts, n)
        @test length(p2) == n
        @test length(q2) == n
        @test 0 <= r2 <= n
    end
end

function _phase0_kernel_rect_checks(A::CuModMatrix)
    m = size(A, 1)
    n = size(A, 2)
    p, q, r = GPUFiniteFieldMatrices.pluq_rectangular_rank_gpu!(copy(A.data), A.N, m, n)
    @test length(p) == m
    @test length(q) == n
    @test 0 <= r <= min(m, n)
end

function _phase0_main_case(reg::Phase0Regime, shape::Symbol, T::DataType, p::Int, rng)
    if shape == :square
        Ahost = _phase0_square_host(reg.square_n, p, T, rng; force_invertible = true)
        A = CuModMatrix(Ahost, p; elem_type = T)
        F = pluq_new(A)
        @test length(F.p) == reg.square_n
        @test length(F.q) == reg.square_n
        @test 0 <= F.rank <= reg.square_n
        if reg.square_n <= 64
            @test pluq_check_identity(F, A)
            L = pluq_extract_L(F)
            U = pluq_extract_U(F)
            @test size(L) == (reg.square_n, reg.square_n)
            @test size(U) == (reg.square_n, reg.square_n)
        end
        if reg.run_inverse && reg.square_n <= 64
            Ainv = inverse_new(A)
            AAinv = mod.(round.(Int, Array(A * Ainv)), p)
            @test AAinv == _phase0_id(Int, reg.square_n)
        end
        _phase0_kernel_square_checks(A)
        return
    end
    Ahost = _phase0_full_row_rank_host(reg.rect_m, reg.rect_n, p, T, rng)
    A = CuModMatrix(Ahost, p; elem_type = T)
    F = pluq_new(A)
    @test length(F.p) == reg.rect_m
    @test length(F.q) == reg.rect_n
    @test 0 <= F.rank <= min(reg.rect_m, reg.rect_n)
    if reg.run_inverse
        Xr = right_inverse_new(A)
        AX = mod.(round.(Int, Array(A * Xr)), p)
        @test AX == _phase0_id(Int, reg.rect_m)
        At = CuModMatrix(
            _phase0_full_col_rank_host(reg.rect_n, reg.rect_m, p, T, rng),
            p;
            elem_type = T,
        )
        Xl = left_inverse_new(At)
        XA = mod.(round.(Int, Array(Xl * At)), p)
        @test XA == _phase0_id(Int, reg.rect_m)
    end
    _phase0_kernel_rect_checks(A)
end

function test_phase0_regime_matrix_grid()
    rng = Random.MersenneTwister(7)
    for reg in _PHASE0_REGIMES
        if reg.long_only && !_phase0_long_enabled()
            @test true
            continue
        end
        @testset "Phase0 regime $(reg.name)" begin
            for (shape, T, p) in _PHASE0_CASES
                @testset "$(shape)-$(T)-p$(p)" begin
                    _phase0_main_case(reg, shape, T, p, rng)
                end
            end
        end
    end
end
