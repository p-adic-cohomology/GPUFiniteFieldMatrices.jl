module MetalOrdinaryPLUQTests

using GPUFiniteFieldMatrices, Metal, Random, Test

const FFM = GPUFiniteFieldMatrices
const PANEL_OPTIONS = PLUQOptions(blocksize = 4, basecase = 3, trsm_mode = :panel)
const LARGE_PRIME = 16_777_213

upload(x, ::Type{S}) where {S} = MtlArray{eltype(x),ndims(x),S}(x)

function padded(input)
    m, n = size(input)
    storage = fill(Float32(-99), m + 3, n + 5)
    storage[1:m, 1:n] .= input
    return storage
end

function check_padding(actual, original, m, n)
    @test actual[(m+1):end, :] == original[(m+1):end, :]
    @test actual[1:m, (n+1):end] == original[1:m, (n+1):end]
end

# Independent exact reference for both a full active rectangle and a clipped
# diagonal panel. A partial panel only searches the current column.
function factor_reference!(A, N, p, q, dinv, start, rowstop, colstop, m, n; partial = false)
    rank = 0
    for k = start:min(rowstop, colstop)
        columns = partial ? (k:k) : (k:colstop)
        domain = CartesianIndices((k:rowstop, columns))
        pivot = findfirst(ij -> !iszero(mod(A[ij[1], ij[2]], N)), domain)
        isnothing(pivot) && break
        i, j = Tuple(domain[pivot])
        A[[k, i], 1:n] = A[[i, k], 1:n]
        A[1:m, [k, j]] = A[1:m, [j, k]]
        p[k], p[i] = p[i], p[k]
        q[k], q[j] = q[j], q[k]
        dinv[k] = invmod(A[k, k], N)
        for row = (k+1):rowstop
            A[row, k] = mod(A[row, k] * dinv[k], N)
        end
        for col = (k+1):colstop, row = (k+1):rowstop
            A[row, col] = mod(A[row, col] - A[row, k] * A[k, col], N)
        end
        rank += 1
    end
    return rank
end

function reference(input, p)
    A = mod.(BigInt.(input), p)
    m, n = size(A)
    rows, cols = collect(1:m), collect(1:n)
    rank = factor_reference!(A, p, rows, cols, zeros(BigInt, min(m, n)), 1, m, n, m, n)
    return A, rows, cols, rank
end

function test_pivot(S)
    p = 11
    m, n = 137, 151
    host = padded(zeros(Float32, m, n))
    host[3, 4] = 22
    host[133, 128] = -1
    host[135, 128] = 5
    host[3, 129] = 4
    A = upload(host, S)
    slot = upload(Int32[-7], S)
    @test FFM._pluq_ka_find_pivot!(slot, A, 3, m, n, p) === slot
    @test only(Array(slot)) == (128 - 3) * (m - 2) + (133 - 3) + 1
    @test Array(A) == host

    @test FFM._pluq_ka_find_pivot!(slot, A, 3, 8, 9, p) === slot
    @test only(Array(slot)) == (8 - 2) * (9 - 2) + 1
    @test FFM._pluq_ka_find_pivot!(slot, A, 1, 0, 9, p) === slot
    @test only(Array(slot)) == 1
    @test_throws ArgumentError FFM._pluq_ka_find_pivot!(slot, A, 0, m, n, p)
    @test only(Array(slot)) == 1
end

function test_fused_panel(S)
    p = 65537
    for (m, n, start, stop, partial) in ((9, 9, 2, 7, false), (137, 144, 3, 134, true))
        input = zeros(Float32, m, n)
        input[stop, start] = p - 2
        input[stop-1, start] = p - 3
        input[stop-1, stop] = p - 5
        input[stop, stop] = p - 7
        input[stop+1, start] = p - 1
        input[start, stop+1] = p - 8
        original = padded(input)
        expected = BigInt.(original)
        rows, cols = collect(1:m), collect(1:n)
        inverses = fill(BigInt(-8), min(m, n))
        expected_rank = factor_reference!(
            expected,
            p,
            rows,
            cols,
            inverses,
            start,
            stop,
            stop,
            m,
            n;
            partial,
        )
        A = upload(original, S)
        rowdev = upload(Int32.(1:m), S)
        coldev = upload(Int32.(1:n), S)
        dinv = upload(fill(-8.0f0, min(m, n)), S)
        slot = upload(Int32[-1], S)
        result = FFM._pluq_ka_panel!(
            A,
            p,
            rowdev,
            coldev,
            dinv,
            slot,
            Vector{Int32}(undef, 1),
            start,
            stop,
            m,
            n,
            Val(partial),
        )
        @test result == expected_rank
        @test Array(rowdev) == rows
        @test Array(coldev) == cols
        @test Array(dinv) == inverses
        @test Array(A) == expected
    end
end

function trsm_reference(input, N, k0, kend, m, n)
    out = BigInt.(input)
    for col = (kend+1):n, row = k0:kend
        acc = out[row, col]
        for k = k0:(row-1)
            acc -= out[row, k] * out[k, col]
        end
        out[row, col] = mod(acc, N)
    end
    for row = (kend+1):m, col = k0:kend
        acc = out[row, col]
        for k = k0:(col-1)
            acc -= out[row, k] * out[k, col]
        end
        diagonal = mod(out[col, col], N)
        out[row, col] = iszero(diagonal) ? 0 : mod(acc * invmod(diagonal, N), N)
    end
    return out
end

function test_trsm(S)
    rng = MersenneTwister(1009)
    m, n, start, stop = 139, 147, 3, 8
    for p in (11, LARGE_PRIME)
        input = rand(rng, (1-p):(p-1), m, n)
        input[start, start] = 2
        input[start+1, start+1] = -3
        input[start+2, start+2] = 0
        for k = (start+3):stop
            input[k, k] = k
        end
        original = padded(input)
        expected = trsm_reference(original, p, start, stop, m, n)
        @test (FFM.find_max_ops(Float32, p) >= stop - start + 1) == (p == 11)

        for cached in (false, true)
            A = upload(original, S)
            diag = fill(-3.0f0, stop + 3)
            for k = start:stop
                d = mod(BigInt(input[k, k]), p)
                diag[k] = iszero(d) ? 0 : invmod(d, p)
            end
            scratch = cached ? upload(diag, S) : nothing
            @test FFM.pluq_trsm_left_lower_unit_gpu!(
                A,
                p,
                start,
                stop,
                n;
                options = PANEL_OPTIONS,
            ) === A
            @test FFM.pluq_trsm_right_upper_gpu!(
                A,
                p,
                start,
                stop,
                m;
                options = PANEL_OPTIONS,
                dinv = scratch,
            ) === A
            @test Array(A) == expected
            if cached
                @test Array(scratch) == diag
            end
        end
    end

    host = padded(ones(Float32, 5, 5))
    A = upload(host, S)
    @test FFM.pluq_trsm_left_lower_unit_gpu!(A, 11, 1, 0, 5) === A
    @test FFM.pluq_trsm_right_upper_gpu!(A, 11, 1, 5, 5) === A
    @test_throws ArgumentError FFM.pluq_trsm_right_upper_gpu!(
        A,
        11,
        1,
        3,
        5;
        dinv = view(A, :, 1),
    )
    @test_throws DimensionMismatch FFM.pluq_trsm_right_upper_gpu!(
        A,
        11,
        2,
        4,
        5;
        dinv = upload(Float32[1, 2, 3], S),
    )
    @test_throws DimensionMismatch FFM.pluq_trsm_right_upper_gpu!(
        A,
        11,
        1,
        3,
        size(A, 1) + 1,
    )
    @test_throws ArgumentError FFM.pluq_trsm_left_lower_unit_gpu!(A, 11, 0, 3, 5)
    @test_throws DomainError FFM.pluq_trsm_left_lower_unit_gpu!(A, 1, 1, 3, 5)
    @test Array(A) == host
end

function check_blocked(input, p, S; options = PANEL_OPTIONS)
    m, n = size(input)
    _, _, _, expected_rank = reference(input, p)
    original = padded(input)
    A = upload(original, S)
    rows, cols, rank = if m == n
        FFM.pluq_blocked_gpu!(A, p, options, n)
    else
        FFM.pluq_rectangular_rank_gpu!(A, p, m, n; options)
    end
    @test sort(rows) == collect(1:m)
    @test sort(cols) == collect(1:n)
    @test rank == expected_rank
    out = Array(A)
    packed = BigInt.(out[1:m, 1:n])
    L = zeros(BigInt, m, rank)
    U = zeros(BigInt, rank, n)
    for k = 1:rank
        L[k, k] = 1
        L[(k+1):m, k] = packed[(k+1):m, k]
        U[k, k:n] = packed[k, k:n]
    end
    @test mod.(L * U, p) == mod.(BigInt.(input[rows, cols]), p)
    check_padding(out, original, m, n)
    return rows, cols, rank
end

function test_complete(S)
    p = LARGE_PRIME
    input = BigInt[0 0 -3 2; 0 0 4 5; p-4 1 p-1 p-2; p-3 2 p-7 p-9; 0 0 0 0]
    expected, rows, cols, rank = reference(input, p)
    original = padded(input)
    A = upload(original, S)
    @test FFM.pluq_rectangular_rank_reference_gpu!(A, p, 5, 4; options = PANEL_OPTIONS) ==
          (rows, cols, rank)
    out = Array(A)
    @test out[1:5, 1:4] == expected
    check_padding(out, original, 5, 4)

    check_blocked(zeros(Int, 0, 5), 11, S)
    check_blocked(zeros(Int, 7, 0), 11, S)
    check_blocked(zeros(Int, 0, 0), 11, S)
    check_blocked(reshape([6], 1, 1), 11, S)
    check_blocked(zeros(Int, 7, 9), 11, S)
    rows, _, rank = check_blocked([0 0 1 0; 0 1 0 0; 1 0 0 1], 11, S)
    @test rows[1] == 3
    @test rank == 3

    host = padded(ones(Float32, 3, 3))
    A = upload(host, S)
    @test_throws DimensionMismatch FFM.pluq_blocked_gpu!(A, 11, PANEL_OPTIONS, 100)
    @test_throws OverflowError FFM.pluq_blocked_gpu!(
        A,
        Int(typemax(Int32)) + 1,
        PANEL_OPTIONS,
        3,
    )
    @test Array(A) == host

    # A wrapped view must operate on its own axes and leave the rest of the
    # backing array untouched. Also exercise Metal's supported Float16 type.
    input16 = Float16[0 2 7 4; 4 1 5 6; 0 3 9 2]
    expected16, rows16, cols16, rank16 = reference(input16, 31)
    host16 = fill(Float16(-31), 8, 11)
    I, J = 3:5, 4:7
    host16[I, J] .= input16
    parent16 = upload(host16, S)
    view16 = view(parent16, I, J)
    @test FFM.pluq_rectangular_rank_reference_gpu!(view16, 31, 3, 4) ==
          (rows16, cols16, rank16)
    host16[I, J] .= expected16
    @test Array(parent16) == host16
end

function test_blocked(S, large_blocked)
    rng = MersenneTwister(92817)
    for (m, n) in ((13, 13), (15, 23), (21, 14))
        p = 101
        regular = rand(rng, 0:(p-1), m, n)
        check_blocked(regular, p, S)

        # The first two pivots succeed, but their local Schur complement
        # contains no pivot. Correct recovery must update all three outer
        # pieces while preserving the already reduced inner piece.
        L = rand(rng, 1:(p-1), m, 2)
        U = rand(rng, 1:(p-1), 2, n)
        L[1:2, :] .= [1 0; 0 1]
        U[:, 1:2] .= [3 2; 0 5]
        late = mod.(L * U, p)
        late[5, 3] += 7
        late[3, 6] += 11
        late[9, 9] += 13
        rows, _, rank = check_blocked(late, p, S)
        @test rows[1:3] == [1, 2, 5]
        @test rank == 5

        # A late nonzero at a panel boundary also needs complete pivots
        # when the failed panel has not accepted any rows.
        very_late = zeros(Int, m, n)
        very_late[m, 1] = 5
        very_late[1, n] = 7
        rows, cols, rank = check_blocked(very_late, p, S)
        @test (rows[1], cols[2], rank) == (m, n, 2)
    end

    if large_blocked
        large = rand(rng, 0:(LARGE_PRIME-1), 7, 11)
        check_blocked(large, LARGE_PRIME, S)
    end
end

function run_tests(; blocked = true, large_blocked = true)
    Metal.allowscalar(false)
    @testset "Metal ordinary PLUQ panels and TRSM" begin
        for storage in (Metal.SharedStorage, Metal.PrivateStorage)
            @testset "$storage" begin
                @testset "Pivot reduction" test_pivot(storage)
                @testset "Fused panels" test_fused_panel(storage)
                @testset "Packed TRSM" test_trsm(storage)
                @testset "Direct complete pivots and boundary cases" test_complete(storage)
                if blocked
                    @testset "Blocked Schur integration" test_blocked(
                        storage,
                        large_blocked,
                    )
                end
            end
        end
    end
end

end
