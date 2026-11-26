using LinearAlgebra: SingularException
using Random: MersenneTwister

function _metal_trsm_fixture(rng, dims, panel, N, unit_diagonal)
    data = rand(rng, 0:(N-1), dims)
    for i in panel
        if unit_diagonal
            data[i, i] = 0
        else
            value = rand(rng, 1:(N-1))
            while gcd(value, N) != 1
                value = rand(rng, 1:(N-1))
            end
            data[i, i] = value
        end
    end
    return Float32.(data)
end

function _metal_trsm_reference(data, panel, rhs_range, N, side, uplo, unit_diagonal)
    result = BigInt.(data)
    M = BigInt(N)
    left = side === :left
    order = left == (uplo === :lower) ? panel : reverse(panel)
    for rhs in rhs_range, (step, pivot) in enumerate(order)
        value = left ? result[pivot, rhs] : result[rhs, pivot]
        for term in Iterators.take(order, step - 1)
            value -=
                left ? result[pivot, term] * result[term, rhs] :
                result[rhs, term] * result[term, pivot]
        end
        if !unit_diagonal
            value *= invmod(result[pivot, pivot], M)
        end
        if left
            result[pivot, rhs] = mod(value, M)
        else
            result[rhs, pivot] = mod(value, M)
        end
    end
    return Float32.(result)
end

@testset "Metal KI subgroup TRSM" begin
    GFM = GPUFiniteFieldMatrices
    rng = MersenneTwister(4571)
    Metal.allowscalar(false)
    try
        for storage in (Metal.SharedStorage, Metal.PrivateStorage)
            @testset "$(nameof(storage))" begin
                cases = (
                    (1, 0, 1, 7),
                    (31, 2, 5, 257),
                    (32, 1, 3, 2^24),
                    (33, 0, 17, 2^24 - 3),
                    (65, 3, 7, 2^24),
                    (97, 1, 9, 257),
                )
                for side in (:left, :right),
                    uplo in (:lower, :upper),
                    unit_diagonal in (true, false)

                    for (depth, offset, nrhs, N) in cases
                        first, last = offset + 1, offset + depth
                        rhs_last = last + nrhs
                        rows, cols =
                            side === :left ? (last + 2, rhs_last + 2) :
                            (rhs_last + 2, last + 2)
                        panel = first:last
                        host =
                            _metal_trsm_fixture(rng, (rows, cols), panel, N, unit_diagonal)

                        # Use a noncontiguous native view with a nonzero offset. The
                        # full backing comparison detects writes to padding, the
                        # diagonal, and any logical region outside the selected RHS.
                        backing = fill(-73.0f0, rows + 3, cols + 5)
                        row_slice, col_slice = 2:(rows+1), 3:(cols+2)
                        backing[row_slice, col_slice] .= host
                        native = Metal.MtlArray{Float32,2,storage}(backing)
                        logical = view(native, row_slice, col_slice)
                        @test GFM._ki_trsm_panel!(
                            logical,
                            N,
                            first,
                            last,
                            rhs_last;
                            side,
                            uplo,
                            unit_diagonal,
                        ) === logical
                        expected = copy(backing)
                        expected[row_slice, col_slice] .= _metal_trsm_reference(
                            host,
                            panel,
                            (last+1):rhs_last,
                            N,
                            side,
                            uplo,
                            unit_diagonal,
                        )
                        @test Array(native) == expected
                    end
                end

                # A value late in the solve order must be diagnosed before any
                # earlier pivot can overwrite the RHS. Include nonunits modulo 8.
                for (side, uplo, bad_index, bad_value, N) in (
                    (:left, :lower, 4, 0.0f0, 7),
                    (:left, :upper, 2, 2.0f0, 8),
                    (:right, :upper, 4, 0.0f0, 7),
                    (:right, :lower, 2, 2.0f0, 8),
                )
                    host = _metal_trsm_fixture(rng, (9, 9), 2:4, N, false)
                    host[bad_index, bad_index] = bad_value
                    native = Metal.MtlArray{Float32,2,storage}(host)
                    @test_throws SingularException GFM._ki_trsm_panel!(
                        native,
                        N,
                        2,
                        4,
                        8;
                        side,
                        uplo,
                        unit_diagonal = false,
                    )
                    @test Array(native) == host
                end

                host = Float32[0 4 5; 6 0 3; 2 1 0]
                native = Metal.MtlArray{Float32,2,storage}(host)
                for side in (:left, :right)
                    @test GFM._ki_trsm_panel!(
                        native,
                        7,
                        1,
                        3,
                        3;
                        side,
                        uplo = :lower,
                        unit_diagonal = false,
                    ) === native
                    @test GFM._ki_trsm_panel!(
                        native,
                        7,
                        2,
                        1,
                        3;
                        side,
                        uplo = :upper,
                        unit_diagonal = false,
                    ) === native
                end
                @test_throws DimensionMismatch GFM._ki_trsm_panel!(
                    native,
                    7,
                    2,
                    2,
                    4;
                    side = :left,
                    uplo = :lower,
                    unit_diagonal = true,
                )
                @test_throws DomainError GFM._ki_trsm_panel!(
                    native,
                    2^24 + 1,
                    1,
                    1,
                    3;
                    side = :left,
                    uplo = :lower,
                    unit_diagonal = true,
                )
                @test Array(native) == host

                zero_ring = Metal.MtlArray{Float32,2,storage}(zeros(Float32, 2, 3))
                @test GFM._ki_trsm_panel!(
                    zero_ring,
                    1,
                    1,
                    2,
                    3;
                    side = :left,
                    uplo = :lower,
                    unit_diagonal = true,
                ) === zero_ring
                @test Array(zero_ring) == zeros(Float32, 2, 3)

                empty_native = Metal.MtlArray{Float32,2,storage}(zeros(Float32, 0, 0))
                @test GFM._ki_trsm_panel!(
                    empty_native,
                    7,
                    1,
                    0,
                    0;
                    side = :right,
                    uplo = :upper,
                    unit_diagonal = false,
                ) === empty_native

                # No explicit synchronization between the producer copy, the
                # solve, and the consumer copy on the native queue.
                work = similar(native)
                result = similar(native)
                copyto!(work, native)
                @test GFM._ki_trsm_panel!(
                    work,
                    7,
                    1,
                    2,
                    3;
                    side = :left,
                    uplo = :lower,
                    unit_diagonal = true,
                ) === work
                copyto!(result, work)
                @test Array(result) ==
                      _metal_trsm_reference(host, 1:2, 3:3, 7, :left, :lower, true)
            end
        end
    finally
        Metal.allowscalar(true)
    end

    host = ones(Float32, 3, 3)
    @test_throws ArgumentError GFM._ki_trsm_panel!(
        host,
        7,
        1,
        2,
        3;
        side = :left,
        uplo = :lower,
        unit_diagonal = true,
    )
    @test host == ones(Float32, 3, 3)
end
