module MetalSubgroupPivotTests

using Test
import GPUFiniteFieldMatrices
import KernelInterface
import Metal

const FFM = GPUFiniteFieldMatrices

metal_array(::Type{S}, A::Array{T,D}) where {S,T,D} = Metal.MtlArray{T,D,S}(A)

function host_pivot(A, k, lastrow, lastcol, modulus_value; row_index = false)
    panel = vec(@view A[k:lastrow, k:lastcol])
    modulus_big = BigInt(modulus_value)
    first = findfirst(value -> !iszero(mod(BigInt(value), modulus_big)), panel)
    offset = row_index ? k - 1 : 0
    return Int32(offset + something(first, length(panel) + 1))
end

function check_pivot(S, mode, A, k, lastrow, lastcol, modulus_value; row_index = false)
    input = metal_array(S, A)
    initial = Int32[-19, 73, 81]
    slot = metal_array(S, initial)
    if mode === :workgroup
        # Exercise the missing-capability kernel directly on this device.
        # The host selector must still use the requested native subgroup path.
        nrows = lastrow - k + 1
        total = nrows * (lastcol - k + 1)
        threads = clamp(total, 1, FFM._PLUQ_PIVOT_FALLBACK_THREADS)
        offset = row_index ? k - 1 : 0
        backend = KernelInterface.get_backend(input)
        KernelInterface.@launch backend numgroups=1 workgroupsize=threads FFM._pluq_pivot_workgroup_kernel!(
            slot,
            input,
            k,
            nrows,
            total,
            Int64(modulus_value),
            offset,
        )
    else
        @test FFM._pluq_subgroup_pivot!(
            slot,
            input,
            k,
            lastrow,
            lastcol,
            modulus_value,
            mode;
            row_index,
        ) === slot
    end
    actual = Array(slot)
    @test actual[1] == host_pivot(A, k, lastrow, lastcol, modulus_value; row_index)
    @test actual[2:end] == initial[2:end]
    @test Array(input) == A
    return actual[1]
end

function check_rank_deficient_panel(S, mode)
    m, n = 7, 11
    prime = big(17)
    A = zeros(BigInt, m, n)
    A[5, 3] = 3
    A[2, 7] = 2
    A[6, :] = 2 .* (A[5, :] + A[2, :])
    p, q = collect(1:m), collect(1:n)
    rank = 0
    # Supply the active panels of exact reference elimination in order.
    for k = 1:m
        first = check_pivot(S, mode, Float32.(A), k, m, n, prime)
        nrows = m - k + 1
        if first > nrows * (n - k + 1)
            break
        end
        col, row = divrem(first - 1, nrows) .+ k
        A[[k, row], :] = A[[row, k], :]
        A[:, [k, col]] = A[:, [col, k]]
        p[k], p[row] = p[row], p[k]
        q[k], q[col] = q[col], q[k]
        invpivot = invmod(A[k, k], prime)
        for i = (k+1):m
            factor = mod(A[i, k] * invpivot, prime)
            for j = (k+1):n
                A[i, j] = mod(A[i, j] - factor * A[k, j], prime)
            end
            A[i, k] = 0
        end
        rank += 1
    end
    @test rank == 2
    @test p == [5, 2, 3, 4, 1, 6, 7]
    @test q == [3, 7, 1, 4, 5, 6, 2, 8, 9, 10, 11]
end

@testset "Standalone Metal PLUQ pivots" begin
    @test Metal.functional()
    Metal.allowscalar(false)
    backend = Metal.MetalBackend()
    @test KernelInterface.supports_subgroups(backend)
    @test KernelInterface.supports_linear_subgroups(backend)
    @test KernelInterface.supports_shuffle(backend, Int32)
    width = KernelInterface.sub_group_size(backend)
    @test 1 < width <= 64
    @test FFM._pluq_pivot_subgroup_width(backend, :ballot) == width
    @test FFM._pluq_pivot_subgroup_width(backend, :shfl) == width

    for S in (Metal.SharedStorage, Metal.PrivateStorage),
        mode in (:ballot, :shfl, :workgroup)

        @testset "$S / $mode" begin
            @testset "No pivot, singleton and empty panels" begin
                check_pivot(S, mode, fill(-17.0f0, width + 1, 3), 1, width + 1, 3, 17)
                check_pivot(S, mode, reshape(Float32[-1], 1, 1), 1, 1, 1, 17)
                check_pivot(S, mode, zeros(Float32, 0, 3), 1, 0, 3, 17)
                check_pivot(S, mode, ones(Float32, 4, 4), 5, 4, 4, 17)
            end
            @testset "Float16 exact residues" begin
                A = reshape(Float16[-2034, 2034, -2033, 0, 113], 5, 1)
                check_pivot(S, mode, A, 1, 5, 1, 113)
            end
            @testset "Last lane and ragged work" begin
                for nrows in (width, max(1, width - 3), 2 * width + 3)
                    A = zeros(Float32, nrows, 1)
                    A[end, 1] = 7
                    check_pivot(S, mode, A, 1, nrows, 1, 17)
                end
                # Last physical lane of the full group and the last lane
                # of a partial group both receive candidates above.
            end
            @testset "Column-major ties with logical bounds" begin
                A = zeros(Float32, width + 2, 9)
                A[3, 4] = -2
                A[width+1, 4] = 99
                A[2, 5] = 1
                A[2, 2] = -514
                A[3, 2] = Float32(257 * 65_280)
                A[1, 2] = 1
                A[width+2, 2] = 1
                A[2, 9] = 1
                check_pivot(S, mode, A, 2, width + 1, 8, 257)
            end
            @testset "Late augmented row and missing row" begin
                k = width + 5
                A = zeros(Float32, k + 6, 2 * (k + 6))
                A[k+2, k] = -3
                A[k+6, k] = 4
                A[1, k] = 1
                A[k, k+1] = 1
                check_pivot(S, mode, A, k, k + 6, k, 17; row_index = true)
                A[k+2, k] = -17
                A[k+6, k] = 34
                check_pivot(S, mode, A, k, k + 6, k, 17; row_index = true)
                check_pivot(S, mode, A, k, k - 1, k, 17; row_index = true)
            end
            @testset "Rank-deficient rectangular sequence" begin
                check_rank_deficient_panel(S, mode)
            end
        end
    end

    @testset "Strided panels and slot padding" begin
        S = Metal.PrivateStorage
        A = fill(1.0f0, 12, 13)
        logical = @view A[2:2:12, 3:2:13]
        fill!(logical, 0)
        logical[6, 5] = 7
        backing = metal_array(S, A)
        panel = @view backing[2:2:12, 3:2:13]
        for mode in (:ballot, :shfl)
            slot_backing = metal_array(S, fill(Int32(-19), 5))
            slot = @view slot_backing[2:2:4]
            @test FFM._pluq_subgroup_pivot!(slot, panel, 2, 6, 6, 17, mode) === slot
            expected = host_pivot(logical, 2, 6, 6, 17)
            @test Array(slot_backing) == Int32[-19, expected, -19, -19, -19]
            @test Array(backing) == A
        end
    end

    @testset "Reject invalid arguments before writing" begin
        S = Metal.PrivateStorage
        A = fill(1.0f0, 3, 4)
        input = metal_array(S, A)
        initial = Int32[-19, 73]
        slot = metal_array(S, initial)
        alias_slot = reinterpret(Int32, vec(input))
        @test Base.mightalias(alias_slot, input)
        @test_throws ArgumentError FFM._pluq_subgroup_pivot!(
            alias_slot,
            input,
            1,
            3,
            4,
            17,
            :ballot,
        )
        @test_throws BoundsError FFM._pluq_subgroup_pivot!(
            slot,
            input,
            0,
            3,
            4,
            17,
            :ballot,
        )
        @test_throws BoundsError FFM._pluq_subgroup_pivot!(
            slot,
            input,
            1,
            3,
            5,
            17,
            :ballot,
        )
        @test_throws BoundsError FFM._pluq_subgroup_pivot!(slot, input, 5, 3, 4, 17, :shfl)
        @test_throws ArgumentError FFM._pluq_subgroup_pivot!(slot, input, 1, 3, 4, 0, :shfl)
        @test_throws ArgumentError FFM._pluq_subgroup_pivot!(
            slot,
            input,
            1,
            3,
            4,
            17,
            :other,
        )
        @test_throws ArgumentError FFM._pluq_subgroup_pivot!(
            slot,
            input,
            1,
            3,
            4,
            17,
            :shfl;
            row_index = true,
        )
        @test_throws DimensionMismatch FFM._pluq_subgroup_pivot!(
            metal_array(S, Int32[]),
            input,
            1,
            3,
            4,
            17,
            :ballot,
        )
        host_slot = copy(initial)
        @test_throws ArgumentError FFM._pluq_subgroup_pivot!(
            host_slot,
            A,
            1,
            3,
            4,
            17,
            :ballot,
        )
        @test_throws ArgumentError FFM._pluq_subgroup_pivot!(
            host_slot,
            input,
            1,
            3,
            4,
            17,
            :ballot,
        )
        @test host_slot == initial
        @test Array(slot) == initial
        @test Array(input) == A
    end
end

end
