# Standalone KernelInterface kernels. Keep this include outside @stable and
# launch exactly one 1-D workgroup with each kernel.

@inline function _pluq_pivot_is_nonzero(A, k, nrows, index, modulus_value)
    col, row = divrem(index - 1, nrows)
    value = @inbounds A[k+row, k+col]
    return !iszero(rem(Int64(value), modulus_value))
end

@inline function _pluq_pivot_lane_min(A, k, nrows, total, modulus_value, lane, width)
    index = lane
    while index <= total
        if _pluq_pivot_is_nonzero(A, k, nrows, index, modulus_value)
            return index % Int32
        end
        index += width
    end
    return (total + 1) % Int32
end

function _pluq_pivot_ballot_kernel!(slot, A, k, nrows, total, modulus_value, offset)
    lane = KernelInterface.get_sub_group_local_id()
    width = KernelInterface.get_sub_group_size()
    first = total + 1
    base = 0
    while base < total
        index = base + lane
        candidate =
            index <= total && _pluq_pivot_is_nonzero(A, k, nrows, index, modulus_value)
        bits = KernelInterface.sub_group_ballot(candidate)
        # The ballot is uniform, so every lane stops at the same chunk.
        if !iszero(bits)
            first = base + trailing_zeros(bits) + 1
            break
        end
        base += width
    end
    if lane == 1
        @inbounds slot[1] = (first + offset) % Int32
    end
    return
end

function _pluq_pivot_shuffle_kernel!(slot, A, k, nrows, total, modulus_value, offset)
    lane = KernelInterface.get_sub_group_local_id()
    width = KernelInterface.get_sub_group_size()
    first = _pluq_pivot_lane_min(A, k, nrows, total, modulus_value, lane, width)
    if KernelInterface.sub_group_any(first <= total)
        # Reduce to lane 1. A shuffle from an absent lane in a partial
        # subgroup has an unspecified result, so discard it.
        delta = 1
        while delta < width
            other = KernelInterface.shfl_down(first, delta)
            if lane + delta <= width
                first = min(first, other)
            end
            delta *= 2
        end
    end
    if lane == 1
        @inbounds slot[1] = (first + offset) % Int32
    end
    return
end

const _PLUQ_PIVOT_FALLBACK_THREADS = 64

function _pluq_pivot_workgroup_kernel!(slot, A, k, nrows, total, modulus_value, offset)
    lane = KernelInterface.get_local_id().x
    width = KernelInterface.get_local_size().x
    candidates = KernelInterface.localmemory(Int32, Val((_PLUQ_PIVOT_FALLBACK_THREADS,)))
    @inbounds candidates[lane] =
        _pluq_pivot_lane_min(A, k, nrows, total, modulus_value, lane, width)
    KernelInterface.barrier()
    if lane == 1
        first = @inbounds candidates[1]
        for i = 2:width
            first = min(first, @inbounds candidates[i])
        end
        @inbounds slot[1] = (first + offset) % Int32
    end
    return
end

function _pluq_pivot_subgroup_width(backend, mode::Symbol)
    KernelInterface.supports_linear_subgroups(backend) || return 0
    width = KernelInterface.sub_group_size(backend)
    if mode === :ballot
        return width <= 64 ? width : 0
    end
    return KernelInterface.supports_shuffle(backend, Int32) ? width : 0
end

"""
    _pluq_subgroup_pivot!(slot, A, k, lastrow, lastcol, modulus, mode; row_index=false)

Queue a first-nonzero search over `A[k:lastrow, k:lastcol]` on the active device.
Write the one-based column-major position relative to that panel into `slot[1]`,
or the panel length plus one when no value is nonzero modulo `modulus`.

For an augmented single-column search, pass `lastcol = k` and
`row_index = true` to write the absolute row or `lastrow + 1`. The explicit
bounds exclude padding and augmented columns. The input must contain exact
integer values representable as `Int64`. The caller can reuse the returned
slot after consuming its result in queue order.
"""
function _pluq_subgroup_pivot!(
    slot::AbstractVector{Int32},
    A::AbstractMatrix{<:AbstractFloat},
    k::Integer,
    lastrow::Integer,
    lastcol::Integer,
    modulus_value::Integer,
    mode::Symbol;
    row_index::Bool = false,
)
    Base.require_one_based_indexing(slot, A)
    isempty(slot) &&
        throw(DimensionMismatch("PLUQ pivot slot must hold at least one Int32"))
    0 <= lastrow <= size(A, 1) && 0 <= lastcol <= size(A, 2) ||
        throw(BoundsError(A, (lastrow, lastcol)))
    1 <= k && k - 1 <= min(lastrow, lastcol) || throw(BoundsError(A, (k, k)))
    (!row_index || lastcol == k) ||
        throw(ArgumentError("an absolute pivot row requires a single column"))
    0 < modulus_value <= typemax(Int32) ||
        throw(ArgumentError("PLUQ pivot modulus must fit in a positive Int32"))
    mode in (:ballot, :shfl) ||
        throw(ArgumentError("PLUQ pivot mode must be :ballot or :shfl"))
    Base.mightalias(slot, A) &&
        throw(ArgumentError("PLUQ pivot slot must not alias the input"))

    start = Int(k)
    nrows = Int(lastrow) - start + 1
    ncols = Int(lastcol) - start + 1
    offset = row_index ? start - 1 : 0
    capacity = Int(typemax(Int32)) - offset - 1
    capacity >= 0 && (ncols == 0 || nrows <= capacity ÷ ncols) ||
        throw(ArgumentError("PLUQ pivot index and sentinel must fit in Int32"))
    total = nrows * ncols

    backend = KernelInterface.get_backend(A)
    backend isa KernelAbstractions.CPU &&
        throw(ArgumentError("PLUQ subgroup pivot search requires a GPU"))
    typeof(backend) === typeof(KernelInterface.get_backend(slot)) &&
    KernelInterface.device(backend, A) ==
    KernelInterface.device(backend, slot) ==
    KernelInterface.device(backend) ||
        throw(ArgumentError("PLUQ pivot input and slot must be on the same active device"))

    width = _pluq_pivot_subgroup_width(backend, mode)
    limit = iszero(width) ? _PLUQ_PIVOT_FALLBACK_THREADS : width
    threads = min(
        limit,
        max(1, total),
        KernelInterface.max_work_group_size(backend),
        KernelInterface.max_work_group_dims(backend)[1],
    )
    modulus_i = Int64(modulus_value)
    # Each KI launch specializes on its concrete kernel function.
    if iszero(width)
        KernelInterface.@launch backend numgroups=1 workgroupsize=threads _pluq_pivot_workgroup_kernel!(
            slot,
            A,
            start,
            nrows,
            total,
            modulus_i,
            offset,
        )
    elseif mode === :ballot
        KernelInterface.@launch backend numgroups=1 workgroupsize=threads _pluq_pivot_ballot_kernel!(
            slot,
            A,
            start,
            nrows,
            total,
            modulus_i,
            offset,
        )
    else
        KernelInterface.@launch backend numgroups=1 workgroupsize=threads _pluq_pivot_shuffle_kernel!(
            slot,
            A,
            start,
            nrows,
            total,
            modulus_i,
            offset,
        )
    end
    return slot
end
