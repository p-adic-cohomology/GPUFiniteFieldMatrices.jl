@inline function _ki_trsm_mul(a::UInt32, b::UInt32, N::UInt32)
    # Canonical Float32 residues are below 2^24, so the product fits in 48 bits.
    return rem(UInt64(a) * UInt64(b), UInt64(N)) % UInt32
end

@inline function _ki_trsm_inv(a::UInt32, N::UInt32)
    iszero(a) && return zero(UInt32)
    t, next_t = Int64(0), Int64(1)
    r, next_r = Int64(N), Int64(a)
    while !iszero(next_r)
        q = div(r, next_r)
        t, next_t = next_t, t - q * next_t
        r, next_r = next_r, r - q * next_r
    end
    r == 1 || return zero(UInt32)
    return (t < 0 ? t + Int64(N) : t) % UInt32
end

@kernel function _ki_trsm_inverse_diagonal_kernel!(inverses, A, panel_first, N)
    idx = @index(Global, Linear)
    if idx <= length(inverses)
        pivot = panel_first + idx - 1
        @inbounds inverses[idx] = _ki_trsm_inv(unsafe_trunc(UInt32, A[pivot, pivot]), N)
    end
end

function _ki_trsm_panel_kernel!(
    A,
    inverses,
    panel_first,
    panel_last,
    rhs_last,
    N,
    ::Val{Left},
    ::Val{Forward},
) where {Left,Forward}
    lane = KernelInterface.get_sub_group_local_id()
    width = KernelInterface.get_sub_group_size()
    group = KernelInterface.get_group_id().x
    subgroup = KernelInterface.get_sub_group_id()
    subgroups = KernelInterface.get_num_sub_groups()
    rhs = panel_last + (group - 1) * subgroups + subgroup
    active = rhs <= rhs_last
    start = Forward ? panel_first : panel_last
    step = Forward ? 1 : -1

    # Every subgroup in the workgroup executes one reduction and fence per pivot.
    # Inactive RHS subgroups and lanes with no terms contribute zero.
    for pivot = start:step:(Forward ? panel_last : panel_first)
        partial = zero(UInt32)
        if active
            term = start + (lane - 1) * step
            while Forward ? term < pivot : term > pivot
                @inbounds coeff = Left ? A[pivot, term] : A[term, pivot]
                @inbounds solved = Left ? A[term, rhs] : A[rhs, term]
                prod = _ki_trsm_mul(
                    unsafe_trunc(UInt32, coeff),
                    unsafe_trunc(UInt32, solved),
                    N,
                )
                total = partial + prod
                partial = ifelse(total >= N, total - N, total)
                term += step * width
            end
        end
        dot = rem(KernelInterface.sub_group_reduce(+, partial), N)
        if active && lane == 1
            @inbounds entry = Left ? A[pivot, rhs] : A[rhs, pivot]
            b = unsafe_trunc(UInt32, entry)
            value = ifelse(b >= dot, b - dot, N - (dot - b))
            if inverses !== nothing
                @inbounds value = _ki_trsm_mul(value, inverses[pivot-panel_first+1], N)
            end
            if Left
                @inbounds A[pivot, rhs] = Float32(value)
            else
                @inbounds A[rhs, pivot] = Float32(value)
            end
        end
        # The next pivot reads previously solved values from device memory.
        KernelInterface.sub_group_barrier()
    end
    return nothing
end

function _ki_trsm_supports_subgroups(backend)
    return KernelInterface.supports_subgroups(backend) &&
           KernelInterface.supports_linear_subgroups(backend) &&
           KernelInterface.supports_shuffle(backend, UInt32)
end

function _ki_trsm_workgroupsize(backend, N)
    if !_ki_trsm_supports_subgroups(backend)
        throw(
            ArgumentError(
                "subgroup TRSM requires linear subgroups with UInt32 shuffle support",
            ),
        )
    end
    width = KernelInterface.sub_group_size(backend)
    max_threads = KernelInterface.max_work_group_size(backend)
    if !(0 < width <= max_threads) || UInt64(width) * UInt64(N - 1) > typemax(UInt32)
        throw(ArgumentError("unsupported subgroup width $width for TRSM at modulus $N"))
    end
    subgroups = min(2, max_threads ÷ width)
    return subgroups * width, subgroups
end

"""
    _ki_trsm_panel!(A, N, panel_first, panel_last, rhs_last; side, uplo, unit_diagonal)

Solve a triangular system in packed native Float32 storage on the caller's active
device. The diagonal block is `A[panel, panel]`. For `side=:left`, solve into
`A[panel, panel_last+1:rhs_last]`; for `side=:right`, solve into
`A[panel_last+1:rhs_last, panel]`. The input contains canonical residues. Only
the indicated RHS block is overwritten, and the function returns `A`.

`uplo` selects the lower or upper triangle. A unit diagonal is implicit and
unread. Otherwise a noninvertible diagonal throws `SingularException` before
any writes to `A`. Empty panels and empty RHS blocks are no-ops.
"""
function _ki_trsm_panel!(
    A::AbstractMatrix{Float32},
    N::Int,
    panel_first::Int,
    panel_last::Int,
    rhs_last::Int;
    side::Symbol,
    uplo::Symbol,
    unit_diagonal::Bool,
)
    side in (:left, :right) || throw(ArgumentError("TRSM side must be :left or :right"))
    uplo in (:lower, :upper) || throw(ArgumentError("TRSM uplo must be :lower or :upper"))
    Base.require_one_based_indexing(A)
    left = side === :left
    rhs_extent = size(A, left ? 2 : 1)
    if panel_first < 1 ||
       !(panel_first - 1 <= panel_last <= min(size(A)...)) ||
       !(panel_last <= rhs_last <= rhs_extent)
        throw(DimensionMismatch("TRSM panel and RHS must fit the logical input dimensions"))
    end
    _validate_storage_modulus(Float32, N)
    _storage_backend(A) === Val(:metal) ||
        throw(ArgumentError("subgroup TRSM requires native Metal storage"))
    (panel_first > panel_last || panel_last == rhs_last) && return A

    backend = KernelInterface.get_backend(A)
    workgroupsize, subgroups = _ki_trsm_workgroupsize(backend, N)
    modulus_u = UInt32(N)
    inverses = if unit_diagonal
        nothing
    else
        depth = panel_last - panel_first + 1
        values = similar(A, UInt32, (depth,))
        _ki_trsm_inverse_diagonal_kernel!(backend)(
            values,
            A,
            panel_first,
            modulus_u;
            ndrange = depth,
        )
        # Transfer only the reduction result to the host before touching the RHS.
        any(iszero, values) && throw(LinearAlgebra.SingularException(0))
        values
    end
    forward = left == (uplo === :lower)
    groups = cld(rhs_last - panel_last, subgroups)
    KernelInterface.@launch backend numgroups=groups workgroupsize=workgroupsize _ki_trsm_panel_kernel!(
        A,
        inverses,
        panel_first,
        panel_last,
        rhs_last,
        modulus_u,
        Val(left),
        Val(forward),
    )
    return A
end
