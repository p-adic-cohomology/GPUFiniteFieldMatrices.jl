import KernelAbstractions: @atomic, @localmem, @synchronize

const _PLUQ_KA_GROUP_SIZE = 128
const _PLUQ_KA_NO_CANDIDATE = typemax(Int32)

@inline _pluq_ka_integer(value::AbstractFloat) = unsafe_trunc(Int64, value)
@inline _pluq_ka_integer(value::Signed) = Int64(value)

@inline function _pluq_ka_residue(value, N)
    x = _pluq_ka_integer(value)
    return 0 <= x < N ? x : mod(x, N)
end

@inline function _pluq_ka_residue(value::Unsigned, N)
    x = UInt64(value)
    p = UInt64(N)
    return Int64(x < p ? x : rem(x, p))
end

@inline function _pluq_ka_inverse(a, N)
    r, next_r = N, _pluq_ka_residue(a, N)
    t, next_t = Int64(0), Int64(1)
    while !iszero(next_r)
        q = r ÷ next_r
        r, next_r = next_r, r - q * next_r
        t, next_t = next_t, t - q * next_t
    end
    return r == 1 ? mod(t, N) : Int64(0)
end

@inline function _pluq_ka_submul(acc, a, b, N)
    return mod(
        _pluq_ka_residue(acc, N) - _pluq_ka_residue(a, N) * _pluq_ka_residue(b, N),
        N,
    )
end

@inline function _pluq_ka_candidate(A, row_start, col_start, rows, area, lane, stride, N)
    pos = lane
    while pos <= area
        col, row = divrem(pos - 1, rows)
        if !iszero(_pluq_ka_residue(@inbounds(A[row_start+row, col_start+col]), N))
            return Base.unsafe_trunc(Int32, pos)
        end
        pos += stride
    end
    return _PLUQ_KA_NO_CANDIDATE
end

# A full one-dimensional workgroup is launched for every reduction below.
# Empty lanes initialize their entries to the same neutral value.
@kernel unsafe_indices = true inbounds = true function _pluq_ka_panel_kernel!(
    A,
    p,
    q,
    dinv,
    rank,
    k0,
    kend,
    m,
    n,
    N,
    ::Val{Partial},
) where {Partial}
    lane = @index(Local, Linear)
    candidates = @localmem Int32 (_PLUQ_KA_GROUP_SIZE,)
    inverse = @localmem Int64 (1,)

    if lane == 1
        rank[1] = Int32(0)
    end
    for k = k0:kend
        span = kend - k + 1
        # Restrict partial row pivoting to the active diagonal panel. A row
        # beyond kend has yet to receive the panel updates.
        area = Partial ? span : span * span
        candidates[lane] =
            _pluq_ka_candidate(A, k, k, span, area, lane, _PLUQ_KA_GROUP_SIZE, N)
        @synchronize()
        for stride in (64, 32, 16, 8, 4, 2, 1)
            if lane <= stride
                candidates[lane] = min(candidates[lane], candidates[lane+stride])
            end
            @synchronize()
        end
        pos = candidates[1]
        # Every lane reads the same reduced position and exits together.
        if pos == _PLUQ_KA_NO_CANDIDATE
            break
        end
        col_offset, row_offset = divrem(Int(pos) - 1, span)
        prow, pcol = k + row_offset, k + col_offset

        if prow != k
            for j = lane:_PLUQ_KA_GROUP_SIZE:n
                A[k, j], A[prow, j] = A[prow, j], A[k, j]
            end
            if lane == 1
                p[k], p[prow] = p[prow], p[k]
            end
        end
        @synchronize()
        if pcol != k
            for i = lane:_PLUQ_KA_GROUP_SIZE:m
                A[i, k], A[i, pcol] = A[i, pcol], A[i, k]
            end
            if lane == 1
                q[k], q[pcol] = q[pcol], q[k]
            end
        end
        @synchronize()

        if lane == 1
            inverse[1] = _pluq_ka_inverse(A[k, k], N)
            dinv[k] = eltype(A)(inverse[1])
            rank[1] = Int32(k - k0 + 1)
        end
        @synchronize()
        for i = (k+lane):_PLUQ_KA_GROUP_SIZE:kend
            A[i, k] = eltype(A)(rem(_pluq_ka_residue(A[i, k], N) * inverse[1], N))
        end
        @synchronize()

        width = kend - k
        for pos = lane:_PLUQ_KA_GROUP_SIZE:(width*width)
            col_offset, row_offset = divrem(pos - 1, width)
            i, j = k + 1 + row_offset, k + 1 + col_offset
            A[i, j] = eltype(A)(_pluq_ka_submul(A[i, j], A[i, k], A[k, j], N))
        end
        @synchronize()
    end
end

function _pluq_ka_panel!(A, N, p, q, dinv, rank, host_rank, k0, kend, m, n, mode::Val)
    depth = kend - k0 + 1
    depth > isqrt(Int(_PLUQ_KA_NO_CANDIDATE) - 1) &&
        throw(OverflowError("PLUQ panel is too large to encode a pivot position"))
    backend = KernelAbstractions.get_backend(A)
    _pluq_ka_panel_kernel!(backend, _PLUQ_KA_GROUP_SIZE)(
        A,
        p,
        q,
        dinv,
        rank,
        k0,
        kend,
        m,
        n,
        Int64(N),
        mode;
        ndrange = _PLUQ_KA_GROUP_SIZE,
    )
    copyto!(host_rank, rank)
    return Int(host_rank[1])
end

function _pluq_ka_check_matrix(A::AbstractMatrix, N::Int, m::Int, n::Int)
    Base.require_one_based_indexing(A)
    0 <= m <= size(A, 1) && 0 <= n <= size(A, 2) ||
        throw(DimensionMismatch("PLUQ logical dimensions must fit the device matrix"))
    max(m, n) < typemax(Int32) || throw(OverflowError("PLUQ dimensions exceed Int32"))
    N >= 2 || throw(DomainError(N, "PLUQ modulus must be at least 2"))
    N <= typemax(Int32) || throw(OverflowError("PLUQ modulus exceeds the Int32 range"))
    _validate_storage_modulus(eltype(A), N)
    _check_elementwise_backend(A, A)
    return nothing
end

function _pluq_ka_check_scratch(A, scratch, ::Type{T}, n) where {T}
    scratch isa AbstractVector{T} ||
        throw(ArgumentError("PLUQ scratch must be a vector with element type $T"))
    Base.require_one_based_indexing(scratch)
    length(scratch) >= n || throw(DimensionMismatch("PLUQ scratch is too short"))
    _check_elementwise_backend(A, scratch)
    Base.mightalias(A, scratch) && throw(ArgumentError("PLUQ scratch aliases its matrix"))
    return nothing
end
