import Atomix

# Reduce one 1-based column-major index per workgroup. Only group leaders
# update the common slot, and only when independent groups are launched.
@kernel unsafe_indices = true inbounds = true function _pluq_ka_pivot_kernel!(
    slot,
    A,
    k,
    first_col,
    rows,
    area,
    stride,
    N,
    ::Val{Mode},
) where {Mode}
    lane = @index(Local, Linear)
    group = @index(Group, Linear)
    candidates = @localmem Int32 (_PLUQ_KA_GROUP_SIZE,)
    first = (group - 1) * _PLUQ_KA_GROUP_SIZE + lane
    candidates[lane] = _pluq_ka_candidate(A, k, first_col, rows, area, first, stride, N)
    @synchronize()
    for width in (64, 32, 16, 8, 4, 2, 1)
        if lane <= width
            candidates[lane] = min(candidates[lane], candidates[lane+width])
        end
        @synchronize()
    end
    if lane == 1
        position = min(candidates[1], Int32(area + 1))
        if Mode === :atomic
            Atomix.@atomic :monotonic min(slot[1], position)
        else
            slot[group] = position
        end
    end
end

const _PLUQ_KA_PIVOT_GROUPS = 128

@kernel inbounds = true function _pluq_ka_merge_minima_kernel!(slot, minima, groups)
    candidate = minima[1]
    for group = 2:groups
        candidate = min(candidate, minima[group])
    end
    slot[1] = candidate
end

function _pluq_ka_launch_pivot!(slot, A, k, first_col, rows, area, N, minima = nothing)
    backend = KernelAbstractions.get_backend(A)
    groups = clamp(cld(area, _PLUQ_KA_GROUP_SIZE), 1, _PLUQ_KA_PIVOT_GROUPS)
    atomic = groups > 1 && KernelInterface.supports_atomics(backend)
    if atomic
        fill!(slot, Int32(area + 1))
    end
    if groups == 1 || atomic
        output = slot
    else
        output = minima === nothing ? similar(slot, Int32, groups) : minima
    end
    lanes = groups * _PLUQ_KA_GROUP_SIZE
    _pluq_ka_pivot_kernel!(backend, _PLUQ_KA_GROUP_SIZE)(
        output,
        A,
        k,
        first_col,
        rows,
        area,
        lanes,
        Int64(N),
        atomic ? Val(:atomic) : Val(:local);
        ndrange = lanes,
    )
    if groups > 1 && !atomic
        _pluq_ka_merge_minima_kernel!(backend, 1)(slot, output, groups; ndrange = 1)
    end
    return slot
end

# A single Int32 slot holds the first active position, or area + 1 if absent.
function _pluq_ka_find_pivot!(slot, A::AbstractMatrix, k::Int, m::Int, n::Int, N::Int)
    _pluq_ka_check_matrix(A, N, m, n)
    _pluq_ka_check_scratch(A, slot, Int32, 1)
    length(slot) == 1 || throw(DimensionMismatch("PLUQ pivot slot must have length one"))
    1 <= k <= min(m, n) + 1 || throw(ArgumentError("invalid PLUQ pivot start"))
    rows, cols = m - k + 1, n - k + 1
    if rows > 0 && cols > (Int(_PLUQ_KA_NO_CANDIDATE) - 1) ÷ rows
        throw(OverflowError("PLUQ pivot position exceeds the Int32 range"))
    end
    return _pluq_ka_launch_pivot!(slot, A, k, k, rows, rows * cols, N)
end

@kernel inbounds = true function _pluq_ka_identity_permutations_kernel!(p, q, m, n)
    i = @index(Global, Linear)
    if i <= m
        p[i] = Int32(i)
    end
    if i <= n
        q[i] = Int32(i)
    end
end

function _pluq_ka_device_permutations(A, m, n)
    native = _elementwise_native_root(A)
    p = similar(native, Int32, m)
    q = similar(native, Int32, n)
    backend = KernelAbstractions.get_backend(A)
    _pluq_ka_identity_permutations_kernel!(backend, _PLUQ_KA_GROUP_SIZE)(
        p,
        q,
        m,
        n;
        ndrange = max(m, n),
    )
    return p, q
end

@kernel inbounds = true function _pluq_ka_swap_kernel!(
    A,
    perm,
    k,
    pivot,
    ::Val{Rows},
) where {Rows}
    i = @index(Global, Linear)
    if Rows
        A[k, i], A[pivot, i] = A[pivot, i], A[k, i]
    else
        A[i, k], A[i, pivot] = A[i, pivot], A[i, k]
    end
    if i == 1
        perm[k], perm[pivot] = perm[pivot], perm[k]
    end
end

@kernel unsafe_indices = true inbounds = true function _pluq_ka_scale_kernel!(A, k, m, N)
    lane = @index(Local, Linear)
    group = @index(Group, Linear)
    invdiag = @localmem Int64 (1,)
    if lane == 1
        invdiag[1] = _pluq_ka_inverse(A[k, k], N)
    end
    @synchronize()
    row = k + (group - 1) * _PLUQ_KA_GROUP_SIZE + lane
    if row <= m
        A[row, k] = eltype(A)(rem(_pluq_ka_residue(A[row, k], N) * invdiag[1], N))
    end
end

@kernel inbounds = true function _pluq_ka_rank1_kernel!(A, k, N)
    row, col = @index(Global, NTuple)
    i, j = k + row, k + col
    A[i, j] = eltype(A)(_pluq_ka_submul(A[i, j], A[i, k], A[k, j], N))
end

# For a very large trailing rectangle, scan consecutive column windows so the
# atomic position still fits Int32. Windows and positions are in pivot order.
function _pluq_ka_complete_pivot(A, slot, minima, host, k, m, n, N, mode, subgroup_width)
    rows = m - k + 1
    max_cols = (Int(_PLUQ_KA_NO_CANDIDATE) - 1) ÷ rows
    for first_col = k:max_cols:n
        cols = min(max_cols, n - first_col + 1)
        area = rows * cols
        if rows <= subgroup_width && cols == n - k + 1
            _pluq_subgroup_pivot!(slot, A, k, m, n, N, mode)
        else
            _pluq_ka_launch_pivot!(slot, A, k, first_col, rows, area, N, minima)
        end
        copyto!(host, slot)
        position = Int(host[1])
        if position <= area
            col, row = divrem(position - 1, rows)
            return k + row, first_col + col
        end
    end
    return 0, 0
end

function _pluq_ka_complete!(A, N, p, q, slot, host, start, m, n; options = PLUQOptions())
    backend = KernelAbstractions.get_backend(A)
    group = _PLUQ_KA_GROUP_SIZE
    swap = _pluq_ka_swap_kernel!(backend, group)
    scale = _pluq_ka_scale_kernel!(backend, group)
    eliminate = _pluq_ka_rank1_kernel!(backend, (16, 8))
    minima =
        KernelInterface.supports_atomics(backend) ? nothing :
        similar(slot, Int32, _PLUQ_KA_PIVOT_GROUPS)
    mode = options.pivot_warp_kernel
    subgroup_width =
        _storage_backend(A) === Val(:metal) ? _pluq_pivot_subgroup_width(backend, mode) : 0
    rank = start - 1
    for k = start:min(m, n)
        prow, pcol =
            _pluq_ka_complete_pivot(A, slot, minima, host, k, m, n, N, mode, subgroup_width)
        prow == 0 && break
        if prow != k
            swap(A, p, k, prow, Val(true); ndrange = n)
        end
        if pcol != k
            swap(A, q, k, pcol, Val(false); ndrange = m)
        end
        if k < m
            scale(A, k, m, Int64(N); ndrange = cld(m - k, group) * group)
            if k < n
                eliminate(A, k, Int64(N); ndrange = (m - k, n - k))
            end
        end
        rank = k
    end
    return rank
end

function pluq_rectangular_rank_reference_gpu!(
    A::AbstractMatrix,
    N::Int,
    m::Int,
    n::Int;
    options::PLUQOptions = PLUQOptions(),
)
    _pluq_ka_check_matrix(A, N, m, n)
    min(m, n) == 0 && return pluq_init_perm(m), pluq_init_perm(n), 0

    logical = _logical_view(A, (m, n))
    _launch_mod_arrays!(logical, logical, N, _PLUQ_KA_GROUP_SIZE)
    p, q = _pluq_ka_device_permutations(A, m, n)
    slot = similar(_elementwise_native_root(A), Int32, 1)
    rank = _pluq_ka_complete!(A, N, p, q, slot, Vector{Int32}(undef, 1), 1, m, n; options)
    return Int.(Array(p)), Int.(Array(q)), rank
end

function pluq_rectangular_rank_gpu!(
    A::AbstractMatrix,
    N::Int,
    m::Int,
    n::Int;
    options::PLUQOptions = PLUQOptions(),
)
    return _pluq_ka_blocked!(A, N, m, n, options, Val(true))
end
