# The delayed floating-point path is used only when an entire dot product
# and the initial residue fit within the storage type's exact integer budget.
@inline function _pluq_ka_trsm_subtract(acc::T, a, b, N, ::Val{true}) where {T}
    return acc - T(_pluq_ka_residue(a, N)) * T(_pluq_ka_residue(b, N))
end

@inline function _pluq_ka_trsm_subtract(acc::T, a, b, N, ::Val{false}) where {T}
    return T(_pluq_ka_submul(acc, a, b, N))
end

@kernel inbounds = true function _pluq_ka_trsm_left_kernel!(A, k0, kend, first, N, mode)
    j = first + (@index(Global, Linear)) - 1
    T = eltype(A)
    for i = k0:kend
        acc = T(_pluq_ka_residue(A[i, j], N))
        for k = k0:(i-1)
            acc = _pluq_ka_trsm_subtract(acc, A[i, k], A[k, j], N, mode)
        end
        A[i, j] = T(_pluq_ka_residue(acc, N))
    end
end

@kernel inbounds = true function _pluq_ka_trsm_diagonal_kernel!(dinv, A, k0, N)
    i = @index(Global, Linear)
    k = k0 + i - 1
    dinv[i] = eltype(dinv)(_pluq_ka_inverse(A[k, k], N))
end

@kernel inbounds = true function _pluq_ka_trsm_right_kernel!(
    A,
    dinv,
    diagonal_start,
    k0,
    kend,
    first,
    N,
    mode,
)
    i = first + (@index(Global, Linear)) - 1
    T = eltype(A)
    for j = k0:kend
        acc = T(_pluq_ka_residue(A[i, j], N))
        for k = k0:(j-1)
            acc = _pluq_ka_trsm_subtract(acc, A[i, k], A[k, j], N, mode)
        end
        invdiag = _pluq_ka_residue(dinv[diagonal_start+j-k0], N)
        A[i, j] = T(rem(_pluq_ka_residue(acc, N) * invdiag, N))
    end
end

function _pluq_ka_trsm_left!(A, N, k0, kend, first, n)
    (k0 > kend || first > n) && return A
    backend = KernelAbstractions.get_backend(A)
    T = eltype(A)
    delayed = T <: AbstractFloat && find_max_ops(T, N) >= kend - k0 + 1
    _pluq_ka_trsm_left_kernel!(backend, _PLUQ_KA_GROUP_SIZE)(
        A,
        k0,
        kend,
        first,
        Int64(N),
        Val(delayed);
        ndrange = n - first + 1,
    )
    return A
end

function _pluq_ka_trsm_right!(A, N, k0, kend, first, m, dinv)
    (k0 > kend || first > m) && return A
    backend = KernelAbstractions.get_backend(A)
    T = eltype(A)
    depth = kend - k0 + 1
    if dinv === nothing
        dinv = similar(_elementwise_native_root(A), T, depth)
        _pluq_ka_trsm_diagonal_kernel!(backend, _PLUQ_KA_GROUP_SIZE)(
            dinv,
            A,
            k0,
            Int64(N);
            ndrange = depth,
        )
        diagonal_start = 1
    else
        diagonal_start = k0
    end
    delayed = T <: AbstractFloat && find_max_ops(T, N) >= depth
    _pluq_ka_trsm_right_kernel!(backend, _PLUQ_KA_GROUP_SIZE)(
        A,
        dinv,
        diagonal_start,
        k0,
        kend,
        first,
        Int64(N),
        Val(delayed);
        ndrange = m - first + 1,
    )
    return A
end

# A leading packed lower-unit panel acts on the columns after kend.
function pluq_trsm_left_lower_unit_gpu!(
    A::AbstractMatrix,
    N::Int,
    k0::Int,
    kend::Int,
    n::Int;
    options::PLUQOptions = PLUQOptions(),
)
    _pluq_ka_check_matrix(A, N, kend, n)
    1 <= k0 <= kend + 1 && kend <= n || throw(ArgumentError("invalid PLUQ TRSM panel"))
    return _pluq_ka_trsm_left!(A, N, k0, kend, kend + 1, n)
end

# The upper-triangular panel acts on the rows after kend. A supplied vector of
# inverse diagonals is indexed by the absolute column of the panel.
function pluq_trsm_right_upper_gpu!(
    A::AbstractMatrix,
    N::Int,
    k0::Int,
    kend::Int,
    m::Int;
    options::PLUQOptions = PLUQOptions(),
    dinv = nothing,
)
    _pluq_ka_check_matrix(A, N, m, kend)
    1 <= k0 <= kend + 1 && kend <= m || throw(ArgumentError("invalid PLUQ TRSM panel"))
    if dinv !== nothing
        _pluq_ka_check_scratch(A, dinv, eltype(A), kend)
    end
    return _pluq_ka_trsm_right!(A, N, k0, kend, kend + 1, m, dinv)
end
