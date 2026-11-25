"""
    pluq_schur_update_rect_gpu!(A, N, k0, kend, m, n; options=PLUQOptions())

Apply the packed-panel Schur update to the logical trailing block of a native
device matrix. Panel factors, preceding entries, and backing padding are
preserved. A panel of zero width or an empty trailing block is a no-op.
"""
function pluq_schur_update_rect_gpu!(
    A::AbstractMatrix{T},
    N::Int,
    k0::Int,
    kend::Int,
    m::Int,
    n::Int;
    options::PLUQOptions = PLUQOptions(),
) where {T<:AbstractFloat}
    Base.require_one_based_indexing(A)
    if !(0 <= m <= size(A, 1) && 0 <= n <= size(A, 2)) ||
       !(1 <= k0 <= kend + 1 && 0 <= kend <= min(m, n))
        throw(
            DimensionMismatch("Schur panel and trailing block must fit the logical input"),
        )
    end
    _storage_backend(A) === nothing &&
        throw(ArgumentError("Schur updates require native GPU storage"))
    q = _validate_elementwise_modulus(T, A, N)
    (k0 > kend || kend == m || kend == n) && return A

    panel = k0:kend
    rows = (kend+1):m
    cols = (kend+1):n
    L = view(A, rows, panel)
    U = view(A, panel, cols)
    trailing = view(A, rows, cols)
    if _ordinary_vendor_budget(A, q) >= length(panel)
        # The destination and factors share one packed backing matrix. A
        # separate product allows the checked vendor API to enforce its alias
        # contract before the exact in-place subtraction.
        product = stripe_mul(GPUModArray(L; modulus = q), GPUModArray(U; modulus = q))
        data = _logical_view(parent(product), size(product))
        _launch_binary_arrays!(
            trailing,
            trailing,
            data,
            q,
            Val(:sub),
            _ELEMENTWISE_WORKGROUP_SIZE,
        )
    else
        tile = min(options.schur_tile, 16)
        _launch_ordinary_matmul!(
            trailing,
            L,
            U,
            q,
            tile,
            options.schur_transpose_u,
            Val(true),
        )
    end
    return A
end

function pluq_schur_update_gpu!(A, N, k0, kend, n; options::PLUQOptions = PLUQOptions())
    return pluq_schur_update_rect_gpu!(A, N, k0, kend, n, n; options)
end
