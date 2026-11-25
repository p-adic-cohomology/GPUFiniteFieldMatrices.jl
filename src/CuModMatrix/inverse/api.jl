"""
    pluq!(A; options=PLUQOptions())

Factor a GPU modular matrix in place. The returned `PLUQ.LU` is `A`, and its
host permutations satisfy `A_original[F.p, F.q] == F.L * F.U` modulo the stored
modulus. Enable `check_prime` to validate the finite-field precondition.
"""
function pluq!(A::GPUModMatrix; options::PLUQOptions = PLUQOptions())
    opts = _inverse_options(options, A)
    m, n = size(A)
    p, q, rank = if isempty(A)
        pluq_init_perm(m), pluq_init_perm(n), 0
    elseif m == n
        pluq_blocked_gpu!(parent(A), modulus(A), opts, n)
    else
        pluq_rectangular_rank_gpu!(parent(A), modulus(A), m, n; options = opts)
    end
    return PLUQ(A, p, q, rank)
end

"""Factor a copy of `A`, preserving the input and its backing storage."""
function pluq(A::GPUModMatrix; options::PLUQOptions = PLUQOptions())
    _inverse_preflight(A, options)
    return pluq!(copy(A); options)
end

"""Return whether the modular matrix is square and has full rank."""
function is_invertible(A::GPUModMatrix; options::PLUQOptions = PLUQOptions())
    _inverse_preflight(A, options)
    return size(A, 1) == size(A, 2) && pluq(A; options).rank == size(A, 1)
end

function _inverse_from_factor(F::PLUQ)
    A = F.LU
    n = size(A, 1)
    F.rank == n || throw(LinearAlgebra.SingularException(0))
    result = similar(A)
    n == 0 && return result
    product = similar(A)
    LinearAlgebra.mul!(
        product,
        upper_triangular_inverse_no_copy(F.U),
        lower_triangular_inverse_no_copy(F.L),
    )
    _inverse_gather!(
        parent(result),
        parent(product),
        pluq_inverse_perm(F.q),
        pluq_inverse_perm(F.p),
    )
    return result
end

"""Compute a square inverse through PLUQ and the ordinary triangular inverses."""
function inverse_pluq(A::GPUModMatrix; options::PLUQOptions = PLUQOptions())
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("matrix must be square"))
    opts = _inverse_options(options, A)
    isempty(A) || _inverse_check_product_budget(A)
    return _inverse_from_factor(pluq(A; options = opts))
end

"""
    is_invertible_with_inverse(A; debug=false)

Return `(false, nothing)` for a singular or nonsquare matrix, and
`(true, inverse)` for a nonsingular square modular matrix.
"""
function is_invertible_with_inverse(A::GPUModMatrix; debug::Bool = false)
    F = pluq(A)
    m, n = size(A)
    m == n == F.rank || return false, nothing
    isempty(A) || _inverse_check_product_budget(A)
    result = _inverse_from_factor(F)
    if debug
        display(A)
        display(result)
        display(A * result)
    end
    return true, result
end

@kernel inbounds = true function _inverse_determinant_kernel!(out, A, n, N)
    value = Int64(1)
    for i = 1:n
        value = rem(value * _pluq_ka_residue(A[i, i], N), N)
    end
    out[1] = value
end

function _inverse_odd_permutation(p)
    visited = falses(length(p))
    odd = false
    for i in eachindex(p)
        visited[i] && continue
        j = p[i]
        while j != i
            visited[j] = true
            odd = !odd
            j = p[j]
        end
        visited[i] = true
    end
    return odd
end

"""Return the determinant modulo the stored modulus, using both permutation signs."""
function LinearAlgebra.det(F::PLUQ)
    A = F.LU
    m, n = size(A)
    m == n || throw(DimensionMismatch("matrix must be square"))
    _inverse_preflight(A, PLUQOptions())
    _inverse_validate_factor(F)
    F.rank == n || return zero(eltype(A))
    n == 0 && return one(eltype(A))
    data = parent(A)
    value = similar(_elementwise_native_root(data), Int64, 1)
    N = modulus(A)
    _inverse_determinant_kernel!(KernelInterface.get_backend(data), 1)(
        value,
        data,
        n,
        N;
        ndrange = 1,
    )
    d = only(Array(value))
    odd = _inverse_odd_permutation(F.p) != _inverse_odd_permutation(F.q)
    return eltype(A)(mod(odd ? -d : d, N))
end

function LinearAlgebra.det(A::GPUModMatrix; options::PLUQOptions = PLUQOptions())
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("matrix must be square"))
    return LinearAlgebra.det(pluq(A; options))
end

@kernel inbounds = true function _inverse_init_augmented!(aug, source, n, N)
    i, j = @index(Global, NTuple)
    aug[i, j] = if j <= n
        eltype(aug)(_pluq_ka_residue(source[i, j], N))
    else
        eltype(aug)(i == j - n)
    end
end

@kernel inbounds = true function _inverse_augmented_swap!(
    A,
    k,
    other,
    ::Val{Rows},
) where {Rows}
    i = @index(Global, Linear)
    if Rows
        A[k, i], A[other, i] = A[other, i], A[k, i]
    else
        A[i, k], A[i, other] = A[i, other], A[i, k]
    end
end

@kernel inbounds = true function _inverse_augmented_diagonal!(dinv, A, k, N)
    dinv[1] = _pluq_ka_inverse(A[k, k], N)
end

@kernel inbounds = true function _inverse_augmented_scale!(A, dinv, k, N)
    i = @index(Global, Linear)
    j = k + i - 1
    A[k, j] = eltype(A)(rem(_pluq_ka_residue(A[k, j], N) * dinv[1], N))
end

@kernel inbounds = true function _inverse_augmented_column!(factors, A, k)
    i = @index(Global, Linear)
    factors[i] = A[i, k]
end

@kernel inbounds = true function _inverse_augmented_eliminate!(A, factors, k, N)
    i, offset = @index(Global, NTuple)
    j = k + offset - 1
    if i != k
        A[i, j] = eltype(A)(_pluq_ka_submul(A[i, j], factors[i], A[k, j], N))
    end
end

@kernel inbounds = true function _inverse_scatter_kernel!(out, A, rows, column_offset)
    i, j = @index(Global, NTuple)
    out[rows[i], j] = A[i, j+column_offset]
end

function _inverse_scatter!(out, data, rows, column_offset, columns)
    (isempty(rows) || columns == 0) && return out
    device_rows = _inverse_device_indices(out, rows)
    _inverse_scatter_kernel!(KernelInterface.get_backend(out))(
        out,
        data,
        device_rows,
        column_offset;
        ndrange = (length(rows), columns),
    )
    return out
end

# The first n columns of [A I] are pivoted on the device. Host transfers
# carry only pivot indices and invertibility status, never matrix values.
function _inverse_augmented(A::GPUModMatrix)
    m, n = size(A)
    result = _inverse_zero_matrix(A, (n, m))
    m == 0 && return result
    source = parent(A)
    native = _elementwise_native_root(source)
    backend = KernelInterface.get_backend(native)
    N = modulus(A)
    width = n + m
    aug = similar(native, eltype(A), (m, width))
    pivot_slot = similar(native, Int32, 1)
    diagonal = similar(native, Int64, 1)
    factors = similar(native, eltype(A), m)
    _inverse_init_augmented!(backend)(aug, source, n, N; ndrange = (m, width))
    permutation = pluq_init_perm(n)
    swap = _inverse_augmented_swap!(backend)
    scale = _inverse_augmented_scale!(backend)
    column = _inverse_augmented_column!(backend)
    eliminate = _inverse_augmented_eliminate!(backend)
    for k = 1:m
        _pluq_ka_find_pivot!(pivot_slot, aug, k, m, n, N)
        pivot = Int(only(Array(pivot_slot)))
        active_rows = m - k + 1
        pivot <= active_rows * (n - k + 1) || throw(LinearAlgebra.SingularException(0))
        col, row = divrem(pivot - 1, active_rows)
        prow, pcol = k + row, k + col
        if prow != k
            swap(aug, k, prow, Val(true); ndrange = width)
        end
        if pcol != k
            swap(aug, k, pcol, Val(false); ndrange = m)
            permutation[k], permutation[pcol] = permutation[pcol], permutation[k]
        end
        _inverse_augmented_diagonal!(backend, 1)(diagonal, aug, k, N; ndrange = 1)
        iszero(only(Array(diagonal))) && throw(LinearAlgebra.SingularException(0))
        remaining = width - k + 1
        scale(aug, diagonal, k, N; ndrange = remaining)
        column(factors, aug, k; ndrange = m)
        eliminate(aug, factors, k, N; ndrange = (m, remaining))
    end
    _inverse_scatter!(parent(result), aug, view(permutation, 1:m), n, m)
    return result
end

"""Compute a square modular inverse with the selected PLUQ or augmented strategy."""
function Base.inv(A::GPUModMatrix; options::PLUQOptions = PLUQOptions())
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("matrix must be square"))
    opts = _inverse_options(options, A)
    return opts.inverse_strategy === :pluq ? inverse_pluq(A; options = opts) :
           _inverse_augmented(A)
end

function _inverse_selected_columns(A, columns, opts)
    m, n = size(A)
    block = _inverse_zero_matrix(A, (m, m))
    _inverse_gather!(parent(block), parent(A), 1:m, columns)
    result = try
        inverse_pluq(block; options = opts)
    catch err
        err isa LinearAlgebra.SingularException || rethrow()
        return nothing
    end
    out = _inverse_zero_matrix(A, (n, m))
    _inverse_scatter!(parent(out), parent(result), columns, 0, m)
    return out
end

"""
    right_inverse(A; options=PLUQOptions())

Find `X` such that `A * X == I` modulo the stored modulus. A full-row-rank
matrix is required. The PLUQ strategy first tries the leading square block
and then the independent columns identified by rectangular PLUQ.
"""
function right_inverse(A::GPUModMatrix; options::PLUQOptions = PLUQOptions())
    opts = _inverse_options(options, A)
    m, n = size(A)
    m <= n || throw(DimensionMismatch("right inverse requires rows(A) <= cols(A)"))
    if 0 < m < n && opts.inverse_strategy === :pluq
        leading = _inverse_selected_columns(A, 1:m, opts)
        leading === nothing || return leading
        F = pluq(A; options = opts)
        F.rank == m || throw(LinearAlgebra.SingularException(0))
        selected = _inverse_selected_columns(A, view(F.q, 1:m), opts)
        selected === nothing || return selected
    end
    return _inverse_augmented(A)
end

@kernel inbounds = true function _inverse_transpose_kernel!(out, A)
    i, j = @index(Global, NTuple)
    out[i, j] = A[j, i]
end

function _inverse_transpose(A)
    out = _inverse_zero_matrix(A, reverse(size(A)))
    isempty(out) || _inverse_transpose_kernel!(KernelInterface.get_backend(parent(out)))(
        parent(out),
        parent(A);
        ndrange = size(out),
    )
    return out
end

"""Find a modular left inverse of a full-column-rank matrix."""
function left_inverse(A::GPUModMatrix; options::PLUQOptions = PLUQOptions())
    size(A, 1) >= size(A, 2) ||
        throw(DimensionMismatch("left inverse requires rows(A) >= cols(A)"))
    _inverse_preflight(A, options)
    return _inverse_transpose(right_inverse(_inverse_transpose(A); options))
end
