@inline function _triangular_diagonal_inverse(a::UInt64, q::UInt64)
    t, next_t = Int64(0), Int64(1)
    r, next_r = Int64(q), Int64(a)
    while !iszero(next_r)
        quotient, remainder = divrem(r, next_r)
        t, next_t = next_t, t - quotient * next_t
        r, next_r = next_r, remainder
    end
    return r == 1 ? UInt64(t < 0 ? t + Int64(q) : t) : zero(UInt64)
end

@kernel function _triangular_inverse_diagonal_kernel!(diagonal, A, modulus_i, q)
    i = @index(Global, Linear)
    @inbounds diagonal[i] =
        _triangular_diagonal_inverse(_residue_u64(A[i, i], modulus_i), q)
end

function _triangular_inverse_diagonal(A, N)
    diagonal = similar(_elementwise_native_root(A), UInt64, (size(A, 1),))
    backend = KernelInterface.get_backend(A)
    _triangular_inverse_diagonal_kernel!(backend, TILE_WIDTH)(
        diagonal,
        A,
        Int64(N),
        UInt64(N);
        ndrange = length(diagonal),
    )
    # Validate all pivots before writing any entry of a caller's destination.
    N > 1 && any(iszero, diagonal) && throw(LinearAlgebra.SingularException(0))
    return diagonal
end

@inline function _substitute_inverse_column!(
    D,
    A,
    diagonal,
    col,
    n,
    modulus_i,
    q,
    ::Val{Lower},
) where {Lower}
    start, step, last = Lower ? (1, 1, n) : (n, -1, 1)
    @inbounds for row = start:step:last
        dot = zero(UInt64)
        for k = start:step:(row-step)
            a = _residue_u64(A[row, k], modulus_i)
            x = _residue_u64(D[k, col], modulus_i)
            dot = _add_mod_u64(dot, _ordinary_residue_product(a, x, q), q)
        end
        rhs = row == col && q > 1 ? one(UInt64) : zero(UInt64)
        value = _ordinary_residue_product(diagonal[row], _sub_mod_u64(rhs, dot, q), q)
        D[row, col] = convert(eltype(D), value)
    end
    return nothing
end

@kernel function _triangular_inverse_tile_kernel!(
    D,
    A,
    diagonal,
    n,
    modulus_i,
    q,
    orientation::Val{Lower},
) where {Lower}
    col = @index(Local, Linear)
    coefficients = @localmem eltype(A) (TILE_WIDTH, TILE_WIDTH)
    for row = 1:n
        inside = col <= n && (Lower ? row > col : row < col)
        @inbounds coefficients[row, col] = inside ? A[row, col] : zero(eltype(A))
    end
    # All 32 lanes load or initialize their entries and reach this barrier.
    # Each active lane then owns a complete solution column.
    @synchronize()
    if col <= n
        _substitute_inverse_column!(
            D,
            coefficients,
            diagonal,
            col,
            n,
            modulus_i,
            q,
            orientation,
        )
    end
end

@kernel function _triangular_inverse_columns_kernel!(
    D,
    A,
    diagonal,
    n,
    modulus_i,
    q,
    orientation,
)
    col = @index(Global, Linear)
    _substitute_inverse_column!(D, A, diagonal, col, n, modulus_i, q, orientation)
end

function _launch_triangular_inverse_tile!(D, A, diagonal, N, orientation)
    backend = KernelInterface.get_backend(D)
    _triangular_inverse_tile_kernel!(backend, TILE_WIDTH)(
        D,
        A,
        diagonal,
        size(A, 1),
        Int64(N),
        UInt64(N),
        orientation;
        ndrange = TILE_WIDTH,
    )
    return D
end

function _triangular_inv_32!(D, A, N, row_shift, col_shift, n, orientation)
    Base.require_one_based_indexing(D, A)
    if !(0 <= n <= TILE_WIDTH && n <= min(size(D)...)) ||
       !(0 <= row_shift <= size(A, 1) - n && 0 <= col_shift <= size(A, 2) - n)
        throw(DimensionMismatch("the triangular inverse tile must fit both device arrays"))
    end
    _check_elementwise_backend(D, A)
    _stripe_mightalias(D, A) &&
        throw(ArgumentError("triangular inverse destination must not alias its input"))
    q = _validate_elementwise_modulus(eltype(A), A, N)
    _validate_elementwise_modulus(eltype(D), D, q)
    iszero(n) && return D

    tile = view(A, (row_shift+1):(row_shift+n), (col_shift+1):(col_shift+n))
    diagonal = _triangular_inverse_diagonal(tile, q)
    return _launch_triangular_inverse_tile!(D, tile, diagonal, q, orientation)
end

"""Write the forward-substitution inverse into the leading square of `D`."""
function _lower_triangular_inv_32!(
    D::AbstractMatrix,
    A::AbstractMatrix,
    N::Int,
    row_shift::Int,
    col_shift::Int,
    n_active::Int,
)
    return _triangular_inv_32!(D, A, N, row_shift, col_shift, n_active, Val(true))
end

"""Write the backward-substitution inverse into the leading square of `D`."""
function _upper_triangular_inv_32!(
    D::AbstractMatrix,
    A::AbstractMatrix,
    N::Int,
    row_shift::Int,
    col_shift::Int,
    n_active::Int,
)
    return _triangular_inv_32!(D, A, N, row_shift, col_shift, n_active, Val(false))
end
