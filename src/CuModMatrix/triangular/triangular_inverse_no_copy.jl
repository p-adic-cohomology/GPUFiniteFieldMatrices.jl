function _triangular_inverse_blocks!(
    D,
    A,
    diagonal,
    q,
    orientation::Val{Lower},
) where {Lower}
    n = size(A, 1)
    n <= TILE_WIDTH &&
        return _launch_triangular_inverse_tile!(D, A, diagonal, q, orientation)

    split = TILE_WIDTH * ((n + TILE_WIDTH) ÷ (2 * TILE_WIDTH))
    leading, trailing = 1:split, (split+1):n
    for block in (leading, trailing)
        _triangular_inverse_blocks!(
            view(D, block, block),
            view(A, block, block),
            view(diagonal, block),
            q,
            orientation,
        )
    end

    # The off-diagonal inverse is -D_first * A_coupling * D_last.
    first, last = Lower ? (trailing, leading) : (leading, trailing)
    first_inverse = GPUModArray(view(D, first, first); modulus = q)
    last_inverse = GPUModArray(view(D, last, last); modulus = q)
    coupling = GPUModArray(view(A, first, last); modulus = q)
    middle = stripe_mul(first_inverse, coupling)
    product = stripe_mul(middle, last_inverse)
    _launch_scalar_arrays!(
        view(D, first, last),
        _logical_view(parent(product), size(product)),
        q,
        zero(UInt64),
        Val(:neg),
        _ELEMENTWISE_WORKGROUP_SIZE,
    )
    return D
end

function _triangular_inverse_no_copy(
    A::GPUModMatrix,
    orientation::Val{Lower},
    debug,
) where {Lower}
    _storage_backend(parent(A)) === nothing &&
        throw(ArgumentError("triangular inversion requires native GPU storage"))
    rows, cols = size(A)
    # The released lower inverse supports square, wide, and single-tile tall inputs.
    Lower && rows > cols > TILE_WIDTH && throw(LinearAlgebra.SingularException(0))
    n = min(rows, cols)
    q = modulus(A)
    debug && @info "Triangular inverse" dimensions = (rows, cols) depth = n lower = Lower

    storage = getfield(A, :storage)
    backend = KernelInterface.get_backend(storage)
    target = KernelInterface.device(backend, storage)
    previous = KernelInterface.device(backend)
    target == previous || KernelInterface.device!(backend, target)
    try
        inverse = _stripe_result(A, (cols, rows), q)
        fill!(parent(inverse), zero(eltype(inverse)))
        (iszero(n) || q == 1) && return inverse

        square = view(parent(A), 1:n, 1:n)
        output = view(parent(inverse), 1:n, 1:n)
        diagonal = _triangular_inverse_diagonal(square, q)
        if n <= TILE_WIDTH || _ordinary_vendor_budget(square, q) > 0
            _triangular_inverse_blocks!(output, square, diagonal, q, orientation)
        else
            # A direct triangular solve covers moduli for which a floating
            # vendor product cannot take an exact contribution.
            _triangular_inverse_columns_kernel!(backend, TILE_WIDTH)(
                output,
                square,
                diagonal,
                n,
                Int64(q),
                UInt64(q),
                orientation;
                ndrange = n,
            )
        end
        return inverse
    finally
        target == previous || KernelInterface.device!(backend, previous)
    end
end

"""
    upper_triangular_inverse_no_copy(A; debug=false)

Invert the leading square of an upper triangular modular matrix while preserving
the input. A wide input returns a right inverse whose extra rows are zero.
"""
function upper_triangular_inverse_no_copy(A::GPUModMatrix; debug::Bool = false)
    return _triangular_inverse_no_copy(A, Val(false), debug)
end

"""
    lower_triangular_inverse_no_copy(A; debug=false)

Invert the leading square of a lower triangular modular matrix while preserving
the input. A wide input returns a right inverse whose extra rows are zero.
"""
function lower_triangular_inverse_no_copy(A::GPUModMatrix; debug::Bool = false)
    return _triangular_inverse_no_copy(A, Val(true), debug)
end
