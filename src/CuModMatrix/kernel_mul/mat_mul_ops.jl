import KernelAbstractions: @localmem, @synchronize

@inline function _ordinary_residue_product(a::UInt64, b::UInt64, q::UInt64)
    if q <= UInt64(1) << 32
        return rem(a * b, q)
    end
    return _mul_mod_u64(a, b, q)
end

@kernel function _ordinary_matmul_kernel!(
    C,
    A,
    B,
    rows,
    inner,
    cols,
    modulus_i,
    q,
    ::Val{Tile},
    transpose_b,
    ::Val{Subtract},
) where {Tile,Subtract}
    i, j = Tuple(@index(Global, Cartesian))
    x, y = Tuple(@index(Local, Cartesian))
    left = @localmem UInt64 (Tile, Tile)
    right = @localmem UInt64 (Tile, Tile)
    dot = zero(UInt64)

    for first_k = 1:Tile:inner
        ka = first_k + y - 1
        kb = first_k + x - 1
        @inbounds left[x, y] =
            i <= rows && ka <= inner ? _residue_u64(A[i, ka], modulus_i) : zero(UInt64)
        @inbounds b =
            j <= cols && kb <= inner ? _residue_u64(B[kb, j], modulus_i) : zero(UInt64)
        @inbounds right[transpose_b ? y : x, transpose_b ? x : y] = b
        @synchronize()

        for k = 1:Tile
            @inbounds a = left[x, k]
            @inbounds b_tile = right[transpose_b ? y : k, transpose_b ? k : y]
            dot = _add_mod_u64(dot, _ordinary_residue_product(a, b_tile, q), q)
        end
        @synchronize()
    end

    if i <= rows && j <= cols
        if Subtract
            @inbounds dot = _sub_mod_u64(_residue_u64(C[i, j], modulus_i), dot, q)
        end
        @inbounds C[i, j] = convert(eltype(C), dot)
    end
end

function _launch_ordinary_matmul!(C, A, B, N, tile, transpose_b, subtract)
    isempty(C) && return C
    # Round up the launch so ragged lanes initialize their tile entries and
    # enter both workgroup barriers. A tile uses at most 4 KiB of local memory.
    rows, cols = size(C)
    backend = KernelInterface.get_backend(C)
    _ordinary_matmul_kernel!(backend, (tile, tile))(
        C,
        A,
        B,
        rows,
        size(A, 2),
        cols,
        Int64(N),
        UInt64(N),
        Val(tile),
        transpose_b,
        subtract;
        ndrange = (cld(rows, tile) * tile, cld(cols, tile) * tile),
    )
    return C
end

"""Write an ordinary tiled modular product into a native device matrix or view."""
function _mat_mul_ops!(
    C::AbstractMatrix,
    A::AbstractMatrix,
    B::AbstractMatrix,
    N::Integer;
    tile::Int = 16,
    transpose_b::Bool = false,
)
    Base.require_one_based_indexing(C, A, B)
    size(C) == _stripe_output_size(A, B) ||
        throw(DimensionMismatch("destination dimensions do not match the matrix product"))
    T = _stripe_check_eltypes(C, A, B)
    _check_elementwise_backend(C, A)
    _check_elementwise_backend(C, B)
    _stripe_check_alias(C, A, B)
    q = _validate_elementwise_modulus(T, C, N)
    tile in (8, 16) || throw(ArgumentError("ordinary product tile must be 8 or 16"))
    return _launch_ordinary_matmul!(C, A, B, q, tile, transpose_b, Val(false))
end

function _ordinary_vendor_budget(A, N)
    N > 1 || return 0
    T = eltype(A)
    if _storage_backend(A) === Val(:metal) && T !== Float32
        return 0
    end
    return find_max_stripe_ops(T, N)
end
