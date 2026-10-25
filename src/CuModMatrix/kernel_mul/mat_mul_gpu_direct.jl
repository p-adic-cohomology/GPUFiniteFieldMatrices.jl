

"""Multiply GPU matrices modulo `N`, reducing between exactly representable GEMM chunks."""
function _exact_mod_matmul_data(A, B, N::Integer)
    size(A, 2) == size(B, 1) ||
        throw(DimensionMismatch("matrix product dimensions do not match"))
    T = eltype(A)
    T == eltype(B) || throw(ArgumentError("matrix product element types do not match"))
    CUDA.math_mode() == CUDA.FAST_MATH &&
        T == Float32 &&
        throw(
            ArgumentError("exact modular GEMM requires CUDA DEFAULT_MATH or PEDANTIC_MATH"),
        )
    chunk = find_max_ops(T, N)
    chunk >= 1 || throw(
        InverseOverflowError("modulus $N is too large for exact multiplication with $T"),
    )
    m, k = size(A)
    n = size(B, 2)
    C = CUDA.zeros(T, m, n)
    tmp = k > chunk ? similar(C) : C
    first_chunk = true
    lo = 1
    while lo <= k
        hi = min(k, lo + chunk - 1)
        Av = @view A[:, lo:hi]
        Bv = @view B[lo:hi, :]
        if first_chunk
            mul!(C, Av, Bv)
            C .= mod.(C, T(N))
            first_chunk = false
        else
            mul!(tmp, Av, Bv)
            C .= mod.(C .+ tmp, T(N))
        end
        lo = hi + 1
    end
    return C
end

"""
    mat_mul_gpu_type(A::CuModMatrix, B::CuModMatrix; modulus_override = nothing)

Matrix multiplication that works directly with `CuModMatrix` objects. A
positive `modulus_override` sets the reduction modulus; omitted, zero, or
negative values use `A.N`.
"""
function mat_mul_gpu_type(
    A::CuModMatrix,
    B::CuModMatrix,
    ;
    modulus_override::Union{Nothing,Integer} = nothing,
    REGIME = "⊠",
    type = nothing,
)
    N = modulus_override === nothing || modulus_override <= 0 ? A.N : modulus_override

    if cols(A) != rows(B)
        throw(CuModArraySizeMismatchException("Matrix dimensions do not match.
            A has $(rows(A)) rows and $(cols(A)) cols, 
            B has $(rows(B)) rows and $(cols(B)) cols."))
    end

    d_C = _exact_mod_matmul_data(A.data, B.data, N)
    return CuModMatrix(d_C, N, new_size = (rows(A), cols(B)))
end

"""
    mat_mul_type_inplace!(C::CuModMatrix, A::CuModMatrix, B::CuModMatrix; modulus_override = nothing)

In-place matrix multiplication that works directly with `CuModMatrix` objects.
A positive `modulus_override` sets the reduction modulus; omitted, zero, or
negative values use `A.N`. Reductions leave `C.N` unchanged.
"""
function mat_mul_type_inplace!(
    C::CuModMatrix,
    A::CuModMatrix,
    B::CuModMatrix,
    ;
    modulus_override::Union{Nothing,Integer} = nothing,
    REGIME = "⊠",
    type = nothing,
)
    # Use the provided modulus if available, otherwise preserve A's stored modulus.
    N = modulus_override === nothing || modulus_override <= 0 ? A.N : modulus_override

    if rows(C) != rows(A) || cols(C) != cols(B)
        throw(MatrixSizeMismatchException("Output matrix C has incorrect dimensions"))
    end

    if cols(A) != rows(B)
        throw(MatrixSizeMismatchException("Matrix dimensions do not match.
            A has $rows(A) rows and $cols(A) cols, 
            B has $rows(B) rows and $cols(B) cols."))
    end

    copyto!(C.data, _exact_mod_matmul_data(A.data, B.data, N))
    return C
end
