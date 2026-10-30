"""
    _stripe_mul(Adata, Bdata; modulus, input_bound=nothing, max_ops=nothing)

Allocating raw-array adapter for callers that need a temporary product. The
stripe iteration and modular accumulation live only in `_stripe_mul!`.
"""
function _stripe_mul(
    Adata::AbstractMatrix,
    Bdata::AbstractMatrix;
    modulus,
    input_bound = nothing,
    max_ops = nothing,
)
    size(Adata, 2) == size(Bdata, 1) ||
        throw(DimensionMismatch("matrix product dimensions do not match"))
    eltype(Adata) === eltype(Bdata) ||
        throw(ArgumentError("matrix product element types do not match"))
    bound = input_bound === nothing ? modulus : input_bound
    find_max_stripe_ops(eltype(Adata), bound; modulus) >= 1 || throw(
        InverseOverflowError(
            "modulus $modulus is too large for exact multiplication with $(eltype(Adata))",
        ),
    )
    Cdata = CUDA.zeros(eltype(Adata), size(Adata, 1), size(Bdata, 2))
    return _stripe_mul!(Cdata, Adata, Bdata; modulus, input_bound, max_ops)
end
