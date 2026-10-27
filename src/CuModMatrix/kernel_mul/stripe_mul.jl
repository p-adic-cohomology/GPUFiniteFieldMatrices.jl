
#function gemv!(transpose::Bool,alpha::Integer,A::CuModMatrix,x::CuModVector,beta::Integer,y::CuModVector)
#
#end

"""
    find_max_stripe_ops(type, input_bound; modulus=input_bound)

Given a type `type` that is supported by cuBLAS,
find the maximum number of product contributions that can be accumulated per output
entry before a modular reduction, for multiplicands in `0:(input_bound - 1)`.
The budget includes the reduced prior accumulator, which is less than `modulus`.
The input bound and reduction modulus are independent. `input_bound > 1` is
required, and the supported exact-integer range follows the significand of the
cuBLAS input type.
"""
function find_max_stripe_ops(type, input_bound; modulus = input_bound)
    bits = if type === Float64
        53
    elseif type === Float32
        24
    elseif type === Float16
        11
    else
        throw(ArgumentError("The input type is not supported for cuBLAS gemm"))
    end

    input_bound isa Integer && input_bound > 1 ||
        throw(ArgumentError("input_bound must be an integer greater than one"))
    modulus isa Integer && modulus > 0 ||
        throw(ArgumentError("modulus must be a positive integer"))

    exact_max = BigInt(2)^bits - 1
    max_product = BigInt(input_bound - 1)^2
    max_prior = BigInt(modulus - 1)

    # Keep the established conservative bound when q == R while accounting for
    # a larger working modulus when the input bound and reduction differ.
    input_budget = exact_max ÷ max_product - 1
    accumulator_budget = (exact_max - max_prior) ÷ max_product
    safe_budget = max(min(input_budget, accumulator_budget), 0)
    return Int(min(safe_budget, typemax(Int)))
end

"""
     unsafe_gemm!(transposeA::Bool,transposeB::Bool,alpha::Integer,A::CuModMatrix,B::CuModMatrix,beta::Integer,C::CuModMatrix)

Light wrapper over the provided interface in CUDA.jl. 

Does not keep track of overflow, so errors could happen if the matrices are too big.

Also does not check to see if the modulus of the varios matrices agree

Since we are working mod N, `alpha` and `beta` must be integers
"""
function unsafe_gemm!(
    transposeA::Bool,
    transposeB::Bool,
    alpha::Integer,
    A::CuModMatrix,
    B::CuModMatrix,
    beta::Integer,
    C::CuModMatrix,
)

    tAchar = transposeA ? 'Y' : 'N'
    tBchar = transposeB ? 'Y' : 'N'

    #TODO: convert alpha and beta to floats?

    _CUDA_BLAS.gemm!(tAchar, tBchar, alpha, A.data, B.data, beta, C.data)

    mod!(C.data, C.data, C.N)
end

const _cublas_scalar_cache_f64 = IdDict{Task,Tuple}()
const _cublas_scalar_cache_f32 = IdDict{Task,Tuple}()

"""
returns (0, 1) as pointers that can be used with low-level cuBLAS APIs
"""
@inline function cublas_scalars_f64()
    t = current_task()
    get!(_cublas_scalar_cache_f64, t) do
        (CuRef(Float64(0.0)), CuRef(Float64(1.0)))
    end
end

"""
returns (0, 1) as pointers that can be used with low-level cuBLAS APIs
"""
@inline function cublas_scalars_f32()
    t = current_task()
    get!(_cublas_scalar_cache_f32, t) do
        (CuRef(Float32(0.0)), CuRef(Float32(1.0)))
    end
end

function _stripe_is_cuda_storage(data::CuArray)
    return true
end

function _stripe_is_cuda_storage(data::SubArray)
    return _stripe_is_cuda_storage(parent(data))
end

_stripe_is_cuda_storage(data) = false

_stripe_storage_root(data::CuArray) = data
_stripe_storage_root(data::SubArray) = _stripe_storage_root(parent(data))
_stripe_mightalias(A, B) = _stripe_storage_root(A) === _stripe_storage_root(B)

function _stripe_validate_raw(Cdata, Adata, Bdata, modulus, input_bound, max_ops)
    _stripe_is_cuda_storage(Cdata) &&
    _stripe_is_cuda_storage(Adata) &&
    _stripe_is_cuda_storage(Bdata) ||
        throw(ArgumentError("stripe multiplication requires CUDA arrays or CUDA views"))

    T = eltype(Cdata)
    T === eltype(Adata) === eltype(Bdata) ||
        throw(ArgumentError("matrix product element types do not match"))
    T in (Float16, Float32, Float64) || throw(
        ArgumentError("element type $T is unsupported for exact stripe multiplication"),
    )
    modulus isa Integer && modulus > 0 ||
        throw(ArgumentError("modulus must be a positive integer"))
    input_bound === nothing ||
        (input_bound isa Integer && input_bound > 1) ||
        throw(ArgumentError("input_bound must be an integer greater than one"))
    max_ops === nothing ||
        (max_ops isa Integer && max_ops > 0) ||
        throw(ArgumentError("max_ops must be a positive integer"))
    if T === Float32 && CUDA.math_mode() == CUDA.FAST_MATH
        throw(
            ArgumentError("exact modular GEMM requires CUDA DEFAULT_MATH or PEDANTIC_MATH"),
        )
    end
    return T
end

function _stripe_safe_chunk(::Type{T}, modulus, input_bound, max_ops) where {T}
    bound = input_bound === nothing ? modulus : input_bound
    safe_ops = find_max_stripe_ops(T, bound; modulus = modulus)
    safe_ops >= 1 || throw(
        ArgumentError(
            "cannot perform one exact product contribution for input bound $bound and modulus $modulus with $T",
        ),
    )
    chunk = max_ops === nothing ? safe_ops : Int(max_ops)
    chunk <= safe_ops || throw(
        ArgumentError(
            "max_ops=$chunk exceeds the exact stripe budget $safe_ops for input bound $bound and modulus $modulus",
        ),
    )
    return chunk
end

function _stripe_preflight(Cdata, Adata, Bdata, modulus, input_bound, max_ops)
    T = _stripe_validate_raw(Cdata, Adata, Bdata, modulus, input_bound, max_ops)
    _stripe_mightalias(Cdata, Adata) && throw(
        ArgumentError(
            "matrix product destination must not share storage with its left input",
        ),
    )
    _stripe_mightalias(Cdata, Bdata) && throw(
        ArgumentError(
            "matrix product destination must not share storage with its right input",
        ),
    )
    return _stripe_safe_chunk(T, modulus, input_bound, max_ops)
end

function _stripe_reduce!(Cdata::CuArray, modulus)
    mod!(Cdata, Cdata, modulus)
    return Cdata
end

function _stripe_reduce!(Cdata::SubArray, modulus)
    modulus_value = convert(eltype(Cdata), modulus)
    @. Cdata = mod(Cdata, modulus_value)
    return Cdata
end

function _stripe_gemv_scalars(::Type{Float64})
    return cublas_scalars_f64()
end

function _stripe_gemv_scalars(::Type{Float32})
    return cublas_scalars_f32()
end

function _stripe_gemv_scalars(::Type{T}) where {T}
    return (CuRef(zero(T)), CuRef(one(T)))
end

function _stripe_accumulate_stripe!(
    Cdata::AbstractMatrix,
    Adata::AbstractMatrix,
    Bdata::AbstractMatrix,
    stripe,
    first_stripe,
    scalars,
)
    beta = first_stripe ? 0 : 1
    _CUDA_BLAS.gemm!(
        'N',
        'N',
        1,
        @view(Adata[:, stripe]),
        @view(Bdata[stripe, :]),
        beta,
        Cdata,
    )
    return Cdata
end

function _stripe_accumulate_stripe!(
    zdata::AbstractVector,
    Adata::AbstractMatrix,
    xdata::AbstractVector,
    stripe,
    first_stripe,
    scalars,
)
    zero_ptr, one_ptr = scalars
    _CUDA_BLAS.gemv!(
        'N',
        one_ptr,
        @view(Adata[:, stripe]),
        @view(xdata[stripe]),
        first_stripe ? zero_ptr : one_ptr,
        zdata,
    )
    return zdata
end

function _stripe_run!(Cdata, Adata, Bdata, modulus, chunk, scalars)
    k = size(Adata, 2)
    if k == 0
        fill!(Cdata, zero(eltype(Cdata)))
        return Cdata
    end

    first_stripe = true
    for lo = 1:chunk:k
        hi = min(lo + chunk - 1, k)
        _stripe_accumulate_stripe!(Cdata, Adata, Bdata, lo:hi, first_stripe, scalars)
        _stripe_reduce!(Cdata, modulus)
        first_stripe = false
    end
    return Cdata
end

"""
    _stripe_mul!(Cdata, Adata, Bdata; modulus, input_bound=nothing, max_ops=nothing)

Destination-writing raw GPU core shared by matrix-matrix and matrix-vector
multiplication. Each stripe is accumulated directly into the destination; after
the first stripe, the prior destination is reduced before the next GEMM/GEMV.
The exactness budget accounts for that value using
`(modulus - 1) + max_ops * (input_bound - 1)^2`.
"""
function _stripe_mul!(
    Cdata::AbstractMatrix,
    Adata::AbstractMatrix,
    Bdata::AbstractMatrix;
    modulus,
    input_bound = nothing,
    max_ops = nothing,
)
    size(Adata, 2) == size(Bdata, 1) ||
        throw(DimensionMismatch("matrix product dimensions do not match"))
    size(Cdata) == (size(Adata, 1), size(Bdata, 2)) ||
        throw(DimensionMismatch("destination dimensions do not match the matrix product"))
    chunk = _stripe_preflight(Cdata, Adata, Bdata, modulus, input_bound, max_ops)
    return _stripe_run!(Cdata, Adata, Bdata, modulus, chunk, nothing)
end

"""Matrix-vector form of the shared destination-writing stripe core."""
function _stripe_mul!(
    zdata::AbstractVector,
    Adata::AbstractMatrix,
    xdata::AbstractVector;
    modulus,
    input_bound = nothing,
    max_ops = nothing,
)
    size(Adata, 2) == length(xdata) ||
        throw(DimensionMismatch("matrix-vector product dimensions do not match"))
    length(zdata) == size(Adata, 1) || throw(
        DimensionMismatch("destination length does not match the matrix-vector product"),
    )
    T = eltype(zdata)
    chunk = _stripe_preflight(zdata, Adata, xdata, modulus, input_bound, max_ops)
    return _stripe_run!(zdata, Adata, xdata, modulus, chunk, _stripe_gemv_scalars(T))
end

function _stripe_logical_data(A::CuModMatrix)
    m, n = size(A)
    size(A.data, 1) >= m && size(A.data, 2) >= n ||
        throw(DimensionMismatch("matrix storage is smaller than its logical dimensions"))
    return @view A.data[1:m, 1:n]
end

function _stripe_logical_data(x::CuModVector)
    n = length(x)
    length(x.data) >= n ||
        throw(DimensionMismatch("vector storage is smaller than its logical length"))
    return @view x.data[1:n]
end

function _validate_positive_modulus_override(modulus_override)
    modulus_override === nothing ||
        (modulus_override isa Integer && modulus_override > 0) ||
        throw(ArgumentError("modulus_override must be a positive integer"))
    return modulus_override
end

function _stripe_effective_modulus(stored_modulus, modulus_override)
    _validate_positive_modulus_override(modulus_override)
    modulus = modulus_override === nothing ? stored_modulus : modulus_override
    modulus isa Integer && modulus > 0 ||
        throw(ArgumentError("modulus must be a positive integer"))
    return Int(modulus)
end

"""
    stripe_mul!(C, A, B; modulus_override=nothing, input_bound=nothing, max_ops=nothing)

Multiply matrices into `C` modulo the effective modulus. The destination's
stored modulus is unchanged. When `modulus_override` is omitted, `C.N` is used
for reduction. A supplied `modulus_override` must be positive.
`input_bound` promises that input entries are nonnegative and strictly below
the given bound. `max_ops` limits product contributions per output entry
between reductions. For compatibility, an explicit matrix `input_bound`
selects one stripe; the entire inner dimension must fit the exact-integer
budget or the call throws `ArgumentError`.
"""
function stripe_mul!(
    C::CuModMatrix,
    A::CuModMatrix,
    B::CuModMatrix;
    modulus_override = nothing,
    input_bound = nothing,
    max_ops = nothing,
)
    modulus = _stripe_effective_modulus(C.N, modulus_override)
    A.N == B.N || throw(
        CuModArrayModulusMismatchException("Input matrices must have the same modulus N"),
    )
    size(A, 2) == size(B, 1) || throw(
        CuModArraySizeMismatchException(
            "Matrix dimensions do not match for multiplication",
        ),
    )
    size(C) == (size(A, 1), size(B, 2)) ||
        throw(CuModArraySizeMismatchException("Output matrix C has incorrect dimensions"))
    eltype(A.data) === eltype(B.data) === eltype(C.data) ||
        throw(ArgumentError("matrix product element types do not match"))
    max_ops === nothing ||
        (max_ops isa Integer && max_ops > 0) ||
        throw(ArgumentError("max_ops must be a positive integer"))

    bound = input_bound === nothing ? A.N : input_bound
    raw_max_ops = input_bound === nothing ? max_ops : max(size(A, 2), 1)
    Cdata = _stripe_logical_data(C)
    Adata = _stripe_logical_data(A)
    Bdata = _stripe_logical_data(B)
    _stripe_preflight(Cdata, Adata, Bdata, modulus, bound, raw_max_ops)

    fill!(C.data, zero(eltype(C.data)))
    _stripe_mul!(Cdata, Adata, Bdata; modulus, input_bound = bound, max_ops = raw_max_ops)
    return C
end

"""
Multiply a matrix and vector into `z`, preserving its stored modulus. A supplied
`modulus_override` must be positive. `respect_max_ops=false` recomputes the safe
stripe budget from `input_bound`.
"""
function stripe_mul!(
    z::CuModVector,
    A::CuModMatrix,
    x::CuModVector;
    modulus_override = nothing,
    input_bound = nothing,
    max_ops = nothing,
    respect_max_ops::Bool = true,
)
    modulus = _stripe_effective_modulus(A.N, modulus_override)
    A.N == x.N || throw(
        CuModArrayModulusMismatchException(
            "Matrix and vector inputs must have the same modulus N",
        ),
    )
    modulus_override === nothing &&
        z.N != A.N &&
        throw(
            CuModArrayModulusMismatchException(
                "Destination must have the input modulus when no override is supplied",
            ),
        )
    size(A, 2) == length(x) ||
        throw(CuModArraySizeMismatchException("Matrix and vector dimensions do not match"))
    length(z) == size(A, 1) ||
        throw(CuModArraySizeMismatchException("Output vector has incorrect length"))
    eltype(A.data) === eltype(x.data) === eltype(z.data) ||
        throw(ArgumentError("matrix-vector product element types do not match"))

    bound = input_bound === nothing ? A.N : input_bound
    zdata = _stripe_logical_data(z)
    Adata = _stripe_logical_data(A)
    xdata = _stripe_logical_data(x)
    effective_max_ops = respect_max_ops ? max_ops : nothing
    _stripe_preflight(zdata, Adata, xdata, modulus, bound, effective_max_ops)
    fill!(z.data, zero(eltype(z.data)))
    _stripe_mul!(
        zdata,
        Adata,
        xdata;
        modulus,
        input_bound = bound,
        max_ops = effective_max_ops,
    )
    return z
end

"""
    stripe_mul(A, B; modulus_override=nothing, input_bound=nothing, max_ops=nothing)

Allocate a matrix or vector result and multiply using the shared stripe core.
The returned object's stored modulus is the effective reduction modulus.
"""
function stripe_mul(
    A::CuModMatrix,
    B::CuModMatrix;
    modulus_override = nothing,
    input_bound = nothing,
    max_ops = nothing,
)
    modulus = _stripe_effective_modulus(A.N, modulus_override)
    A.N == B.N || throw(
        CuModArrayModulusMismatchException("Input matrices must have the same modulus N"),
    )
    size(A, 2) == size(B, 1) || throw(
        CuModArraySizeMismatchException(
            "Matrix dimensions do not match for multiplication",
        ),
    )
    C = zeros(eltype(A.data), size(A, 1), size(B, 2), modulus)
    return stripe_mul!(C, A, B; modulus_override = modulus, input_bound, max_ops)
end

function stripe_mul(
    A::CuModMatrix,
    x::CuModVector;
    modulus_override = nothing,
    input_bound = nothing,
    max_ops = nothing,
)
    modulus = _stripe_effective_modulus(A.N, modulus_override)
    A.N == x.N || throw(
        CuModArrayModulusMismatchException(
            "Matrix and vector inputs must have the same modulus N",
        ),
    )
    size(A, 2) == length(x) ||
        throw(CuModArraySizeMismatchException("Matrix and vector dimensions do not match"))
    z = zeros(eltype(A.data), size(A, 1), modulus)
    return stripe_mul!(z, A, x; modulus_override = modulus, input_bound, max_ops)
end
