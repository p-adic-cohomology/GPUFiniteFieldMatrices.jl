function _stripe_safe_chunk(
    ::Type{T},
    modulus,
    input_bound,
    max_ops;
    empty_inner::Bool = false,
) where {T}
    max_ops === nothing ||
        (max_ops isa Integer && max_ops > 0) ||
        throw(ArgumentError("max_ops must be a positive integer"))
    bound = input_bound === nothing ? modulus : input_bound
    safe_ops = find_max_stripe_ops(T, bound; modulus)
    if empty_inner && max_ops === nothing
        return max(safe_ops, 1)
    end
    safe_ops >= 1 || throw(
        ArgumentError(
            "cannot perform one exact product contribution for input bound $bound and modulus $modulus with $T",
        ),
    )
    max_ops === nothing && return safe_ops
    max_ops <= safe_ops || throw(
        ArgumentError(
            "max_ops=$max_ops exceeds the exact stripe budget $safe_ops for input bound $bound and modulus $modulus",
        ),
    )
    return Int(max_ops)
end

function _stripe_output_size(A::AbstractMatrix, B::AbstractMatrix)
    size(A, 2) == size(B, 1) ||
        throw(DimensionMismatch("matrix product dimensions do not match"))
    return (size(A, 1), size(B, 2))
end

function _stripe_output_size(A::AbstractMatrix, x::AbstractVector)
    size(A, 2) == length(x) ||
        throw(DimensionMismatch("matrix-vector product dimensions do not match"))
    return (size(A, 1),)
end

function _stripe_check_eltypes(C, A, B)
    T = eltype(C)
    T === eltype(A) === eltype(B) ||
        throw(ArgumentError("matrix product element types do not match"))
    T in (Float16, Float32, Float64) || throw(
        ArgumentError("element type $T is unsupported for exact stripe multiplication"),
    )
    return T
end

function _stripe_check_devices(C, A, B)
    vendor = _storage_backend(_native_storage(C))
    vendor !== nothing &&
    vendor ===
    _storage_backend(_native_storage(A)) ===
    _storage_backend(_native_storage(B)) || throw(
        ArgumentError("stripe multiplication requires arrays on the same GPU backend"),
    )
    backend = KernelInterface.get_backend(C)
    _stripe_require_same_device(
        KernelInterface.device(backend, C),
        KernelInterface.device(backend, A),
        KernelInterface.device(backend, B),
    )
    return nothing
end

function _stripe_require_same_device(output_device, left_device, right_device)
    output_device == left_device == right_device ||
        throw(ArgumentError("matrix product arrays must use the same GPU device"))
    return nothing
end

function _stripe_mightalias(A, B)
    return _elementwise_native_root(A) === _elementwise_native_root(B) ||
           Base.mightalias(A, B)
end

function _stripe_check_alias(Cdata, Adata, Bdata)
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
    return nothing
end

function _stripe_logical_data(A::GPUModArray)
    return _logical_view(parent(A), size(A))
end

function _stripe_effective_modulus(stored_modulus, modulus_override)
    modulus_override === nothing && return stored_modulus
    modulus_override isa Integer ||
        throw(ArgumentError("modulus_override must be a positive integer"))
    return _validate_positive_modulus_override(modulus_override)
end

function _stripe_product_modulus(C::GPUModMatrix, A, B, modulus_override)
    effective_modulus = _stripe_effective_modulus(modulus(C), modulus_override)
    modulus(A) == modulus(B) ||
        throw(ModulusMismatchError("Input matrices must have the same modulus N"))
    return effective_modulus
end

function _stripe_product_modulus(z::GPUModVector, A, x, modulus_override)
    effective_modulus = _stripe_effective_modulus(modulus(A), modulus_override)
    modulus(A) == modulus(x) ||
        throw(ModulusMismatchError("Matrix and vector inputs must have the same modulus N"))
    modulus_override === nothing &&
        modulus(z) != modulus(A) &&
        throw(
            ModulusMismatchError(
                "Destination must have the input modulus when no override is supplied",
            ),
        )
    return effective_modulus
end

function _stripe_product_args(
    C::GPUModArray,
    A::GPUModMatrix,
    B::GPUModArray;
    modulus_override = nothing,
    input_bound = nothing,
    max_ops = nothing,
)
    effective_modulus = _stripe_product_modulus(C, A, B, modulus_override)
    size(C) == _stripe_output_size(A, B) ||
        throw(DimensionMismatch("destination dimensions do not match the matrix product"))
    T = _stripe_check_eltypes(C, A, B)
    _stripe_check_devices(
        getfield(C, :storage),
        getfield(A, :storage),
        getfield(B, :storage),
    )
    Cdata, Adata, Bdata =
        _stripe_logical_data(C), _stripe_logical_data(A), _stripe_logical_data(B)
    _stripe_check_alias(Cdata, Adata, Bdata)

    bound = input_bound === nothing ? modulus(A) : input_bound
    chunk = _stripe_safe_chunk(
        T,
        effective_modulus,
        bound,
        max_ops;
        empty_inner = size(A, 2) == 0,
    )
    return Cdata, Adata, Bdata, effective_modulus, chunk
end

_stripe_ranges(k, chunk) = (lo:(lo+min(chunk-1, k-lo)) for lo = 1:chunk:k)

function _stripe_reduce!(Cdata, modulus::Integer)
    # Preflight bounds every accumulator to an exactly representable integer.
    return _launch_mod_arrays!(Cdata, Cdata, Int(modulus), _ELEMENTWISE_WORKGROUP_SIZE)
end

function _stripe_accumulate_stripe!(Cdata, Adata, Bdata, stripe)
    T = eltype(Cdata)
    alpha = one(T)
    beta = first(stripe) == 1 ? zero(T) : one(T)
    return LinearAlgebra.mul!(
        Cdata,
        view(Adata, :, stripe),
        selectdim(Bdata, 1, stripe),
        alpha,
        beta,
    )
end

# The caller provides native operands whose views dispatch to vendor BLAS.
# The previous stripe has already been reduced when beta is one.
function _stripe_run!(Cdata, Adata, Bdata, modulus, chunk)
    isempty(Cdata) && return Cdata
    k = size(Adata, 2)
    k == 0 && return fill!(Cdata, zero(eltype(Cdata)))

    for stripe in _stripe_ranges(k, chunk)
        _stripe_accumulate_stripe!(Cdata, Adata, Bdata, stripe)
        _stripe_reduce!(Cdata, modulus)
    end
    return Cdata
end

"""
    stripe_mul!(C, A, B; modulus_override=nothing, input_bound=nothing, max_ops=nothing)
    stripe_mul!(z, A, x; modulus_override=nothing, input_bound=nothing,
                max_ops=nothing, respect_max_ops=true)

Multiply into an existing modular matrix or vector. For matrices, the default
reduction modulus is `modulus(C)`. For vectors, the destination and inputs must
have the same stored modulus unless a positive `modulus_override` is supplied.
Input moduli must agree and the destination retains its stored modulus.

Each nonnegative input entry must be below `input_bound`, which defaults to
the input modulus. The product is reduced after each stripe. `max_ops` limits
the number of terms per stripe and must fit the exact accumulation budget.
For vectors, `respect_max_ops=false` uses the computed budget.

An empty inner dimension produces zeros after validation. The default budget
may be zero when no contributions are needed. An explicit `max_ops` still
requires a positive safe budget.
"""
function stripe_mul!(C::GPUModArray, A::GPUModMatrix, B::GPUModArray; kwargs...)
    throw(
        ArgumentError(
            "stripe multiplication requires three arrays on a supported GPU backend",
        ),
    )
end
