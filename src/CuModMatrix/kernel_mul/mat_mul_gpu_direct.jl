function _stripe_result(A::GPUModMatrix{T}, dims::Dims, effective_modulus) where {T}
    source = getfield(A, :storage)
    _storage_backend(parent(A)) !== nothing ||
        throw(ArgumentError("stripe multiplication requires GPU storage"))
    effective_modulus = _validate_storage_modulus(T, effective_modulus)
    backend = KernelInterface.get_backend(source)
    target = KernelInterface.device(backend, source)
    previous = KernelInterface.device(backend)

    # Native similar preserves the storage mode and allocates on the active device.
    target == previous || KernelInterface.device!(backend, target)
    native = try
        similar(_elementwise_native_root(parent(A)), T, _padded_size(dims))
    finally
        target == previous || KernelInterface.device!(backend, previous)
    end
    return GPUModArray(native; modulus = effective_modulus, new_size = dims)
end

"""
    stripe_mul(A, B; modulus_override=nothing, input_bound=nothing, max_ops=nothing)

Allocate a modular matrix or vector for the exact stripe product of `A` and
`B`. Its stored modulus is the effective reduction modulus.
"""
function stripe_mul(
    A::GPUModMatrix,
    B::Union{GPUModMatrix,GPUModVector};
    modulus_override = nothing,
    input_bound = nothing,
    max_ops = nothing,
)
    effective_modulus = _stripe_effective_modulus(modulus(A), modulus_override)
    modulus(A) == modulus(B) ||
        throw(ModulusMismatchError("Matrix product inputs must have the same modulus N"))
    dims = _stripe_output_size(A, B)
    _stripe_check_eltypes(A, A, B)
    _stripe_check_devices(
        getfield(A, :storage),
        getfield(A, :storage),
        getfield(B, :storage),
    )
    C = _stripe_result(A, dims, effective_modulus)
    return stripe_mul!(C, A, B; modulus_override = effective_modulus, input_bound, max_ops)
end

Base.:*(A::GPUModMatrix, B::GPUModMatrix) = stripe_mul(A, B)
Base.:*(A::GPUModMatrix, x::GPUModVector) = stripe_mul(A, x)

"""
    mul!(C, A, B; max_ops=nothing, input_bound=nothing)

Multiply modular matrices in place. All three matrices must have the same
modulus. The exact stripe budget uses the exclusive input bound and modulus.
"""
function LinearAlgebra.mul!(
    C::GPUModMatrix,
    A::GPUModMatrix,
    B::GPUModMatrix;
    max_ops = nothing,
    input_bound = nothing,
)
    modulus(C) == modulus(A) == modulus(B) ||
        throw(ModulusMismatchError("All matrices must have the same modulus N"))
    return stripe_mul!(C, A, B; max_ops, input_bound)
end

"""
    mul!(z, A, x; input_bound=nothing, modulus_override=nothing)

Multiply a modular matrix and vector in place. A positive `modulus_override`
sets the reduction modulus for this call. By default the destination and both
inputs must have the same modulus.
"""
function LinearAlgebra.mul!(
    z::GPUModVector,
    A::GPUModMatrix,
    x::GPUModVector;
    input_bound = nothing,
    modulus_override = nothing,
)
    return stripe_mul!(z, A, x; input_bound, modulus_override)
end
