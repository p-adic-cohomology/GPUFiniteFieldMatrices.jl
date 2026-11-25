module GPUFiniteFieldMatricesMetalExt

import Metal
import GPUFiniteFieldMatrices
import GPUFiniteFieldMatrices:
    _backend_available,
    _backend_instance,
    _validate_backend_storage,
    _validate_storage_modulus
import GPUFiniteFieldMatrices: _storage_modulus_limit
import KernelInterface

_backend_available(::Val{:metal}) = Metal.functional()
_backend_instance(::Val{:metal}) = Metal.MetalBackend()

function _validate_backend_storage(::Val{:metal}, ::Type{Float64}, modulus::Integer)
    throw(
        ArgumentError(
            "Metal modular storage does not support Float64; supported storage is Float16 (modulus ≤ 2048) or Float32 (modulus ≤ 16777216)",
        ),
    )
end

function _validate_backend_storage(::Val{:metal}, ::Type{T}, modulus::Integer) where {T}
    if !(T === Float16 || T === Float32)
        throw(
            ArgumentError(
                "Metal modular storage supports Float16 (modulus ≤ 2048) and Float32 (modulus ≤ 16777216); got $T",
            ),
        )
    end
    try
        _validate_storage_modulus(T, modulus)
    catch err
        if err isa DomainError
            limit = _storage_modulus_limit(T)
            throw(
                DomainError(
                    modulus,
                    "modulus $modulus is not exactly representable in Metal $T storage; supported maximum is $limit",
                ),
            )
        end
        rethrow()
    end
    return nothing
end

end
