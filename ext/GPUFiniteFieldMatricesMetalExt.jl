module GPUFiniteFieldMatricesMetalExt

import Metal
import GPUFiniteFieldMatrices
import GPUFiniteFieldMatrices:
    _backend_available, _backend_instance, _validate_backend_storage
import KernelInterface

_backend_available(::Val{:metal}) = Metal.functional()
_backend_instance(::Val{:metal}) = Metal.MetalBackend()

function _validate_backend_storage(::Val{:metal}, ::Type{Float64}, modulus::Integer)
    throw(ArgumentError("Metal modular storage does not support Float64"))
end

function _validate_backend_storage(::Val{:metal}, ::Type{T}, modulus::Integer) where {T}
    GPUFiniteFieldMatrices._validate_storage_modulus(T, modulus)
    return nothing
end

end
