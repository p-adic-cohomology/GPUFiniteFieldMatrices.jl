module BackendTestSupport

import GPUFiniteFieldMatrices

const G = GPUFiniteFieldMatrices

"""Construct a modular array using a named backend's supported public factory."""
function modular_array(backend::Symbol, values::AbstractArray; modulus::Integer, kwargs...)
    T = eltype(values)
    D = ndims(values)
    if backend === :cuda
        return G.CuModArray{T,D}(values; modulus, kwargs...)
    elseif backend === :metal
        return G.MtlModArray{T,D}(values; modulus, kwargs...)
    end
    throw(ArgumentError("unsupported test backend $backend"))
end

"""Wait for work queued by a modular wrapper to finish."""
synchronize_array!(array) = G.synchronize_backend!(array)

"""Compute exact modular results on the host with unbounded integers."""
function exact_modular_reference(values, modulus::Integer)
    q = BigInt(modulus)
    return map(value -> mod(BigInt(value), q), values)
end

function exact_modular_product(
    left::AbstractMatrix,
    right::AbstractMatrix,
    modulus::Integer,
)
    size(left, 2) == size(right, 1) ||
        throw(DimensionMismatch("matrix inner dimensions differ"))
    q = BigInt(modulus)
    result = Matrix{BigInt}(undef, size(left, 1), size(right, 2))
    for j in axes(right, 2), i in axes(left, 1)
        value = BigInt(0)
        for k in axes(left, 2)
            value += BigInt(left[i, k]) * BigInt(right[k, j])
        end
        result[i, j] = mod(value, q)
    end
    return result
end

"""Shared modulus and floating-point exactness boundaries for backend tests."""
function precision_fixture(backend::Symbol)
    backend === :cuda && return (;
        storage_types = (Float32, Float64),
        exact_bits = (Float32 => 24, Float64 => 53),
    )
    backend === :metal &&
        return (; storage_types = (Float32,), exact_bits = (Float32 => 24,))
    throw(ArgumentError("unsupported test backend $backend"))
end

function modulus_fixture(backend::Symbol)
    precision_fixture(backend)
    return (; modulus = 7, first_unsupported_float32_modulus = BigInt(2)^24 + 1)
end

"""Return only capabilities that this test process has actually established."""
function capability_fixture(;
    cuda_loaded::Bool,
    cuda_functional::Bool,
    metal_loaded::Bool,
    metal_functional::Bool,
)
    return (
        cuda = (; loaded = cuda_loaded, functional = cuda_functional),
        metal = (; loaded = metal_loaded, functional = metal_functional),
    )
end

end
