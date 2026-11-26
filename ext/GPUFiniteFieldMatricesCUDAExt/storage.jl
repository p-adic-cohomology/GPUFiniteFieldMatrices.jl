_cuda_root(array::CuArray) = array
function _cuda_root(array::AbstractArray)
    native = parent(array)
    native === array && throw(MethodError(_cuda_root, (array,)))
    return _cuda_root(native)
end
_cuda_memory_type(::CuArray{T,D,M}) where {T,D,M} = M

KernelInterface.get_backend(storage::FFM.CuStorage) =
    KernelInterface.get_backend(parent(storage))
KernelInterface.device(backend::CUDA.CUDABackend, storage::FFM.CuStorage) =
    KernelInterface.device(backend, _cuda_root(parent(storage)))

function _cuda_storage(array::AbstractArray{T,D}) where {T,D}
    memory_type = _cuda_memory_type(_cuda_root(array))
    return FFM.CuStorage{T,D,memory_type,typeof(array)}(array)
end

FFM._storage_backend(::CuArray) = Val(:cuda)

function _cuda_storage_mode(storage)
    storage === :default ||
        throw(ArgumentError("CUDA modular storage supports only the default storage mode"))
    return nothing
end

_cuda_device_index(::Nothing) = nothing

function _cuda_device_index(device::Integer)
    device isa Bool &&
        throw(ArgumentError("CUDA device indices must be one-based integers, got Bool"))
    1 <= device <= typemax(Int) || throw(
        ArgumentError("CUDA device indices must be positive and fit in Int, got $device"),
    )
    return Int(device)
end

function _cuda_device_index(device::CUDA.CuDevice)
    ordinal = CUDA.deviceid(device)
    0 <= ordinal < typemax(Int) || throw(
        ArgumentError("CUDA device handles must refer to a GPU, got device id $ordinal"),
    )
    return ordinal + 1
end

function _cuda_device_index(device)
    throw(
        ArgumentError(
            "CUDA device must be a one-based integer or CUDA.CuDevice, got $(typeof(device))",
        ),
    )
end

function _with_cuda_device(f::Function, device)
    index = _cuda_device_index(device)
    index === nothing && return f()
    target = device isa CUDA.CuDevice ? device : CUDA.CuDevice(index - 1)
    return CUDA.device!(target) do
        f()
    end
end

function _validate_backend_storage(::Val{:cuda}, ::Type{T}, modulus::Integer) where {T}
    if !(T === Float16 || T === Float32 || T === Float64)
        throw(
            ArgumentError(
                "CUDA modular storage supports Float16 (modulus ≤ 2048), Float32 (modulus ≤ 16777216), and Float64 (modulus ≤ 9007199254740992); got $T",
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
                    "modulus $modulus is not exactly representable in CUDA $T storage; supported maximum is $limit",
                ),
            )
        end
        rethrow()
    end
    return nothing
end

function _cuda_dimensions(dims::Tuple)
    return ntuple(length(dims)) do i
        dim = Int(dims[i])
        dim >= 0 || throw(
            ArgumentError("CUDA modular storage dimensions must be nonnegative, got $dims"),
        )
        dim
    end
end

function _cuda_factory_preflight(
    ::Type{T},
    dims::Tuple,
    modulus::Integer,
    logical_size = nothing,
) where {T}
    normalized_dims = _cuda_dimensions(dims)
    _validate_backend_storage(Val(:cuda), T, modulus)
    N = _validate_storage_modulus(T, modulus)
    normalized_logical_size = _normalize_logical_size(normalized_dims, logical_size)
    return normalized_dims, normalized_logical_size, N
end

function _construct_cuda_storage(
    ::Type{T},
    array::AbstractArray{T,D},
    logical_size::Dims{D},
    modulus::Integer,
) where {T,D}
    _validate_backend_storage(Val(:cuda), T, modulus)
    normalized_size = _normalize_logical_size(size(array), logical_size)
    storage = _cuda_storage(array)
    return GPUModArray{T,D,typeof(storage)}(storage, normalized_size, modulus)
end

function _wrap_cuda_storage(
    array::AbstractArray{T,D},
    new_size,
    modulus::Integer;
    storage = :default,
    device = nothing,
) where {T,D}
    requested_device = _cuda_device_index(device)
    _, dims, N = _cuda_factory_preflight(T, size(array), modulus, new_size)
    _cuda_storage_mode(storage)
    requested_device === nothing ||
        requested_device ==
        KernelInterface.device(KernelInterface.get_backend(array), _cuda_root(array)) ||
        throw(
            ArgumentError(
                "raw CUDA storage belongs to a different device than the requested device",
            ),
        )
    return _construct_cuda_storage(T, array, dims, N)
end

function (::Type{CuModArray{T,D}})(
    array::Union{CUDA.CuArray{T,D},SubArray{T,D,<:CUDA.CuArray}};
    modulus::Integer,
    mod::Bool = false,
    new_size = nothing,
    storage = :default,
    device = nothing,
) where {T,D}
    return _wrap_cuda_storage(array, new_size, modulus; storage, device)
end

function FFM._construct_modular_array(
    ::Type{T},
    values::AbstractArray{S},
    ::CUDA.CuArray;
    backend = nothing,
    modulus::Integer,
    mod::Bool = true,
    new_size = nothing,
    storage = :default,
    device = nothing,
) where {T,S}
    FFM._validate_native_wrap(T, S, backend, :cuda)
    return _wrap_cuda_storage(values, new_size, modulus; storage, device)
end

function (::Type{CuModArray{T,D}})(
    values::AbstractArray{S,D};
    modulus::Integer,
    mod::Bool = true,
    new_size = nothing,
    storage = :default,
    device = nothing,
) where {T,D,S}
    _cuda_device_index(device)
    source_backend = FFM._storage_backend(values)
    if source_backend !== nothing
        source_backend === Val(:cuda) || throw(
            ArgumentError(
                "CuModArray cannot wrap storage from the $(FFM._backend_symbol(source_backend)) backend",
            ),
        )
        T === S || throw(
            ArgumentError(
                "raw CUDA storage has element type $S and cannot be converted to $T without copying",
            ),
        )
        return _wrap_cuda_storage(values, new_size, modulus; storage, device)
    end
    _cuda_storage_mode(storage)
    dims, logical_size, N = _cuda_factory_preflight(T, size(values), modulus, new_size)
    padded = FFM._padded_size(dims)
    converted = FFM._modular_host_values(T, values, N, mod)
    data = _with_cuda_device(device) do
        data = CUDA.zeros(T, padded...)
        if !isempty(values)
            inds = CartesianIndices(values)
            copyto!(data, inds, converted, inds)
        end
        data
    end
    return _construct_cuda_storage(T, data, logical_size, N)
end

function (::Type{CuModArray})(
    values::AbstractArray{S,D};
    modulus::Integer,
    mod::Bool = true,
    new_size = nothing,
    storage = :default,
    device = nothing,
) where {S,D}
    return CuModArray{DEFAULT_TYPE,D}(values; modulus, mod, new_size, storage, device)
end

function (::Type{CuModArray})(array::CUDA.CuArray{T,D,M}; kwargs...) where {T,D,M}
    return CuModArray{T,D}(array; kwargs...)
end

function (::Type{CuModArray})(array::SubArray{T,D,P}; kwargs...) where {T,D,P<:CUDA.CuArray}
    return CuModArray{T,D}(array; kwargs...)
end

function (::Type{CuModArray})(array::SubArray{T,D}; kwargs...) where {T,D}
    if FFM._storage_backend(array) === Val(:cuda)
        return CuModArray{T,D}(array; kwargs...)
    end
    return CuModArray{DEFAULT_TYPE,D}(array; kwargs...)
end

CuModMatrix(values::AbstractMatrix; kwargs...) =
    CuModArray{DEFAULT_TYPE,2}(values; kwargs...)
CuModVector(values::AbstractVector; kwargs...) =
    CuModArray{DEFAULT_TYPE,1}(values; kwargs...)

CuModMatrix(array::CUDA.CuArray{T,2}; kwargs...) where {T} =
    CuModArray{T,2}(array; kwargs...)
CuModVector(array::CUDA.CuArray{T,1}; kwargs...) where {T} =
    CuModArray{T,1}(array; kwargs...)
CuModMatrix(array::SubArray{T,2,P}; kwargs...) where {T,P<:CUDA.CuArray} =
    CuModArray{T,2}(array; kwargs...)
CuModVector(array::SubArray{T,1,P}; kwargs...) where {T,P<:CUDA.CuArray} =
    CuModArray{T,1}(array; kwargs...)
function CuModMatrix(array::SubArray{T,2}; kwargs...) where {T}
    if FFM._storage_backend(array) === Val(:cuda)
        return CuModArray{T,2}(array; kwargs...)
    end
    return CuModArray{DEFAULT_TYPE,2}(array; kwargs...)
end
function CuModVector(array::SubArray{T,1}; kwargs...) where {T}
    if FFM._storage_backend(array) === Val(:cuda)
        return CuModArray{T,1}(array; kwargs...)
    end
    return CuModArray{DEFAULT_TYPE,1}(array; kwargs...)
end

function (::Type{CuModArray{T,2}})(
    ::LinearAlgebra.UniformScaling,
    rows::Integer,
    cols::Integer;
    modulus::Integer,
) where {T}
    return CuModArray{T,2}(Matrix{T}(I, rows, cols); modulus)
end

identity_matrix(::Type{CuModArray{T,2}}, n::Integer; modulus::Integer) where {T} =
    CuModArray{T,2}(I, n, n; modulus)
