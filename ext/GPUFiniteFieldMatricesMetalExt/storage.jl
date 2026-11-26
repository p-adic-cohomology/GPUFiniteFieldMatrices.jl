function _metal_root(array::Metal.MtlArray)
    return array
end

function _metal_root(array::AbstractArray)
    native = parent(array)
    native === array && throw(MethodError(_metal_root, (array,)))
    return _metal_root(native)
end

_metal_modeof(::Metal.MtlArray{T,D,S}) where {T,D,S} = S
function _metal_modeof(array::AbstractArray)
    native = parent(array)
    native === array && throw(MethodError(_metal_modeof, (array,)))
    return _metal_modeof(native)
end

FFM._storage_backend(::Metal.MtlArray) = Val(:metal)

KernelInterface.get_backend(storage::FFM.MtlStorage) =
    KernelInterface.get_backend(parent(storage))
KernelInterface.device(backend::Metal.MetalBackend, storage::FFM.MtlStorage) =
    _metal_device_index(Metal.device(_metal_root(parent(storage))))

function _metal_mode(storage)
    storage === :default && return :default
    (storage === Metal.SharedStorage || storage === Metal.PrivateStorage) && return storage
    throw(
        ArgumentError(
            "Metal storage must be :default, Metal.SharedStorage, or Metal.PrivateStorage",
        ),
    )
end

_metal_mode(::Type{Metal.SharedStorage}) = Metal.SharedStorage
_metal_mode(::Type{Metal.PrivateStorage}) = Metal.PrivateStorage

function _checked_metal_mode(storage)
    (storage === Metal.SharedStorage || storage === Metal.PrivateStorage) || throw(
        ArgumentError(
            "Metal modular arrays support only shared or private storage, got $storage",
        ),
    )
    return storage
end

_checked_metal_mode(::Type{Metal.SharedStorage}) = Metal.SharedStorage
_checked_metal_mode(::Type{Metal.PrivateStorage}) = Metal.PrivateStorage

function _with_metal_device(f::Function, device)
    device === nothing && return f()
    device isa typeof(Metal.device()) || throw(
        ArgumentError("Metal device must be a Metal.MTLDevice, got $(typeof(device))"),
    )
    previous = Metal.device()
    Metal.device!(device)
    try
        return f()
    finally
        Metal.device!(previous)
    end
end

function _metal_zeros(::Type{T}, dims::Dims, storage::Symbol) where {T}
    storage === :default || throw(ArgumentError("unsupported Metal storage mode $storage"))
    return Metal.zeros(T, dims...)
end

_metal_zeros(::Type{T}, dims::Dims, ::Type{Metal.SharedStorage}) where {T} =
    Metal.zeros(T, dims...; storage = Metal.SharedStorage)
_metal_zeros(::Type{T}, dims::Dims, ::Type{Metal.PrivateStorage}) where {T} =
    Metal.zeros(T, dims...; storage = Metal.PrivateStorage)

function _metal_rand(::Type{T}, dims::Dims, storage::Symbol) where {T}
    storage === :default || throw(ArgumentError("unsupported Metal storage mode $storage"))
    return Metal.rand(T, dims...)
end

_metal_rand(::Type{T}, dims::Dims, ::Type{Metal.SharedStorage}) where {T} =
    Metal.rand(T, dims...; storage = Metal.SharedStorage)
_metal_rand(::Type{T}, dims::Dims, ::Type{Metal.PrivateStorage}) where {T} =
    Metal.rand(T, dims...; storage = Metal.PrivateStorage)

function _metal_storage(array::AbstractArray{T,D}) where {T,D}
    root = _metal_root(array)
    mode = _checked_metal_mode(_metal_modeof(root))
    return FFM.MtlStorage{T,D,mode,typeof(array)}(array)
end

function _wrap_metal_storage(
    array::AbstractArray{T,D},
    logical_size,
    modulus::Integer;
    storage = :default,
    device = nothing,
) where {T,D}
    native = _metal_root(array)
    mode = _checked_metal_mode(_metal_modeof(native))
    requested_mode = _metal_mode(storage)
    requested_mode === :default ||
        requested_mode === mode ||
        throw(
            ArgumentError(
                "raw Metal storage uses $mode; wrapping cannot change it to $requested_mode",
            ),
        )
    native_device = Metal.device(native)
    device === nothing ||
        device == native_device ||
        throw(
            ArgumentError(
                "raw Metal storage belongs to a different device than the requested device",
            ),
        )
    wrapped_storage = _metal_storage(array)
    return GPUModArray{T,D,typeof(wrapped_storage)}(
        wrapped_storage,
        FFM._normalize_logical_size(size(array), logical_size),
        modulus,
    )
end

function _metal_dimensions(dims::Tuple)
    return ntuple(length(dims)) do i
        dim = Int(dims[i])
        dim >= 0 || throw(
            ArgumentError(
                "Metal modular storage dimensions must be nonnegative, got $dims",
            ),
        )
        dim
    end
end

function _metal_factory_preflight(
    ::Type{T},
    dims::Tuple,
    modulus::Integer,
    logical_size = nothing,
) where {T}
    normalized_dims = _metal_dimensions(dims)
    _validate_backend_storage(Val(:metal), T, modulus)
    N = FFM._validate_storage_modulus(T, modulus)
    normalized_logical_size = FFM._normalize_logical_size(normalized_dims, logical_size)
    return normalized_dims, normalized_logical_size, N
end

function _construct_metal_host(
    ::Type{T},
    values::AbstractArray{S,D},
    logical_size,
    modulus::Integer;
    mod::Bool = true,
    storage = :default,
    device = nothing,
) where {T,S,D}
    dims, logical_dims, N = _metal_factory_preflight(T, size(values), modulus, logical_size)
    Metal.functional() ||
        throw(ArgumentError("the selected Metal backend is not functional"))
    converted = FFM._modular_host_values(T, values, N, mod)
    padded = FFM._padded_size(dims)
    data = _with_metal_device(device) do
        result = _metal_zeros(T, padded, storage)
        if !isempty(values)
            copyto!(FFM._logical_view(result, dims), converted)
        end
        result
    end
    return _wrap_metal_storage(data, logical_dims, N)
end

function FFM._construct_modular_array(
    ::Type{T},
    values::AbstractArray{S},
    ::Metal.MtlArray;
    backend = nothing,
    modulus::Integer,
    mod::Bool = true,
    new_size = nothing,
    storage = :default,
    device = nothing,
) where {T,S}
    FFM._validate_native_wrap(T, S, backend, :metal)
    return _wrap_metal_storage(values, new_size, modulus; storage, device)
end

function (::Type{MtlModArray{T,D}})(
    values::AbstractArray{S,D};
    modulus::Integer,
    mod::Bool = true,
    new_size = nothing,
    storage = :default,
    device = nothing,
) where {T,S,D}
    source_backend = FFM._storage_backend(values)
    if source_backend !== nothing
        source_backend === Val(:metal) || throw(
            ArgumentError(
                "MtlModArray cannot wrap storage from the $(FFM._backend_symbol(source_backend)) backend",
            ),
        )
        T === S || throw(
            ArgumentError(
                "raw Metal storage has element type $S and cannot be converted to $T without copying",
            ),
        )
        return _wrap_metal_storage(values, new_size, modulus; storage, device)
    end
    return _construct_metal_host(T, values, new_size, modulus; mod, storage, device)
end

function (::Type{MtlModArray})(values::AbstractArray{S,D}; kwargs...) where {S,D}
    return MtlModArray{S,D}(values; kwargs...)
end

MtlModMatrix(values::AbstractMatrix; kwargs...) =
    MtlModArray{DEFAULT_TYPE,2}(values; kwargs...)
MtlModVector(values::AbstractVector; kwargs...) =
    MtlModArray{DEFAULT_TYPE,1}(values; kwargs...)
MtlModMatrix(array::Metal.MtlArray{T,2}; kwargs...) where {T} =
    MtlModArray{T,2}(array; kwargs...)
MtlModVector(array::Metal.MtlArray{T,1}; kwargs...) where {T} =
    MtlModArray{T,1}(array; kwargs...)
function MtlModMatrix(array::SubArray{T,2}; kwargs...) where {T}
    if FFM._storage_backend(array) === Val(:metal)
        return MtlModArray{T,2}(array; kwargs...)
    end
    return MtlModArray{DEFAULT_TYPE,2}(array; kwargs...)
end
function MtlModVector(array::SubArray{T,1}; kwargs...) where {T}
    if FFM._storage_backend(array) === Val(:metal)
        return MtlModArray{T,1}(array; kwargs...)
    end
    return MtlModArray{DEFAULT_TYPE,1}(array; kwargs...)
end

function (::Type{MtlModArray{T,2}})(
    ::UniformScaling,
    rows::Integer,
    cols::Integer;
    modulus::Integer,
) where {T}
    return MtlModArray{T,2}(Matrix{T}(I, rows, cols); modulus)
end

identity_matrix(::Type{MtlModArray{T,2}}, n::Integer; modulus::Integer) where {T} =
    MtlModArray{T,2}(I, n, n; modulus)

function Base.zeros(
    ::Type{MtlModArray{T,D}},
    dims::Vararg{Integer,D};
    modulus::Integer,
    new_size = nothing,
    storage = :default,
    device = nothing,
) where {T,D}
    normalized_dims, logical_dims, N = _metal_factory_preflight(T, dims, modulus, new_size)
    Metal.functional() ||
        throw(ArgumentError("the selected Metal backend is not functional"))
    data = _with_metal_device(device) do
        _metal_zeros(T, FFM._padded_size(normalized_dims), storage)
    end
    return _wrap_metal_storage(data, logical_dims, N)
end

function Base.rand(
    ::Type{MtlModArray{T,D}},
    dims::Vararg{Integer,D};
    modulus::Integer,
    new_size = nothing,
    storage = :default,
    device = nothing,
) where {T,D}
    normalized_dims, logical_dims, N = _metal_factory_preflight(T, dims, modulus, new_size)
    Metal.functional() ||
        throw(ArgumentError("the selected Metal backend is not functional"))
    padded = FFM._padded_size(normalized_dims)
    data = _with_metal_device(device) do
        result = _metal_zeros(T, padded, storage)
        if !any(iszero, logical_dims)
            random_values = _metal_rand(T, logical_dims, storage)
            if T <: AbstractFloat
                random_values .= mod.(floor.(random_values .* N), N)
            end
            copyto!(FFM._logical_view(result, logical_dims), random_values)
        end
        result
    end
    return _wrap_metal_storage(data, logical_dims, N)
end

function _similar_metal(A::MtlModArray{T,D,S}, modulus::Integer) where {T,D,S}
    dims, logical_dims, N = _metal_factory_preflight(T, size(A), modulus, size(A))
    device = Metal.device(_metal_root(parent(A)))
    data = _with_metal_device(device) do
        _metal_zeros(T, FFM._padded_size(dims), S)
    end
    return _wrap_metal_storage(data, logical_dims, N)
end

Base.similar(A::MtlModArray) = _similar_metal(A, modulus(A))
similar_with_modulus(A::MtlModArray, N::Integer) = _similar_metal(A, N)
