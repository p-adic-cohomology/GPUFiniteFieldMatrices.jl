function _metal_undef(::Type{T}, dims::Dims{D}, storage::Symbol) where {T,D}
    storage === :default || throw(ArgumentError("unsupported Metal storage mode $storage"))
    return Metal.MtlArray{T,D}(undef, dims)
end

_metal_undef(::Type{T}, dims::Dims{D}, ::Type{Metal.SharedStorage}) where {T,D} =
    Metal.MtlArray{T,D,Metal.SharedStorage}(undef, dims)
_metal_undef(::Type{T}, dims::Dims{D}, ::Type{Metal.PrivateStorage}) where {T,D} =
    Metal.MtlArray{T,D,Metal.PrivateStorage}(undef, dims)

function FFM.allocate_storage(
    ::Metal.MetalBackend,
    ::Type{T},
    dims::Dims;
    storage = :default,
    device = nothing,
) where {T}
    normalized_dims = _metal_dimensions(dims)
    return _with_metal_device(device) do
        _metal_undef(T, normalized_dims, storage)
    end
end

FFM.backendof(::FFM.MtlStorage) = Metal.MetalBackend()
FFM.backendof(::Metal.MtlArray) = Metal.MetalBackend()
function FFM.backendof(array::SubArray)
    try
        _metal_root(array)
        return Metal.MetalBackend()
    catch err
        err isa MethodError || rethrow()
        return invoke(FFM.backendof, Tuple{AbstractArray}, array)
    end
end

function FFM.deviceof(storage::FFM.MtlStorage)
    return (:metal, Metal.device(_metal_root(parent(storage))))
end

function FFM.deviceof(storage::Metal.MtlArray)
    return (:metal, Metal.device(storage))
end

function FFM.deviceof(storage::SubArray)
    try
        return (:metal, Metal.device(_metal_root(storage)))
    catch err
        err isa MethodError || rethrow()
        return invoke(FFM.deviceof, Tuple{AbstractArray}, storage)
    end
end

function FFM.similar_storage(
    template::Metal.MtlArray{S,N,Mode},
    ::Type{T},
    dims::Dims{D},
) where {S,N,Mode,T,D}
    return _with_metal_device(Metal.device(template)) do
        Metal.MtlArray{T,D,Mode}(undef, dims)
    end
end

function FFM.similar_storage(
    template::SubArray{S,N,P},
    ::Type{T},
    dims::Dims{D},
) where {S,N,P<:Metal.MtlArray,T,D}
    return FFM.similar_storage(_metal_root(template), T, dims)
end

function FFM.copy_storage!(dest::FFM.MtlStorage, src)
    source = src isa FFM.MtlStorage ? parent(src) : src
    copyto!(parent(dest), source)
    return dest
end

function FFM.fill_storage!(dest::FFM.MtlStorage, value)
    fill!(parent(dest), value)
    return dest
end

function FFM.synchronize_backend!(storage::Union{FFM.MtlStorage,Metal.MtlArray,SubArray})
    array = storage isa FFM.MtlStorage ? parent(storage) : storage
    native = try
        _metal_root(array)
    catch err
        err isa MethodError || rethrow()
        return invoke(FFM.synchronize_backend!, Tuple{AbstractArray}, array)
    end
    return _with_metal_device(Metal.device(native)) do
        Metal.synchronize()
        nothing
    end
end
