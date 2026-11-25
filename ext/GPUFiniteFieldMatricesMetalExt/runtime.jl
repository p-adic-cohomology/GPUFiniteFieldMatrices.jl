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

function FFM.similar_storage(
    template::Metal.MtlArray{S,N,Mode},
    ::Type{T},
    dims::Dims{D},
) where {S,N,Mode,T,D}
    return _with_metal_device(Metal.device(template)) do
        Metal.MtlArray{T,D,Mode}(undef, dims)
    end
end

function _metal_with_native_device(f::F, storage) where {F}
    native = FFM._runtime_storage_root(storage)
    return _with_metal_device(Metal.device(native)) do
        f(storage)
    end
end

function FFM.copy_storage!(dest::FFM.MtlStorage, src)
    source = src isa FFM.MtlStorage ? parent(src) : src
    return _metal_with_native_device(parent(dest)) do native_dest
        copyto!(native_dest, source)
        dest
    end
end

function FFM.fill_storage!(dest::FFM.MtlStorage, value)
    return _metal_with_native_device(parent(dest)) do native_dest
        fill!(native_dest, value)
        dest
    end
end

function FFM.rand_storage!(dest::FFM.MtlStorage, rng = nothing)
    return _metal_with_native_device(parent(dest)) do native_dest
        if rng === nothing
            Random.rand!(Metal.default_rng(), native_dest)
        elseif rng isa typeof(Metal.default_rng())
            Random.rand!(rng, native_dest)
        else
            throw(
                ArgumentError(
                    "explicit Metal RNGs must be Metal.KernelRNG; got $(typeof(rng))",
                ),
            )
        end
        dest
    end
end

function FFM._host_scalar(value::Metal.MtlArray)
    return _with_metal_device(Metal.device(value)) do
        only(Array(value))
    end
end

FFM._host_scalar(storage::FFM.MtlStorage) =
    FFM._host_scalar(FFM._runtime_storage_root(storage))

function FFM._synchronize_runtime_storage!(::Metal.MetalBackend, storage::Metal.MtlArray)
    return _with_metal_device(Metal.device(storage)) do
        Metal.synchronize()
        nothing
    end
end

function FFM._with_backend_streams(
    f::F,
    ::Metal.MetalBackend,
    storage::Metal.MtlArray,
    n::Int,
) where {F}
    n == 1 || throw(
        ArgumentError("Metal supports only one command queue; batch_streams must equal 1"),
    )
    return _with_metal_device(Metal.device(storage)) do
        f(nothing)
    end
end
