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

function FFM._copy_storage!(::Metal.MetalBackend, dest, src)
    native_dest = FFM._runtime_storage_value(dest)
    native_source = FFM._runtime_storage_value(src)
    return _metal_with_native_device(native_dest) do selected_dest
        copyto!(selected_dest, native_source)
        dest
    end
end

function FFM._fill_storage!(::Metal.MetalBackend, dest, value)
    native_dest = FFM._runtime_storage_value(dest)
    return _metal_with_native_device(native_dest) do selected_dest
        fill!(selected_dest, value)
        dest
    end
end

function FFM._rand_storage!(::Metal.MetalBackend, dest, rng)
    supported_rng_type = typeof(Metal.default_rng())
    if rng !== nothing && !(rng isa supported_rng_type)
        throw(
            ArgumentError(
                "explicit Metal RNGs must use Metal.default_rng() (type $supported_rng_type); got $(typeof(rng))",
            ),
        )
    end
    native_dest = FFM._runtime_storage_value(dest)
    return _metal_with_native_device(native_dest) do selected_dest
        selected_rng = rng === nothing ? Metal.default_rng() : rng
        Random.rand!(selected_rng, selected_dest)
        dest
    end
end

function FFM._host_scalar(::Metal.MetalBackend, value::AbstractArray)
    native = FFM._runtime_storage_root(value)
    return _with_metal_device(Metal.device(native)) do
        if value isa Metal.MtlArray
            return only(Array(value))
        end
        staging = FFM.similar_storage(native, eltype(value), size(value))
        FFM.copy_storage!(staging, value)
        only(Array(staging))
    end
end

FFM._host_scalar(backend::Metal.MetalBackend, value::LinearAlgebra.Transpose) =
    FFM._host_scalar(backend, parent(value))
FFM._host_scalar(backend::Metal.MetalBackend, value::LinearAlgebra.Adjoint) =
    adjoint(FFM._host_scalar(backend, parent(value)))

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
