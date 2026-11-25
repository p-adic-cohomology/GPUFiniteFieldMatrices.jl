function FFM._backend_available(::Val{:cuda})
    return CUDA.functional()
end

FFM._backend_instance(::Val{:cuda}) = CUDA.CUDABackend()

KernelInterface.get_backend(storage::FFM.CuStorage) =
    KernelInterface.get_backend(parent(storage))

function FFM.allocate_storage(
    backend::CUDA.CUDABackend,
    ::Type{T},
    dims::Dims;
    storage = :default,
    device = nothing,
) where {T}
    storage === :default || throw(ArgumentError("unsupported CUDA storage mode $storage"))
    requested_device = _cuda_device_index(device)
    if requested_device === nothing || KernelInterface.device(backend) == requested_device
        return CUDA.CuArray{T}(undef, dims)
    end
    return _with_cuda_device(device) do
        CUDA.CuArray{T}(undef, dims)
    end
end

function FFM.similar_storage(template::CUDA.CuArray, ::Type{T}, dims::Dims) where {T}
    return _with_cuda_device(CUDA.device(template)) do
        similar(template, T, dims)
    end
end

function _cuda_with_native_device(f::F, storage) where {F}
    native = FFM._runtime_storage_root(storage)
    return _with_cuda_device(CUDA.device(native)) do
        f(storage)
    end
end

function FFM.copy_storage!(dest::FFM.CuStorage, src)
    source = src isa FFM.CuStorage ? parent(src) : src
    return _cuda_with_native_device(parent(dest)) do native_dest
        copyto!(native_dest, source)
        dest
    end
end

function FFM.fill_storage!(dest::FFM.CuStorage, value)
    return _cuda_with_native_device(parent(dest)) do native_dest
        fill!(native_dest, value)
        dest
    end
end

function FFM.rand_storage!(dest::FFM.CuStorage, rng = nothing)
    return _cuda_with_native_device(parent(dest)) do native_dest
        if rng === nothing
            Random.rand!(native_dest)
        elseif rng isa CUDA.RNG
            Random.rand!(rng, native_dest)
        else
            throw(ArgumentError("explicit CUDA RNGs must be CUDA.RNG; got $(typeof(rng))"))
        end
        dest
    end
end

function FFM._host_scalar(value::CUDA.CuArray)
    return _with_cuda_device(CUDA.device(value)) do
        CUDA.scalar(value)
    end
end

FFM._host_scalar(storage::FFM.CuStorage) =
    FFM._host_scalar(FFM._runtime_storage_root(storage))

FFM.pinned_host_buffer(::CUDA.CUDABackend, ::Type{T}, dims::Dims) where {T} =
    CUDA.pin(Array{T}(undef, dims))

function FFM._synchronize_runtime_storage!(backend::CUDA.CUDABackend, storage::CUDA.CuArray)
    return _with_cuda_device(CUDA.device(storage)) do
        KernelInterface.synchronize(backend)
        nothing
    end
end

struct CUDAStreamContext
    streams::Vector{CuStream}
    device::CUDA.CuDevice
end

function _synchronize_owned_streams!(
    streams::Vector{CuStream};
    suppress_errors::Bool = false,
)
    first_error = nothing
    for stream in streams
        try
            synchronize(stream)
        catch err
            if first_error === nothing
                first_error = err
            end
        end
    end
    first_error === nothing && return nothing
    suppress_errors && return nothing
    throw(first_error)
end

function FFM._with_backend_streams(
    f::F,
    backend::CUDA.CUDABackend,
    storage::CUDA.CuArray,
    n::Int,
) where {F}
    target = CUDA.device(storage)
    return _with_cuda_device(target) do
        n == 1 && return f(nothing)

        streams = CuStream[]
        sizehint!(streams, n)
        for _ = 1:n
            push!(streams, CuStream())
        end
        context = CUDAStreamContext(streams, target)

        result = try
            f(context)
        catch
            _synchronize_owned_streams!(streams; suppress_errors = true)
            rethrow()
        end
        _synchronize_owned_streams!(streams)
        return result
    end
end

function FFM._with_backend_stream(f::F, context::CUDAStreamContext, index::Int) where {F}
    1 <= index <= length(context.streams) || throw(
        ArgumentError("CUDA stream index $index is outside 1:$(length(context.streams))"),
    )
    stream = context.streams[index]
    return _with_cuda_device(context.device) do
        CUDA.stream!(f, stream)
    end
end
