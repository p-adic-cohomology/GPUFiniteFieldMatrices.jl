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
    if device === nothing || KernelInterface.device(backend) == device
        return CUDA.CuArray{T}(undef, dims)
    end
    return CUDA.device!(Int(device) - 1) do
        CUDA.CuArray{T}(undef, dims)
    end
end

FFM.backendof(storage::FFM.CuStorage) = KernelInterface.get_backend(parent(storage))
FFM.backendof(storage::CUDA.CuArray) = KernelInterface.get_backend(storage)
FFM.deviceof(storage::Union{FFM.CuStorage,CUDA.CuArray}) =
    (:cuda, KernelInterface.device(FFM.backendof(storage), parent(storage)))
FFM.host_scalar(value::CUDA.CuArray) = CUDA.scalar(value)
FFM.pinned_host_buffer(::CUDA.CUDABackend, ::Type{T}, dims::Dims) where {T} =
    CUDA.pin(Array{T}(undef, dims))

function FFM.synchronize_backend!(storage::Union{FFM.CuStorage,CUDA.CuArray})
    KernelInterface.synchronize(FFM.backendof(storage))
    return nothing
end

function FFM.with_backend_streams(
    f::Function,
    storage::Union{FFM.CuStorage,CUDA.CuArray},
    n::Integer,
)
    n == 1 || throw(ArgumentError("CUDA stream support is not initialized in E1a"))
    return f(nothing)
end
