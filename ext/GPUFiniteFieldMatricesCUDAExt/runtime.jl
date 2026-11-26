# CUDA-specific stream orchestration for batched fallback kernels. The package
# core exposes no backend runtime protocol; native CUDA device and stream APIs
# stay inside this extension.
struct _CUDAStreamBatch
    streams::Vector{CuStream}
    device::CUDA.CuDevice
end

function _sync_cuda_streams!(streams::Vector{CuStream}; suppress_errors::Bool = false)
    first_error = nothing
    for stream in streams
        try
            synchronize(stream)
        catch err
            first_error === nothing && (first_error = err)
        end
    end
    first_error === nothing && return nothing
    suppress_errors && return nothing
    throw(first_error)
end

function _with_cuda_streams(f::F, storage, n::Int) where {F}
    n > 0 || throw(ArgumentError("CUDA stream count must be positive"))
    native = _cuda_root(storage)
    target = CUDA.device(native)
    return _with_cuda_device(target) do
        # A single stream uses the caller's current CUDA task stream. It needs
        # no package-wide fence or owned stream lifecycle.
        n == 1 && return f(nothing)

        streams = CuStream[]
        sizehint!(streams, n)
        try
            for _ = 1:n
                push!(streams, CuStream())
            end
            context = _CUDAStreamBatch(streams, target)
            result = f(context)
            _sync_cuda_streams!(streams)
            return result
        catch
            # CuStream carries CUDA's native finalizer. Synchronize queued work
            # before the owned stream vector leaves scope on both exit paths.
            _sync_cuda_streams!(streams; suppress_errors = true)
            rethrow()
        end
    end
end

function _with_cuda_stream(f::F, context::_CUDAStreamBatch, index::Int) where {F}
    1 <= index <= length(context.streams) || throw(
        ArgumentError("CUDA stream index $index is outside 1:$(length(context.streams))"),
    )
    stream = context.streams[index]
    return _with_cuda_device(context.device) do
        CUDA.stream!(f, stream)
    end
end

function _with_cuda_stream(f::F, ::Nothing, index::Int) where {F}
    index == 1 || throw(ArgumentError("the current CUDA stream has only index 1"))
    return f()
end
