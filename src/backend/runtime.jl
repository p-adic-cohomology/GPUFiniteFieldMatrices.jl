function backendof end
function allocate_storage end
function similar_storage end
function copy_storage! end
function fill_storage! end
function rand_storage! end
function synchronize_backend! end
function deviceof end
function host_scalar end
function pinned_host_buffer end
function with_backend_streams end
function with_backend_stream end
function check_product_backend end
function vendor_mul! end

function _runtime_storage_root(storage::AbstractArray)
    native = parent(storage)
    native === storage && return storage
    return _runtime_storage_root(native)
end
_runtime_storage_root(storage) = storage

backendof(storage::AbstractArray) =
    KernelInterface.get_backend(_runtime_storage_root(storage))

function similar_storage(template::AbstractArray, ::Type{T}, dims::Dims) where {T}
    native = _runtime_storage_root(template)
    native === template && return similar(template, T, dims)
    return similar_storage(native, T, dims)
end
copy_storage!(dest, src) = copyto!(dest, src)
fill_storage!(dest, value) = fill!(dest, value)
rand_storage!(dest, rng = nothing) =
    rng === nothing ? Random.rand!(dest) : Random.rand!(rng, dest)

function _synchronize_runtime_storage!(backend::KernelInterface.Backend, storage)
    KernelInterface.synchronize(backend)
    return nothing
end

function synchronize_backend!(storage::AbstractArray)
    native = _runtime_storage_root(storage)
    backend = KernelInterface.get_backend(native)
    return _synchronize_runtime_storage!(backend, native)
end

function deviceof(storage::AbstractArray)
    native = _runtime_storage_root(storage)
    backend = KernelInterface.get_backend(native)
    return (typeof(backend), KernelInterface.device(backend, native))
end

_host_scalar(value) = value
host_scalar(value) = _host_scalar(value)
host_scalar(value::AbstractArray) = _host_scalar(value)
host_scalar(value::BackendStorage) = _host_scalar(_runtime_storage_root(value))

pinned_host_buffer(::KernelInterface.Backend, ::Type{T}, dims::Dims) where {T} =
    Array{T}(undef, dims)

function _validate_stream_count(n::Integer)
    (n isa Bool || n < 1 || n > typemax(Int)) &&
        throw(ArgumentError("stream count must be a positive integer that fits in Int"))
    return Int(n)
end

function _with_backend_streams(
    f::F,
    backend::KernelInterface.Backend,
    storage,
    n::Int,
) where {F}
    n == 1 || throw(ArgumentError("this backend supports only one stream"))
    return f(nothing)
end

function with_backend_streams(f::F, storage, n::Integer) where {F}
    count = _validate_stream_count(n)
    native = _runtime_storage_root(storage)
    backend = KernelInterface.get_backend(native)
    return _with_backend_streams(f, backend, native, count)
end

_with_backend_stream(f::F, ::Nothing, index::Int) where {F} =
    index == 1 ? f() :
    throw(ArgumentError("the selected backend supports only stream index 1"))

function with_backend_stream(f::F, stream_context, index::Integer) where {F}
    (index isa Bool || index < 1 || index > typemax(Int)) &&
        throw(ArgumentError("stream index must be a positive integer that fits in Int"))
    return _with_backend_stream(f, stream_context, Int(index))
end

check_product_backend(C, A, B) = nothing
