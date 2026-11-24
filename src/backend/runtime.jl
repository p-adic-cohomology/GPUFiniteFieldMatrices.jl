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

backendof(storage::BackendStorage) = KernelInterface.get_backend(parent(storage))
backendof(storage::AbstractArray) = KernelInterface.get_backend(storage)

similar_storage(template::AbstractArray, ::Type{T}, dims::Dims) where {T} =
    similar(parent(template), T, dims)
copy_storage!(dest, src) = copyto!(dest, src)
fill_storage!(dest, value) = fill!(dest, value)
rand_storage!(dest, rng = nothing) = rng === nothing ? rand!(dest) : rand!(rng, dest)
synchronize_backend!(storage::AbstractArray) =
    KernelInterface.synchronize(backendof(storage))
deviceof(storage::AbstractArray) = KernelInterface.device(backendof(storage))
host_scalar(value) = value
pinned_host_buffer(::KernelInterface.Backend, ::Type{T}, dims::Dims) where {T} =
    Array{T}(undef, dims)

function with_backend_streams(f::Function, storage, n::Integer)
    n == 1 || throw(ArgumentError("this backend supports only one stream"))
    return f(nothing)
end

with_backend_stream(f::Function, stream_context, index::Integer) = f()
check_product_backend(C, A, B) = nothing
