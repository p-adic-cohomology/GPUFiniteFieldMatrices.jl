abstract type BackendStorage{T,D} <: AbstractArray{T,D} end

struct CuStorage{T,D,M,A<:AbstractArray{T,D}} <: BackendStorage{T,D}
    array::A
end

struct MtlStorage{T,D,S,A<:AbstractArray{T,D}} <: BackendStorage{T,D}
    array::A
end

Base.size(storage::BackendStorage) = size(getfield(storage, :array))
Base.axes(storage::BackendStorage) = axes(getfield(storage, :array))
Base.length(storage::BackendStorage) = length(getfield(storage, :array))
Base.IndexStyle(::Type{<:BackendStorage}) = IndexLinear()
Base.getindex(storage::BackendStorage, i::Int) = getindex(getfield(storage, :array), i)
Base.setindex!(storage::BackendStorage, value, i::Int) =
    setindex!(getfield(storage, :array), value, i)
Base.parent(storage::BackendStorage) = getfield(storage, :array)

_storage_backend(::CuStorage) = Val(:cuda)
_storage_backend(::MtlStorage) = Val(:metal)
_storage_backend(storage::SubArray) = _storage_backend(parent(storage))
_storage_backend(::Any) = nothing
