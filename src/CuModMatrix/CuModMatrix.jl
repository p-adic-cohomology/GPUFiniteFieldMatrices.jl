const TILE_WIDTH = 32
const DEFAULT_TYPE = Float32

@inline _padded_extent(d::Integer) = cld(d, TILE_WIDTH) * TILE_WIDTH
@inline _padded_size(dims::Dims{D}) where {D} = ntuple(i -> _padded_extent(dims[i]), D)

@inline function _logical_view(data::AbstractArray{T,D}, dims::Dims{D}) where {T,D}
    return view(data, ntuple(i -> 1:dims[i], D)...)
end

"""Operand moduli are incompatible for the requested operation."""
struct ModulusMismatchError <: Exception
    message::String
end

Base.showerror(io::IO, err::ModulusMismatchError) = print(io, err.message)

function _storage_modulus_limit(::Type{Float16})
    return BigInt(1) << 11
end

function _storage_modulus_limit(::Type{Float32})
    return BigInt(1) << 24
end

function _storage_modulus_limit(::Type{Float64})
    return BigInt(1) << 53
end

function _storage_modulus_limit(::Type{T}) where {T<:Integer}
    return BigInt(typemax(T)) + 1
end

_storage_modulus_limit(::Type{T}) where {T} =
    throw(ArgumentError("unsupported modular storage element type $T"))

function _validate_storage_modulus(::Type{T}, modulus::Integer) where {T}
    modulus > 0 || throw(ArgumentError("modulus must be positive, got $modulus"))
    limit = _storage_modulus_limit(T)
    BigInt(modulus) <= limit || throw(
        DomainError(
            modulus,
            "modulus $modulus is not exactly representable in $T storage; supported maximum is $limit",
        ),
    )
    modulus <= typemax(Int) || throw(OverflowError("modulus $modulus does not fit Int"))
    return Int(modulus)
end

function _normalize_logical_size(backing_dims::Dims{D}, requested) where {D}
    requested === nothing && return backing_dims
    requested isa Tuple ||
        throw(ArgumentError("logical size must be a tuple, got $(typeof(requested))"))
    length(requested) == D || throw(
        DimensionMismatch(
            "logical dimensions $(length(requested)) do not match array rank $D",
        ),
    )
    logical_dims = ntuple(i -> Int(requested[i]), D)
    all(i -> 0 <= logical_dims[i] <= backing_dims[i], 1:D) || throw(
        DimensionMismatch(
            "logical dimensions $logical_dims must fit backing storage $backing_dims",
        ),
    )
    return logical_dims
end

_validate_backend_storage(::Any, ::Type, ::Integer) = nothing

function _modular_host_values(
    ::Type{T},
    values::AbstractArray,
    modulus::Int,
    normalize::Bool,
) where {T}
    normalize || return convert.(T, values)

    if eltype(values) <: Integer
        big_modulus = BigInt(modulus)
        return map(value -> convert(T, mod(BigInt(value), big_modulus)), values)
    end

    return map(value -> convert(T, mod(value, modulus)), values)
end

"""
    GPUModArray{T,D,A}

Modular array metadata over storage `A`. GPU-specific storage is wrapped in an
internal adapter so the core package can define the public type without loading
CUDA or Metal.
"""
struct GPUModArray{T,D,A<:AbstractArray{T,D}} <: AbstractArray{T,D}
    storage::A
    logical_size::Dims{D}
    N::Int

    function GPUModArray{T,D,A}(
        storage::A,
        logical_size::Dims{D},
        modulus::Integer,
    ) where {T,D,A<:AbstractArray{T,D}}
        backend = _storage_backend(storage)
        backend === nothing || _validate_backend_storage(backend, T, modulus)
        N = _validate_storage_modulus(T, modulus)
        normalized_size = _normalize_logical_size(size(storage), logical_size)
        return new{T,D,A}(storage, normalized_size, N)
    end
end

const GPUModMatrix{T} = GPUModArray{T,2}
const GPUModVector{T} = GPUModArray{T,1}
const CuModArray{T,D,M} = GPUModArray{T,D,A} where {A<:CuStorage{T,D,M}}
const CuModMatrix{T} = CuModArray{T,2}
const CuModVector{T} = CuModArray{T,1}

const MtlModArray{T,D,S} = GPUModArray{T,D,A} where {A<:MtlStorage{T,D,S}}
const MtlModMatrix{T} = MtlModArray{T,2}
const MtlModVector{T} = MtlModArray{T,1}

function _requested_backend(backend)
    backend === nothing && return default_backend()
    backend === :auto && return default_backend()
    backend isa Symbol || throw(ArgumentError("backend must be :cuda, :metal, or :auto"))
    return _check_backend_symbol(backend)
end

function _constructor_backend(values::AbstractArray, backend)
    source = _storage_backend(values)
    if source !== nothing
        source_backend = _backend_symbol(source)
        if backend !== nothing && backend !== :auto
            requested = _requested_backend(backend)
            requested === source_backend || throw(
                ArgumentError(
                    "cannot wrap $source_backend storage with backend=$requested; raw backend storage is wrapped without migration",
                ),
            )
        end
        return source_backend, true
    end
    selected = _requested_backend(backend)
    _require_backend(selected)
    return selected, false
end

function _construct_backend_array(
    ::Val{:cuda},
    ::Type{T},
    values::AbstractArray{S,D};
    kwargs...,
) where {T,S,D}
    return CuModArray{T,D}(values; kwargs...)
end

function _construct_backend_array(
    ::Val{:metal},
    ::Type{T},
    values::AbstractArray{S,D};
    kwargs...,
) where {T,S,D}
    return MtlModArray{T,D}(values; kwargs...)
end

function _construct_modular_array(
    ::Type{T},
    values::AbstractArray{S,D};
    backend = nothing,
    modulus::Integer,
    mod::Bool = true,
    new_size = nothing,
    storage = :default,
    device = nothing,
) where {T,S,D}
    selected, is_backend_storage = _constructor_backend(values, backend)
    if is_backend_storage && T !== S
        throw(
            ArgumentError(
                "raw $selected storage has element type $S and cannot be converted to $T without copying",
            ),
        )
    end
    return _construct_backend_array(
        Val(selected),
        T,
        values;
        modulus,
        mod = is_backend_storage ? false : mod,
        new_size,
        storage,
        device,
    )
end

function GPUModArray(
    values::AbstractArray{T,D};
    modulus::Integer,
    new_size = nothing,
    backend = nothing,
    mod::Bool = true,
    storage = :default,
    device = nothing,
) where {T,D}
    return _construct_modular_array(
        T,
        values;
        modulus,
        new_size,
        backend,
        mod,
        storage,
        device,
    )
end

function (::Type{GPUModArray{T,D}})(
    values::AbstractArray{S,D};
    modulus::Integer,
    new_size = nothing,
    backend = nothing,
    mod::Bool = true,
    storage = :default,
    device = nothing,
) where {T,S,D}
    return _construct_modular_array(
        T,
        values;
        modulus,
        new_size,
        backend,
        mod,
        storage,
        device,
    )
end

_backend_zeros(::Val{:cuda}, ::Type{T}, dims::Dims{D}; kwargs...) where {T,D} =
    Base.zeros(CuModArray{T,D}, dims...; kwargs...)
_backend_zeros(::Val{:metal}, ::Type{T}, dims::Dims{D}; kwargs...) where {T,D} =
    Base.zeros(MtlModArray{T,D}, dims...; kwargs...)

_backend_rand(::Val{:cuda}, ::Type{T}, dims::Dims{D}; kwargs...) where {T,D} =
    Base.rand(CuModArray{T,D}, dims...; kwargs...)
_backend_rand(::Val{:metal}, ::Type{T}, dims::Dims{D}; kwargs...) where {T,D} =
    Base.rand(MtlModArray{T,D}, dims...; kwargs...)

function Base.zeros(
    ::Type{GPUModArray{T,D}},
    dims::Vararg{Integer,D};
    modulus::Integer,
    backend = nothing,
    kwargs...,
) where {T,D}
    selected = _requested_backend(backend)
    _require_backend(selected)
    return _backend_zeros(
        Val(selected),
        T,
        ntuple(i -> Int(dims[i]), D);
        modulus,
        kwargs...,
    )
end

function Base.rand(
    ::Type{GPUModArray{T,D}},
    dims::Vararg{Integer,D};
    modulus::Integer,
    backend = nothing,
    kwargs...,
) where {T,D}
    selected = _requested_backend(backend)
    _require_backend(selected)
    return _backend_rand(Val(selected), T, ntuple(i -> Int(dims[i]), D); modulus, kwargs...)
end

(::Type{GPUModMatrix})(storage::AbstractMatrix; kwargs...) = GPUModArray(storage; kwargs...)
(::Type{GPUModVector})(storage::AbstractVector; kwargs...) = GPUModArray(storage; kwargs...)

"""Return the modulus associated with `A`."""
modulus(A::GPUModArray) = getfield(A, :N)

_native_storage(storage::BackendStorage) = parent(storage)
_native_storage(storage) = storage

"""Return the native storage or view backing `A`, preserving object identity."""
Base.parent(A::GPUModArray) = _native_storage(getfield(A, :storage))
backendof(A::GPUModArray) = backendof(getfield(A, :storage))
deviceof(A::GPUModArray) = deviceof(getfield(A, :storage))

Base.size(A::GPUModArray) = getfield(A, :logical_size)
Base.size(A::GPUModArray, i::Int) = i > ndims(A) ? 1 : size(A)[i]
Base.axes(A::GPUModArray) = ntuple(i -> Base.OneTo(size(A, i)), ndims(A))
Base.IndexStyle(::Type{<:GPUModArray}) = IndexLinear()

@inline function Base.getindex(A::GPUModArray, i::Int)
    return getindex(_logical_view(parent(A), size(A)), i)
end

@inline function Base.getindex(A::GPUModArray{T,D}, I::Vararg{Int,D}) where {T,D}
    return getindex(_logical_view(parent(A), size(A)), I...)
end

@inline function Base.setindex!(A::GPUModArray, value, i::Int)
    setindex!(_logical_view(parent(A), size(A)), mod(value, modulus(A)), i)
    return A
end

@inline function Base.setindex!(A::GPUModArray{T,D}, value, I::Vararg{Int,D}) where {T,D}
    setindex!(_logical_view(parent(A), size(A)), mod(value, modulus(A)), I...)
    return A
end

Base.length(A::GPUModArray) = prod(size(A))

function Base.Array(A::GPUModArray)
    return Array(_logical_view(parent(A), size(A)))
end

function Base.getproperty(A::GPUModArray, name::Symbol)
    name === :data && return parent(A)
    name === :size && return getfield(A, :logical_size)
    return getfield(A, name)
end

function Base.show(io::IO, A::GPUModArray)
    print(io, nameof(typeof(A)), "(", join(size(A), "×"), "; modulus=", modulus(A), ")")
end

function Base.show(io::IO, ::MIME"text/plain", A::GPUModArray)
    show(io, A)
    print(io, '\n')
    show(io, Array(A))
end
