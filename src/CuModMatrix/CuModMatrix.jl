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
    modulus <= typemax(Int) || throw(OverflowError("modulus $modulus does not fit Int"))
    limit = _storage_modulus_limit(T)
    BigInt(modulus) <= limit || throw(
        DomainError(
            modulus,
            "modulus $modulus is not exactly representable in $T storage; supported maximum is $limit",
        ),
    )
    return Int(modulus)
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
        N = _validate_storage_modulus(T, modulus)
        all(i -> 0 <= logical_size[i] <= size(storage, i), 1:D) || throw(
            DimensionMismatch(
                "logical dimensions $logical_size must fit backing storage $(size(storage))",
            ),
        )
        _validate_backend_storage(_storage_backend(storage), T, N)
        return new{T,D,A}(storage, logical_size, N)
    end
end

const CuModArray{T,D,M} = GPUModArray{T,D,A} where {A<:CuStorage{T,D,M}}
const CuModMatrix{T} = CuModArray{T,2}
const CuModVector{T} = CuModArray{T,1}

const MtlModArray{T,D,S} = GPUModArray{T,D,A} where {A<:MtlStorage{T,D,S}}
const MtlModMatrix{T} = MtlModArray{T,2}
const MtlModVector{T} = MtlModArray{T,1}

GPUModArray(storage::AbstractArray{T,D}; modulus::Integer) where {T,D} =
    GPUModArray{T,D,typeof(storage)}(storage, size(storage), modulus)

"""Return the modulus associated with `A`."""
modulus(A::GPUModArray) = getfield(A, :N)

"""Return the native storage or view backing `A`, preserving object identity."""
Base.parent(A::GPUModArray) = parent(getfield(A, :storage))

Base.size(A::GPUModArray) = getfield(A, :logical_size)
Base.size(A::GPUModArray, i::Int) = size(A)[i]
Base.axes(A::GPUModArray) = ntuple(i -> Base.OneTo(size(A, i)), ndims(A))
Base.IndexStyle(::Type{<:GPUModArray}) = IndexLinear()
Base.getindex(A::GPUModArray, i::Int) = getindex(parent(A), i)
Base.setindex!(A::GPUModArray, value, i::Int) = setindex!(parent(A), value, i)
Base.length(A::GPUModArray) = prod(size(A))

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
