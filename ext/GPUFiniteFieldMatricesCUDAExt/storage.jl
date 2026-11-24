function _cuda_storage(array::CuArray{T,D,M}) where {T,D,M}
    return FFM.CuStorage{T,D,M,typeof(array)}(array)
end

function _validate_backend_storage(::Val{:cuda}, ::Type{T}, modulus::Integer) where {T}
    FFM._validate_storage_modulus(T, modulus)
    return nothing
end

function _construct_cuda_storage(
    ::Type{T},
    array::CUDA.CuArray{T,D,M},
    logical_size::Dims{D},
    modulus::Integer,
) where {T,D,M}
    storage = _cuda_storage(array)
    return GPUModArray{T,D,typeof(storage)}(storage, logical_size, modulus)
end

function (::Type{CuModArray{T,D}})(
    array::CUDA.CuArray{T,D,M};
    modulus::Integer,
    mod::Bool = false,
    new_size = nothing,
) where {T,D,M}
    dims = new_size === nothing ? size(array) : Tuple(Int.(new_size))
    return _construct_cuda_storage(T, array, dims, modulus)
end

function (::Type{CuModArray})(
    array::CUDA.CuArray{T,D,M};
    modulus::Integer,
    mod::Bool = false,
    new_size = nothing,
) where {T,D,M}
    return CuModArray{T,D}(array; modulus, mod, new_size)
end

function (::Type{CuModArray{T,D}})(
    values::AbstractArray{S,D};
    modulus::Integer,
    mod::Bool = true,
    new_size = nothing,
) where {T,D,S}
    N = _validate_storage_modulus(T, modulus)
    dims = size(values)
    padded = FFM._padded_size(dims)
    converted = FFM._modular_host_values(T, values, N, mod)
    data = CUDA.zeros(T, padded...)
    if !isempty(values)
        inds = CartesianIndices(values)
        copyto!(data, inds, converted, inds)
    end
    logical_size = new_size === nothing ? dims : Tuple(Int.(new_size))
    return _construct_cuda_storage(T, data, logical_size, N)
end

function (::Type{CuModArray})(
    values::AbstractArray{S,D};
    modulus::Integer,
    mod::Bool = true,
    new_size = nothing,
) where {S,D}
    return CuModArray{DEFAULT_TYPE,D}(values; modulus, mod, new_size)
end

CuModMatrix(values::AbstractMatrix; kwargs...) =
    CuModArray{DEFAULT_TYPE,2}(values; kwargs...)
CuModVector(values::AbstractVector; kwargs...) =
    CuModArray{DEFAULT_TYPE,1}(values; kwargs...)

CuModMatrix(array::CUDA.CuArray{T,2}; kwargs...) where {T} =
    CuModArray{T,2}(array; kwargs...)
CuModVector(array::CUDA.CuArray{T,1}; kwargs...) where {T} =
    CuModArray{T,1}(array; kwargs...)

function (::Type{CuModArray{T,2}})(
    ::LinearAlgebra.UniformScaling,
    rows::Integer,
    cols::Integer;
    modulus::Integer,
) where {T}
    return CuModArray{T,2}(Matrix{T}(I, rows, cols); modulus)
end

identity_matrix(::Type{CuModArray{T,2}}, n::Integer; modulus::Integer) where {T} =
    CuModArray{T,2}(I, n, n; modulus)
