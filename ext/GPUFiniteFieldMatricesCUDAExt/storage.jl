function _cuda_storage(array::CuArray{T,D,M}) where {T,D,M}
    return FFM.CuStorage{T,D,M,typeof(array)}(array)
end

function _validate_backend_storage(::Val{:cuda}, ::Type{T}, modulus::Integer) where {T}
    if !(T === Float16 || T === Float32 || T === Float64)
        throw(
            ArgumentError(
                "CUDA modular storage supports Float16 (modulus ≤ 2048), Float32 (modulus ≤ 16777216), and Float64 (modulus ≤ 9007199254740992); got $T",
            ),
        )
    end
    try
        _validate_storage_modulus(T, modulus)
    catch err
        if err isa DomainError
            limit = _storage_modulus_limit(T)
            throw(
                DomainError(
                    modulus,
                    "modulus $modulus is not exactly representable in CUDA $T storage; supported maximum is $limit",
                ),
            )
        end
        rethrow()
    end
    return nothing
end

function _cuda_dimensions(dims::Tuple)
    return ntuple(length(dims)) do i
        dim = Int(dims[i])
        dim >= 0 || throw(
            ArgumentError("CUDA modular storage dimensions must be nonnegative, got $dims"),
        )
        dim
    end
end

function _cuda_factory_preflight(
    ::Type{T},
    dims::Tuple,
    modulus::Integer,
    logical_size = nothing,
) where {T}
    normalized_dims = _cuda_dimensions(dims)
    _validate_backend_storage(Val(:cuda), T, modulus)
    N = _validate_storage_modulus(T, modulus)
    normalized_logical_size = _normalize_logical_size(normalized_dims, logical_size)
    return normalized_dims, normalized_logical_size, N
end

function _construct_cuda_storage(
    ::Type{T},
    array::CUDA.CuArray{T,D,M},
    logical_size::Dims{D},
    modulus::Integer,
) where {T,D,M}
    _validate_backend_storage(Val(:cuda), T, modulus)
    normalized_size = _normalize_logical_size(size(array), logical_size)
    storage = _cuda_storage(array)
    return GPUModArray{T,D,typeof(storage)}(storage, normalized_size, modulus)
end

function (::Type{CuModArray{T,D}})(
    array::CUDA.CuArray{T,D,M};
    modulus::Integer,
    mod::Bool = false,
    new_size = nothing,
) where {T,D,M}
    _, dims, N = _cuda_factory_preflight(T, size(array), modulus, new_size)
    return _construct_cuda_storage(T, array, dims, N)
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
    dims, logical_size, N = _cuda_factory_preflight(T, size(values), modulus, new_size)
    padded = FFM._padded_size(dims)
    converted = FFM._modular_host_values(T, values, N, mod)
    data = CUDA.zeros(T, padded...)
    if !isempty(values)
        inds = CartesianIndices(values)
        copyto!(data, inds, converted, inds)
    end
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
