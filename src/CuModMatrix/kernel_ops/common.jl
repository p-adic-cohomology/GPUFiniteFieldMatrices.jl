const DEFAULT_THREADS = 256

function Base.similar(A::CuModArray)
    T = eltype(parent(A))
    D = ndims(parent(A))
    data = CUDA.zeros(T, _padded_size(size(A))...)
    return CuModArray{T,D}(data; modulus = modulus(A), mod = false, new_size = size(A))
end

"""
    similar_with_modulus(A, N)

Allocate a same-logical-shape array with modulus N. Its storage rounds each
dimension up to a multiple of `TILE_WIDTH` and starts with zero padding.
"""
function similar_with_modulus(A::CuModArray, N::Integer)
    T = eltype(parent(A))
    D = ndims(parent(A))
    data = CUDA.zeros(T, _padded_size(size(A))...)
    return CuModArray{T,D}(data; modulus = N, mod = false, new_size = size(A))
end
