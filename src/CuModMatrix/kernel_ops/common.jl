const DEFAULT_THREADS = 256

function Base.similar(A::CuModArray)
    T = eltype(parent(A))
    D = ndims(parent(A))
    data = similar(parent(A))
    return CuModArray{T,D}(data; modulus = modulus(A), mod = false, new_size = size(A))
end

"""
    similar_with_modulus(A, N)

Allocate a same-shaped array with modulus N.
"""
function similar_with_modulus(A::CuModArray, N::Integer)
    T = eltype(parent(A))
    D = ndims(parent(A))
    data = similar(parent(A))
    return CuModArray{T,D}(data; modulus = N, mod = false, new_size = size(A))
end
