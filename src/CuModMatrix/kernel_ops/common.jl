const DEFAULT_THREADS = 256

function Base.similar(A::CuModArray)
    T = eltype(A.data)
    D = ndims(A.data)
    data = similar(A.data)
    return CuModArray{T,D}(data; modulus = A.N, mod = false, new_size = size(A))
end

"""
    similar_with_modulus(A, N)

Allocate a same-shaped array with modulus N.
"""
function similar_with_modulus(A::CuModArray, N::Integer)
    T = eltype(A.data)
    D = ndims(A.data)
    data = similar(A.data)
    return CuModArray{T,D}(data; modulus = N, mod = false, new_size = size(A))
end
