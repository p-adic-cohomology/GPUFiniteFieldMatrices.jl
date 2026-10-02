const DEFAULT_THREADS = 256

function Base.similar(A::CuModArray)
    T = eltype(A.data)
    D = ndims(A.data)
    data = similar(A.data)
    return CuModArray{T,D}(data, A.N; mod = false, new_size = size(A))
end

function Base.similar(A::CuModArray, N::Integer)
    T = eltype(A.data)
    D = ndims(A.data)
    data = similar(A.data)
    return CuModArray{T,D}(data, N; mod = false, new_size = size(A))
end
