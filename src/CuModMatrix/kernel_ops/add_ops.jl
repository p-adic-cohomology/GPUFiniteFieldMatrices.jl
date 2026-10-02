import Base: +

function k_add_matrix!(out, A, B, m)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    len = length(A)
    @inbounds for i = idx:stride:len
        out[i] = mod(A[i] + B[i], m)
    end
    return
end

function k_add_scalar!(out, A, s, m)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    len = length(A)
    @inbounds for i = idx:stride:len
        out[i] = mod(A[i] + s, m)
    end
    return
end

function add!(
    C::CuModArray,
    A::CuModArray,
    B::CuModArray;
    mod_N::Integer = -1,
    threads::Int = DEFAULT_THREADS,
)
    m = mod_N > 0 ? mod_N : C.N
    mT = convert(eltype(C.data), m)
    len = length(A.data)
    blocks = min(cld(len, threads), 65535)
    @cuda threads=threads blocks=blocks k_add_matrix!(C.data, A.data, B.data, mT)
    return C
end

function scalar_add!(
    C::CuModArray,
    A::CuModArray,
    s::Number,
    mod_N::Integer = -1;
    threads::Int = DEFAULT_THREADS,
)
    m = mod_N > 0 ? mod_N : C.N
    mT = convert(eltype(C.data), m)
    sT = convert(eltype(C.data), s)
    len = length(A.data)
    blocks = min(cld(len, threads), 65535)
    @cuda threads=threads blocks=blocks k_add_scalar!(C.data, A.data, sT, mT)
    return C
end

function +(A::CuModArray, B::CuModArray)
    C = similar(A, A.N)
    add!(C, A, B)
    return C
end

function +(A::CuModArray, s::Number)
    C = similar(A, A.N)
    scalar_add!(C, A, s)
    return C
end

function +(s::Number, A::CuModArray)
    C = similar(A, A.N)
    scalar_add!(C, A, s)
    return C
end
