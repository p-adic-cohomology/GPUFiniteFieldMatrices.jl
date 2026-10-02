import Base: -

function k_sub_matrix!(out, A, B, m)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    len = length(A)
    @inbounds for i = idx:stride:len
        out[i] = mod(A[i] - B[i], m)
    end
    return
end

function k_sub_scalar!(out, A, s, m)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    len = length(A)
    @inbounds for i = idx:stride:len
        out[i] = mod(A[i] - s, m)
    end
    return
end

function k_rsub_scalar!(out, A, s, m)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    len = length(A)
    @inbounds for i = idx:stride:len
        out[i] = mod(s - A[i], m)
    end
    return
end

function sub!(
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
    @cuda threads=threads blocks=blocks k_sub_matrix!(C.data, A.data, B.data, mT)
    return C
end

function scalar_sub!(
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
    @cuda threads=threads blocks=blocks k_sub_scalar!(C.data, A.data, sT, mT)
    return C
end

function rscalar_sub!(
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
    @cuda threads=threads blocks=blocks k_rsub_scalar!(C.data, A.data, sT, mT)
    return C
end

function rscalar_sub!(
    C::CuArray,
    A::CuArray,
    s::Number,
    m::Integer;
    threads::Int = DEFAULT_THREADS,
)
    mT = convert(eltype(C), m)
    sT = convert(eltype(C), s)
    len = length(A)
    blocks = min(cld(len, threads), 65535)
    @cuda threads=threads blocks=blocks k_rsub_scalar!(C, A, sT, mT)
    return C
end

function -(A::CuModArray, B::CuModArray)
    C = similar(A)
    sub!(C, A, B)
    return C
end

function -(A::CuModArray, s::Number)
    C = similar(A)
    scalar_sub!(C, A, s)
    return C
end

function -(s::Number, A::CuModArray)
    C = similar(A)
    rscalar_sub!(C, A, s)
    return C
end
