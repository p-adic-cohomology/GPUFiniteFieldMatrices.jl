import Base: mod

function k_mod!(out, A, m)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    len = length(A)
    @inbounds for i = idx:stride:len
        out[i] = mod(A[i], m)
    end
    return
end

function mod!(C::CuModArray, A::CuModArray, m::Integer; threads::Int = DEFAULT_THREADS)
    mT = convert(eltype(C.data), m)
    len = length(A.data)
    blocks = min(cld(len, threads), 65535)
    @cuda threads=threads blocks=blocks k_mod!(C.data, A.data, mT)
    return C
end

function mod!(C::CuArray, A::CuArray, m::Integer; threads::Int = DEFAULT_THREADS)
    mT = convert(eltype(C), m)
    len = length(A)
    blocks = min(cld(len, threads), 65535)
    @cuda threads=threads blocks=blocks k_mod!(C, A, mT)
    return C
end

function mod(A::CuModArray, m::Integer)
    C = similar(A, m)
    mod!(C, A, m)
    return C
end
