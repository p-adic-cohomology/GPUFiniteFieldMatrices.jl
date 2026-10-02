import Base: /

function k_div_scalar!(out, A, s_inv, m)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    len = length(A)
    @inbounds for i = idx:stride:len
        out[i] = mod(A[i] * s_inv, m)
    end
    return
end

function div!(
    C::CuModArray,
    A::CuModArray,
    s::Integer,
    mod_N::Integer = -1;
    threads::Int = DEFAULT_THREADS,
)
    m = mod_N > 0 ? mod_N : C.N
    s_inv = mod_inv(s, m)
    mT = convert(eltype(C.data), m)
    sT = convert(eltype(C.data), s_inv)
    len = length(A.data)
    blocks = min(cld(len, threads), 65535)
    @cuda threads=threads blocks=blocks k_div_scalar!(C.data, A.data, sT, mT)
    return C
end

function /(A::CuModArray, s::Integer)
    C = similar(A)
    div!(C, A, s)
    return C
end
