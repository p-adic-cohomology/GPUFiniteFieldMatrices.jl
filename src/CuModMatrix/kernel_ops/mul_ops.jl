import Base: *

function k_mul_scalar!(out, A, s, m)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    len = length(A)
    @inbounds for i = idx:stride:len
        out[i] = mod(A[i] * s, m)
    end
    return
end

function k_mul_matrix!(out, A, B, m)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    len = length(A)
    @inbounds for i = idx:stride:len
        out[i] = mod(A[i] * B[i], m)
    end
    return
end

function mul_elementwise!(
    C::CuModArray,
    A::CuModArray,
    B::CuModArray,
    ;
    modulus_override::Union{Nothing,Integer} = nothing,
    threads::Int = DEFAULT_THREADS,
)
    _validate_positive_modulus_override(modulus_override)
    m = modulus_override === nothing ? modulus(C) : modulus_override
    mT = convert(eltype(parent(C)), m)
    len = length(parent(A))
    blocks = min(cld(len, threads), 65535)
    @cuda threads=threads blocks=blocks k_mul_matrix!(parent(C), parent(A), parent(B), mT)
    return C
end

function LinearAlgebra.mul!(
    C::CuModArray,
    A::CuModArray,
    s::Number,
    ;
    modulus_override::Union{Nothing,Integer} = nothing,
    threads::Int = DEFAULT_THREADS,
)
    _validate_positive_modulus_override(modulus_override)
    m = modulus_override === nothing ? modulus(C) : modulus_override
    mT = convert(eltype(parent(C)), m)
    sT = convert(eltype(parent(C)), s)
    len = length(parent(A))
    blocks = min(cld(len, threads), 65535)
    @cuda threads=threads blocks=blocks k_mul_scalar!(parent(C), parent(A), sT, mT)
    return C
end

function *(A::CuModArray, s::Number)
    C = similar(A)
    mul!(C, A, s)
    return C
end

function *(s::Number, A::CuModArray)
    C = similar(A)
    mul!(C, A, s)
    return C
end

function *(A::CuModMatrix, B::CuModMatrix)
    return stripe_mul(A, B)
end
