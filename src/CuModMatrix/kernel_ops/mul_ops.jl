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

Base.@propagate_inbounds function mul_elementwise!(
    C::CuModArray,
    A::CuModArray,
    B::CuModArray,
    ;
    modulus_override::Union{Nothing,Integer} = nothing,
    threads::Int = DEFAULT_THREADS,
)
    _validate_positive_modulus_override(modulus_override)
    _check_elementwise_compatible(C, A, B)
    m = modulus_override === nothing ? modulus(C) : modulus_override
    mT = convert(eltype(parent(C)), m)
    output = _logical_view(parent(C), size(C))
    left = _logical_view(parent(A), size(A))
    right = _logical_view(parent(B), size(B))
    len = length(left)
    if len > 0
        blocks = min(cld(len, threads), 65535)
        @cuda threads=threads blocks=blocks k_mul_matrix!(output, left, right, mT)
    end
    return C
end

Base.@propagate_inbounds function LinearAlgebra.mul!(
    C::CuModArray,
    A::CuModArray,
    s::Number,
    ;
    modulus_override::Union{Nothing,Integer} = nothing,
    threads::Int = DEFAULT_THREADS,
)
    _validate_positive_modulus_override(modulus_override)
    _check_elementwise_compatible(C, A)
    m = modulus_override === nothing ? modulus(C) : modulus_override
    mT = convert(eltype(parent(C)), m)
    sT = convert(eltype(parent(C)), s)
    output = _logical_view(parent(C), size(C))
    input = _logical_view(parent(A), size(A))
    len = length(input)
    if len > 0
        blocks = min(cld(len, threads), 65535)
        @cuda threads=threads blocks=blocks k_mul_scalar!(output, input, sT, mT)
    end
    return C
end

Base.@propagate_inbounds function *(A::CuModArray, s::Number)
    _check_elementwise_allocating_input(A)
    C = similar(A)
    mul!(C, A, s)
    return C
end

Base.@propagate_inbounds function *(s::Number, A::CuModArray)
    _check_elementwise_allocating_input(A)
    C = similar(A)
    mul!(C, A, s)
    return C
end

function *(A::CuModMatrix, B::CuModMatrix)
    return stripe_mul(A, B)
end
