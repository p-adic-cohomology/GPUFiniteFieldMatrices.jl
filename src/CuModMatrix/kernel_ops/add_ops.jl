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
    modulus_override::Union{Nothing,Integer} = nothing,
    threads::Int = DEFAULT_THREADS,
)
    m =
        modulus_override === nothing || modulus_override <= 0 ? modulus(C) :
        modulus_override
    mT = convert(eltype(parent(C)), m)
    output = _logical_view(parent(C), size(C))
    left = _logical_view(parent(A), size(A))
    right = _logical_view(parent(B), size(B))
    len = length(left)
    if len > 0
        blocks = min(cld(len, threads), 65535)
        @cuda threads=threads blocks=blocks k_add_matrix!(output, left, right, mT)
    end
    return C
end

function scalar_add!(
    C::CuModArray,
    A::CuModArray,
    s::Number,
    ;
    modulus_override::Union{Nothing,Integer} = nothing,
    threads::Int = DEFAULT_THREADS,
)
    m =
        modulus_override === nothing || modulus_override <= 0 ? modulus(C) :
        modulus_override
    mT = convert(eltype(parent(C)), m)
    sT = convert(eltype(parent(C)), s)
    output = _logical_view(parent(C), size(C))
    input = _logical_view(parent(A), size(A))
    len = length(input)
    if len > 0
        blocks = min(cld(len, threads), 65535)
        @cuda threads=threads blocks=blocks k_add_scalar!(output, input, sT, mT)
    end
    return C
end

function +(A::CuModArray, B::CuModArray)
    C = similar_with_modulus(A, modulus(A))
    add!(C, A, B)
    return C
end

function +(A::CuModArray, s::Number)
    C = similar_with_modulus(A, modulus(A))
    scalar_add!(C, A, s)
    return C
end

function +(s::Number, A::CuModArray)
    C = similar_with_modulus(A, modulus(A))
    scalar_add!(C, A, s)
    return C
end
