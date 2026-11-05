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
        @cuda threads=threads blocks=blocks k_sub_matrix!(output, left, right, mT)
    end
    return C
end

function scalar_sub!(
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
        @cuda threads=threads blocks=blocks k_sub_scalar!(output, input, sT, mT)
    end
    return C
end

function rscalar_sub!(
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
        @cuda threads=threads blocks=blocks k_rsub_scalar!(output, input, sT, mT)
    end
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
