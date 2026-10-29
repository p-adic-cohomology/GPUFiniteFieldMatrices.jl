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

Base.@propagate_inbounds function div!(
    C::CuModArray,
    A::CuModArray,
    s::Integer,
    ;
    modulus_override::Union{Nothing,Integer} = nothing,
    threads::Int = DEFAULT_THREADS,
)
    _validate_positive_modulus_override(modulus_override)
    _check_elementwise_compatible(C, A)
    m = modulus_override === nothing ? modulus(C) : modulus_override
    s_inv = mod_inv(s, m)
    mT = convert(eltype(parent(C)), m)
    sT = convert(eltype(parent(C)), s_inv)
    output = _logical_view(parent(C), size(C))
    input = _logical_view(parent(A), size(A))
    len = length(input)
    if len > 0
        blocks = min(cld(len, threads), 65535)
        @cuda threads=threads blocks=blocks k_div_scalar!(output, input, sT, mT)
    end
    return C
end

Base.@propagate_inbounds function /(A::CuModArray, s::Integer)
    _check_elementwise_allocating_input(A)
    C = similar(A)
    div!(C, A, s)
    return C
end
