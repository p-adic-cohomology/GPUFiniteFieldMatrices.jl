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

"""
    mod!(C::CuModArray, A::CuModArray, m)

Reduce the logical elements of `A` into `C` using the explicit working modulus
`m`. This operation retains its existing semantics and does not compare the
arrays' stored moduli. Logical sizes and backing lengths must match; callers
using `@inbounds` are responsible for compatibility because those checks may be
elided.
"""
Base.@propagate_inbounds function mod!(
    C::CuModArray,
    A::CuModArray,
    m::Integer;
    threads::Int = DEFAULT_THREADS,
)
    _check_elementwise_storage_compatible(C, A)
    mT = convert(eltype(parent(C)), m)
    output = _logical_view(parent(C), size(C))
    input = _logical_view(parent(A), size(A))
    len = length(input)
    if len > 0
        blocks = min(cld(len, threads), 65535)
        @cuda threads=threads blocks=blocks k_mod!(output, input, mT)
    end
    return C
end

function mod!(C::CuArray, A::CuArray, m::Integer; threads::Int = DEFAULT_THREADS)
    mT = convert(eltype(C), m)
    len = length(A)
    blocks = min(cld(len, threads), 65535)
    @cuda threads=threads blocks=blocks k_mod!(C, A, mT)
    return C
end

Base.@propagate_inbounds function mod(A::CuModArray, m::Integer)
    _check_elementwise_allocating_input(A)
    C = similar_with_modulus(A, m)
    mod!(C, A, m)
    return C
end
