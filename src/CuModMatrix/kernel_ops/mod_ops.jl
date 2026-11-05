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

@inline function _launch_logical_mod!(C::CuModArray, A::CuModArray, m, threads::Int)
    output = _logical_view(parent(C), size(C))
    input = _logical_view(parent(A), size(A))
    len = length(input)
    if len > 0
        blocks = min(cld(len, threads), 65535)
        @cuda threads=threads blocks=blocks k_mod!(output, input, m)
    end
    return C
end

"""
    mod!(C::CuModArray, A::CuModArray, m)

Reduce the logical elements of `A` into `C` using the explicit working modulus
`m`. The stored moduli of `C` and `A` must match; `m` remains an independent
working modulus and does not change either array's stored metadata. Logical
sizes and backing lengths must also match. Callers using `@inbounds` are
responsible for compatibility because those checks may be elided.
"""
Base.@propagate_inbounds @inline function mod!(
    C::CuModArray,
    A::CuModArray,
    m::Integer;
    threads::Int = DEFAULT_THREADS,
)
    @boundscheck _check_elementwise_compatible(C, A)
    mT = convert(eltype(parent(C)), m)
    return _launch_logical_mod!(C, A, mT, threads)
end

function mod!(C::CuArray, A::CuArray, m::Integer; threads::Int = DEFAULT_THREADS)
    mT = convert(eltype(C), m)
    len = length(A)
    blocks = min(cld(len, threads), 65535)
    @cuda threads=threads blocks=blocks k_mod!(C, A, mT)
    return C
end

Base.@propagate_inbounds @inline function mod(A::CuModArray, m::Integer)
    @boundscheck _check_elementwise_allocating_input(A)
    C = similar_with_modulus(A, m)
    mT = convert(eltype(parent(C)), m)
    _launch_logical_mod!(C, A, mT, DEFAULT_THREADS)
    return C
end
