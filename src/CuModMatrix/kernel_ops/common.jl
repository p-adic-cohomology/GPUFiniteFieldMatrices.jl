const DEFAULT_THREADS = 256

Base.@noinline function _throw_elementwise_size_mismatch()
    throw(
        CuModArraySizeMismatchException(
            "Elementwise arrays must have equal logical sizes and backing lengths",
        ),
    )
end

Base.@noinline function _throw_elementwise_modulus_mismatch()
    throw(
        CuModArrayModulusMismatchException(
            "Elementwise arrays must have the same modulus N",
        ),
    )
end

Base.@propagate_inbounds @inline function _check_elementwise_metadata(
    size_C::Dims,
    size_A::Dims,
    size_B::Dims,
    modulus_C::Int,
    modulus_A::Int,
    modulus_B::Int,
    length_C::Int,
    length_A::Int,
    length_B::Int,
)
    @boundscheck begin
        size_C == size_A == size_B || _throw_elementwise_size_mismatch()
        modulus_C == modulus_A == modulus_B || _throw_elementwise_modulus_mismatch()
        length_C == length_A == length_B || _throw_elementwise_size_mismatch()
    end
    return nothing
end

Base.@propagate_inbounds @inline function _check_elementwise_metadata(
    size_C::Dims,
    size_A::Dims,
    modulus_C::Int,
    modulus_A::Int,
    length_C::Int,
    length_A::Int,
)
    @boundscheck begin
        size_C == size_A || _throw_elementwise_size_mismatch()
        modulus_C == modulus_A || _throw_elementwise_modulus_mismatch()
        length_C == length_A || _throw_elementwise_size_mismatch()
    end
    return nothing
end

Base.@propagate_inbounds @inline function _check_elementwise_storage(
    size_C::Dims,
    size_A::Dims,
    length_C::Int,
    length_A::Int,
)
    @boundscheck begin
        size_C == size_A || _throw_elementwise_size_mismatch()
        length_C == length_A || _throw_elementwise_size_mismatch()
    end
    return nothing
end

Base.@propagate_inbounds @inline function _check_elementwise_compatible(
    C::CuModArray,
    A::CuModArray,
    B::CuModArray,
)
    _check_elementwise_metadata(
        size(C),
        size(A),
        size(B),
        modulus(C),
        modulus(A),
        modulus(B),
        length(parent(C)),
        length(parent(A)),
        length(parent(B)),
    )
    return nothing
end

Base.@propagate_inbounds @inline function _check_elementwise_compatible(
    C::CuModArray,
    A::CuModArray,
)
    _check_elementwise_metadata(
        size(C),
        size(A),
        modulus(C),
        modulus(A),
        length(parent(C)),
        length(parent(A)),
    )
    return nothing
end

Base.@propagate_inbounds @inline function _check_elementwise_storage_compatible(
    C::CuModArray,
    A::CuModArray,
)
    _check_elementwise_storage(size(C), size(A), length(parent(C)), length(parent(A)))
    return nothing
end

Base.@propagate_inbounds @inline function _check_elementwise_allocating_metadata(
    size_A::Dims,
    size_B::Dims,
    modulus_A::Int,
    modulus_B::Int,
    length_A::Int,
    length_B::Int,
)
    @boundscheck begin
        size_A == size_B || _throw_elementwise_size_mismatch()
        modulus_A == modulus_B || _throw_elementwise_modulus_mismatch()
        length_A == length_B || _throw_elementwise_size_mismatch()
        length_A == prod(_padded_size(size_A)) || _throw_elementwise_size_mismatch()
    end
    return nothing
end

Base.@propagate_inbounds @inline function _check_elementwise_allocating_input(
    size_A::Dims,
    length_A::Int,
)
    @boundscheck length_A == prod(_padded_size(size_A)) ||
                 _throw_elementwise_size_mismatch()
    return nothing
end

Base.@propagate_inbounds @inline function _check_elementwise_allocating_input(A::CuModArray)
    _check_elementwise_allocating_input(size(A), length(parent(A)))
    return nothing
end

Base.@propagate_inbounds @inline function _check_elementwise_allocating_inputs(
    A::CuModArray,
    B::CuModArray,
)
    _check_elementwise_allocating_metadata(
        size(A),
        size(B),
        modulus(A),
        modulus(B),
        length(parent(A)),
        length(parent(B)),
    )
    return nothing
end

function Base.similar(A::CuModArray)
    T = eltype(parent(A))
    D = ndims(parent(A))
    data = CUDA.zeros(T, _padded_size(size(A))...)
    return CuModArray{T,D}(data; modulus = modulus(A), mod = false, new_size = size(A))
end

"""
    similar_with_modulus(A, N)

Allocate a same-logical-shape array with modulus N. Its storage rounds each
dimension up to a multiple of `TILE_WIDTH` and starts with zero padding.
"""
function similar_with_modulus(A::CuModArray, N::Integer)
    T = eltype(parent(A))
    D = ndims(parent(A))
    data = CUDA.zeros(T, _padded_size(size(A))...)
    return CuModArray{T,D}(data; modulus = N, mod = false, new_size = size(A))
end
