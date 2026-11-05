"""
    -(A::CuModArray)

Return a new array containing the additive inverse of each logical entry of
`A`, reduced modulo `modulus(A)`. The input must use the established padded
storage geometry accepted by allocating elementwise operations.
"""
Base.@propagate_inbounds @inline function Base.:-(A::CuModArray)
    @boundscheck _check_elementwise_allocating_input(A)
    return CUDA.device!(CUDA.device(parent(A))) do
        B = similar_with_modulus(A, modulus(A))
        neg!(B, A)
        B
    end
end

"""
    -(A::KaratsubaArray)

Return a new Karatsuba array containing the additive inverse of `A`. The
existing copy and `negate!` paths preserve the component and plan conventions.
GPU-backed components are copied and negated in their input CUDA context;
other storage keeps the existing generic path.
"""
function Base.:-(A::KaratsubaArray)
    compute_negative = () -> begin
        B = copy(A)
        negate!(B, A)
        B
    end

    component = low_part(A)
    component_storage = component isa CuModArray ? parent(component) : component
    if component_storage isa CUDA.CuArray
        return CUDA.device!(compute_negative, CUDA.device(component_storage))
    end

    return compute_negative()
end

"""
    A == B

Compare the logical entries and stored moduli of two `CuModArray`s. Physical
padding and backing layouts are ignored. A shape or modulus mismatch returns
`false` without launching a comparison.
"""
function Base.:(==)(A::CuModArray, B::CuModArray)
    size(A) == size(B) || return false
    modulus(A) == modulus(B) || return false
    isempty(A) && return true

    logical_A = _logical_view(parent(A), size(A))
    logical_B = _logical_view(parent(B), size(B))
    return all(logical_A .== logical_B)
end

"""Return the rank recorded in a `PLUQ` factorization."""
LinearAlgebra.rank(F::PLUQ) = getfield(F, :rank)

"""
    LinearAlgebra.rank(A::CuModMatrix; options=PLUQOptions())

Return the rank of `A` over its stored modulus using the existing non-mutating
PLUQ path and its options.
"""
function LinearAlgebra.rank(A::CuModMatrix; options::PLUQOptions = PLUQOptions())
    return LinearAlgebra.rank(pluq(A; options = options))
end
