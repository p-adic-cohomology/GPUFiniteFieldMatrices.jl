"""
    -(A::KaratsubaArray)

Return a negated copy of the modular value represented by `A`.
"""
function Base.:-(A::KaratsubaArray)
    result = copy(A)
    negate!(result, A)
    return result
end

"""
    -(A::CuModArray)

Return a new modular array containing the additive inverse of each logical
entry of `A`.
"""
function Base.:-(A::CuModArray)
    result = similar_with_modulus(A, modulus(A))
    neg!(result, A)
    return result
end

"""
    A == B

Compare the logical entries and stored moduli of two `GPUModArray`s. Physical
padding and backing layouts are ignored.
"""
function Base.:(==)(A::GPUModArray, B::GPUModArray)
    size(A) == size(B) || return false
    modulus(A) == modulus(B) || return false
    isempty(A) && return true

    logical_A = _logical_view(parent(A), size(A))
    logical_B = _logical_view(parent(B), size(B))
    return mapreduce(==, &, logical_A, logical_B; init = true)
end

"""Return the rank recorded in a `PLUQ` factorization."""
LinearAlgebra.rank(F::PLUQ) = getfield(F, :rank)

"""
    LinearAlgebra.rank(A::CuModMatrix; options=PLUQOptions())

Return the rank of `A` over its stored modulus using the nonmutating PLUQ
path and the supplied options.
"""
function LinearAlgebra.rank(A::GPUModMatrix; options::PLUQOptions = PLUQOptions())
    return LinearAlgebra.rank(pluq(A; options = options))
end
