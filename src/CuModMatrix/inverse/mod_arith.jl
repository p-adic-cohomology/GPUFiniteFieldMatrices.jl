"""
    pluq_mod_reduce(x, N)

Return `x mod N` in `[0, N-1]`.

Example:
```julia
pluq_mod_reduce(-3, 11) == 8
```
"""
function pluq_mod_reduce(x, N::Int)
    return mod(x, N)
end

"""
    pluq_mod_add(a, b, N)

Return `(a + b) mod N`.
"""
function pluq_mod_add(a, b, N::Int)
    return pluq_mod_reduce(a + b, N)
end

"""
    pluq_mod_sub(a, b, N)

Return `(a - b) mod N`.
"""
function pluq_mod_sub(a, b, N::Int)
    return pluq_mod_reduce(a - b, N)
end

"""
    pluq_mod_mul(a, b, N)

Return `(a * b) mod N`.
"""
function pluq_mod_mul(a, b, N::Int)
    return pluq_mod_reduce(a * b, N)
end

"""
    pluq_mod_inv(a, N)

Compute the modular inverse of `a` modulo `N` using extended Euclid.
Throws if no inverse exists.
"""
function pluq_mod_inv(a::Number, N::Int)
    aa = Int(pluq_mod_reduce(round(Int, a), N))
    if aa == 0
        throw(LinearAlgebra.SingularException(0))
    end
    t = 0
    newt = 1
    r = N
    newr = aa
    while newr != 0
        q = r ÷ newr
        t, newt = newt, t - q * newt
        r, newr = newr, r - q * newr
    end
    if r != 1
        throw(LinearAlgebra.SingularException(0))
    end
    if t < 0
        t += N
    end
    return t
end
