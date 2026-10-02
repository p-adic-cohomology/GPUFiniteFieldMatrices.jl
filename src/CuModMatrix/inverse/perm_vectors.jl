"""
    pluq_init_perm(n)

Create the identity permutation vector `1:n`.
"""
function pluq_init_perm(n::Int)
    return collect(1:n)
end

"""
    pluq_inverse_perm(p)

Compute the inverse permutation of `p`.

Example:
```julia
p = [3, 1, 2]
pluq_inverse_perm(p) == [2, 3, 1]
```
"""
function pluq_inverse_perm(p::Vector{Int})
    pinv = Vector{Int}(undef, length(p))
    for i in eachindex(p)
        pinv[p[i]] = i
    end
    return pinv
end

"""
    pluq_compose_segment!(perm, offset, locperm)

Compose a local gather permutation into a segment of a global gather vector.

If `seg = perm[offset:last]`, this performs:
`seg_new[i] = seg_old[locperm[i]]`.

Example:
```julia
perm = [1, 2, 3, 4, 5]
pluq_compose_segment!(perm, 2, [2, 1, 3])
perm == [1, 3, 2, 4, 5]
```
"""
function pluq_compose_segment!(perm::Vector{Int}, offset::Int, locperm::Vector{Int})
    lastidx = offset + length(locperm) - 1
    tmp = copy(perm[offset:lastidx])
    for i in eachindex(locperm)
        perm[offset+i-1] = tmp[locperm[i]]
    end
    return perm
end
