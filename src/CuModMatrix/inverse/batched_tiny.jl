_tiny_inverse_size(n) = n in (4, 8, 16, 32)

function _inverse_batch_default_pluq(options)
    defaults = PLUQOptions()
    return options.blocksize == defaults.blocksize &&
           options.basecase == defaults.basecase &&
           options.pivot_policy == defaults.pivot_policy &&
           options.lazy_q == defaults.lazy_q &&
           options.nftb == defaults.nftb &&
           options.pivot_warp_kernel == defaults.pivot_warp_kernel &&
           options.trsm_mode == defaults.trsm_mode &&
           options.trsm_warp_threshold == defaults.trsm_warp_threshold &&
           options.schur_tile == defaults.schur_tile &&
           options.schur_transpose_u == defaults.schur_transpose_u &&
           options.mod_backend == defaults.mod_backend &&
           !options.autotune
end

_inverse_batch_default_augmented(options) =
    options.inverse_strategy === :augmented && _inverse_batch_default_pluq(options)

function _inverse_batch_inputs(mats, options)
    Base.require_one_based_indexing(mats)
    isempty(mats) && return nothing
    options.batch_streams == 1 || throw(
        ArgumentError("portable inverse batches support one queue; use batch_streams=1"),
    )
    first_A = first(mats)
    for A in mats
        _inverse_preflight(A, options)
        _check_elementwise_backend(first_A, A)
    end
    return nothing
end

_inverse_batch_alias(A, B) = Base.mightalias(parent(A), parent(B))

function _inverse_distinct_matrices(mats)
    for i in eachindex(mats), j = 1:(i-1)
        _inverse_batch_alias(mats[i], mats[j]) &&
            throw(ArgumentError("batch matrices must not alias each other"))
    end
    return nothing
end

function _inverse_batch_homogeneous(mats)
    isempty(mats) && return nothing
    T = eltype(first(mats))
    all(A -> eltype(A) === T, mats) ||
        throw(ArgumentError("batch matrices must have the same element type"))
    return nothing
end

function _inverse_batch_tiny_shape(mats, n)
    isempty(mats) && return nothing
    _inverse_batch_homogeneous(mats)
    N = modulus(first(mats))
    for A in mats
        size(A) == (n, n) || throw(DimensionMismatch("all matrices must be $(n)x$(n)"))
        modulus(A) == N ||
            throw(ModulusMismatchError("batch matrices must have the same modulus"))
    end
    return nothing
end

function _inverse_batch_destinations(dests, mats; inplace = false)
    Base.require_one_based_indexing(dests)
    length(dests) == length(mats) ||
        throw(DimensionMismatch("destination and input batches must have the same length"))
    if inplace
        dests === mats ||
            throw(ArgumentError("in-place inversion requires the input wrappers"))
        _inverse_distinct_matrices(mats)
        return nothing
    end
    for (dest, src) in zip(dests, mats)
        size(dest) == size(src) ||
            throw(DimensionMismatch("destination must match input shape"))
        eltype(dest) === eltype(src) ||
            throw(ArgumentError("destination must match input element type"))
        modulus(dest) == modulus(src) ||
            throw(ModulusMismatchError("destination must match input modulus"))
        _check_elementwise_backend(dest, src)
    end
    _inverse_distinct_matrices(dests)
    for dest in dests, src in mats
        _inverse_batch_alias(dest, src) &&
            throw(ArgumentError("destination storage must not alias any input storage"))
    end
    return nothing
end

function _inverse_batch_square_size(mats)
    isempty(mats) && return nothing
    n = size(first(mats), 1)
    return all(A -> size(A) == (n, n), mats) ? n : nothing
end

_inverse_empty_factors(::AbstractVector{<:GPUModMatrix{T}}) where {T} = PLUQ{T}[]
_inverse_empty_factors(::AbstractVector) = PLUQ[]

@kernel inbounds = true function _inverse_pack_matrix!(packed, A, batch)
    i, j = @index(Global, NTuple)
    packed[i, j, batch] = A[i, j]
end

@kernel inbounds = true function _inverse_unpack_matrix!(A, packed, batch, offset)
    i, j = @index(Global, NTuple)
    A[i, j] = packed[i, j+offset, batch]
end

function _inverse_pack_matrices(mats, n)
    native = _elementwise_native_root(parent(first(mats)))
    packed = similar(native, eltype(native), (n, n, length(mats)))
    pack = _inverse_pack_matrix!(KernelInterface.get_backend(packed))
    for i in eachindex(mats)
        pack(packed, parent(mats[i]), i; ndrange = (n, n))
    end
    return packed
end

function _inverse_unpack_matrices!(mats, packed, n, offset)
    unpack = _inverse_unpack_matrix!(KernelInterface.get_backend(packed))
    for i in eachindex(mats)
        unpack(parent(mats[i]), packed, i, offset; ndrange = (n, n))
    end
    return mats
end

@kernel inbounds = true function _inverse_tiny_factor_kernel!(packed, p, q, ranks, n, N)
    batch = @index(Global, Linear)
    T = eltype(packed)
    for i = 1:n
        p[i, batch] = q[i, batch] = i % Int32
    end
    for j = 1:n, i = 1:n
        packed[i, j, batch] = T(_pluq_ka_residue(packed[i, j, batch], N))
    end
    rank = Int32(0)
    for k = 1:n
        prow, pcol = 0, 0
        for j = k:n
            for i = k:n
                if !iszero(packed[i, j, batch])
                    prow, pcol = i, j
                    break
                end
            end
            prow > 0 && break
        end
        prow == 0 && break
        inverse = _pluq_ka_inverse(packed[prow, pcol, batch], N)
        iszero(inverse) && break
        if prow != k
            for j = 1:n
                packed[k, j, batch], packed[prow, j, batch] =
                    packed[prow, j, batch], packed[k, j, batch]
            end
            p[k, batch], p[prow, batch] = p[prow, batch], p[k, batch]
        end
        if pcol != k
            for i = 1:n
                packed[i, k, batch], packed[i, pcol, batch] =
                    packed[i, pcol, batch], packed[i, k, batch]
            end
            q[k, batch], q[pcol, batch] = q[pcol, batch], q[k, batch]
        end
        for i = (k+1):n
            packed[i, k, batch] =
                T(rem(_pluq_ka_residue(packed[i, k, batch], N) * inverse, N))
        end
        for j = (k+1):n, i = (k+1):n
            packed[i, j, batch] = T(
                _pluq_ka_submul(
                    packed[i, j, batch],
                    packed[i, k, batch],
                    packed[k, j, batch],
                    N,
                ),
            )
        end
        rank += Int32(1)
    end
    ranks[batch] = rank
end

@kernel inbounds = true function _inverse_tiny_augmented_kernel!(aug, source, ok, n, N)
    batch = @index(Global, Linear)
    T = eltype(aug)
    for j = 1:n, i = 1:n
        aug[i, j, batch] = T(_pluq_ka_residue(source[i, j, batch], N))
        aug[i, j+n, batch] = T(i == j)
    end
    full_rank = true
    for k = 1:n
        row = k
        while row <= n && iszero(aug[row, k, batch])
            row += 1
        end
        if row > n
            full_rank = false
            break
        end
        inverse = _pluq_ka_inverse(aug[row, k, batch], N)
        if iszero(inverse)
            full_rank = false
            break
        end
        if row != k
            for j = k:(2n)
                aug[k, j, batch], aug[row, j, batch] = aug[row, j, batch], aug[k, j, batch]
            end
        end
        for j = k:(2n)
            aug[k, j, batch] = T(rem(_pluq_ka_residue(aug[k, j, batch], N) * inverse, N))
        end
        for i = 1:n
            i == k && continue
            multiplier = aug[i, k, batch]
            for j = k:(2n)
                aug[i, j, batch] =
                    T(_pluq_ka_submul(aug[i, j, batch], multiplier, aug[k, j, batch], N))
            end
        end
    end
    ok[batch] = full_rank
end

function _inverse_batch_factor_tiny!(mats, n)
    packed = _inverse_pack_matrices(mats, n)
    b = length(mats)
    p = similar(packed, Int32, (n, b))
    q = similar(packed, Int32, (n, b))
    ranks = similar(packed, Int32, b)
    _inverse_tiny_factor_kernel!(KernelInterface.get_backend(packed))(
        packed,
        p,
        q,
        ranks,
        n,
        modulus(first(mats));
        ndrange = b,
    )
    ph, qh, rh = Array(p), Array(q), Array(ranks)
    _inverse_unpack_matrices!(mats, packed, n, 0)
    return [
        PLUQ(mats[i], Int.(ph[:, i]), Int.(qh[:, i]), Int(rh[i])) for i in eachindex(mats)
    ]
end

function _inverse_batch_factor!(mats, options; fixed_size = nothing)
    _inverse_batch_inputs(mats, options)
    isempty(mats) && return _inverse_empty_factors(mats)
    _inverse_batch_homogeneous(mats)
    _inverse_distinct_matrices(mats)
    n = isnothing(fixed_size) ? _inverse_batch_square_size(mats) : fixed_size
    if n !== nothing && _tiny_inverse_size(n)
        _inverse_batch_tiny_shape(mats, n)
        _inverse_batch_default_pluq(options) && return _inverse_batch_factor_tiny!(mats, n)
    end
    return [pluq!(A; options) for A in mats]
end

"""
Factor a batch in place and return factorizations wrapping those inputs.
Metal batches use one queue (`batch_streams=1`).
"""
pluq_batched!(mats::AbstractVector{<:GPUModMatrix}; options::PLUQOptions = PLUQOptions()) =
    _inverse_batch_factor!(mats, options)

"""Factor input copies, preserving all input storage."""
function pluq_batched(
    mats::AbstractVector{<:GPUModMatrix};
    options::PLUQOptions = PLUQOptions(),
)
    _inverse_batch_inputs(mats, options)
    return pluq_batched!(map(copy, mats); options)
end

"""Factor into distinct destination matrices after validating the whole batch."""
function pluq_batched!(
    dests::AbstractVector{<:GPUModMatrix},
    mats::AbstractVector{<:GPUModMatrix};
    options::PLUQOptions = PLUQOptions(),
)
    _inverse_batch_inputs(mats, options)
    _inverse_batch_destinations(dests, mats)
    factors = pluq_batched(mats; options)
    isempty(factors) && return _inverse_empty_factors(dests)
    for i in eachindex(dests)
        copyto!(dests[i], factors[i].LU)
    end
    return [
        PLUQ(dests[i], factors[i].p, factors[i].q, factors[i].rank) for
        i in eachindex(dests)
    ]
end

function pluq_batched_NxN!(
    mats::AbstractVector{<:GPUModMatrix};
    options::PLUQOptions = PLUQOptions(),
)
    _inverse_batch_square_size(mats) === nothing &&
        !isempty(mats) &&
        throw(DimensionMismatch("all matrices must have the same square shape"))
    return _inverse_batch_factor!(mats, options)
end

function _inverse_batch_inverse(A, options)
    m, n = size(A)
    m == n && return inv(A; options)
    return m < n ? right_inverse(A; options) : left_inverse(A; options)
end

function _inverse_batch_tiny!(dests, mats, n, options; inplace = false)
    _inverse_batch_inputs(mats, options)
    _inverse_batch_destinations(dests, mats; inplace)
    isempty(mats) && return dests
    _inverse_batch_tiny_shape(mats, n)
    if _inverse_batch_default_augmented(options)
        packed = _inverse_pack_matrices(mats, n)
        b = length(mats)
        aug = similar(packed, eltype(packed), (n, 2n, b))
        ok = similar(packed, Bool, b)
        _inverse_tiny_augmented_kernel!(KernelInterface.get_backend(packed))(
            aug,
            packed,
            ok,
            n,
            modulus(first(mats));
            ndrange = b,
        )
        all(Array(ok)) || throw(LinearAlgebra.SingularException(0))
        return _inverse_unpack_matrices!(dests, aug, n, n)
    end
    inverses = [_inverse_batch_inverse(A, options) for A in mats]
    for i in eachindex(dests)
        copyto!(dests[i], inverses[i])
    end
    return dests
end

"""
    inverse_batched(mats; options=PLUQOptions())

Allocate inverses of a batch, using one-sided inverses for rectangular inputs.
The fixed-size augmented path supports 4, 8, 16 and 32. Portable Metal batches
support one queue (`batch_streams=1`).
"""
function inverse_batched(
    mats::AbstractVector{<:GPUModMatrix};
    options::PLUQOptions = PLUQOptions(),
)
    _inverse_batch_inputs(mats, options)
    isempty(mats) && return GPUModMatrix[]
    n = _inverse_batch_square_size(mats)
    if n !== nothing && _tiny_inverse_size(n)
        _inverse_batch_tiny_shape(mats, n)
        if _inverse_batch_default_augmented(options)
            return _inverse_batch_tiny!(map(similar, mats), mats, n, options)
        end
    end
    return [_inverse_batch_inverse(A, options) for A in mats]
end

pluq_batched_4x4!(
    mats::AbstractVector{<:GPUModMatrix};
    options::PLUQOptions = PLUQOptions(),
) = _inverse_batch_factor!(mats, options; fixed_size = 4)

inverse_batched_4x4!(
    mats::AbstractVector{<:GPUModMatrix};
    options::PLUQOptions = PLUQOptions(),
) = _inverse_batch_tiny!(mats, mats, 4, options; inplace = true)

inverse_batched_4x4!(
    dests::AbstractVector{<:GPUModMatrix},
    mats::AbstractVector{<:GPUModMatrix};
    options::PLUQOptions = PLUQOptions(),
) = _inverse_batch_tiny!(dests, mats, 4, options)

pluq_batched_8x8!(
    mats::AbstractVector{<:GPUModMatrix};
    options::PLUQOptions = PLUQOptions(),
) = _inverse_batch_factor!(mats, options; fixed_size = 8)

inverse_batched_8x8!(
    mats::AbstractVector{<:GPUModMatrix};
    options::PLUQOptions = PLUQOptions(),
) = _inverse_batch_tiny!(mats, mats, 8, options; inplace = true)

inverse_batched_8x8!(
    dests::AbstractVector{<:GPUModMatrix},
    mats::AbstractVector{<:GPUModMatrix};
    options::PLUQOptions = PLUQOptions(),
) = _inverse_batch_tiny!(dests, mats, 8, options)

pluq_batched_16x16!(
    mats::AbstractVector{<:GPUModMatrix};
    options::PLUQOptions = PLUQOptions(),
) = _inverse_batch_factor!(mats, options; fixed_size = 16)

inverse_batched_16x16!(
    mats::AbstractVector{<:GPUModMatrix};
    options::PLUQOptions = PLUQOptions(),
) = _inverse_batch_tiny!(mats, mats, 16, options; inplace = true)

inverse_batched_16x16!(
    dests::AbstractVector{<:GPUModMatrix},
    mats::AbstractVector{<:GPUModMatrix};
    options::PLUQOptions = PLUQOptions(),
) = _inverse_batch_tiny!(dests, mats, 16, options)

pluq_batched_32x32!(
    mats::AbstractVector{<:GPUModMatrix};
    options::PLUQOptions = PLUQOptions(),
) = _inverse_batch_factor!(mats, options; fixed_size = 32)

inverse_batched_32x32!(
    mats::AbstractVector{<:GPUModMatrix};
    options::PLUQOptions = PLUQOptions(),
) = _inverse_batch_tiny!(mats, mats, 32, options; inplace = true)

inverse_batched_32x32!(
    dests::AbstractVector{<:GPUModMatrix},
    mats::AbstractVector{<:GPUModMatrix};
    options::PLUQOptions = PLUQOptions(),
) = _inverse_batch_tiny!(dests, mats, 32, options)
