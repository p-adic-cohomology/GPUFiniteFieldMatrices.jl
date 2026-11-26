function _inverse_zero_matrix(A::GPUModMatrix{T}, dims::Dims{2}) where {T}
    dims == size(A) && return similar(A)
    result = _stripe_result(A, dims, modulus(A))
    fill!(parent(result), zero(T))
    return result
end

function _inverse_device_indices(A, indices)
    device_indices = similar(_elementwise_native_root(A), Int32, length(indices))
    copyto!(device_indices, Int32.(indices))
    return device_indices
end

@kernel inbounds = true function _inverse_extract_kernel!(
    out,
    packed,
    rank,
    ::Val{Lower},
) where {Lower}
    i, j = @index(Global, NTuple)
    T = eltype(out)
    value = if Lower
        i == j ? one(T) : (i > j && j <= rank ? packed[i, j] : zero(T))
    else
        i <= rank && i <= j ? packed[i, j] : zero(T)
    end
    out[i, j] = value
end

function _inverse_extract(F::PLUQ, lower::Val)
    A = F.LU
    _inverse_preflight(A, PLUQOptions())
    m, n = size(A)
    0 <= F.rank <= min(m, n) || throw(ArgumentError("invalid PLUQ rank"))
    dims = (m, lower isa Val{true} ? m : n)
    out = _inverse_zero_matrix(A, dims)
    if !isempty(out)
        native = parent(out)
        _inverse_extract_kernel!(KernelInterface.get_backend(native))(
            native,
            parent(A),
            F.rank,
            lower;
            ndrange = dims,
        )
    end
    return out
end

_pluq_extract_L(F::PLUQ) = _inverse_extract(F, Val(true))
_pluq_extract_U(F::PLUQ) = _inverse_extract(F, Val(false))

@kernel inbounds = true function _inverse_gather_kernel!(out, A, p, q)
    i, j = @index(Global, NTuple)
    out[i, j] = A[p[i], q[j]]
end

function _inverse_gather!(out, source, p, q)
    (isempty(p) || isempty(q)) && return out
    dp, dq = _inverse_device_indices(out, p), _inverse_device_indices(out, q)
    _inverse_gather_kernel!(KernelInterface.get_backend(out))(
        out,
        source,
        dp,
        dq;
        ndrange = (length(p), length(q)),
    )
    return out
end
