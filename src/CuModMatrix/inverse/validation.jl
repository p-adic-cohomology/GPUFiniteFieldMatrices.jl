function _inverse_is_prime(N::Int)
    N >= 2 || return false
    N == 2 && return true
    iseven(N) && return false
    for divisor = 3:2:isqrt(N)
        iszero(rem(N, divisor)) && return false
    end
    return true
end

function _inverse_preflight(A::GPUModMatrix, options::PLUQOptions)
    N = modulus(A)
    N >= 2 || throw(DomainError(N, "inverse algorithms require a modulus of at least 2"))
    N <= typemax(Int32) || throw(OverflowError("inverse modulus exceeds the Int32 limit"))
    _storage_backend(parent(A)) !== nothing ||
        throw(ArgumentError("inverse algorithms require CUDA or Metal storage"))
    eltype(A) <: AbstractFloat ||
        throw(ArgumentError("inverse algorithms require floating-point residue storage"))
    options.check_prime &&
        !_inverse_is_prime(N) &&
        throw(DomainError(N, "inverse modulus must be prime"))
    return nothing
end

function _inverse_check_product_budget(A)
    find_max_ops(eltype(A), modulus(A)) >= 1 || throw(
        OverflowError(
            "modulus $(modulus(A)) is too large for one modular product in $(eltype(A)) storage",
        ),
    )
    return nothing
end

function _inverse_options(options, A)
    _inverse_preflight(A, options)
    options.autotune || return options
    n = max(size(A)...)
    T = eltype(A)
    blocksize, nftb, warp_threshold, tile, trsm_mode = if n <= 32
        (32, 4, 16, 8, :auto)
    elseif n <= 256
        (64, T === Float32 ? 8 : 4, 24, 16, :auto)
    elseif n <= 1536
        (32, T === Float32 ? 8 : 6, 32, 16, :auto)
    else
        (n <= 3072 ? 32 : 64, T === Float32 ? 8 : 6, 32, 16, :panel)
    end
    return PLUQOptions(
        options;
        blocksize,
        nftb,
        schur_tile = tile,
        trsm_mode,
        trsm_warp_threshold = warp_threshold,
        basecase = 32,
        autotune = false,
    )
end

function _inverse_valid_permutation(p, n)
    length(p) == n || return false
    seen = falses(n)
    for i in p
        1 <= i <= n && !seen[i] || return false
        seen[i] = true
    end
    return true
end

function _inverse_validate_factor(F::PLUQ)
    m, n = size(F.LU)
    0 <= F.rank <= min(m, n) || throw(ArgumentError("invalid PLUQ rank"))
    _inverse_valid_permutation(F.p, m) ||
        throw(ArgumentError("invalid PLUQ row permutation"))
    _inverse_valid_permutation(F.q, n) ||
        throw(ArgumentError("invalid PLUQ column permutation"))
    return nothing
end

"""
    pluq_check_identity(F, A)

Check the modular factorization identity `A[F.p, F.q] = F.L * F.U` on GPU.
"""
function pluq_check_identity(F::PLUQ, A::GPUModMatrix)
    size(F.LU) == size(A) || throw(DimensionMismatch("PLUQ shape must match the input"))
    modulus(F.LU) == modulus(A) ||
        throw(ModulusMismatchError("PLUQ modulus must match the input"))
    _check_elementwise_backend(F.LU, A)
    _inverse_validate_factor(F)
    isempty(A) && return true
    permuted = _inverse_zero_matrix(A, size(A))
    _inverse_gather!(parent(permuted), parent(A), F.p, F.q)
    mod_elements!(permuted)
    product = similar(A)
    LinearAlgebra.mul!(product, F.L, F.U)
    return product == permuted
end
