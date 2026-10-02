@inline _to_i32(x::Integer) = Int32(x)

const _PLUQ_MAX_SAFE_MODULUS = Int(typemax(Int32))

function _pluq_is_prime_host(n::Integer)
    n < 2 && return false
    n == 2 && return true
    iseven(n) && return false
    d = 3
    limit = isqrt(n)
    while d <= limit
        n % d == 0 && return false
        d += 2
    end
    return true
end

function _validate_inverse_modulus(options::PLUQOptions, A::CuModMatrix)
    N = A.N
    if N < 2
        throw(CuModMatrixModulusNotPrimeException("modulus $(N) must be at least 2"))
    end
    if N > _PLUQ_MAX_SAFE_MODULUS
        throw(
            InverseOverflowError(
                "modulus $(N) exceeds the Int32 CUDA-kernel limit $(_PLUQ_MAX_SAFE_MODULUS)",
            ),
        )
    end
    if options.check_prime && !_pluq_is_prime_host(N)
        throw(CuModMatrixModulusNotPrimeException("modulus $(N) is not prime"))
    end
    return nothing
end

function _validate_inverse_matmul_modulus(A::CuModMatrix)
    if find_max_ops(eltype(A.data), A.N) < 1
        throw(
            InverseOverflowError(
                "modulus $(A.N) is too large for one modular matrix multiplication with datatype $(eltype(A.data))",
            ),
        )
    end
    return nothing
end

function _pluq_host_i32_buffer()
    return CUDA.pin(Vector{Int32}(undef, 1))
end

function _pluq_read_i32!(hostbuf::Vector{Int32}, devbuf)
    copyto!(hostbuf, devbuf)
    return Int(hostbuf[1])
end

function _pluq_autotune_options(options::PLUQOptions, n::Int, T::DataType)
    if !options.autotune
        return options
    end
    if n <= 32
        return PLUQOptions(
            options;
            blocksize = 32,
            basecase = 32,
            nftb = 4,
            trsm_mode = :auto,
            trsm_warp_threshold = 16,
            schur_tile = 8,
            autotune = false,
            batch_streams = max(1, options.batch_streams),
        )
    elseif n <= 256
        return PLUQOptions(
            options;
            blocksize = 64,
            basecase = 32,
            nftb = T == Float32 ? 8 : 4,
            trsm_mode = :auto,
            trsm_warp_threshold = 24,
            schur_tile = 16,
            autotune = false,
            batch_streams = max(1, options.batch_streams),
        )
    elseif n <= 1536
        return PLUQOptions(
            options;
            blocksize = 32,
            basecase = 32,
            nftb = T == Float32 ? 8 : 6,
            trsm_mode = :auto,
            trsm_warp_threshold = 32,
            schur_tile = 16,
            autotune = false,
            batch_streams = max(1, options.batch_streams),
        )
    elseif n <= 3072
        return PLUQOptions(
            options;
            blocksize = 32,
            basecase = 32,
            nftb = T == Float32 ? 8 : 6,
            trsm_mode = :panel,
            trsm_warp_threshold = 32,
            schur_tile = 16,
            autotune = false,
            batch_streams = max(1, options.batch_streams),
        )
    end
    return PLUQOptions(
        options;
        blocksize = 64,
        basecase = 32,
        nftb = T == Float32 ? 8 : 6,
        trsm_mode = :panel,
        trsm_warp_threshold = 32,
        schur_tile = 16,
        autotune = false,
        batch_streams = max(1, options.batch_streams),
    )
end

function _resolve_options(options::PLUQOptions, A::CuModMatrix)
    _validate_inverse_modulus(options, A)
    return _pluq_autotune_options(options, max(rows(A), cols(A)), eltype(A.data))
end

"""
    pluq_new!(A; options=PLUQOptions())

Compute a blocked PLUQ factorization in place on GPU-resident data and return
`PLUQFactorization` with packed LU and permutation vectors.

`A.N` must be prime. Pass `check_prime=true` in `PLUQOptions` to validate this
precondition on the host before launching GPU kernels.
"""
function pluq_new!(A::CuModMatrix; options::PLUQOptions = PLUQOptions())
    opts = _resolve_options(options, A)
    m = rows(A)
    n = cols(A)
    p, q, rank = if m == n
        pluq_blocked_gpu!(A.data, A.N, opts, n)
    else
        # Normal dense panels complete entirely on device.  Retain an exact
        # copy only to recover the rare zero-column event with the established
        # complete-pivot reference factorization.
        original = copy(A.data)
        bp, bq, brank = pluq_rectangular_rank_gpu!(A.data, A.N, m, n, options = opts)
        if brank == min(m, n)
            bp, bq, brank
        else
            # The failed panel has queued writes on the default stream. Make
            # the restore explicit before the exceptional reference fallback.
            CUDA.synchronize()
            copyto!(A.data, original)
            CUDA.synchronize()
            pluq_rectangular_rank_reference_gpu!(A.data, A.N, m, n, options = opts)
        end
    end
    return PLUQFactorization(A, p, q, rank)
end

"""
    pluq_new(A; options=PLUQOptions())

Compute a blocked PLUQ factorization on a copy of `A`.

`A.N` must be prime. Pass `check_prime=true` in `PLUQOptions` to validate this
precondition on the host before launching GPU kernels.
"""
function pluq_new(A::CuModMatrix; options::PLUQOptions = PLUQOptions())
    Adata = copy(A.data)
    Awork = CuModMatrix(Adata, A.N; new_size = size(A))
    return pluq_new!(Awork, options = options)
end

"""
    is_invertible_new(A; options=PLUQOptions())

Return `true` iff `A` has full rank under the new GPU PLUQ path.

`A.N` must be prime. Pass `check_prime=true` in `PLUQOptions` to validate this
precondition on the host before launching GPU kernels.
"""
function is_invertible_new(A::CuModMatrix; options::PLUQOptions = PLUQOptions())
    m = rows(A)
    n = cols(A)
    F = pluq_new(A, options = options)
    return m == n && F.rank == m
end

"""
    pluq_init_aug_kernel!(aug, Adata, n)

Initialize augmented matrix `[A | I]` in `aug` for Gauss-Jordan inversion.
Internal kernel used by `inverse_new`.
"""
function pluq_init_aug_kernel!(aug, Adata, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    if i <= n && j <= 2n
        if j <= n
            aug[i, j] = Adata[i, j]
        else
            aug[i, j] = (j - n == i) ? one(eltype(aug)) : zero(eltype(aug))
        end
    end
    return
end

"""
    pluq_aug_find_pivot_kernel!(aug, pivot_slot, k, n, N)

Find first nonzero pivot row in column `k` of augmented matrix.
Internal kernel used by `inverse_new`.
"""
function pluq_aug_find_pivot_kernel!(aug, pivot_slot, k::Int32, n::Int32, N::Int32)
    gtid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ltid = Int(threadIdx().x)
    span = n - k + 1
    total = span
    stride = blockDim().x * gridDim().x
    local_min = Int32(n + 1)
    idx = gtid
    while idx <= total
        i = k + idx - 1
        v = _pluq_mod_t(aug[i, k], N)
        if v != zero(eltype(aug))
            local_min = min(local_min, i)
        end
        idx += stride
    end
    smins = CuStaticSharedArray(Int32, 256)
    smins[ltid] = local_min
    sync_threads()
    step = Int(blockDim().x) >>> 1
    while step >= 1
        if ltid <= step
            smins[ltid] = min(smins[ltid], smins[ltid+step])
        end
        sync_threads()
        step >>>= 1
    end
    if ltid == 1
        CUDA.@atomic pivot_slot[1] = min(pivot_slot[1], smins[1])
    end
    return
end

function pluq_aug_find_pivot_warp_kernel!(aug, pivot_slot, k::Int32, n::Int32, N::Int32)
    lane = Int(threadIdx().x)
    if lane > 32
        return
    end
    span = n - k + 1
    row = k + Int32(lane - 1)
    pred = Int32(lane) <= span && _pluq_mod_t(aug[row, k], N) != zero(eltype(aug))
    bits = CUDA.vote_ballot_sync(CUDA.FULL_MASK, pred)
    if lane == 1
        if bits == UInt32(0)
            pivot_slot[1] = Int32(n + 1)
        else
            pivot_slot[1] = k + Int32(trailing_zeros(bits))
        end
    end
    return
end

function pluq_aug_find_pivot_warp_shfl_kernel!(
    aug,
    pivot_slot,
    k::Int32,
    n::Int32,
    N::Int32,
)
    lane = Int32(threadIdx().x)
    if lane > Int32(32)
        return
    end
    span = n - k + Int32(1)
    local_min = n + Int32(1)
    if lane <= span
        row = k + lane - Int32(1)
        if _pluq_mod_t(aug[row, k], N) != zero(eltype(aug))
            local_min = row
        end
    end
    wmin = _pluq_warp_min_shfl_i32(local_min)
    if lane == Int32(1)
        pivot_slot[1] = wmin
    end
    return
end

"""
    pluq_aug_scale_row_kernel!(aug, row, jstart, n2, invpivot, N)

Scale one augmented row by the modular inverse of the pivot.
Internal kernel used by `inverse_new`.
"""
function pluq_aug_scale_row_from_diag_kernel!(
    aug,
    row::Int32,
    jstart::Int32,
    n2::Int32,
    N::Int32,
)
    invslot = CuStaticSharedArray(eltype(aug), 1)
    if threadIdx().x == 1
        invslot[1] = _pluq_mod_inv_t(aug[row, row], N)
    end
    sync_threads()
    invpivot = invslot[1]
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x + jstart - 1
    stride = blockDim().x * gridDim().x
    while j <= n2
        aug[row, j] = _pluq_mod_mul_t(aug[row, j], invpivot, N)
        j += stride
    end
    return
end

"""
    pluq_aug_elim_kernel!(aug, k, n, n2, N)

Eliminate column `k` from all rows except the pivot row in augmented matrix.
Internal kernel used by `inverse_new`.
"""
function pluq_aug_elim_kernel!(aug, k::Int32, n::Int32, n2::Int32, N::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y + k - Int32(1)
    if i <= n && j <= n2 && i != k
        f = _pluq_mod_t(aug[i, k], N)
        if f != zero(eltype(aug))
            aug[i, j] = _pluq_mod_t(aug[i, j] - _pluq_mod_mul_t(f, aug[k, j], N), N)
        end
    end
    return
end

"""
    pluq_copy_block_kernel!(dest, src, n)

Copy the top-left `n x n` block from `src` to `dest`.
Internal utility used by `inverse_new`.
"""
function pluq_copy_block_kernel!(dest, src, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    if i <= n && j <= n
        dest[i, j] = src[i, j]
    end
    return
end

"""
    inverse_new(A; options=PLUQOptions())

Compute the inverse over `mod N`.

The default `PLUQOptions` route uses the PLUQ factorization pipeline. Passing
`PLUQOptions(inverse_strategy=:augmented)` selects the GPU Gauss-Jordan
reference path on `[A I]`.

`A.N` must be prime. Pass `check_prime=true` in `PLUQOptions` to validate this
precondition on the host before launching GPU kernels.
"""
function inverse_new(A::CuModMatrix; options::PLUQOptions = PLUQOptions())
    if rows(A) != cols(A)
        throw(CuModMatrixNotSquareException("matrix must be square"))
    end
    opts = _resolve_options(options, A)
    if opts.inverse_strategy == :pluq
        return inverse_pluq_new(A, options = opts)
    end
    n = rows(A)
    N = A.N
    n2 = 2 * n
    aug = CUDA.zeros(eltype(A.data), size(A.data, 1), size(A.data, 2) + n + TILE_WIDTH)
    tx = 16
    ty = 16
    bx = max(1, cld(n, tx))
    by = max(1, cld(n2, ty))
    n32 = _to_i32(n)
    n232 = _to_i32(n2)
    N32 = _to_i32(N)
    @cuda threads=(tx, ty) blocks=(bx, by) pluq_init_aug_kernel!(aug, A.data, n32)
    threads = 256
    pivot_slot = CUDA.fill(_to_i32(n + 1), 1)
    pivot_host = _pluq_host_i32_buffer()
    for k = 1:n
        k32 = _to_i32(k)
        fill!(pivot_slot, _to_i32(n + 1))
        if n - k + 1 <= 32
            if opts.pivot_warp_kernel == :shfl
                @cuda threads=32 blocks=1 pluq_aug_find_pivot_warp_shfl_kernel!(
                    aug,
                    pivot_slot,
                    k32,
                    n32,
                    N32,
                )
            else
                @cuda threads=32 blocks=1 pluq_aug_find_pivot_warp_kernel!(
                    aug,
                    pivot_slot,
                    k32,
                    n32,
                    N32,
                )
            end
        else
            @cuda threads=threads blocks=max(1, cld(n - k + 1, threads)) pluq_aug_find_pivot_kernel!(
                aug,
                pivot_slot,
                k32,
                n32,
                N32,
            )
        end
        prow = _pluq_read_i32!(pivot_host, pivot_slot)
        if prow > n
            throw(InverseNotDefinedException("matrix is singular modulo $(A.N)"))
        end
        if prow != k
            @cuda threads=threads blocks=max(1, cld(n2, threads)) pluq_swap_rows_kernel!(
                aug,
                k32,
                _to_i32(prow),
                n232,
            )
        end
        @cuda threads=threads blocks=max(1, cld(n2 - k + 1, threads)) pluq_aug_scale_row_from_diag_kernel!(
            aug,
            k32,
            k32,
            n232,
            N32,
        )
        bx2 = max(1, cld(n, tx))
        by2 = max(1, cld(n2 - k + 1, ty))
        @cuda threads=(tx, ty) blocks=(bx2, by2) pluq_aug_elim_kernel!(
            aug,
            k32,
            n32,
            n232,
            N32,
        )
    end
    invdata = @view aug[1:n, (n+1):n2]
    out = CUDA.zeros(eltype(A.data), size(A.data, 1), size(A.data, 2))
    bx3 = max(1, cld(n, tx))
    by3 = max(1, cld(n, ty))
    @cuda threads=(tx, ty) blocks=(bx3, by3) pluq_copy_block_kernel!(out, invdata, n32)
    return CuModMatrix(out, N; new_size = (n, n))
end

"""
    inverse_pluq_new(A; options=PLUQOptions())

Compute the inverse over `mod N` through the PLUQ factorization pipeline:
factor `A`, extract `L` and `U`, invert the triangular factors, multiply them,
and apply row/column permutations.

`A.N` must be prime. Pass `check_prime=true` in `PLUQOptions` to validate this
precondition on the host before launching GPU kernels.
"""
function inverse_pluq_new(A::CuModMatrix; options::PLUQOptions = PLUQOptions())
    if rows(A) != cols(A)
        throw(CuModMatrixNotSquareException("matrix must be square"))
    end
    opts = _resolve_options(options, A)
    _validate_inverse_matmul_modulus(A)
    n = rows(A)
    F = pluq_new(A, options = opts)
    if F.rank != n
        throw(InverseNotDefinedException("matrix is singular modulo $(A.N)"))
    end
    L = pluq_extract_L(F)
    U = pluq_extract_U(F)
    Linv = lower_triangular_inverse_no_copy(L)
    Uinv = upper_triangular_inverse_no_copy(U)
    M = zeros(eltype(A.data), n, n, A.N)
    mul!(M, Uinv, Linv)
    # P*A*Q = L*U implies A^-1 = Q*U^-1*L^-1*P.  The stored vectors are
    # gather permutations, so applying Q and P on these opposite sides needs
    # their inverse gather maps.
    pdev = CuArray(Int32.(pluq_inverse_perm(F.q)))
    qdev = CuArray(Int32.(pluq_inverse_perm(F.p)))
    out = zeros(eltype(A.data), n, n, A.N)
    tx = 16
    ty = 16
    @cuda threads=(tx, ty) blocks=(max(1, cld(n, tx)), max(1, cld(n, ty))) pluq_apply_paq_kernel!(
        out.data,
        M.data,
        pdev,
        qdev,
        Int32(n),
    )
    return out
end

function pluq_init_rect_aug_kernel!(aug, Adata, m::Int32, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    if i <= m && j <= (n + m)
        if j <= n
            aug[i, j] = Adata[i, j]
        else
            aug[i, j] = (j - n == i) ? one(eltype(aug)) : zero(eltype(aug))
        end
    end
    return
end

function pluq_scale_row_rect_aug_from_diag_kernel!(
    aug,
    row::Int32,
    jstart::Int32,
    w::Int32,
    N::Int32,
)
    invslot = CuStaticSharedArray(eltype(aug), 1)
    if threadIdx().x == 1
        invslot[1] = _pluq_mod_inv_t(aug[row, row], N)
    end
    sync_threads()
    invpivot = invslot[1]
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x + jstart - 1
    stride = blockDim().x * gridDim().x
    while j <= w
        aug[row, j] = _pluq_mod_mul_t(aug[row, j], invpivot, N)
        j += stride
    end
    return
end

"""Capture one stable elimination factor per row before modifying column `k`."""
function pluq_rect_elim_factors_kernel!(factors, aug, k::Int32, m::Int32, N::Int32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if i <= m
        factors[i] = i == k ? zero(eltype(factors)) : _pluq_mod_t(aug[i, k], N)
    end
    return
end

function pluq_elim_rect_aug_kernel!(aug, factors, k::Int32, m::Int32, w::Int32, N::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y + k - Int32(1)
    if i <= m && j <= w && i != k
        f = factors[i]
        if f != zero(eltype(aug))
            aug[i, j] = _pluq_mod_t(aug[i, j] - _pluq_mod_mul_t(f, aug[k, j], N), N)
        end
    end
    return
end

function pluq_load_z_kernel!(Z, Y, m::Int32, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    if i <= n && j <= m
        if i <= m
            Z[i, j] = Y[i, j]
        else
            Z[i, j] = zero(eltype(Z))
        end
    end
    return
end

function pluq_scatter_solution_kernel!(X, Z, qdev, n::Int32, m::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    if i <= n && j <= m
        dst = qdev[i]
        X[dst, j] = Z[i, j]
    end
    return
end

"""Copy the logical `m × n` leading block of `src` into `dest` on device."""
function pluq_copy_rect_block_kernel!(dest, src, m::Int32, n::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    if i <= m && j <= n
        dest[i, j] = src[i, j]
    end
    return
end

"""Gather `m` selected columns into a square matrix, with row-contiguous lanes."""
function pluq_gather_selected_columns_kernel!(dest, src, qdev, m::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    if i <= m && j <= m
        dest[i, j] = src[i, qdev[j]]
    end
    return
end

"""Scatter the rows of a selected square inverse into a wide right inverse."""
function pluq_scatter_selected_inverse_kernel!(dest, src, qdev, m::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y
    if i <= m && j <= m
        dest[qdev[i], j] = src[i, j]
    end
    return
end

"""
    _right_inverse_leading_block(A, opts)

Try the inexpensive right inverse induced by the leading square block of a
wide matrix.  When that block is nonsingular, `[B⁻¹; 0]` is a right inverse of
`A = [B C]`.  Return `nothing` if the block is singular so the rank-revealing
general path can select columns elsewhere in `A`.
"""
function _right_inverse_leading_block(A::CuModMatrix, opts::PLUQOptions)
    m = rows(A)
    n = cols(A)
    m == n && return nothing
    T = eltype(A.data)
    B = zeros(T, m, m, A.N)
    tx = 16
    ty = 16
    @cuda threads=(tx, ty) blocks=(max(1, cld(m, tx)), max(1, cld(m, ty))) pluq_copy_rect_block_kernel!(
        B.data,
        A.data,
        Int32(m),
        Int32(m),
    )
    Binv = try
        inverse_pluq_new(B, options = PLUQOptions(opts; inverse_strategy = :pluq))
    catch err
        if err isa InverseNotDefinedException
            return nothing
        end
        rethrow()
    end
    X = zeros(T, n, m, A.N)
    @cuda threads=(tx, ty) blocks=(max(1, cld(m, tx)), max(1, cld(m, ty))) pluq_copy_rect_block_kernel!(
        X.data,
        Binv.data,
        Int32(m),
        Int32(m),
    )
    return X
end

"""Construct a right inverse from the independent columns selected by PLUQ."""
function _right_inverse_selected_columns(A::CuModMatrix, opts::PLUQOptions)
    m = rows(A)
    n = cols(A)
    F = pluq_new(A, options = opts)
    F.rank == m || return nothing
    qdev = CuArray(Int32.(F.q[1:m]))
    T = eltype(A.data)
    B = zeros(T, m, m, A.N)
    tx = 16
    ty = 16
    @cuda threads=(tx, ty) blocks=(max(1, cld(m, tx)), max(1, cld(m, ty))) pluq_gather_selected_columns_kernel!(
        B.data,
        A.data,
        qdev,
        Int32(m),
    )
    Binv = try
        inverse_pluq_new(B, options = PLUQOptions(opts; inverse_strategy = :pluq))
    catch err
        err isa InverseNotDefinedException || rethrow()
        return nothing
    end
    X = zeros(T, n, m, A.N)
    @cuda threads=(tx, ty) blocks=(max(1, cld(m, tx)), max(1, cld(m, ty))) pluq_scatter_selected_inverse_kernel!(
        X.data,
        Binv.data,
        qdev,
        Int32(m),
    )
    return X
end

"""
    right_inverse_new(A; options=PLUQOptions())

Compute a right inverse `X` such that `A*X = I` for full row-rank rectangular
`A` with `rows(A) <= cols(A)` over `GF(N)`.

When the leading `m × m` block is nonsingular, this uses the blocked PLUQ
inverse of that block and returns `[B⁻¹; 0]`.  Otherwise it falls back to
Gauss-Jordan elimination on `[A | I_m]` with row and column pivoting. If rank
is smaller than `m`, no right inverse exists and an exception is thrown.
`inverse_strategy` controls the square fast path; the fallback is augmented
rank-revealing elimination.

`A.N` must be prime. Pass `check_prime=true` in `PLUQOptions` to validate this
precondition on the host before launching GPU kernels.

Example:
```julia
A = CuModMatrix([1 2 3; 0 1 4], 101)
X = right_inverse_new(A)
Array(A * X)
```
"""
function right_inverse_new(A::CuModMatrix; options::PLUQOptions = PLUQOptions())
    opts = _resolve_options(options, A)
    m = rows(A)
    n = cols(A)
    if m > n
        throw(CuModArraySizeMismatchException("right inverse requires rows(A) <= cols(A)"))
    end
    # A dense random wide matrix almost always has a nonsingular leading
    # square block.  Use the blocked PLUQ inverse in that case; the augmented
    # fallback below remains necessary for arbitrary full-row-rank matrices.
    if m < n
        fast = _right_inverse_leading_block(A, opts)
        fast === nothing || return fast
        selected = _right_inverse_selected_columns(A, opts)
        selected === nothing || return selected
    end
    N = A.N
    w = n + m
    aug = CUDA.zeros(eltype(A.data), size(A.data, 1), w + TILE_WIDTH)
    tx = 16
    ty = 16
    m32 = _to_i32(m)
    n32 = _to_i32(n)
    w32 = _to_i32(w)
    N32 = _to_i32(N)
    @cuda threads=(tx, ty) blocks=(max(1, cld(m, tx)), max(1, cld(w, ty))) pluq_init_rect_aug_kernel!(
        aug,
        A.data,
        m32,
        n32,
    )
    q = collect(1:n)
    lq = opts.lazy_q ? collect(1:n) : Int[]
    threads = 256
    pivot_slot = CUDA.fill(_to_i32(m * n + 1), 1)
    pivot_host = _pluq_host_i32_buffer()
    elim_factors = CUDA.zeros(eltype(A.data), m)
    rank = 0
    for k = 1:m
        span_r = m - k + 1
        span_c = n - k + 1
        if span_r <= 0 || span_c <= 0
            break
        end
        total = span_r * span_c
        k32 = _to_i32(k)
        fill!(pivot_slot, _to_i32(total + 1))
        if span_r <= 32
            if opts.pivot_warp_kernel == :shfl
                @cuda threads=32 blocks=1 pluq_find_pivot_rect_warp_shfl_kernel!(
                    aug,
                    pivot_slot,
                    k32,
                    m32,
                    n32,
                    N32,
                )
            else
                @cuda threads=32 blocks=1 pluq_find_pivot_rect_warp_kernel!(
                    aug,
                    pivot_slot,
                    k32,
                    m32,
                    n32,
                    N32,
                )
            end
        else
            @cuda threads=threads blocks=max(1, cld(total, threads)) pluq_find_pivot_rect_kernel!(
                aug,
                pivot_slot,
                k32,
                m32,
                n32,
                N32,
            )
        end
        pivlin = _pluq_read_i32!(pivot_host, pivot_slot)
        if pivlin > total
            break
        end
        joff = (pivlin - 1) ÷ span_r
        ioff = (pivlin - 1) % span_r
        prow = k + ioff
        pcol = k + joff
        if prow != k
            @cuda threads=threads blocks=max(1, cld(w, threads)) pluq_swap_rows_kernel!(
                aug,
                k32,
                _to_i32(prow),
                w32,
            )
        end
        if pcol != k
            @cuda threads=threads blocks=max(1, cld(m, threads)) pluq_swap_cols_kernel!(
                aug,
                k32,
                _to_i32(pcol),
                m32,
            )
            if opts.lazy_q
                lq[k], lq[pcol] = lq[pcol], lq[k]
            else
                q[k], q[pcol] = q[pcol], q[k]
            end
        end
        @cuda threads=threads blocks=max(1, cld(w - k + 1, threads)) pluq_scale_row_rect_aug_from_diag_kernel!(
            aug,
            k32,
            k32,
            w32,
            N32,
        )
        @cuda threads=threads blocks=max(1, cld(m, threads)) pluq_rect_elim_factors_kernel!(
            elim_factors,
            aug,
            k32,
            m32,
            N32,
        )
        @cuda threads=(tx, ty) blocks=(max(1, cld(m, tx)), max(1, cld(w - k + 1, ty))) pluq_elim_rect_aug_kernel!(
            aug,
            elim_factors,
            k32,
            m32,
            w32,
            N32,
        )
        rank += 1
    end
    if rank != m
        throw(
            InverseNotDefinedException(
                "matrix is not full row-rank modulo $(A.N); right inverse undefined",
            ),
        )
    end
    Y = @view aug[1:m, (n+1):(n+m)]
    Z = CUDA.zeros(eltype(A.data), n + TILE_WIDTH, m + TILE_WIDTH)
    X = CUDA.zeros(eltype(A.data), n + TILE_WIDTH, m + TILE_WIDTH)
    @cuda threads=(tx, ty) blocks=(max(1, cld(n, tx)), max(1, cld(m, ty))) pluq_load_z_kernel!(
        Z,
        Y,
        m32,
        n32,
    )
    if opts.lazy_q
        q = lq
    end
    qdev = CuArray(Int32.(q))
    @cuda threads=(tx, ty) blocks=(max(1, cld(n, tx)), max(1, cld(m, ty))) pluq_scatter_solution_kernel!(
        X,
        Z,
        qdev,
        n32,
        m32,
    )
    return CuModMatrix(X, N; new_size = (n, m))
end

"""
    left_inverse_new(A; options=PLUQOptions())

Compute a left inverse `X` such that `X*A = I` for full column-rank
rectangular `A` with `rows(A) >= cols(A)` over `GF(N)`.
`inverse_strategy` only controls square `inverse_new`; this rectangular path is
always augmented rank-revealing elimination via transposition.

`A.N` must be prime. Pass `check_prime=true` in `PLUQOptions` to validate this
precondition on the host before launching GPU kernels.

Example:
```julia
A = CuModMatrix([1 2; 0 1; 3 4], 101)
X = left_inverse_new(A)
Array(X * A)
```
"""
function left_inverse_new(A::CuModMatrix; options::PLUQOptions = PLUQOptions())
    m = rows(A)
    n = cols(A)
    if m < n
        throw(CuModArraySizeMismatchException("left inverse requires rows(A) >= cols(A)"))
    end
    ATdata = permutedims(A.data, (2, 1))
    AT = CuModMatrix(ATdata, A.N; new_size = (n, m))
    R = right_inverse_new(AT, options = options)
    Ldata = permutedims(R.data, (2, 1))
    return CuModMatrix(Ldata, A.N; new_size = (n, m))
end

"""
    pluq_new_batch(mats; options=PLUQOptions())

Run `pluq_new` across a batch of matrices and return factorization objects.

Each modulus must be prime. Pass `check_prime=true` in `PLUQOptions` to validate
the precondition for the first matrix before launching GPU kernels.
"""
function pluq_new_batch(
    mats::AbstractVector{<:CuModMatrix};
    options::PLUQOptions = PLUQOptions(),
)
    if !isempty(mats)
        n = rows(mats[1])
        if n == cols(mats[1]) && all(A -> rows(A) == n && cols(A) == n, mats)
            if n == 4
                return pluq_batched_4x4!(mats)
            elseif n == 8
                return pluq_batched_8x8!(mats)
            elseif n == 16
                return pluq_batched_16x16!(mats)
            elseif n == 32
                return pluq_batched_32x32!(mats)
            end
        end
    end
    isempty(mats) && return Any[]
    opts = _resolve_options(options, mats[1])
    firstF = pluq_new(mats[1], options = opts)
    out = Vector{typeof(firstF)}(undef, length(mats))
    out[1] = firstF
    nstreams = min(opts.batch_streams, length(mats))
    if nstreams == 1
        for i = 2:length(mats)
            out[i] = pluq_new(mats[i], options = opts)
        end
        return out
    end
    streams = [CuStream() for _ = 1:nstreams]
    for i = 2:length(mats)
        s = streams[(i-1)%nstreams+1]
        CUDA.stream!(s) do
            out[i] = pluq_new(mats[i], options = opts)
        end
    end
    for s in streams
        synchronize(s)
    end
    return out
end

"""
    inverse_new_batch(mats; options=PLUQOptions())

Run the fastest inverse path across a batch. For rectangular matrices this
dispatches to one-sided inverses.

Each modulus must be prime. Pass `check_prime=true` in `PLUQOptions` to validate
the precondition for the first matrix before launching GPU kernels.
"""
function inverse_new_batch(
    mats::AbstractVector{<:CuModMatrix};
    options::PLUQOptions = PLUQOptions(),
)
    if !isempty(mats)
        n = rows(mats[1])
        if n == cols(mats[1]) && all(A -> rows(A) == n && cols(A) == n, mats)
            if n == 4
                return inverse_batched_4x4!(mats)
            elseif n == 8
                return inverse_batched_8x8!(mats)
            elseif n == 16
                return inverse_batched_16x16!(mats)
            elseif n == 32
                return inverse_batched_32x32!(mats)
            end
        end
    end
    isempty(mats) && return CuModMatrix[]
    opts = _resolve_options(options, mats[1])
    out = Vector{CuModMatrix}(undef, length(mats))
    nstreams = min(opts.batch_streams, length(mats))
    if nstreams == 1
        for i in eachindex(mats)
            A = mats[i]
            if rows(A) == cols(A)
                out[i] = inverse_new(A, options = opts)
            elseif rows(A) < cols(A)
                out[i] = right_inverse_new(A, options = opts)
            else
                out[i] = left_inverse_new(A, options = opts)
            end
        end
        return out
    end
    streams = [CuStream() for _ = 1:nstreams]
    for i in eachindex(mats)
        A = mats[i]
        s = streams[(i-1)%nstreams+1]
        CUDA.stream!(s) do
            if rows(A) == cols(A)
                out[i] = inverse_new(A, options = opts)
            elseif rows(A) < cols(A)
                out[i] = right_inverse_new(A, options = opts)
            else
                out[i] = left_inverse_new(A, options = opts)
            end
        end
    end
    for s in streams
        synchronize(s)
    end
    return out
end
