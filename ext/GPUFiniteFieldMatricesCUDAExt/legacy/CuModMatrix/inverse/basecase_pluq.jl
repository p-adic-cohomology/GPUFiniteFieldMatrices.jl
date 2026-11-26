"""
    _pluq_mod_t(x, N)

Device-friendly modulo reduction returning the same element type as `x`.

`Int64` intermediates are used to avoid overflow in intermediate integer
products/sums before reducing modulo `N`, then values are converted back to
the matrix element type (`Float32`/`Float64` in this project).
"""
@inline function _pluq_mod_t(x::T, N::Int32) where {T}
    v = Int64(x)
    r = rem(v, Int64(N))
    if r < 0
        r += Int64(N)
    end
    return T(r)
end

"""
    _pluq_mod_mul_t(a, b, N)

Device-friendly modular multiplication returning the same element type.
"""
@inline function _pluq_mod_mul_baseline_t(a::T, b::T, N::Int32) where {T}
    av = Int64(a)
    bv = Int64(b)
    r = rem(av * bv, Int64(N))
    if r < 0
        r += Int64(N)
    end
    return T(r)
end

@inline function _pluq_mod_mul_barrett_t(a::T, b::T, N::Int32) where {T}
    av = UInt64(Int64(_pluq_mod_t(a, N)))
    bv = UInt64(Int64(_pluq_mod_t(b, N)))
    n = UInt64(UInt32(N))
    x = av * bv
    mu = (UInt64(1) << 32) ÷ n
    q = (x * mu) >>> 32
    r = x - q * n
    if r >= n
        r -= n
    end
    if r >= n
        r -= n
    end
    return T(Int64(r))
end

@inline function _pluq_mod_mul_t(a::T, b::T, N::Int32) where {T}
    if N <= Int32(65535)
        return _pluq_mod_mul_barrett_t(a, b, N)
    end
    return _pluq_mod_mul_baseline_t(a, b, N)
end

@inline function _pluq_mod_inv_t(a::T, N::Int32) where {T}
    aa = Int32(rem(Int64(a), Int64(N)))
    if aa < 0
        aa += N
    end
    if aa == 0
        return zero(T)
    end
    t = Int32(0)
    newt = Int32(1)
    r = N
    newr = aa
    while newr != 0
        q = r ÷ newr
        t, newt = newt, t - q * newt
        r, newr = newr, r - q * newr
    end
    if t < 0
        t += N
    end
    return T(t)
end

"""
    pluq_find_pivot_kernel!(A, pivot_slot, k, kend, N)

Search for the first nonzero pivot in the active block `[k:kend, k:kend]`.
Writes the earliest linearized position into `pivot_slot[1]`.
"""
function pluq_find_pivot_kernel!(A, pivot_slot, k::Int32, kend::Int32, N::Int32)
    gtid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ltid = Int(threadIdx().x)
    span = kend - k + 1
    total = span * span
    stride = blockDim().x * gridDim().x
    idx = gtid
    local_min = Int32(total + 1)
    while idx <= total
        joff = (idx - 1) ÷ span
        ioff = (idx - 1) % span
        i = k + ioff
        j = k + joff
        val = _pluq_mod_t(A[i, j], N)
        if val != zero(eltype(A))
            local_min = min(local_min, idx)
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

function pluq_find_pivot_warp_kernel!(A, pivot_slot, k::Int32, kend::Int32, N::Int32)
    lane = Int(threadIdx().x)
    span = kend - k + 1
    if lane > 32
        return
    end
    for joff = 0:(span-1)
        row = k + lane - 1
        pred = lane <= span && _pluq_mod_t(A[row, k+joff], N) != zero(eltype(A))
        bits = CUDA.vote_ballot_sync(CUDA.FULL_MASK, pred)
        if bits != UInt32(0)
            if lane == 1
                first_lane = trailing_zeros(bits) + 1
                pivot_slot[1] = Int32(joff * span + first_lane)
            end
            return
        end
    end
    return
end

@inline function _pluq_warp_min_shfl_i32(v::Int32)
    v = min(v, CUDA.shfl_down_sync(CUDA.FULL_MASK, v, Int32(16), Int32(32)))
    v = min(v, CUDA.shfl_down_sync(CUDA.FULL_MASK, v, Int32(8), Int32(32)))
    v = min(v, CUDA.shfl_down_sync(CUDA.FULL_MASK, v, Int32(4), Int32(32)))
    v = min(v, CUDA.shfl_down_sync(CUDA.FULL_MASK, v, Int32(2), Int32(32)))
    v = min(v, CUDA.shfl_down_sync(CUDA.FULL_MASK, v, Int32(1), Int32(32)))
    return v
end

function pluq_find_pivot_warp_shfl_kernel!(A, pivot_slot, k::Int32, kend::Int32, N::Int32)
    lane = Int32(threadIdx().x)
    if lane > Int32(32)
        return
    end
    span = kend - k + Int32(1)
    local_min = span * span + Int32(1)
    if lane <= span
        row = k + lane - Int32(1)
        joff = Int32(0)
        while joff < span
            if _pluq_mod_t(A[row, k+joff], N) != zero(eltype(A))
                cand = joff * span + lane
                local_min = min(local_min, cand)
            end
            joff += Int32(1)
        end
    end
    wmin = _pluq_warp_min_shfl_i32(local_min)
    if lane == Int32(1)
        pivot_slot[1] = wmin
    end
    return
end

"""
    pluq_swap_rows_kernel!(A, r1, r2, ncols)

Swap two rows in a dense device matrix.
"""
function pluq_swap_rows_kernel!(A, r1::Int32, r2::Int32, ncols::Int32)
    col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    while col <= ncols
        tmp = A[r1, col]
        A[r1, col] = A[r2, col]
        A[r2, col] = tmp
        col += stride
    end
    return
end

"""
    pluq_swap_cols_kernel!(A, c1, c2, nrows)

Swap two columns in a dense device matrix.
"""
function pluq_swap_cols_kernel!(A, c1::Int32, c2::Int32, nrows::Int32)
    row = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    while row <= nrows
        tmp = A[row, c1]
        A[row, c1] = A[row, c2]
        A[row, c2] = tmp
        row += stride
    end
    return
end

"""
    pluq_scale_column_kernel!(A, k, kend, invpivot, N)

Scale `A[k+1:kend, k]` by `invpivot` modulo `N`.
"""
function pluq_scale_column_from_diag_kernel!(A, k::Int32, kend::Int32, N::Int32)
    invslot = CuStaticSharedArray(eltype(A), 1)
    if threadIdx().x == 1
        invslot[1] = _pluq_mod_inv_t(A[k, k], N)
    end
    sync_threads()
    invpivot = invslot[1]
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + k
    stride = blockDim().x * gridDim().x
    while i <= kend
        A[i, k] = _pluq_mod_mul_t(A[i, k], invpivot, N)
        i += stride
    end
    return
end

function pluq_compute_invpivot_kernel!(invpivot_slot, A, k::Int32, N::Int32)
    if blockIdx().x == 1 && threadIdx().x == 1
        invpivot_slot[1] = _pluq_mod_inv_t(A[k, k], N)
    end
    return
end

"""
    pluq_rank1_update_kernel!(A, k, kend, N)

Apply one elimination update step to the trailing active block.
"""
function pluq_rank1_update_kernel!(A, k::Int32, kend::Int32, N::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x + k
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y + k
    if i <= kend && j <= kend
        aik = A[i, k]
        akj = A[k, j]
        A[i, j] = _pluq_mod_t(A[i, j] - _pluq_mod_mul_t(aik, akj, N), N)
    end
    return
end

"""
    pluq_panel_fused_kernel!(A, p, q, rank_slot, k0, kend, n, N)

Factor one diagonal panel in a single cooperative thread block.  Pivot search,
global row/column swaps, multiplier scaling, and the panel rank-one updates stay
on the device.  This removes the per-pivot device-to-host synchronization and
the four kernel launches used by the reference basecase.

`p` and `q` are device-resident gather permutations and are updated at the same
time as the corresponding matrix swaps.  `rank_slot[1]` is the only value the
blocked driver must read after the whole panel has completed.
"""
function pluq_panel_fused_kernel!(
    A,
    p,
    q,
    dinv,
    rank_slot,
    k0::Int32,
    kend::Int32,
    n::Int32,
    N::Int32,
)
    tid = Int32(threadIdx().x)
    nt = Int32(blockDim().x)
    candidates = CuStaticSharedArray(Int32, 256)
    pivot = CuStaticSharedArray(Int32, 2)
    invpivot = CuStaticSharedArray(eltype(A), 1)

    if tid == Int32(1)
        rank_slot[1] = Int32(0)
    end
    sync_threads()

    k = k0
    while k <= kend
        span = kend - k + Int32(1)
        total = span * span
        local_min = total + Int32(1)
        idx = tid
        while idx <= total
            joff = (idx - Int32(1)) ÷ span
            ioff = (idx - Int32(1)) % span
            if _pluq_mod_t(A[k+ioff, k+joff], N) != zero(eltype(A))
                local_min = min(local_min, idx)
            end
            idx += nt
        end
        candidates[Int(tid)] = local_min
        sync_threads()

        step = nt >>> 1
        while step >= Int32(1)
            if tid <= step
                candidates[Int(tid)] =
                    min(candidates[Int(tid)], candidates[Int(tid + step)])
            end
            sync_threads()
            step >>>= 1
        end
        if tid == Int32(1)
            pos = candidates[1]
            if pos <= total
                pivot[1] = k + (pos - Int32(1)) % span
                pivot[2] = k + (pos - Int32(1)) ÷ span
                rank_slot[1] += Int32(1)
            else
                pivot[1] = Int32(0)
                pivot[2] = Int32(0)
            end
        end
        sync_threads()
        if pivot[1] == Int32(0)
            return
        end

        prow = pivot[1]
        pcol = pivot[2]
        if prow != k
            col = tid
            while col <= n
                tmp = A[k, col]
                A[k, col] = A[prow, col]
                A[prow, col] = tmp
                col += nt
            end
            if tid == Int32(1)
                tmp = p[k]
                p[k] = p[prow]
                p[prow] = tmp
            end
        end
        sync_threads()

        if pcol != k
            row = tid
            while row <= n
                tmp = A[row, k]
                A[row, k] = A[row, pcol]
                A[row, pcol] = tmp
                row += nt
            end
            if tid == Int32(1)
                tmp = q[k]
                q[k] = q[pcol]
                q[pcol] = tmp
            end
        end
        sync_threads()

        if tid == Int32(1)
            invpivot[1] = _pluq_mod_inv_t(A[k, k], N)
            dinv[k] = invpivot[1]
        end
        sync_threads()
        row = k + tid
        while row <= kend
            A[row, k] = _pluq_mod_mul_t(A[row, k], invpivot[1], N)
            row += nt
        end
        sync_threads()

        width = kend - k
        idx = tid
        while idx <= width * width
            joff = (idx - Int32(1)) ÷ width + Int32(1)
            ioff = (idx - Int32(1)) % width + Int32(1)
            row = k + ioff
            col = k + joff
            product = _pluq_mod_mul_t(A[row, k], A[k, col], N)
            A[row, col] = _pluq_mod_t(A[row, col] - product, N)
            idx += nt
        end
        sync_threads()
        k += Int32(1)
    end
    return
end

function pluq_panel_fused_gpu!(
    Adata::CuArray{T,2},
    N::Int,
    pdev,
    qdev,
    dinv,
    rank_slot,
    rank_host,
    k0::Int,
    kend::Int,
    n::Int,
) where {T}
    @cuda threads=256 blocks=1 pluq_panel_fused_kernel!(
        Adata,
        pdev,
        qdev,
        dinv,
        rank_slot,
        Int32(k0),
        Int32(kend),
        Int32(n),
        Int32(N),
    )
    return _pluq_read_i32!(rank_host, rank_slot)
end

"""
    pluq_basecase_gpu!(Adata, N, p, q, k0, kend, n)

Perform in-place PLUQ elimination on a block `[k0:kend, k0:kend]` of `Adata`
using CUDA kernels for pivot search, swaps, scaling, and rank-1 updates.
Returns the rank contributed by this block.
"""
function pluq_basecase_gpu!(
    Adata::CuArray{T,2},
    N::Int,
    p::Vector{Int},
    q::Vector{Int},
    k0::Int,
    kend::Int,
    n::Int;
    options::PLUQOptions = PLUQOptions(),
) where {T}
    rank = 0
    n32 = Int32(n)
    N32 = Int32(N)
    # The shared-memory tree reduction requires a power-of-two block size.
    threads = min(256, max(32, nextpow(2, 32 * options.nftb)))
    maxspan = kend - k0 + 1
    pivot_slot = CUDA.fill(Int32(maxspan * maxspan + 1), 1)
    pivot_host = _pluq_host_i32_buffer()
    locp = options.lazy_q ? collect(1:maxspan) : Int[]
    locq = options.lazy_q ? collect(1:maxspan) : Int[]
    for k = k0:kend
        kk = Int32(k)
        kend32 = Int32(kend)
        span = kend - k + 1
        total = span * span
        fill!(pivot_slot, Int32(total + 1))
        if span <= 32
            if options.pivot_warp_kernel == :shfl
                @cuda threads=32 blocks=1 pluq_find_pivot_warp_shfl_kernel!(
                    Adata,
                    pivot_slot,
                    kk,
                    kend32,
                    N32,
                )
            else
                @cuda threads=32 blocks=1 pluq_find_pivot_warp_kernel!(
                    Adata,
                    pivot_slot,
                    kk,
                    kend32,
                    N32,
                )
            end
        else
            blocks = max(1, cld(total, threads))
            @cuda threads=threads blocks=blocks pluq_find_pivot_kernel!(
                Adata,
                pivot_slot,
                kk,
                kend32,
                N32,
            )
        end
        pivot_lin = _pluq_read_i32!(pivot_host, pivot_slot)
        if pivot_lin > total
            break
        end
        # Decode pivot index from linearized active-submatrix coordinates.
        joff = (pivot_lin - 1) ÷ span
        ioff = (pivot_lin - 1) % span
        prow = k + ioff
        pcol = k + joff
        if prow != k
            @cuda threads=threads blocks=max(1, cld(n, threads)) pluq_swap_rows_kernel!(
                Adata,
                Int32(k),
                Int32(prow),
                n32,
            )
            if options.lazy_q
                lock = k - k0 + 1
                locprow = prow - k0 + 1
                locp[lock], locp[locprow] = locp[locprow], locp[lock]
            else
                p[k], p[prow] = p[prow], p[k]
            end
        end
        if pcol != k
            @cuda threads=threads blocks=max(1, cld(n, threads)) pluq_swap_cols_kernel!(
                Adata,
                Int32(k),
                Int32(pcol),
                n32,
            )
            if options.lazy_q
                lock = k - k0 + 1
                locpcol = pcol - k0 + 1
                locq[lock], locq[locpcol] = locq[locpcol], locq[lock]
            else
                q[k], q[pcol] = q[pcol], q[k]
            end
        end
        # Normalize pivot row/column step in packed LU form.
        if k < kend
            @cuda threads=threads blocks=max(1, cld(kend - k, threads)) pluq_scale_column_from_diag_kernel!(
                Adata,
                Int32(k),
                Int32(kend),
                N32,
            )
            tx = 16
            ty = 16
            bx = max(1, cld(kend - k, tx))
            by = max(1, cld(kend - k, ty))
            @cuda threads=(tx, ty) blocks=(bx, by) pluq_rank1_update_kernel!(
                Adata,
                Int32(k),
                Int32(kend),
                N32,
            )
        end
        rank += 1
    end
    if options.lazy_q
        pluq_compose_segment!(p, k0, locp)
        pluq_compose_segment!(q, k0, locq)
    end
    return rank
end
