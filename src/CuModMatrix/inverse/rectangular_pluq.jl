"""
    pluq_find_pivot_rect_kernel!(A, pivot_slot, k, m, n, N)

Find the first nonzero pivot in the active rectangular submatrix
`A[k:m, k:n]`, encoded as a linear position in `pivot_slot[1]`.
"""
function pluq_find_pivot_rect_kernel!(A, pivot_slot, k::Int32, m::Int32, n::Int32, N::Int32)
    gtid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    ltid = Int(threadIdx().x)
    span_r = m - k + 1
    span_c = n - k + 1
    total = span_r * span_c
    stride = blockDim().x * gridDim().x
    idx = gtid
    local_min = Int32(total + 1)
    while idx <= total
        joff = (idx - 1) ÷ span_r
        ioff = (idx - 1) % span_r
        i = k + ioff
        j = k + joff
        v = _pluq_mod_t(A[i, j], N)
        if v != zero(eltype(A))
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

function pluq_find_pivot_rect_warp_kernel!(
    A,
    pivot_slot,
    k::Int32,
    m::Int32,
    n::Int32,
    N::Int32,
)
    lane = Int(threadIdx().x)
    if lane > 32
        return
    end
    span_r = m - k + 1
    span_c = n - k + 1
    joff = Int32(0)
    while joff < span_c
        row = k + Int32(lane - 1)
        pred = Int32(lane) <= span_r && _pluq_mod_t(A[row, k+joff], N) != zero(eltype(A))
        bits = CUDA.vote_ballot_sync(CUDA.FULL_MASK, pred)
        if bits != UInt32(0)
            if lane == 1
                first_lane = Int32(trailing_zeros(bits) + 1)
                pivot_slot[1] = joff * span_r + first_lane
            end
            return
        end
        joff += 1
    end
    return
end

function pluq_find_pivot_rect_warp_shfl_kernel!(
    A,
    pivot_slot,
    k::Int32,
    m::Int32,
    n::Int32,
    N::Int32,
)
    lane = Int32(threadIdx().x)
    if lane > Int32(32)
        return
    end
    span_r = m - k + Int32(1)
    span_c = n - k + Int32(1)
    local_min = span_r * span_c + Int32(1)
    if lane <= span_r
        row = k + lane - Int32(1)
        joff = Int32(0)
        while joff < span_c
            if _pluq_mod_t(A[row, k+joff], N) != zero(eltype(A))
                cand = joff * span_r + lane
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
    pluq_scale_column_rect_kernel!(A, k, m, invpivot, N)

Scale entries below the pivot in column `k`:
`A[k+1:m, k] *= invpivot (mod N)`.
"""
function pluq_scale_column_rect_kernel!(A, k::Int32, m::Int32, invpivot, N::Int32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + k
    stride = blockDim().x * gridDim().x
    while i <= m
        A[i, k] = _pluq_mod_mul_t(A[i, k], invpivot, N)
        i += stride
    end
    return
end

function pluq_scale_column_rect_from_diag_kernel!(A, k::Int32, m::Int32, N::Int32)
    invslot = CuStaticSharedArray(eltype(A), 1)
    if threadIdx().x == 1
        invslot[1] = _pluq_mod_inv_t(A[k, k], N)
    end
    sync_threads()
    invpivot = invslot[1]
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + k
    stride = blockDim().x * gridDim().x
    while i <= m
        A[i, k] = _pluq_mod_mul_t(A[i, k], invpivot, N)
        i += stride
    end
    return
end

"""
    pluq_rank1_update_rect_kernel!(A, k, m, n, N)

Apply rank-1 elimination update on rectangular trailing block:
`A[k+1:m, k+1:n] -= A[k+1:m,k] * A[k,k+1:n]`.
"""
function pluq_rank1_update_rect_kernel!(A, k::Int32, m::Int32, n::Int32, N::Int32)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x + k
    j = (blockIdx().y - Int32(1)) * blockDim().y + threadIdx().y + k
    if i <= m && j <= n
        A[i, j] = _pluq_mod_t(A[i, j] - _pluq_mod_mul_t(A[i, k], A[k, j], N), N)
    end
    return
end

"""
    pluq_rectangular_rank_gpu!(Adata, N, m, n)

In-place rank-revealing PLUQ-style elimination for rectangular matrices.
Returns `(p, q, rank)` where:
- `p` is row permutation vector (length `m`)
- `q` is column permutation vector (length `n`)
- `rank` is computed rank over `GF(N)`.
"""
function pluq_rectangular_rank_reference_gpu!(
    Adata::CuArray{T,2},
    N::Int,
    m::Int,
    n::Int;
    options::PLUQOptions = PLUQOptions(),
) where {T}
    rmax = min(m, n)
    p = collect(1:m)
    q = collect(1:n)
    lp = options.lazy_q ? collect(1:m) : Int[]
    lq = options.lazy_q ? collect(1:n) : Int[]
    rank = 0
    threads = 256
    N32 = Int32(N)
    m32 = Int32(m)
    n32 = Int32(n)
    pivot_slot = CUDA.fill(Int32(max(1, m * n + 1)), 1)
    pivot_host = _pluq_host_i32_buffer()
    for k = 1:rmax
        span_r = m - k + 1
        span_c = n - k + 1
        total = span_r * span_c
        fill!(pivot_slot, Int32(total + 1))
        blocks = max(1, cld(total, threads))
        k32 = Int32(k)
        if span_r <= 32
            if options.pivot_warp_kernel == :shfl
                @cuda threads=32 blocks=1 pluq_find_pivot_rect_warp_shfl_kernel!(
                    Adata,
                    pivot_slot,
                    k32,
                    m32,
                    n32,
                    N32,
                )
            else
                @cuda threads=32 blocks=1 pluq_find_pivot_rect_warp_kernel!(
                    Adata,
                    pivot_slot,
                    k32,
                    m32,
                    n32,
                    N32,
                )
            end
        else
            @cuda threads=threads blocks=blocks pluq_find_pivot_rect_kernel!(
                Adata,
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
            @cuda threads=threads blocks=max(1, cld(n, threads)) pluq_swap_rows_kernel!(
                Adata,
                k32,
                Int32(prow),
                n32,
            )
            if options.lazy_q
                lp[k], lp[prow] = lp[prow], lp[k]
            else
                p[k], p[prow] = p[prow], p[k]
            end
        end
        if pcol != k
            @cuda threads=threads blocks=max(1, cld(m, threads)) pluq_swap_cols_kernel!(
                Adata,
                k32,
                Int32(pcol),
                m32,
            )
            if options.lazy_q
                lq[k], lq[pcol] = lq[pcol], lq[k]
            else
                q[k], q[pcol] = q[pcol], q[k]
            end
        end
        if k < m
            @cuda threads=threads blocks=max(1, cld(m - k, threads)) pluq_scale_column_rect_from_diag_kernel!(
                Adata,
                k32,
                m32,
                N32,
            )
        end
        if k < n
            tx = 16
            ty = 16
            bx = max(1, cld(m - k, tx))
            by = max(1, cld(n - k, ty))
            @cuda threads=(tx, ty) blocks=(bx, by) pluq_rank1_update_rect_kernel!(
                Adata,
                k32,
                m32,
                n32,
                N32,
            )
        end
        rank += 1
    end
    if options.lazy_q
        p = lp
        q = lq
    end
    return p, q, rank
end

"""
    pluq_rect_panel_fused_kernel!(A, p, q, dinv, rank_slot, k0, kend, m, n, N)

Factor one rectangular PLUQ panel without host interaction.  The normal path
uses the first nonzero entry in the current column (partial row pivoting),
which is coalesced in Julia's column-major storage.  If a column is exhausted,
the panel reports its partial rank; the public wrapper restores the original
matrix and uses the complete-pivot reference path for that exceptional case.
"""
function pluq_rect_panel_fused_kernel!(
    A,
    p,
    q,
    dinv,
    rank_slot,
    k0::Int32,
    kend::Int32,
    m::Int32,
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
        # First search the current column.  Adjacent lanes access adjacent
        # rows, which is contiguous in Julia/CUDA column-major matrices.
        local_row = m + Int32(1)
        row = k + tid - Int32(1)
        while row <= m
            if _pluq_mod_t(A[row, k], N) != zero(eltype(A))
                local_row = min(local_row, row)
            end
            row += nt
        end
        candidates[Int(tid)] = local_row
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
            pivot[1] = candidates[1]
            pivot[2] = k
        end
        sync_threads()

        # A zero active column cannot be completed in-place without replaying
        # this panel's local updates. Report it to the host wrapper instead.
        if pivot[1] > m
            if tid == Int32(1)
                pivot[1] = Int32(0)
                pivot[2] = Int32(0)
            end
            sync_threads()
        end

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
            while row <= m
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
            rank_slot[1] += Int32(1)
        end
        sync_threads()
        row = k + tid
        while row <= kend
            A[row, k] = _pluq_mod_mul_t(A[row, k], invpivot[1], N)
            row += nt
        end
        sync_threads()

        # Column-major linearization makes neighbouring lanes neighbouring
        # rows of a panel column, avoiding the old strided 2-D mapping.
        width = kend - k
        idx = tid
        while idx <= width * width
            joff = (idx - Int32(1)) ÷ width + Int32(1)
            ioff = (idx - Int32(1)) % width + Int32(1)
            row = k + ioff
            col = k + joff
            A[row, col] =
                _pluq_mod_t(A[row, col] - _pluq_mod_mul_t(A[row, k], A[k, col], N), N)
            idx += nt
        end
        sync_threads()
        k += Int32(1)
    end
    return
end

function pluq_rect_panel_fused_gpu!(
    Adata::CuArray{T,2},
    N::Int,
    pdev,
    qdev,
    dinv,
    rank_slot,
    rank_host,
    k0::Int,
    kend::Int,
    m::Int,
    n::Int,
) where {T}
    @cuda threads=256 blocks=1 pluq_rect_panel_fused_kernel!(
        Adata,
        pdev,
        qdev,
        dinv,
        rank_slot,
        Int32(k0),
        Int32(kend),
        Int32(m),
        Int32(n),
        Int32(N),
    )
    return _pluq_read_i32!(rank_host, rank_slot)
end


"""Blocked rank-revealing PLUQ for a wide matrix in column-major storage."""
function pluq_rectangular_rank_gpu!(
    Adata::CuArray{T,2},
    N::Int,
    m::Int,
    n::Int;
    options::PLUQOptions = PLUQOptions(),
) where {T}
    rmax = min(m, n)
    pdev = CuArray(Int32.(1:m))
    qdev = CuArray(Int32.(1:n))
    dinv = CUDA.zeros(T, rmax)
    rank_slot = CUDA.zeros(Int32, 1)
    rank_host = _pluq_host_i32_buffer()
    rank = 0
    start = 1
    while start <= rmax
        kend = min(start + options.blocksize - 1, rmax)
        panel_width = kend - start + 1
        panel_rank = pluq_rect_panel_fused_gpu!(
            Adata,
            N,
            pdev,
            qdev,
            dinv,
            rank_slot,
            rank_host,
            start,
            kend,
            m,
            n,
        )
        if panel_rank < panel_width
            # The panel has touched its local Schur block, so completing a
            # pivot outside that panel would require replaying its trailing
            # updates.  The public wrapper restarts the rare case with the
            # complete-pivot reference factorization from an untouched copy.
            return Int.(Array(pdev)), Int.(Array(qdev)), rank + panel_rank
        end
        rank += panel_rank
        pluq_trsm_left_lower_unit_gpu!(Adata, N, start, kend, n, options = options)
        pluq_trsm_right_upper_gpu!(Adata, N, start, kend, m, options = options, dinv = dinv)
        pluq_schur_update_rect_gpu!(Adata, N, start, kend, m, n, options = options)
        start = kend + 1
    end
    return Int.(Array(pdev)), Int.(Array(qdev)), rank
end
