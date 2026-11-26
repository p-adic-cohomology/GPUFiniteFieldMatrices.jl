"""
    pluq_blocked_recursive_gpu!(Adata, N, opts, p, q, start, stop, n)

Recursively apply blocked PLUQ on active index interval `[start, stop]`.
Uses basecase elimination, triangular solves, and Schur updates on GPU.

The recursion implements:
1. Base/panel PLUQ on the diagonal block
2. `L11 * U12 = A12` (left TRSM)
3. `L21 * U11 = A21` (right TRSM)
4. Schur update `A22 -= L21*U12`
5. Recurse on the trailing block
"""
function pluq_blocked_recursive_gpu!(
    Adata::CuArray{T,2},
    N::Int,
    opts::PLUQOptions,
    pdev,
    qdev,
    dinv,
    rank_slot,
    rank_host,
    start::Int,
    stop::Int,
    n::Int,
) where {T}
    if start > stop
        return 0
    end
    seglen = stop - start + 1
    if seglen <= opts.basecase
        return pluq_panel_fused_gpu!(
            Adata,
            N,
            pdev,
            qdev,
            dinv,
            rank_slot,
            rank_host,
            start,
            stop,
            n,
        )
    end
    b = min(opts.blocksize, seglen)
    kend = min(start + b - 1, stop)
    rank = pluq_panel_fused_gpu!(
        Adata,
        N,
        pdev,
        qdev,
        dinv,
        rank_slot,
        rank_host,
        start,
        kend,
        n,
    )
    panel_width = kend - start + 1
    if rank < panel_width
        return rank
    end
    pluq_trsm_left_lower_unit_gpu!(Adata, N, start, kend, stop, options = opts)
    pluq_trsm_right_upper_gpu!(Adata, N, start, kend, stop, options = opts, dinv = dinv)
    pluq_schur_update_gpu!(Adata, N, start, kend, stop, options = opts)
    rank += pluq_blocked_recursive_gpu!(
        Adata,
        N,
        opts,
        pdev,
        qdev,
        dinv,
        rank_slot,
        rank_host,
        kend + 1,
        stop,
        n,
    )
    return rank
end

"""
    pluq_blocked_gpu!(Adata, N, opts, n)

Run recursive blocked PLUQ on `Adata` and return `(p, q, rank)`.
"""
function pluq_blocked_gpu!(Adata::CuArray{T,2}, N::Int, opts::PLUQOptions, n::Int) where {T}
    pdev = CuArray(Int32.(1:n))
    qdev = CuArray(Int32.(1:n))
    dinv = CUDA.zeros(T, n)
    rank_slot = CUDA.zeros(Int32, 1)
    rank_host = _pluq_host_i32_buffer()
    rank = pluq_blocked_recursive_gpu!(
        Adata,
        N,
        opts,
        pdev,
        qdev,
        dinv,
        rank_slot,
        rank_host,
        1,
        n,
        n,
    )
    p = Int.(Array(pdev))
    q = Int.(Array(qdev))
    return p, q, rank
end
