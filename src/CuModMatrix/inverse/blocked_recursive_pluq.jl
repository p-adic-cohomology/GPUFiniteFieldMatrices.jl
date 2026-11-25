# The fused panel has already reduced the portion of the Schur complement
# inside kend. Finish the missing outer solves, then restore that small
# portion after the ordinary Schur update over the whole trailing rectangle.
function _pluq_ka_finish_panel!(A, N, opts, start, last, kend, m, n, dinv)
    _pluq_ka_trsm_left!(A, N, start, last, kend + 1, n)
    _pluq_ka_trsm_right!(A, N, start, last, kend + 1, m, dinv)
    if last == min(m, n) || kend == max(m, n)
        return A
    end
    if last == kend
        pluq_schur_update_rect_gpu!(A, N, start, last, m, n; options = opts)
    else
        unfinished = (last+1):kend
        remainder = view(A, unfinished, unfinished)
        saved = similar(_elementwise_native_root(A), eltype(A), size(remainder))
        saved .= remainder
        pluq_schur_update_rect_gpu!(A, N, start, last, m, n; options = opts)
        remainder .= saved
    end
    return A
end

function _pluq_ka_blocked!(A, N, m, n, opts, rectangular::Val)
    _pluq_ka_check_matrix(A, N, m, n)
    stop = min(m, n)
    stop == 0 && return pluq_init_perm(m), pluq_init_perm(n), 0

    logical = _logical_view(A, (m, n))
    _launch_mod_arrays!(logical, logical, N, _PLUQ_KA_GROUP_SIZE)
    native = _elementwise_native_root(A)
    p, q = _pluq_ka_device_permutations(A, m, n)
    dinv = similar(native, eltype(A), stop)
    slot = similar(native, Int32, 1)
    host = Vector{Int32}(undef, 1)
    max_panel = isqrt(Int(_PLUQ_KA_NO_CANDIDATE) - 1)
    start, rank = 1, 0
    while start <= stop
        remaining = stop - start + 1
        use_basecase = rectangular isa Val{false} && remaining <= opts.basecase
        depth = use_basecase ? remaining : min(opts.blocksize, remaining)
        kend = start + min(depth, max_panel) - 1
        panel_rank =
            _pluq_ka_panel!(A, N, p, q, dinv, slot, host, start, kend, m, n, rectangular)
        rank = start + panel_rank - 1
        if panel_rank > 0
            _pluq_ka_finish_panel!(A, N, opts, start, rank, kend, m, n, dinv)
        end
        if rank < kend
            # A zero diagonal panel does not imply zero rank outside it.
            # Continue from the fully updated remainder, with complete pivots.
            rank = _pluq_ka_complete!(A, N, p, q, slot, host, rank + 1, m, n)
            break
        end
        start = kend + 1
    end
    return Int.(Array(p)), Int.(Array(q)), rank
end

function pluq_blocked_gpu!(A::AbstractMatrix, N::Int, opts::PLUQOptions, n::Int)
    return _pluq_ka_blocked!(A, N, n, n, opts, Val(false))
end
