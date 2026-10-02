"""
    pluq_trsm_left_panel_kernel!(A, k0, kend, n, N)

Kernel for left lower-unit triangular solve on trailing columns:
`L11 * U12 = A12`, writing `U12` in place.

Each thread processes one trailing column and solves all panel rows in order.
"""
function pluq_trsm_left_panel_kernel!(A, k0::Int32, kend::Int32, n::Int32, N::Int32)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x + kend
    stride = blockDim().x * gridDim().x
    while j <= n
        i = k0
        while i <= kend
            acc = _pluq_mod_t(A[i, j], N)
            t = k0
            while t < i
                acc = _pluq_mod_t(acc - _pluq_mod_mul_t(A[i, t], A[t, j], N), N)
                t += 1
            end
            A[i, j] = acc
            i += 1
        end
        j += stride
    end
    return
end

function pluq_trsm_left_panel_delayed_kernel!(A, k0::Int32, kend::Int32, n::Int32, N::Int32)
    j = (blockIdx().x - 1) * blockDim().x + threadIdx().x + kend
    stride = blockDim().x * gridDim().x
    while j <= n
        i = k0
        while i <= kend
            acc = A[i, j]
            t = k0
            while t < i
                acc -= A[i, t] * A[t, j]
                t += 1
            end
            A[i, j] = _pluq_mod_t(acc, N)
            i += 1
        end
        j += stride
    end
    return
end

function pluq_trsm_left_panel_warp_kernel!(A, k0::Int32, kend::Int32, n::Int32, N::Int32)
    lane = Int32(((threadIdx().x - 1) % 32) + 1)
    wid = Int32(((threadIdx().x - 1) >>> 5) + 1)
    nwarps = Int32(Int(blockDim().x) >>> 5)
    j = kend + wid + (blockIdx().x - 1) * nwarps
    stride = gridDim().x * nwarps
    while j <= n
        i = k0
        while i <= kend
            psum = Int64(0)
            t = k0 + lane - 1
            while t < i
                psum += Int64(_pluq_mod_mul_t(A[i, t], A[t, j], N))
                t += 32
            end
            v = rem(psum, Int64(N))
            v = rem(v + CUDA.shfl_down_sync(CUDA.FULL_MASK, v, 16, 32), Int64(N))
            v = rem(v + CUDA.shfl_down_sync(CUDA.FULL_MASK, v, 8, 32), Int64(N))
            v = rem(v + CUDA.shfl_down_sync(CUDA.FULL_MASK, v, 4, 32), Int64(N))
            v = rem(v + CUDA.shfl_down_sync(CUDA.FULL_MASK, v, 2, 32), Int64(N))
            v = rem(v + CUDA.shfl_down_sync(CUDA.FULL_MASK, v, 1, 32), Int64(N))
            if lane == 1
                acc = _pluq_mod_t(A[i, j] - eltype(A)(v), N)
                A[i, j] = acc
            end
            CUDA.sync_warp(CUDA.FULL_MASK)
            i += 1
        end
        j += stride
    end
    return
end

function pluq_trsm_left_panel_warp_delayed_kernel!(
    A,
    k0::Int32,
    kend::Int32,
    n::Int32,
    N::Int32,
)
    lane = Int32(((threadIdx().x - 1) % 32) + 1)
    wid = Int32(((threadIdx().x - 1) >>> 5) + 1)
    nwarps = Int32(Int(blockDim().x) >>> 5)
    j = kend + wid + (blockIdx().x - 1) * nwarps
    stride = gridDim().x * nwarps
    while j <= n
        i = k0
        while i <= kend
            psum = zero(eltype(A))
            t = k0 + lane - Int32(1)
            while t < i
                psum += A[i, t] * A[t, j]
                t += Int32(32)
            end
            psum += CUDA.shfl_down_sync(CUDA.FULL_MASK, psum, Int32(16), Int32(32))
            psum += CUDA.shfl_down_sync(CUDA.FULL_MASK, psum, Int32(8), Int32(32))
            psum += CUDA.shfl_down_sync(CUDA.FULL_MASK, psum, Int32(4), Int32(32))
            psum += CUDA.shfl_down_sync(CUDA.FULL_MASK, psum, Int32(2), Int32(32))
            psum += CUDA.shfl_down_sync(CUDA.FULL_MASK, psum, Int32(1), Int32(32))
            if lane == Int32(1)
                A[i, j] = _pluq_mod_t(A[i, j] - psum, N)
            end
            CUDA.sync_warp(CUDA.FULL_MASK)
            i += Int32(1)
        end
        j += stride
    end
    return
end

"""
    pluq_trsm_right_panel_kernel!(A, k0, kend, n, N)

Kernel for right upper-triangular solve on trailing rows:
`L21 * U11 = A21`, writing `L21` in place.

Each thread processes one trailing row and solves panel columns forward.
"""
function pluq_trsm_right_panel_kernel!(A, k0::Int32, kend::Int32, n::Int32, N::Int32)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + kend
    stride = blockDim().x * gridDim().x
    while i <= n
        j = k0
        while j <= kend
            acc = _pluq_mod_t(A[i, j], N)
            t = k0
            while t < j
                acc = _pluq_mod_t(acc - _pluq_mod_mul_t(A[i, t], A[t, j], N), N)
                t += 1
            end
            invdiag = _pluq_mod_inv_t(A[j, j], N)
            A[i, j] = _pluq_mod_mul_t(acc, invdiag, N)
            j += 1
        end
        i += stride
    end
    return
end

function pluq_panel_inverse_diagonal_kernel!(dinv, A, k0::Int32, kend::Int32, N::Int32)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    depth = kend - k0 + Int32(1)
    if idx <= depth
        dinv[idx] = _pluq_mod_inv_t(A[k0+idx-Int32(1), k0+idx-Int32(1)], N)
    end
    return
end

function pluq_trsm_right_panel_delayed_kernel!(
    A,
    dinv,
    dinv_offset::Int32,
    k0::Int32,
    kend::Int32,
    n::Int32,
    N::Int32,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x + kend
    stride = blockDim().x * gridDim().x
    while i <= n
        j = k0
        while j <= kend
            acc = A[i, j]
            t = k0
            while t < j
                acc -= A[i, t] * A[t, j]
                t += Int32(1)
            end
            reduced = _pluq_mod_t(acc, N)
            A[i, j] = _pluq_mod_mul_t(reduced, dinv[dinv_offset+j-k0+Int32(1)], N)
            j += Int32(1)
        end
        i += stride
    end
    return
end

function pluq_trsm_right_panel_warp_kernel!(A, k0::Int32, kend::Int32, n::Int32, N::Int32)
    lane = Int32(((threadIdx().x - 1) % 32) + 1)
    wid = Int32(((threadIdx().x - 1) >>> 5) + 1)
    nwarps = Int32(Int(blockDim().x) >>> 5)
    i = kend + wid + (blockIdx().x - 1) * nwarps
    stride = gridDim().x * nwarps
    while i <= n
        j = k0
        while j <= kend
            psum = Int64(0)
            t = k0 + lane - Int32(1)
            while t < j
                psum += Int64(_pluq_mod_mul_t(A[i, t], A[t, j], N))
                t += 32
            end
            v = rem(psum, Int64(N))
            v = rem(v + CUDA.shfl_down_sync(CUDA.FULL_MASK, v, 16, 32), Int64(N))
            v = rem(v + CUDA.shfl_down_sync(CUDA.FULL_MASK, v, 8, 32), Int64(N))
            v = rem(v + CUDA.shfl_down_sync(CUDA.FULL_MASK, v, 4, 32), Int64(N))
            v = rem(v + CUDA.shfl_down_sync(CUDA.FULL_MASK, v, 2, 32), Int64(N))
            v = rem(v + CUDA.shfl_down_sync(CUDA.FULL_MASK, v, 1, 32), Int64(N))
            if lane == 1
                acc = _pluq_mod_t(A[i, j] - eltype(A)(v), N)
                invdiag = _pluq_mod_inv_t(A[j, j], N)
                A[i, j] = _pluq_mod_mul_t(acc, invdiag, N)
            end
            CUDA.sync_warp(CUDA.FULL_MASK)
            j += 1
        end
        i += stride
    end
    return
end

function pluq_trsm_right_panel_warp_delayed_kernel!(
    A,
    dinv,
    dinv_offset::Int32,
    k0::Int32,
    kend::Int32,
    n::Int32,
    N::Int32,
)
    lane = Int32(((threadIdx().x - 1) % 32) + 1)
    wid = Int32(((threadIdx().x - 1) >>> 5) + 1)
    nwarps = Int32(Int(blockDim().x) >>> 5)
    i = kend + wid + (blockIdx().x - 1) * nwarps
    stride = gridDim().x * nwarps
    while i <= n
        j = k0
        while j <= kend
            psum = zero(eltype(A))
            t = k0 + lane - Int32(1)
            while t < j
                psum += A[i, t] * A[t, j]
                t += Int32(32)
            end
            psum += CUDA.shfl_down_sync(CUDA.FULL_MASK, psum, Int32(16), Int32(32))
            psum += CUDA.shfl_down_sync(CUDA.FULL_MASK, psum, Int32(8), Int32(32))
            psum += CUDA.shfl_down_sync(CUDA.FULL_MASK, psum, Int32(4), Int32(32))
            psum += CUDA.shfl_down_sync(CUDA.FULL_MASK, psum, Int32(2), Int32(32))
            psum += CUDA.shfl_down_sync(CUDA.FULL_MASK, psum, Int32(1), Int32(32))
            if lane == Int32(1)
                reduced = _pluq_mod_t(A[i, j] - psum, N)
                didx = dinv_offset + j - k0 + Int32(1)
                A[i, j] = _pluq_mod_mul_t(reduced, dinv[didx], N)
            end
            CUDA.sync_warp(CUDA.FULL_MASK)
            j += Int32(1)
        end
        i += stride
    end
    return
end

"""
    pluq_trsm_left_lower_unit_gpu!(Adata, N, k0, kend, n)

Compute the left solve on trailing block columns:
`L11 * U12 = A12`, writing `U12` in place in `Adata`.

`L11` is interpreted as unit-lower from the packed LU panel.
Rows `k0:kend` are solved independently over columns `kend+1:n`.

Example:
```julia
pluq_trsm_left_lower_unit_gpu!(A.data, A.N, 1, 16, rows(A))
```
"""
function pluq_trsm_left_lower_unit_gpu!(
    Adata::CuArray{T,2},
    N::Int,
    k0::Int,
    kend::Int,
    n::Int;
    options::PLUQOptions = PLUQOptions(),
) where {T}
    if kend >= n
        return
    end
    panel_depth = kend - k0 + 1
    use_warp =
        options.trsm_mode == :warp ||
        (options.trsm_mode == :auto && panel_depth <= options.trsm_warp_threshold)
    threads = use_warp ? 128 : 256
    N32 = Int32(N)
    n32 = Int32(n)
    k032 = Int32(k0)
    kend32 = Int32(kend)
    delayed = find_max_ops(T, N) >= panel_depth
    if delayed
        warp_threads = 128
        nwarps = warp_threads >>> 5
        @cuda threads=warp_threads blocks=max(1, cld(n - kend, nwarps)) pluq_trsm_left_panel_warp_delayed_kernel!(
            Adata,
            k032,
            kend32,
            n32,
            N32,
        )
    elseif use_warp
        nwarps = max(1, threads >>> 5)
        @cuda threads=threads blocks=max(1, cld(n - kend, nwarps)) pluq_trsm_left_panel_warp_kernel!(
            Adata,
            k032,
            kend32,
            n32,
            N32,
        )
    else
        @cuda threads=threads blocks=max(1, cld(n - kend, threads)) pluq_trsm_left_panel_kernel!(
            Adata,
            k032,
            kend32,
            n32,
            N32,
        )
    end
    return
end

"""
    pluq_trsm_right_upper_gpu!(Adata, N, k0, kend, n)

Compute the right solve on trailing block rows:
`L21 * U11 = A21`, writing `L21` in place in `Adata`.

`U11` is interpreted as upper-triangular from packed LU panel.
Columns `k0:kend` are solved forward over rows `kend+1:n`.

Example:
```julia
pluq_trsm_right_upper_gpu!(A.data, A.N, 1, 16, rows(A))
```
"""
function pluq_trsm_right_upper_gpu!(
    Adata::CuArray{T,2},
    N::Int,
    k0::Int,
    kend::Int,
    n::Int;
    options::PLUQOptions = PLUQOptions(),
    dinv = nothing,
) where {T}
    if kend >= n
        return
    end
    panel_depth = kend - k0 + 1
    use_warp =
        options.trsm_mode == :warp ||
        (options.trsm_mode == :auto && panel_depth <= options.trsm_warp_threshold)
    threads = use_warp ? 128 : 256
    N32 = Int32(N)
    n32 = Int32(n)
    k032 = Int32(k0)
    kend32 = Int32(kend)
    delayed = find_max_ops(T, N) >= panel_depth
    if delayed
        if dinv === nothing
            panel_dinv = CUDA.zeros(T, panel_depth)
            @cuda threads=128 blocks=max(1, cld(panel_depth, 128)) pluq_panel_inverse_diagonal_kernel!(
                panel_dinv,
                Adata,
                k032,
                kend32,
                N32,
            )
            @cuda threads=128 blocks=max(1, cld(n - kend, 4)) pluq_trsm_right_panel_warp_delayed_kernel!(
                Adata,
                panel_dinv,
                Int32(0),
                k032,
                kend32,
                n32,
                N32,
            )
        else
            @cuda threads=128 blocks=max(1, cld(n - kend, 4)) pluq_trsm_right_panel_warp_delayed_kernel!(
                Adata,
                dinv,
                k032 - Int32(1),
                k032,
                kend32,
                n32,
                N32,
            )
        end
    elseif use_warp
        nwarps = max(1, threads >>> 5)
        @cuda threads=threads blocks=max(1, cld(n - kend, nwarps)) pluq_trsm_right_panel_warp_kernel!(
            Adata,
            k032,
            kend32,
            n32,
            N32,
        )
    else
        @cuda threads=threads blocks=max(1, cld(n - kend, threads)) pluq_trsm_right_panel_kernel!(
            Adata,
            k032,
            kend32,
            n32,
            N32,
        )
    end
    return
end
