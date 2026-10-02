@inline function _batch_supported_n(n::Int)
    return n == 4 || n == 8 || n == 16 || n == 32
end

function _pack_square_batch(mats::AbstractVector{<:CuModMatrix}, n::Int, T::DataType)
    b = length(mats)
    A = CUDA.zeros(T, n, n, b)
    for i = 1:b
        m = mats[i]
        if rows(m) != n || cols(m) != n
            throw(ArgumentError("all matrices must be $(n)x$(n)"))
        end
        A[:, :, i] .= @view m.data[1:n, 1:n]
    end
    return A
end

function _unpack_square_batch!(mats::AbstractVector{<:CuModMatrix}, A::CuArray, n::Int)
    for i in eachindex(mats)
        mats[i].data[1:n, 1:n] .= @view A[:, :, i]
    end
    return mats
end

function _tiny_pluq_batched_kernel!(A, pdev, qdev, rdev, n::Int32, N::Int32)
    bid = Int(blockIdx().x)
    if bid > size(A, 3) || threadIdx().x != 1
        return
    end
    for t = 1:n
        pdev[t, bid] = Int32(t)
        qdev[t, bid] = Int32(t)
    end
    rank = Int32(0)
    k = Int32(1)
    while k <= n
        piv_i = Int32(0)
        piv_j = Int32(0)
        j = k
        while j <= n && piv_i == 0
            i = k
            while i <= n
                if _pluq_mod_t(A[i, j, bid], N) != zero(eltype(A))
                    piv_i = i
                    piv_j = j
                    break
                end
                i += 1
            end
            j += 1
        end
        if piv_i == 0
            break
        end
        if piv_i != k
            c = Int32(1)
            while c <= n
                tmp = A[k, c, bid]
                A[k, c, bid] = A[piv_i, c, bid]
                A[piv_i, c, bid] = tmp
                c += 1
            end
            tp = pdev[k, bid]
            pdev[k, bid] = pdev[piv_i, bid]
            pdev[piv_i, bid] = tp
        end
        if piv_j != k
            r = Int32(1)
            while r <= n
                tmp = A[r, k, bid]
                A[r, k, bid] = A[r, piv_j, bid]
                A[r, piv_j, bid] = tmp
                r += 1
            end
            tq = qdev[k, bid]
            qdev[k, bid] = qdev[piv_j, bid]
            qdev[piv_j, bid] = tq
        end
        invpiv = _pluq_mod_inv_t(A[k, k, bid], N)
        if invpiv == zero(eltype(A))
            break
        end
        i = k + 1
        while i <= n
            A[i, k, bid] = _pluq_mod_mul_t(A[i, k, bid], invpiv, N)
            i += 1
        end
        j = k + 1
        while j <= n
            i2 = k + 1
            while i2 <= n
                A[i2, j, bid] = _pluq_mod_t(
                    A[i2, j, bid] - _pluq_mod_mul_t(A[i2, k, bid], A[k, j, bid], N),
                    N,
                )
                i2 += 1
            end
            j += 1
        end
        rank += 1
        k += 1
    end
    rdev[bid] = rank
    return
end

function _tiny_inverse_batched_kernel!(A, invA, ok, n::Int32, N::Int32)
    bid = Int(blockIdx().x)
    if bid > size(A, 3) || threadIdx().x != 1
        return
    end
    aug = CuStaticSharedArray(eltype(A), (32, 64))
    i = Int32(1)
    while i <= n
        j = Int32(1)
        while j <= n
            aug[i, j] = A[i, j, bid]
            j += 1
        end
        j2 = Int32(1)
        while j2 <= n
            aug[i, n+j2] = i == j2 ? one(eltype(A)) : zero(eltype(A))
            j2 += 1
        end
        i += 1
    end
    okflag = Int32(1)
    k = Int32(1)
    while k <= n
        prow = Int32(0)
        i3 = k
        while i3 <= n
            if _pluq_mod_t(aug[i3, k], N) != zero(eltype(A))
                prow = i3
                break
            end
            i3 += 1
        end
        if prow == 0
            okflag = 0
            break
        end
        if prow != k
            j3 = k
            while j3 <= 2n
                tmp = aug[k, j3]
                aug[k, j3] = aug[prow, j3]
                aug[prow, j3] = tmp
                j3 += 1
            end
        end
        invpiv = _pluq_mod_inv_t(aug[k, k], N)
        if invpiv == zero(eltype(A))
            okflag = 0
            break
        end
        j4 = k
        while j4 <= 2n
            aug[k, j4] = _pluq_mod_mul_t(aug[k, j4], invpiv, N)
            j4 += 1
        end
        i4 = Int32(1)
        while i4 <= n
            if i4 != k
                f = _pluq_mod_t(aug[i4, k], N)
                if f != zero(eltype(A))
                    j5 = k
                    while j5 <= 2n
                        aug[i4, j5] =
                            _pluq_mod_t(aug[i4, j5] - _pluq_mod_mul_t(f, aug[k, j5], N), N)
                        j5 += 1
                    end
                end
            end
            i4 += 1
        end
        k += 1
    end
    ok[bid] = okflag
    if okflag == 1
        i6 = Int32(1)
        while i6 <= n
            j6 = Int32(1)
            while j6 <= n
                invA[i6, j6, bid] = aug[i6, n+j6]
                j6 += 1
            end
            i6 += 1
        end
    end
    return
end

function _pluq_batched_tiny!(mats::AbstractVector{<:CuModMatrix}, n::Int)
    isempty(mats) && return PLUQFactorization[]
    !_batch_supported_n(n) && throw(ArgumentError("supported sizes are 4, 8, 16, 32"))
    N = mats[1].N
    _validate_inverse_modulus(PLUQOptions(), mats[1])
    T = eltype(mats[1].data)
    for m in mats
        if m.N != N || eltype(m.data) != T
            throw(
                CuModArrayModulusMismatchException(
                    "batch matrices must have same modulus and element type",
                ),
            )
        end
    end
    A = _pack_square_batch(mats, n, T)
    b = length(mats)
    pdev = CUDA.zeros(Int32, n, b)
    qdev = CUDA.zeros(Int32, n, b)
    rdev = CUDA.zeros(Int32, b)
    @cuda threads=32 blocks=b _tiny_pluq_batched_kernel!(
        A,
        pdev,
        qdev,
        rdev,
        Int32(n),
        Int32(N),
    )
    _unpack_square_batch!(mats, A, n)
    p = Array(pdev)
    q = Array(qdev)
    r = Array(rdev)
    out = Vector{PLUQFactorization{T}}(undef, b)
    for i = 1:b
        out[i] = PLUQFactorization(mats[i], Int.(p[:, i]), Int.(q[:, i]), Int(r[i]))
    end
    return out
end

function _inverse_batched_tiny(mats::AbstractVector{<:CuModMatrix}, n::Int)
    isempty(mats) && return CuModMatrix[]
    !_batch_supported_n(n) && throw(ArgumentError("supported sizes are 4, 8, 16, 32"))
    N = mats[1].N
    _validate_inverse_modulus(PLUQOptions(), mats[1])
    T = eltype(mats[1].data)
    for m in mats
        if rows(m) != n || cols(m) != n
            throw(CuModArraySizeMismatchException("all matrices must be $(n)x$(n)"))
        end
        if m.N != N || eltype(m.data) != T
            throw(
                CuModArrayModulusMismatchException(
                    "batch matrices must have same modulus and element type",
                ),
            )
        end
    end
    A = _pack_square_batch(mats, n, T)
    invA = CUDA.zeros(T, n, n, length(mats))
    ok = CUDA.zeros(Int32, length(mats))
    @cuda threads=32 blocks=length(mats) _tiny_inverse_batched_kernel!(
        A,
        invA,
        ok,
        Int32(n),
        Int32(N),
    )
    okh = Array(ok)
    out = Vector{CuModMatrix{T}}(undef, length(mats))
    for i in eachindex(mats)
        if okh[i] != 1
            throw(
                InverseNotDefinedException("matrix $(i) in batch is singular modulo $(N)"),
            )
        end
        data = CUDA.zeros(T, size(mats[i].data, 1), size(mats[i].data, 2))
        data[1:n, 1:n] .= @view invA[:, :, i]
        out[i] = CuModMatrix(data, N; new_size = (n, n))
    end
    return out
end

"""
    pluq_batched_4x4!(mats)

Run the tiny batched PLUQ factorization for fixed-size square matrices over a
prime modulus. Variants are available for 4, 8, 16, and 32.
"""
pluq_batched_4x4!(mats::AbstractVector{<:CuModMatrix}) = _pluq_batched_tiny!(mats, 4)
pluq_batched_8x8!(mats::AbstractVector{<:CuModMatrix}) = _pluq_batched_tiny!(mats, 8)
pluq_batched_16x16!(mats::AbstractVector{<:CuModMatrix}) = _pluq_batched_tiny!(mats, 16)
pluq_batched_32x32!(mats::AbstractVector{<:CuModMatrix}) = _pluq_batched_tiny!(mats, 32)

"""
    inverse_batched_4x4!(mats)

Run the tiny batched inverse kernel for fixed-size square matrices over a prime
modulus. Variants are available for 4, 8, 16, and 32.
"""
inverse_batched_4x4!(mats::AbstractVector{<:CuModMatrix}) = _inverse_batched_tiny(mats, 4)
inverse_batched_8x8!(mats::AbstractVector{<:CuModMatrix}) = _inverse_batched_tiny(mats, 8)
inverse_batched_16x16!(mats::AbstractVector{<:CuModMatrix}) =
    _inverse_batched_tiny(mats, 16)
inverse_batched_32x32!(mats::AbstractVector{<:CuModMatrix}) =
    _inverse_batched_tiny(mats, 32)
