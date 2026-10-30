"""
    _lower_triangular_inv_32!(A_inv, A, N, row_shift, col_shift, n_active)

Launch a forward-substitution inverse kernel for at most 32 columns of `A`,
writing into the caller-provided `A_inv` storage. `A_inv` must not alias `A`.
"""
function _lower_triangular_inv_32!(
    A_inv::AbstractMatrix,
    A::AbstractMatrix,
    N::Int,
    row_shift::Int,
    col_shift::Int,
    n_active::Int,
)
    @cuda threads=TILE_WIDTH blocks=1 _lower_triangular_inv_32_kernel!(
        A_inv,
        A,
        N,
        row_shift,
        col_shift,
        n_active,
    )
    return A_inv
end

"""
    _lower_triangular_inv_32_kernel!(A_inv, A, N, row_shift, col_shift, n_active)

Perform forward substitution on one bounded block of `A`, writing its inverse
columns into `A_inv`.
"""
function _lower_triangular_inv_32_kernel!(
    A_inv::CuDeviceMatrix{T2},
    A::CuDeviceMatrix{T1},
    N::Int,
    row_shift::Int,
    col_shift::Int,
    n_active::Int,
) where {T1,T2}
    col = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if col > n_active
        return
    end

    for row = 1:n_active
        sum = zero(eltype(A))
        for k = 1:(row-1)
            sum = mod(sum + A[row+row_shift, k+col_shift] * A_inv[k, col], N)
        end

        rhs = (row == col) ? one(eltype(A)) : zero(eltype(A))
        diag_inv = mod_inv(A[row+row_shift, row+row_shift], N)
        A_inv[row, col] = mod(diag_inv * (rhs - sum + N), N)
    end

    return
end

"""
    _upper_triangular_inv_32!(A_inv, A, N, row_shift, col_shift, n_active)

Launch a backward-substitution inverse kernel for at most 32 columns of `A`,
writing into the caller-provided `A_inv` storage. `A_inv` must not alias `A`.
"""
function _upper_triangular_inv_32!(
    A_inv::AbstractMatrix,
    A::AbstractMatrix,
    N::Int,
    row_shift::Int,
    col_shift::Int,
    n_active::Int,
)
    @cuda threads=TILE_WIDTH blocks=1 _upper_triangular_inv_32_kernel!(
        A_inv,
        A,
        N,
        row_shift,
        col_shift,
        n_active,
    )
    return A_inv
end

"""
    _upper_triangular_inv_32_kernel!(A_inv, A, N, row_shift, col_shift, n_active)

Perform backward substitution on one bounded block of `A`, writing its inverse
columns into `A_inv`.
"""
function _upper_triangular_inv_32_kernel!(
    A_inv::CuDeviceMatrix{T2},
    A::CuDeviceMatrix{T1},
    N::Int,
    row_shift::Int,
    col_shift::Int,
    n_active::Int,
) where {T1,T2}
    bid = blockIdx().x
    tid = threadIdx().x

    tid = (bid - 1) * blockDim().x + tid
    if tid > n_active
        return
    end

    for row = n_active:-1:1
        CUDA.sync_threads()

        sum = zero(eltype(A))
        for j = (row+1):n_active
            sum += A[row+row_shift, j+col_shift] * A_inv[j, tid]
        end
        sum = mod(sum, N)

        rhs = (tid == row ? one(eltype(A)) : zero(eltype(A)))
        diag = A[row+row_shift, row+row_shift]
        A_inv[row, tid] = mod(mod_inv(diag, N) * (rhs - sum + N), N)
    end

    return
end
