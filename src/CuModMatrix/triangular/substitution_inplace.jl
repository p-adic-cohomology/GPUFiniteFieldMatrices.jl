"""
    forward_sub_kernel_32(A::CuDeviceMatrix{T1}, A_inv::CuDeviceMatrix{T2}, N::Int)

Performs forward substitution on a 32x32 submatrix of a CuModMatrix. This is meant to be used in the recursive algorithm for triangular inverse.
"""
function forward_sub_kernel_32(
    A::CuDeviceMatrix{T1},
    A_inv::CuDeviceMatrix{T2},
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
    forward_sub_gpu_type(A::CuModMatrix)

Performs forward substitution on a 32x32 submatrix of a CuModMatrix. This is meant to be used in the recursive algorithm for triangular inverse.
"""
function forward_sub_gpu_type_32(A::CuModMatrix, row_shift::Int, col_shift::Int)

    d_A_inv = CUDA.zeros(eltype(A.data), TILE_WIDTH + size(A, 2), TILE_WIDTH + size(A, 1))
    n_active = min(TILE_WIDTH, size(A, 1) - row_shift, size(A, 2) - col_shift)
    @cuda threads=(TILE_WIDTH) blocks=1 forward_sub_kernel_32(
        A.data,
        d_A_inv,
        A.N,
        row_shift,
        col_shift,
        n_active,
    )
    return CuModMatrix(d_A_inv, A.N, new_size = (size(A, 2), size(A, 1)))
end

"""
    backward_sub_gpu_type_32(A::CuModMatrix)

Performs backward substitution on a CuModMatrix.

"""
function backward_sub_gpu_type_32(A::CuModMatrix, row_shift::Int, col_shift::Int)
    d_A_inv = CUDA.zeros(eltype(A.data), TILE_WIDTH + size(A, 2), TILE_WIDTH + size(A, 1))
    n_active = min(TILE_WIDTH, size(A, 1) - row_shift, size(A, 2) - col_shift)
    @cuda threads=(TILE_WIDTH) blocks=1 backward_sub_kernel_32(
        A.data,
        d_A_inv,
        A.N,
        row_shift,
        col_shift,
        n_active,
    )
    return CuModMatrix(d_A_inv, A.N, new_size = (size(A, 2), size(A, 1)))
end

"""
    backward_sub_kernel_32(A::CuDeviceMatrix{T1}, A_inv::CuDeviceMatrix{T2}, N::Int)

Performs backward substitution on a 32x32 submatrix of a CuModMatrix. This is meant to be used in the recursive algorithm for triangular inverse.
"""
function backward_sub_kernel_32(
    A::CuDeviceMatrix{T1},
    A_inv::CuDeviceMatrix{T2},
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

end
