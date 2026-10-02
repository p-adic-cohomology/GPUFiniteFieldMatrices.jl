# """
#     forward_sub_kernel(A::CuDeviceMatrix{T1}, A_inv::CuDeviceMatrix{T2}, N::Int)

# Performs forward substitution on a CuModMatrix.

# """
# function forward_sub_kernel(
#     A::CuDeviceMatrix{T1},
#     A_inv::CuDeviceMatrix{T2},
#     N::Int
# ) where {T1, T2}

#     n = size(A, 2)

#     col = (blockIdx().x - 1) * blockDim().x + threadIdx().x

#     for row in 1:n
#         sum = 0
#         for k in 1:(row-1)
#             sum = mod(sum + A[row, k] * A_inv[k, col], N)
#         end

#         rhs = (row == col) ? 1 : 0
#         diag_inv = mod_inv(A[row, row], N)
#         A_inv[row, col] = mod(diag_inv * (rhs - sum + N), N)

#         # CUDA.sync_threads()
#     end

#     return
# end

# """
#     forward_sub_gpu_type(A::CuModMatrix)

# Performs forward substitution on a CuModMatrix.

# """
# function forward_sub_gpu_type(A::CuModMatrix)
#     padded_rows = size(A.data, 1)
#     padded_cols = size(A.data, 2)
#     d_A_inv = CUDA.zeros(eltype(A.data), padded_cols, padded_rows)

#     @cuda threads=(TILE_WIDTH) blocks=(ceil(Int, size(A, 2) / TILE_WIDTH)) forward_sub_kernel(A.data, d_A_inv, A.N)
#     return CuModMatrix(d_A_inv, A.N, new_size=(cols(A), rows(A)))
# end

# """
#     backward_sub_gpu_type(A::CuModMatrix)

# Performs backward substitution on a CuModMatrix.

# """
# function backward_sub_gpu_type(A::CuModMatrix)
#     padded_rows = size(A.data, 1)
#     padded_cols = size(A.data, 2)
#     d_A_inv = CUDA.zeros(eltype(A.data), padded_cols, padded_rows)

#     @cuda threads=(TILE_WIDTH) blocks=(ceil(Int, size(A, 1) / TILE_WIDTH)) backward_sub_kernel(A.data, d_A_inv, A.N)
#     return CuModMatrix(d_A_inv, A.N, new_size=(cols(A), rows(A)))
# end

# """
#     backward_sub_kernel(A::CuDeviceMatrix{T1}, A_inv::CuDeviceMatrix{T2}, N::Int)

# Performs backward substitution on a CuModMatrix.

# """
# function backward_sub_kernel(
#     A::CuDeviceMatrix{T1},
#     A_inv::CuDeviceMatrix{T2},
#     N::Int
# ) where {T1, T2}
#     bid = blockIdx().x
#     tid = threadIdx().x
#     n = size(A, 1)

#     tid = (bid - 1) * blockDim().x + tid

#     for row in n:-1:1
#         CUDA.sync_threads()

#         sum = 0
#         for j in row+1:n
#             sum += A[row, j] * A_inv[j, tid]
#         end
#         sum = mod(sum, N)

#         rhs = (tid == row ? 1 : 0)
#         diag = A[row, row]
#         A_inv[row, tid] = mod(mod_inv(diag, N) * (rhs - sum + N), N)
#     end

# end
