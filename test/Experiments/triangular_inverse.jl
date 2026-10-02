# MAX_INVERSE_SIZE = TILE_WIDTH * 1

# """
#     _recursive_upper_triangular_inverse(
#         A::CuArray{T},
#         A_inv::CuArray{T},
#         i::Int,
#         j::Int,
#         N::Int
#     )

# Computes the inverse of an upper triangular matrix using a recursive algorithm.
# Namely, it find the inverse in place for the submatrix A[i:j, i:j].

# NOTE: This only accepts square or wide matrices!
# (A tall matrix can simply be cut off at the bottom.)
# """
# function _recursive_upper_triangular_inverse(
#     A::CuArray,
#     A_inv::CuArray,
#     row_lower::Int, row_upper::Int,
#     col_lower::Int, col_upper::Int,
#     N::Int;
#     debug::Bool=false
# )
#     if row_upper - row_lower < MAX_INVERSE_SIZE

#         if debug
#             println("First branch: Input is already small enough!")
#             println("row_lower: $row_lower, row_upper: $row_upper")
#             println("col_lower: $col_lower, col_upper: $col_upper")
#         end

#         A_all = CUDA.zeros(eltype(A), TILE_WIDTH, col_upper - col_lower + 1)
#         copyto!(A_all, @view A[row_lower:row_upper, col_lower:col_upper])
#         A_all_inv = CUDA.zeros(eltype(A), TILE_WIDTH, col_upper - col_lower + 1)

#         if debug
#             println("A_all:")
#             display(A_all)
#             @assert size(A_all) == size(A_all_inv)
#         end

#         @cuda threads=TILE_WIDTH blocks=(ceil(Int, size(A_all, 1) / TILE_WIDTH)) backward_sub_kernel(A_all, A_all_inv, N)

#         if debug
#             println("A_all_inv:")
#             display(A_all_inv)
#         end

#         A_inv[row_lower:row_upper, col_lower:col_upper] = A_all_inv
#         return

#     end

#     # Find a split such that A_21 is all zeros
#     # This means A_11 must be square no matter what
#     # We also want the sizes to be multiples of TILE_WIDTH
#     row_mid = row_lower + min((row_upper - row_lower + 1) ÷ TILE_WIDTH * TILE_WIDTH, TILE_WIDTH) - 1
#     col_mid = row_mid

#     if debug
#         println("row_mid: $row_mid, col_mid: $col_mid")
#         println("row_lower: $row_lower, row_upper: $row_upper")
#         println("col_lower: $col_lower, col_upper: $col_upper")

#         @assert (row_mid - row_lower + 1) % TILE_WIDTH == 0
#         @assert (col_mid - col_lower + 1) % TILE_WIDTH == 0
#         # @assert (row_upper - row_mid) % TILE_WIDTH == 0
#         # @assert (col_upper - col_mid) % TILE_WIDTH == 0
#     end

#     if row_mid - row_lower <= MAX_INVERSE_SIZE

#         if debug
#             println("Second branch: Each triangle is small enough!")
#         end

#         # Find inverse of upper left block
#         A_11 = CUDA.zeros(eltype(A), row_mid - row_lower + 1, col_mid - col_lower + 1)
#         copyto!(A_11, @view A[row_lower:row_mid, col_lower:col_mid])
#         A_11_inv = CUDA.zeros(eltype(A), col_mid - col_lower + 1, row_mid - row_lower + 1)

#         if debug
#             println("A_11:")
#             display(A_11)
#             @assert size(A_11) == size(A_11_inv)
#         end

#         @cuda threads=TILE_WIDTH blocks=(ceil(Int, size(A_11, 1) / TILE_WIDTH)) backward_sub_kernel(A_11, A_11_inv, N)

#         if debug
#             println("A_11_inv:")
#             display(A_11_inv)
#             println("A_11 * A_11_inv:")
#             display(mod.(A_11 * A_11_inv, N))
#             @assert Array(mod.(A_11 * A_11_inv, N)) ≈ Array(Matrix{eltype(A)}(I, size(A_11)))
#         end

#         A_inv[col_lower:col_mid, row_lower:row_mid] = A_11_inv

#         if debug
#             println("A_inv:")
#             display(A_inv)
#         end

#         # Find inverse of lower right block
#         A_22 = CUDA.zeros(eltype(A), max(row_upper - row_mid, TILE_WIDTH), max(col_upper - col_mid, TILE_WIDTH))
#         copyto!(A_22, @view A[row_mid+1:row_upper, col_mid+1:col_upper])
#         A_22_inv = CUDA.zeros(eltype(A), max(col_upper - col_mid, TILE_WIDTH), max(row_upper - row_mid, TILE_WIDTH))

#         if debug
#             println("A_22:")
#             display(A_22)
#             println(row_mid+1:row_upper)
#             println(col_mid+1:col_upper)
#             @assert size(A_22) == size(A_22_inv)
#         end

#         @cuda threads=TILE_WIDTH blocks=(div(size(A_22, 1), TILE_WIDTH)) backward_sub_kernel(A_22, A_22_inv, N)

#         if debug
#             println("A_22_inv:")
#             display(A_22_inv)
#             println("A_22 * A_22_inv:")
#             display(mod.(A_22 * A_22_inv, N))
#             @assert Array(mod.(A_22 * A_22_inv, N))[1:(row_upper - row_mid), 1:(col_upper - col_mid)] ≈ Array(Matrix{eltype(A)}(I, row_upper - row_mid, col_upper - col_mid))
#         end

#         A_inv[col_mid+1:col_upper, row_mid+1:row_upper] = A_22_inv[1:(row_upper - row_mid), 1:(col_upper - col_mid)]

#         if debug
#             println("A_inv:")
#             display(A_inv)
#         end

#         # Find inverse of upper right block
#         A_12 = @view A[row_lower:row_mid, col_mid+1:col_upper]
#         # A_inv[row_lower:row_mid, col_mid+1:col_upper] = mod.(mod.(-A_11_inv * A_12 * A_22_inv, N) .+ N, N)
#         A_inv[row_mid+1:row_upper, col_lower:col_mid] = N .- mod.(mod.(A_22_inv[1:(row_upper - row_mid), 1:(col_upper - col_mid)] * A_12, N) * A_11_inv, N)

#         if debug
#             println("A_12:")
#             display(A_12)
#             println("-A_11_inv * A_12 * A_22_inv:")
#             display(mod.(mod.(-A_11_inv * A_12 * A_22_inv[1:(row_upper - row_mid), 1:(col_upper - col_mid)], N) .+ N, N))
#             println("A_inv:")
#             display(A_inv)
#         end

#         return

#     else

#         if debug
#             println("Third branch: Recursive call!")
#         end

#         _recursive_upper_triangular_inverse(
#             A, A_inv, row_lower, row_mid, col_lower, col_mid, N; debug=debug)
#         _recursive_upper_triangular_inverse(
#             A, A_inv, row_mid+1, row_upper, col_mid+1, col_upper, N; debug=debug)
#         A_12 = @view A[row_lower:row_mid, col_mid+1:col_upper]
#         A_11_inv = @view A_inv[col_lower:col_mid, row_lower:row_mid]
#         A_22_inv = @view A_inv[col_mid+1:col_upper, row_mid+1:row_upper]
#         A_inv[row_mid+1:row_upper, col_lower:col_mid] = N .- mod.(mod.(A_22_inv * A_12, N) * A_11_inv, N)

#         return
#     end
# end

# """
#     upper_triangular_inverse(A::CuModMatrix)

# Computes the inverse of an upper triangular matrix.
# """
# function upper_triangular_inverse(A::CuModMatrix; debug::Bool=false)
#     rows, cols = size(A)

#     if rows <= MAX_INVERSE_SIZE
#         return backward_sub_gpu_type(A)
#     end

#     A_inv = GPUFiniteFieldMatrices.zeros(eltype(A.data), size(A)[2], size(A)[1], A.N)

#     if debug
#         println("size(A.data): ", size(A.data))
#     end

#     if rows <= cols #square or wide
#         _recursive_upper_triangular_inverse(
#             A.data, A_inv.data, 1, rows, 1, cols, A.N; debug=debug)
#     else #tall
#         _recursive_upper_triangular_inverse(
#             A.data, A_inv.data, 1, cols, 1, cols, A.N; debug=debug)
#     end

#     if debug
#         println("A.data * A_inv.data:")
#         display(mod.(A.data * A_inv.data, A.N))
#         println("A_inv.data:")
#         display(A_inv.data)
#     end

#     return A_inv
# end

# function _recursive_lower_triangular_inverse(
#     A::CuArray,
#     A_inv::CuArray,
#     row_lower::Int, row_upper::Int,
#     col_lower::Int, col_upper::Int,
#     N::Int;
#     debug::Bool=false
# )
#     if col_upper - col_lower < MAX_INVERSE_SIZE

#         if debug
#             println("First branch: Input is already small enough!")
#             println("row_lower: $row_lower, row_upper: $row_upper")
#             println("col_lower: $col_lower, col_upper: $col_upper")
#         end

#         A_all = CUDA.zeros(eltype(A), row_upper - row_lower + 1, TILE_WIDTH)
#         copyto!(A_all, @view A[col_lower:col_upper, row_lower:row_upper])
#         A_all_inv = CUDA.zeros(eltype(A), row_upper - row_lower + 1, TILE_WIDTH)

#         if debug
#             println("A_all:")
#             display(A_all)
#             @assert size(A_all) == size(A_all_inv)
#         end

#         @cuda threads=TILE_WIDTH blocks=(ceil(Int, size(A_all, 2) / TILE_WIDTH)) forward_sub_kernel(A_all, A_all_inv, N)

#         if debug
#             println("A_all_inv:")
#             display(A_all_inv)
#         end

#         A_inv[col_lower:col_upper, row_lower:row_upper] = A_all_inv
#         return

#     end

#     # Find a split such that A_21 is all zeros
#     # This means A_11 must be square no matter what
#     # We also want the sizes to be multiples of TILE_WIDTH
#     col_mid = col_lower + min((col_upper - col_lower + 1) ÷ TILE_WIDTH * TILE_WIDTH, TILE_WIDTH) - 1
#     row_mid = col_mid

#     if debug
#         println("row_mid: $row_mid, col_mid: $col_mid")
#         println("row_lower: $row_lower, row_upper: $row_upper")
#         println("col_lower: $col_lower, col_upper: $col_upper")

#         @assert (row_mid - row_lower + 1) % TILE_WIDTH == 0
#         @assert (col_mid - col_lower + 1) % TILE_WIDTH == 0
#         # @assert (row_upper - row_mid) % TILE_WIDTH == 0
#         # @assert (col_upper - col_mid) % TILE_WIDTH == 0
#     end

#     if col_mid - col_lower <= MAX_INVERSE_SIZE

#         if debug
#             println("Second branch: Each triangle is small enough!")
#         end

#         # Find inverse of upper left block
#         A_11 = CUDA.zeros(eltype(A), row_mid - row_lower + 1, col_mid - col_lower + 1)
#         copyto!(A_11, @view A[row_lower:row_mid, col_lower:col_mid])
#         A_11_inv = CUDA.zeros(eltype(A), col_mid - col_lower + 1, row_mid - row_lower + 1)

#         if debug
#             println("A_11:")
#             display(A_11)
#             @assert size(A_11) == size(A_11_inv)
#         end

#         @cuda threads=TILE_WIDTH blocks=(ceil(Int, size(A_11, 2) / TILE_WIDTH)) forward_sub_kernel(A_11, A_11_inv, N)

#         if debug
#             println("A_11_inv:")
#             display(A_11_inv)
#             println("A_11 * A_11_inv:")
#             display(mod.(A_11 * A_11_inv, N))
#             @assert Array(mod.(A_11 * A_11_inv, N)) ≈ Array(Matrix{eltype(A)}(I, size(A_11)))
#         end

#         A_inv[col_lower:col_mid, row_lower:row_mid] = A_11_inv

#         if debug
#             println("A_inv:")
#             display(A_inv)
#         end

#         # Find inverse of lower right block
#         A_22 = CUDA.zeros(eltype(A), max(row_upper - row_mid, TILE_WIDTH), max(col_upper - col_mid, TILE_WIDTH))
#         copyto!(A_22, @view A[row_mid+1:row_upper, col_mid+1:col_upper])
#         A_22_inv = CUDA.zeros(eltype(A), max(col_upper - col_mid, TILE_WIDTH), max(row_upper - row_mid, TILE_WIDTH))

#         if debug
#             println("A_22:")
#             display(A_22)
#             println(row_mid+1:row_upper)
#             println(col_mid+1:col_upper)
#             @assert size(A_22) == size(A_22_inv)
#         end

#         @cuda threads=TILE_WIDTH blocks=(div(size(A_22, 2), TILE_WIDTH)) forward_sub_kernel(A_22, A_22_inv, N)

#         if debug
#             println("A_22_inv:")
#             display(A_22_inv)
#             println("A_22 * A_22_inv:")
#             display(mod.(A_22 * A_22_inv, N))
#             @assert Array(mod.(A_22 * A_22_inv, N))[1:(row_upper - row_mid), 1:(col_upper - col_mid)] ≈ Array(Matrix{eltype(A)}(I, row_upper - row_mid, col_upper - col_mid))
#         end

#         A_inv[col_mid+1:col_upper, row_mid+1:row_upper] = A_22_inv[1:(row_upper - row_mid), 1:(col_upper - col_mid)]

#         if debug
#             println("A_inv:")
#             display(A_inv)
#         end

#         # Find inverse of lower left block
#         A_21 = @view A[row_mid+1:row_upper, col_lower:col_mid]
#         A_inv[col_mid+1:col_upper, row_lower:row_mid] = N .- mod.(mod.(A_22_inv[1:(row_upper - row_mid), 1:(col_upper - col_mid)] * A_21, N) * A_11_inv, N)

#         if debug
#             println("A_21:")
#             display(A_21)
#             println("A_22_inv * A_21 * A_11_inv:")
#             display(mod.(mod.(A_22_inv[1:(row_upper - row_mid), 1:(col_upper - col_mid)] * A_21, N) * A_11_inv, N))
#             println("A_inv:")
#             display(A_inv)
#         end

#         return

#     else

#         if debug
#             println("Third branch: Recursive call!")
#         end

#         _recursive_lower_triangular_inverse(
#             A, A_inv, row_lower, row_mid, col_lower, col_mid, N; debug=debug)
#         _recursive_lower_triangular_inverse(
#             A, A_inv, row_mid+1, row_upper, col_mid+1, col_upper, N; debug=debug)
#         A_21 = @view A[row_mid+1:row_upper, col_lower:col_mid]
#         A_11_inv = @view A_inv[col_lower:col_mid, row_lower:row_mid]
#         A_22_inv = @view A_inv[col_mid+1:col_upper, row_mid+1:row_upper]
#         A_inv[col_mid+1:col_upper, row_lower:row_mid] = N .- mod.(mod.(A_22_inv * A_21, N) * A_11_inv, N)

#         return
#     end
# end

# function lower_triangular_inverse(A::CuModMatrix; debug::Bool=false)
#     rows, cols = size(A)

#     if cols <= MAX_INVERSE_SIZE
#         return forward_sub_gpu_type(A)
#     end

#     A_inv = GPUFiniteFieldMatrices.zeros(eltype(A.data), size(A)[2], size(A)[1], A.N)

#     if rows <= cols #square or wide
#         _recursive_lower_triangular_inverse(
#             A.data, A_inv.data, 1, rows, 1, cols, A.N; debug=debug)
#     else #tall
#         throw(InverseNotDefinedException("(Right) pseudoinverse not defined for tall lower triangular matrices"))
#         # _recursive_lower_triangular_inverse(
#         #     A.data, A_inv.data, 1, rows, 1, rows, A.N; debug=debug)
#     end

#     if debug
#         println("A.data * A_inv.data:")
#         display(mod.(A.data * A_inv.data, A.N))
#         println("A_inv.data:")
#         display(A_inv.data)
#     end

#     return A_inv
# end
