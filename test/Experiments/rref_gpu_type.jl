# """
#     rref_gpu_type(A::CuModMatrix, [mod_N])

# Row reduction (RREF) that works directly with CuModMatrix objects.
# Returns a new CuModMatrix in row reduced echelon form.
# If mod_N is provided, it will be used instead of A.N for the modulus.
# """
# function rref_gpu_type(A::CuModMatrix, mod_N::Integer=-1)
#     N = mod_N > 0 ? mod_N : A.N

#     A_rows, A_cols = size(A.data)

#     d_A = CUDA.zeros(Int, (A_rows, A_cols))
#     copyto!(d_A, A.data)

#     row = 1
#     col = 1

#     while row <= rows(A) && col <= cols(A)

#         p = find_pivot_val(d_A, rows(A), row, col)

#         if p == 0
#             col += 1
#             continue
#         end

#         # Find pivot row
#         k = find_pivot_idx(d_A, rows(A), row, col) + row - 1

#         # Only swap if needed
#         p_inv = mod_inv(p, N)
#         if k != row
#             swap_and_mod_rref(d_A, k, row, p_inv, N)
#         end

#         # Only normalize if needed
#         if p != 1
#             normalize_broadcast(d_A, col, p_inv, N)
#         end

#         if row < rows(A) && col < cols(A)
#             @cuda threads=(TILE_WIDTH) blocks=(div(rows(A)-row,TILE_WIDTH)+1) update_sub_matrix_row(d_A, row, col, N)
#         end

#         row += 1
#         col += 1
#     end

#     return CuModMatrix(d_A, N; new_size=(rows(A),cols(A)))
# end

# """
#     lu_gpu_type(A::CuModMatrix, [mod_N])

# LU decomposition that works directly with CuModMatrix objects.
# Returns matrices in CuModMatrix format.
# If mod_N is provided, it will be used instead of A.N for the modulus.
# """
# function lu_gpu_type(A::CuModMatrix, mod_N::Integer=-1)
#     N = mod_N > 0 ? mod_N : A.N

#     A_rows, A_cols = size(A.data)

#     d_A = CUDA.zeros(Int, (A_rows, A_cols))
#     copyto!(d_A, A.data)
#     d_L = CUDA.zeros(Int, (A_rows, A_rows))
#     Perm = Array(1:A_rows)

#     row = 1
#     L_col = 1
#     col = 1

#     while row <= rows(A) && col <= cols(A)
#         p = find_pivot_val(d_A, rows(A), row, col)

#         if p == 0
#             col += 1
#             continue
#         end

#         k = find_pivot_idx(d_A, rows(A), row, col)

#         p_inv = mod_inv(p, N)
#         swap_and_mod_lu(d_A, d_L, k, row, p_inv, N, Perm)

#         normalize_lu_broadcast(d_A, d_L, A_rows, row, L_col, p_inv, p, N)

#         if row == rows(A) || col == cols(A)
#             break
#         end

#         @cuda threads=(TILE_WIDTH) blocks=(div(rows(A)-row,TILE_WIDTH)+1) update_sub_matrix_row(d_A, row, col, N)

#         row += 1
#         L_col += 1
#         col += 1
#     end

#     U = CuModMatrix(d_A, N; new_size=(rows(A), cols(A)))
#     L = CuModMatrix(d_L, N; new_size=(rows(A), rows(A)))
#     return (U, L, Perm)
# end

# function pluq_gpu_type(A::CuModMatrix; perm_stack::Bool=false, perm_array::Bool=true, debug::Bool=false)

#     function _print_plup_debug(stage)
#         if debug
#             println("Stage: $stage")
#             println("Perm_col_idx: ", Perm_col_idx)
#             println("Perm_cols: ", Perm_cols)
#             println("Perm_rows: ", Perm_rows)
#             println("row: ", row)
#             println("col: ", col)
#             println("d_A:")
#             display(@view d_A[1:rows(A),1:cols(A)])
#             println("d_L:")
#             display(@view d_L[1:rows(A),1:rows(A)])
#         end
#     end

#     N = A.N
#     A_padded_rows = size(A.data, 1)
#     A_padded_cols = size(A.data, 2)

#     NVTX.@range "Init PLUQ d_A" begin
#     d_A = copy(A.data)
#     d_L = CUDA.CuArray{CUDA.eltype(A.data)}(undef, (A_padded_rows, A_padded_rows))
#     if perm_stack
#         Perm_rows = Array{Tuple{Int,Int}}(undef, 0)
#         Perm_cols = Array{Tuple{Int,Int}}(undef, 0)
#     else
#         Perm_rows = Array(1:A_padded_rows)
#         Perm_cols = Array(1:A_padded_cols)
#     end
#     end

#     row = 1
#     col = 1
#     Perm_col_idx = cols(A)

#     println("Starting pluq_gpu_type")
#     println("size d_A: ", size(d_A))

#     while row <= rows(A) && col <= cols(A)

#         _print_plup_debug("New iteration")

#         # println("Starting find_zero_col_and_swap")
#         # @time begin
#         NVTX.@range "Find Zero Col and Swap" begin
#         while find_zero_col_and_swap(d_A, rows(A), row, col, Perm_cols, Perm_col_idx; perm_stack)
#             _print_plup_debug("Swapped zero-columns")
#             Perm_col_idx -= 1
#             if Perm_col_idx < col
#                 print("Ran out of non-zero columns")
#                 return CuModMatrix(d_A, N; new_size=(rows(A),cols(A))), CuModMatrix(d_L, N; new_size=(rows(A),rows(A))), Perm_rows, Perm_cols
#             end
#         end
#         end
#         # end

#         # println("Starting find_pivot_idx")
#         # @time begin
#         NVTX.@range "Find Pivot" begin
#         k = find_pivot_idx(d_A, rows(A), row, col)
#         p = find_pivot_val(d_A, rows(A), row, col)
#         end
#         # end

#         NVTX.@range "Check if Pivot is Zero" begin
#         if p == 0
#             d_L[row:end,col] .= 1
#             d_L[row+1:end,col] .= 0
#             col += 1
#             continue
#         end
#         end

#         # println("Starting swap_and_mod_lu")
#         # @time begin
#         NVTX.@range "Swap and Mod" begin
#             p_inv = mod_inv(p, N)
#             swap_and_mod_lu(d_A, d_L, row+k-1, row, p_inv, N, Perm_rows; perm_stack)
#         end
#         # end

#         # println("Starting normalize_lu_broadcast")
#         # @time begin
#         _print_plup_debug("Swapped and modded")
#         NVTX.@range "Normalize LU Broadcast" begin
#             normalize_lu_broadcast(d_A, d_L, rows(A), row, col, p_inv, p, N)
#         end
#         # end

#         _print_plup_debug("Normalized")

#         NVTX.@range "Check if Done" begin
#         if row == rows(A) || col == cols(A)
#             break
#         end
#         end
#         # println("d_A:")
#         # display(@view d_A[1:rows(A),1:cols(A)])

#         # # @time begin
#         NVTX.@range "Update Sub Matrix Row Shared" begin
#             @cuda threads=(TILE_WIDTH) blocks=(ceil(Int, (cols(A)+1-col)/TILE_WIDTH)) update_sub_matrix_row_2dshared(d_A, row, col, N, rows(A))
#         end
#         # # end

#         NVTX.@range "Zero out Sub Matrix" begin
#             d_A[row+1:end,col] .= 0
#         end

#         # println("d_A:")
#         # display(@view d_A[1:rows(A),1:cols(A)])

#         # CUDA.synchronize()

#         # println("Starting update_sub_matrix_col_shared_tiled")
#         # @time begin
#         # @cuda threads=(TILE_WIDTH, 1) blocks=(ceil(Int, (cols(A)+1-col)/TILE_WIDTH), ceil(Int, (rows(A)+1-row)/TILE_WIDTH)) update_sub_matrix_col_shared_tiled(d_A, row, col, N)
#         # end

#         # CUDA.synchronize()

#         # println("d_A:")
#         # display(@view d_A[1:rows(A),1:cols(A)])

#         _print_plup_debug("Updated Sub Matrix")

#         row += 1
#         col += 1
#     end

#     NVTX.@range "End PLUQ" begin
#     U = CuModMatrix(d_A, N; new_size=(rows(A),cols(A)))
#     L = CuModMatrix(d_L, N; new_size=(rows(A),rows(A)))
#     end

#     if perm_array
#         return (U, L, Perm_rows, Perm_cols)
#     else
#         return (
#             U,
#             L,
#             perm_array_to_matrix(Perm_rows, N, (rows(A), rows(A)), perm_stack=perm_stack),
#             perm_array_to_matrix(Perm_cols, N, (cols(A), cols(A)), perm_stack=perm_stack)
#         )
#     end
# end
