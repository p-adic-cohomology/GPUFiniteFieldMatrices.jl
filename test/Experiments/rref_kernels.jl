using CUDA, LinearAlgebra

# """
#     old_rref_gpu(A::Array, P::Int)

# Row reduces matrix A mod P on the GPU.
# """
# function old_rref_gpu(A, P)

#     # Find dimensions and padded
#     A_rows, A_cols = size(A)
#     A_padded_rows = ceil(Int, A_rows / TILE_WIDTH) * TILE_WIDTH
#     A_padded_cols = ceil(Int, A_cols / TILE_WIDTH) * TILE_WIDTH

#     # Create padded GPU CuArray
#     d_A = CUDA.CuArray{Int}(undef, (A_padded_rows+TILE_WIDTH, A_padded_cols+TILE_WIDTH))

#     # Copy over matrix to GPU
#     A_inds = CartesianIndices(A)
#     d_A_inds = CartesianIndices((1:A_rows,1:A_cols))
#     copyto!(d_A, d_A_inds, A, A_inds)

#     # For every row
#     min_dim = min(A_rows,A_cols)
#     for k = 1:min_dim
#         # println("k: $k")
#         # println(d_A)

#         # Find pivot and its p_row and p_col
#         CUDA.@allowscalar p, p_row, p_col = find_pivot(d_A, k, A_rows)

#         # println("PIVOT DONE")
#         # println(d_A)

#         # Move p_row to the kth row
#         swap(d_A, k, p_row)
#         # Also swap_and_mod the pivot row

#         # println("SWAP DONE")
#         # println(d_A)

#         # Normalize the p_row
#         p_inv = mod_inv(p, P)
#         @cuda threads=(TILE_WIDTH) blocks=(div(A_padded_rows,TILE_WIDTH)) normalize(d_A, k, p_inv, P, A_rows)
#         # normalize pivot column return/alloc mult. inv. array
#         # put mult inv inside of d_A itself?

#         # println("NORMALIZE DONE")
#         # println(d_A)

#         # Update the lower submatrix
#         blocks_x = div(A_rows-k,TILE_WIDTH)+1
#         blocks_y = div(A_cols-k,TILE_WIDTH)+1
#         shared_mem_size = sizeof(TYPE) * TILE_WIDTH * (blocks_x*2 + blocks_y)
#         @cuda threads=(TILE_WIDTH,TILE_WIDTH) blocks=(blocks_x,blocks_y) shmem=shared_mem_size update_sub_matrix(d_A, k, P, blocks_x, blocks_y)
#         # Use the mult. inv. to update rows below pivot

#         # println("UPDATE DONE")
#         # println(d_A)
#     end

#     return Array(d_A)[1:A_rows,1:A_cols]
# end

# """
#     find_pivot(d_A::CUDA.CuArray,k::Int)

# Finds the pivot in the kth column.
# Looks only at rows k:A_rows.
# Returns the pivot, p_row, p_col.
# """
# function find_pivot(d_A, k, A_rows)

#     # Technically we do not need to find the largest in column
#     # Since we do not care about numerical stability
#     range = Array(d_A[:,k])
#     # println(range)
#     # println(typeof(range))
#     range = range[k:A_rows]
#     p_row = argmax(range) + k - 1

#     # Ideas:
#     # use CUDA.jl mapreduce()
#     # Write our own kernel

#     return maximum(range), p_row, k
# end

# """
#     swap(d_A::CUDA.CuArray,k::Int,p_row::Int)

# Swaps the kth row and the p_row row on the GPU.
# """
# function swap(d_A, k, p_row)

#     # see other implementations and compare
#     # mod pivot row as well.

#     d_A[k,:], d_A[p_row,:] = d_A[p_row,:], d_A[k,:]

#     return
# end

"""
    mod_inv(p::Int,P::int)

Returns the modular inverse of p mod P.
Uses Extended Euclidean Algorithm.
"""
function mod_inv(p, P)

    # Remark: Since p is prime, we are in a field
    # And thus an inverse is guaranteed to exist
    inv = 0
    new_inv = 1
    rem = P
    new_rem = p

    while new_rem != 0
        quotient = div(rem, new_rem)
        inv, new_inv = new_inv, inv - quotient * new_inv
        rem, new_rem = new_rem, rem - quotient * new_rem
    end

    # TODO Check stock implementation and what CUDA.jl does
    # If worse, consider PR

    if inv < 0
        inv += P
    end

    # Consider doing longer no-if Version
    # TODO compare

    return inv
end

# """
#     normalize(d_A::CUDA.CuArray,k::Int,A_rows::Int)

# Normalizes the kth row with the given pivot.
# We do this using the mult. inv. of the pivot.
# """
# function normalize(d_A, k, p_inv, P, A_rows)

#     idx = CUDA.threadIdx().x
#     width = CUDA.blockDim().x

#     i = k+idx-1
#     while i <= A_rows
#         # Idea: To compute d_A[i,k] =/ p is the same
#         # as finding p^-1 mod P and then computing
#         # d_A[i,k] = (d_A[i,k] * p^-1) % P

#         # CUDA.@cuprintln("i: $i")
#         # CUDA.@cuprintln(d_A[k,i])

#         d_A[k,i] = (d_A[k,i] * p_inv) % P
#         i += width
#     end

#     return
# end

#function update_sub_matrix_col()
#    # Each thread does a whole column
#
#    p_row_shared = ...
#    mult_shared = ...
#
#    # load mult inv into local variable
#
#    CUDA.sync_threads()
#
#    # loop though
#    for i=k:A_rows/TILE_WIDTH
#
#        # Load a single elem from d_A[:,k] into some shared mem
#
#        CUDA.sync_threads()
#
#
#
#    end
#
#end

# """
#     updateSubMatrix(d_A::CUDA.CuArray,k::Int,P::Int,A_rows::Int,A_cols::Int)

# Updates d_A[k:A_rows,k:A_cols] to make the pivot columns all 0 below the pivot.
# """
# function update_sub_matrix(d_A, k, P, block_x, block_y)

#     # Define shared matrices for pivot row and col
#     p_row_shared = CUDA.CuDynamicSharedArray(TYPE, block_x*TILE_WIDTH)
#     p_col_shared = CUDA.CuDynamicSharedArray(TYPE, block_y*TILE_WIDTH)
#     mult_shared = CUDA.CuDynamicSharedArray(TYPE, block_y*TILE_WIDTH)

#     # TODO only need row shared, mult inv inside of d_A
#     # Why use Dynamic? Make Shared mem size of tile...

#     tx = threadIdx().x
#     ty = threadIdx().y
#     bx = blockIdx().x
#     by = blockIdx().y

#     row = (by-1)*TILE_WIDTH + ty
#     col = (bx-1)*TILE_WIDTH + tx
#     # CUDA.@cuprintln("row: $row, col: $col")

#     # Collaboratively put the pivot row and col into shared memory
#     p_row_shared[col] = d_A[k, k+col]
#     p_col_shared[row] = d_A[k+row, k]
#     # CUDA.@cuprintln("shared_row: $(p_row_shared[col])")

#     CUDA.sync_threads()

#     # Find mult. inv. of the p_col for reduction

#     inv = 0
#     new_inv = 1
#     rem = P
#     new_rem = p_col_shared[row]

#     while new_rem != 0
#         quotient = div(rem,new_rem)

#         inv, new_inv = new_inv, inv - quotient * new_inv
#         rem, new_rem = new_rem, rem - quotient * new_rem
#     end

#     if inv < 0
#         inv += P
#     end

#     # CUDA.@cuprintln("ty: $ty, inv: $inv")
#     mult_shared[row] = (inv * p_col_shared[row]) % P

#     CUDA.sync_threads()

#     # We want to calculate, for row and col,
#     # d_A[row,col] = d_A[row,col] - p_row_shared[tx]*mult[ty] mod P

#     # TODO

#     result = d_A[k+row,k+col] - p_row_shared[col]*mult_shared[row]

#     while result < 0
#         result += P
#     end

#     d_A[k+row,k+col] = result
#     d_A[k+row,k] = 0

#     CUDA.sync_threads()

#     return
# end
