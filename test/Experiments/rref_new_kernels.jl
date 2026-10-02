#TODO: Add docstring
function rref_gpu(A, P)

    A_rows, A_cols = size(A)
    A_padded_rows = (ceil(Int, A_rows / TILE_WIDTH)+1) * TILE_WIDTH
    A_padded_cols = (ceil(Int, A_cols / TILE_WIDTH)+1) * TILE_WIDTH

    d_A = CUDA.CuArray{Int}(undef, (A_padded_rows+TILE_WIDTH, A_padded_cols+TILE_WIDTH))

    A_inds = CartesianIndices(A)
    d_A_inds = CartesianIndices((1:A_rows, 1:A_cols))
    copyto!(d_A, d_A_inds, A, A_inds)

    row = 1
    col = 1

    while row <= A_rows && col <= A_cols

        p = find_pivot_val(d_A, A_rows, row, col)

        if p == 0
            col += 1
            continue
        end

        p_inv = mod_inv(p, P)
        swap_and_mod(d_A, k, row, p_inv, P)

        normalize_broadcast(d_A, col, p_inv, P)

        @cuda threads=(TILE_WIDTH) blocks=(div(A_rows, TILE_WIDTH)) update_sub_matrix_row(
            d_A,
            row,
            col,
            div(A_cols-col, TILE_WIDTH),
            P,
        )

        row += 1
        col += 1

    end

    return Array(d_A)
end

function lu_gpu(A, P)

    # Find padded dimensions
    A_rows, A_cols = size(A)
    A_padded_rows = A_rows + TILE_WIDTH + 1
    A_padded_cols = A_cols + TILE_WIDTH + 1

    # Deifne gpu matrices
    # We add another TILE_WDITH for the reduction of the last few rows
    d_A = CUDA.CuArray{Int}(undef, (A_padded_rows, A_padded_cols))
    d_L = CUDA.CuArray{Int}(undef, (A_rows, A_rows))
    Perm = Array(1:A_rows)

    A_inds = CartesianIndices(A)
    d_A_inds = CartesianIndices((1:A_rows, 1:A_cols))
    copyto!(d_A, d_A_inds, A, A_inds)

    row = 1
    L_col = 1
    col = 1

    while row <= A_rows && col <= A_cols
        writedlm("checkpoints/plu_dA_$row.csv", Array(d_A), ',')

        k = find_pivot_idx(d_A, A_rows, row, col)
        p = find_pivot_val(d_A, A_rows, row, col)
        # if DEBUG
        #     println("Finding pivots")
        #     println("k: ", k)
        #     println("p: ", p)
        #     println("d_A: ", d_A)
        # end

        if p == 0
            col += 1
            continue
        end

        p_inv = mod_inv(p, P)
        swap_and_mod_lu(d_A, d_L, row+k-1, row, p_inv, P, Perm)
        # if DEBUG
        #     println("Swap and mod")
        #     println("p_inv: ",p_inv)
        #     println("d_A: ", d_A)
        # end

        normalize_lu_broadcast(d_A, d_L, A_rows, row, L_col, p_inv, p, P)
        # if DEBUG
        #     println("Normalize")
        #     println("d_L: ", d_L)
        #     println("d_A: ", d_A)
        # end

        if row == A_rows || col == A_cols
            break
        end

        @cuda threads=(TILE_WIDTH, TILE_WIDTH) blocks=(div(A_rows-row, TILE_WIDTH)+1, 1) update_sub_matrix_row(
            d_A,
            row,
            col,
            div(A_cols-col, TILE_WIDTH),
            P,
        )
        # if DEBUG
        #     println("Update Sub Matrix")
        #     println("d_A: ", d_A)
        # end

        row += 1
        L_col += 1
        col += 1

    end
    return (Array(d_A)[1:A_rows, 1:A_cols], Array(d_L)[1:A_rows, 1:A_rows], Perm)
end

function plup_gpu(A, P)

    A_rows, A_cols = size(A)
    A_padded_rows = A_rows + TILE_WIDTH + 1
    A_padded_cols = A_cols + TILE_WIDTH + 1

    d_A = CUDA.CuArray{Int}(undef, (A_padded_rows, A_padded_cols))
    d_L = CUDA.CuArray{Int}(undef, (A_rows, A_rows))
    Perm_rows = Array(1:A_rows)
    Perm_cols = Array(1:A_cols)

    A_inds = CartesianIndices(A)
    d_A_inds = CartesianIndices((1:A_rows, 1:A_cols))
    copyto!(d_A, d_A_inds, A, A_inds)

    row = 1
    col = 1
    Perm_col_idx = A_cols

    d_A = d_A .% P

    while row <= A_rows && col <= A_cols
        if DEBUG
            writedlm("checkpoints/dA/plup_dA_$row.csv", Array(d_A), ',')
            writedlm("checkpoints/dL/plup_dL_$row.csv", Array(d_L), ',')
            writedlm("checkpoints/Prows/plup_Prows_$row.csv", Array(Perm_rows), ',')
            writedlm("checkpoints/Pcols/plup_Prows_$row.csv", Array(Perm_cols), ',')
        end

        while find_zero_col_and_swap(d_A, A_rows, row, col, Perm_cols, Perm_col_idx)
            # if DEBUG
            #     println("Swapping columns")
            #     println("Perm_col_idx: ", Perm_col_idx)
            # end
            Perm_col_idx -= 1
        end

        k = find_pivot_idx(d_A, A_rows, row, col)
        p = find_pivot_val(d_A, A_rows, row, col)
        # if DEBUG
        #     println("Finding pivots")
        #     println("k: ", k)
        #     println("p: ", p)
        #     println("d_A: ", d_A)
        # end

        if p == 0
            d_L[row:end, col] .= 1
            d_L[(row+1):end, col] .= 0
            col += 1
            continue
        end

        p_inv = mod_inv(p, P)
        swap_and_mod_lu(d_A, d_L, row+k-1, row, p_inv, P, Perm_rows)
        # if DEBUG
        #     println("Swap and mod")
        #     println("p_inv: ",p_inv)
        #     println("d_A: ", d_A)
        #     println("d_L: ", d_L)
        # end

        normalize_lu_broadcast(d_A, d_L, A_rows, row, col, p_inv, p, P)
        # if DEBUG
        #     println("Normalize")
        #     println("d_L: ", d_L)
        #     println("d_A: ", d_A)
        # end

        if row == A_rows || col == A_cols
            break
        end

        @cuda threads=(TILE_WIDTH, TILE_WIDTH) blocks=(div(A_rows-row, TILE_WIDTH)+1, 1) update_sub_matrix_row(
            d_A,
            row,
            col,
            div(A_cols-col, TILE_WIDTH),
            P,
        )
        # if DEBUG
        #     println("Update Sub Matrix")
        #     println("d_A: ", d_A)
        # end

        row += 1
        col += 1

    end
    return (
        Array(d_A)[1:A_rows, 1:A_cols],
        Array(d_L)[1:A_rows, 1:A_rows],
        Perm_rows,
        Perm_cols,
    )
end

function find_zero_col_and_swap(
    d_A,
    A_rows,
    row,
    col,
    Perm_cols,
    Perm_col_idx;
    perm_stack = false,
)
    max_val = @view d_A[row:A_rows, col]
    # If we have a column of all zeroes, swap it with the last available column.
    if maximum(max_val) == 0
        d_A[:, col], d_A[:, Perm_col_idx] = d_A[:, Perm_col_idx], d_A[:, col]
        if perm_stack
            push!(Perm_cols, (col, Perm_col_idx))
        else
            Perm_cols[col], Perm_cols[Perm_col_idx] =
                Perm_cols[Perm_col_idx], Perm_cols[col]
        end
        return true
    end
    return false
end

function find_pivot(d_A, A_rows, row, col)
    A_temp = @view d_A[row:A_rows, col]
    return argmax(A_temp)
end

function find_pivot_idx(d_A::CUDA.CuArray, A_rows::Int, row::Int, col::Int)
    A_temp = @view d_A[row:A_rows, col]
    return argmax(A_temp)
end

function find_pivot_val(d_A, A_rows, row, col)
    A_temp = @view d_A[row:A_rows, col]
    return maximum(A_temp)
end

function find_pivot_custom(d_A, A_rows, row, col, res)
    tidx = threadIdx().x
    bidx = blockIdx().x
    idx = tid + (bid - 1) * blockDim().x

    if arr[idx, col] != 0
        res[1] = idx
        return
    end

    return
end

function mod_inv(p, P)

    # Remark: Since p is prime, we are in a field
    # And thus an inverse is guaranteed to exist
    inv, new_inv = 0, 1
    rem, new_rem = P, p

    while new_rem != 0
        quotient = div(rem, new_rem)
        inv, new_inv = new_inv, inv - quotient * new_inv
        rem, new_rem = new_rem, rem - quotient * new_rem
    end

    if inv < 0
        inv += P
    end

    return inv
end

function mod_inv_no_if(p, P)

    # Remark: Since p is prime, we are in a field
    # And thus an inverse is guaranteed to exist
    inv, new_inv = 0, 1
    rem, new_rem = P, p

    while new_rem != 0
        quotient = div(rem, new_rem)
        inv, new_inv = new_inv, inv - quotient * new_inv
        rem, new_rem = new_rem, rem - quotient * new_rem
    end

    return mod(inv, P)
end

function swap_and_mod_threaded(d_A, k, p_row, inv)

    col = (blockIdx().x-1)*TILE_WIDTH + threadIdx().x

    # swap k and p_row
    d_A[k, col], d_A[p_row, col] = d_A[p_row, col], d_A[k, col]

    # normalize p_row
    d_A[p_row, col] = inv * d_A[p_row, col] % P

    return
end

# function swap_and_mod(d_A, k, p_row, inv, P)

#     # swap k and p_row
#     d_A[k,:], d_A[p_row,:] = d_A[p_row,:], d_A[k,:]

#     # normalize p_row
#     d_A[p_row,:] = (d_A[p_row,:] .* inv) .% P

#     return
# end

function swap_and_mod_lu(d_A, d_L, k, p_row, p_inv, N, Perm; perm_stack = false)

    d_A[k, :], d_A[p_row, :] = d_A[p_row, :], d_A[k, :]
    d_L[k, :], d_L[p_row, :] = d_L[p_row, :], d_L[k, :]

    if perm_stack
        push!(Perm, (k, p_row))
    else
        Perm[k], Perm[p_row] = Perm[p_row], Perm[k]
    end

    @. d_A[p_row, :] = mod((d_A[p_row, :] * p_inv), N)
    return
end

function normalize(d_A, A_rows, col, p_inv, P)

    idx = threadIdx().x
    bx = blockIdx().x

    i = (bx-1)*TILE_WIDTH + col + idx
    while i <= A_rows
        d_A[i, col] = (d_A[i, col] * p_inv) % P
        i += 1
    end

    return
end

function normalize_lu(d_A, A_rows, col, p_inv, P, d_L)

    idx = threadIdx().x
    bx = blockIdx().x

    i = (bx-1)*TILE_WIDTH + col + idx
    while i <= A_rows
        res = (d_A[i, col] * p_inv) % P
        d_A[i, col] = res
        d_L[i, col] = res
        i += 1
    end

    return
end

function normalize_broadcast(d_A, col, p_inv, P)

    d_A[:, col] = (d_A[:, col] * p_inv) .% P

    return
end

"""
    normalize_lu_broadcast(d_A, d_L, A_rows, row, L_col, p_inv, P)

Update the L matrix values during LU decomposition.
"""

#     d_L[row:end,L_col] .= p
#     d_L[row+1:end,L_col] = d_A[row+1:A_rows,L_col]
#     d_L[row+1:end,L_col] = mod_inv.(Array(d_A[row+1:A_rows,L_col]), P)

function normalize_lu_broadcast(d_A, d_L, A_rows, row, L_col, p_inv, p, N)
    # Set the diagonal element in L to the pivot value
    d_L[row:end, L_col] .= p

    # Copy the column values from d_A to d_L for the lower part
    @. d_L[(row+1):end, L_col] = d_A[(row+1):end, L_col]

    # Normalize the pivot row in d_A
    # @. d_A[row, :] = mod(d_A[row, :] * p_inv, N)

    return
end

function update_sub_matrix_row(d_A, p_row, p_col, P)
    tid = threadIdx().x  # Thread ID within block
    bid = blockIdx().x   # Block ID

    # Calculate which row this thread is responsible for
    row_idx = tid + (bid - 1) * blockDim().x + p_row

    # Skip if we're beyond the matrix size
    # if row_idx <= p_row || row_idx > size(d_A, 1)
    #     return
    # end

    # Get the value in the pivot column for this row
    pivot_col_val = d_A[row_idx, p_col]

    # # If the value is already 0, no need to update
    # if pivot_col_val == 0
    #     return
    # end

    # Calculate the multiplier for elimination
    # For subtraction in modular arithmetic: a - b ≡ a + (P - b) (mod P)
    multiplier = P - pivot_col_val

    # Process all columns from pivot column to the end of the matrix
    for col_idx = p_col:size(d_A, 2)
        if col_idx <= size(d_A, 2)
            pivot_val = d_A[p_row, col_idx]
            d_A[row_idx, col_idx] = mod((d_A[row_idx, col_idx] + multiplier * pivot_val), P)
        end

        CUDA.sync_threads()
    end

    # Zero out the pivot column value
    d_A[row_idx, p_col] = 0

    return
end

function update_sub_matrix_col_shared(d_A, p_row, p_col, N)

    tx = threadIdx().x
    bx = blockIdx().x
    bdx = blockDim().x

    # Each thread corresponds to one column
    col_idx = (bx - 1) * bdx + tx + p_col - 1

    # Shared memory for the pivot row (size = bdx)
    shared_pivot_row = CUDA.CuStaticSharedArray(Float64, TILE_WIDTH)

    # Load pivot row values into shared memory (once, by top-row block)
    if tx < bdx
        shared_pivot_row[tx] = d_A[p_row, col_idx]
    end

    CUDA.sync_threads()

    num_rows = size(d_A, 1)

    # Loop over rows beneath the pivot
    row_idx = p_row + 1
    while row_idx <= num_rows
        # Get the pivot column value for this row
        pivot_col_val = d_A[row_idx, p_col]
        # If zero, nothing to do


        multiplier = N - pivot_col_val
        pivot_val = shared_pivot_row[tx]
        d_A[row_idx, col_idx] = mod(d_A[row_idx, col_idx] + multiplier * pivot_val, N)

        row_idx += 1
    end

    return
end

function update_sub_matrix_col_2dshared(d_A, p_row, p_col, N, num_rows)

    tx = threadIdx().x
    bx = blockIdx().x
    bdx = blockDim().x

    t_shift = (bx - 1) * bdx + tx
    b_shift = (bx - 1) * bdx
    row_idx = t_shift + p_row

    # Shared memory for the pivot row (size = bdx)
    shared_pivot_row = CUDA.CuStaticSharedArray(Float64, TILE_WIDTH)

    @inbounds shared_pivot_row[tx] = d_A[p_row, t_shift+p_col]

    CUDA.sync_threads()

    while row_idx <= num_rows + TILE_WIDTH - p_row

        multiplier = d_A[row_idx, p_col]
        multiplier = N - multiplier

        @unroll for col_idx = 1:TILE_WIDTH
            @inbounds d_A[row_idx, col_idx+b_shift+p_col] = mod(
                d_A[row_idx, col_idx+b_shift+p_col] +
                multiplier * shared_pivot_row[col_idx],
                N,
            )
            # CUDA.sync_threads()
        end

        row_idx += TILE_WIDTH
    end

    return
end

function update_sub_matrix_row_2dshared(d_A, p_row, p_col, N, num_rows)

    tx = threadIdx().x
    bx = blockIdx().x
    bdx = blockDim().x

    t_shift = (bx - 1) * bdx + tx
    b_shift = (bx - 1) * bdx
    row_idx = t_shift + p_row

    # Shared memory for the pivot col (size = bdx)
    shared_pivot_col = CUDA.CuStaticSharedArray(Float64, TILE_WIDTH)

    shared_pivot_col[tx] = N - d_A[t_shift+p_row, p_col]

    CUDA.sync_threads()

    while row_idx <= num_rows + TILE_WIDTH - p_row

        multiplier = d_A[t_shift+p_row, p_col]

        @unroll for col_idx = 1:TILE_WIDTH
            @inbounds d_A[col_idx+b_shift+p_col, row_idx] = mod(
                d_A[col_idx+b_shift+p_col, row_idx] +
                multiplier * shared_pivot_col[col_idx],
                N,
            )
        end

        row_idx += TILE_WIDTH
    end

    return
end

function update_sub_matrix_col_shared_tiled(d_A, p_row, p_col, N)

    tx = threadIdx().x
    bx = blockIdx().x
    by = blockIdx().y

    col_idx = (bx-1)*TILE_WIDTH + tx + p_col - 1
    row_idx = (by-1)*TILE_WIDTH + p_row

    shared_pivot_row = CUDA.CuStaticSharedArray(Float64, TILE_WIDTH)

    if tx < TILE_WIDTH && by == 1
        shared_pivot_row[tx] = d_A[p_row, col_idx]
    end

    CUDA.sync_threads()

    row_shift = 1
    while row_shift <= TILE_WIDTH
        multiplier = N - d_A[row_idx+row_shift, p_col]
        d_A[row_idx+row_shift, col_idx] =
            mod(d_A[row_idx+row_shift, col_idx] + multiplier * shared_pivot_row[tx], N)
        row_shift += 1
    end

    return
end

function update_sub_matrix_col(d_A, p_row, p_col, bound, P)

    tid = threadIdx().x
    bid = blockIdx().x
    idx = tid + (bid - 1) * blockDim().x

    row_val = d_A[p_row, p_col+idx]
    shared_col = CUDA.CuStaticSharedArray(Int, (TILE_WIDTH))

    m = 0
    while m < bound
        shared_col[tid] = d_A[p_row+tid+m*TILE_WIDTH, p_col]
        d_A[p_row+tid+m*TILE_WIDTH, p_col+idx] =
            (d_A[p_row+tid+m*TILE_WIDTH, p_col+idx] + shared_col[tid] * row_val) % P
        m += 1
    end
    d_A[p_row+idx, p_col] = 0

    return
end

function update_sub_matrix_elem(d_A, p_row, p_col, P)

    tidx = threadIdx().x
    bidx = blockIdx().x
    tidy = threadIdx().y
    bidy = blockIdx().y
    idx = tidx + (bidx - 1) * blockDim().x
    idy = tidy + (bidy - 1) * blockDim().y

    d_A[p_row+idy, p_col+idx] =
        (d_A[p_row+idy, p_col+idx] + d_A[p_row, p_col+idx] * d_A[p_row+idy, p_col]) % P

    return
end

function update_sub_matrix_square(d_A, p_row, p_col, P)

    idx = threadIdx().x
    idy = threadIdx().y
    bx = blockIdx().x
    by = blockIdx().y

    row = (by-1)*TILE_WIDTH + idy
    col = (bx-1)*TILE_WIDTH + idx

    d_A[row, col] = (d_A[row, col] - d_A[row, p_col] * d_A[p_row, col]) % P
    # CUDA.sync_threads()

    return
end

function update_sub_matrix_broadcast(d_A, A_rows, p_row, p_col, P)

    for row = p_row:A_rows
        CUDA.@allowscalar d_A[row, :] =
            (d_A[row, :] + d_A[p_row, :] .* d_A[row, p_col]) .% P
    end

    return
end

function update_sub_matrix_broadcast2(d_A, A_rows, p_row, p_col, P)

    temp = zeros(A_rows)
    temp .= Array(d_A[:, p_col])

    for row = p_row:A_rows
        CUDA.@allowscalar d_A[row, :] = (d_A[row, :] + d_A[p_row, :] .* temp[row]) .% P
    end

    return
end

function update_sub_matrix_broadcast4(d_A, A_rows, p_row, p_col, P)

    temp = zeros(A_rows-p_row+1, 1)
    temp_inds = CartesianIndices((A_rows, 1))
    d_A_inds = CartesianIndices((p_row:A_rows, p_col))
    CUDA.@allowscalar copyto!(d_A, d_A_inds, temp, temp_inds)

    for row = p_row:A_rows
        d_A[row, :] = (d_A[row, :] + d_A[p_row, :] .* temp[row-p_row+1]) .% P
    end

    return
end
