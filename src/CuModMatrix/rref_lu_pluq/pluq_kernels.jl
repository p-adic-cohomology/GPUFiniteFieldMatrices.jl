using GPUFiniteFieldMatrices
using Unroll
using CUDA

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

    if inv < 0
        inv += P
    end

    return inv
end

"""
Kernel to perform PLUQ factorization on a CuModMatrix.

# Arguments
- `A`: The CuModMatrix to factorize.
- `debug`: Whether to print debug information.

# Returns
- `U`: The upper triangular factor.
- `L`: The lower triangular factor.
- `Perm_rows`: The array of row permutation tuples.
- `Perm_cols`: The array of column permutation tuples.
"""
function pluq_gpu_kernel(A::CuModMatrix; debug::Bool = false)

    function _print_plup_debug(stage)
        if debug
            println("Stage: $stage")
            println("Perm_col_idx: ", Perm_col_idx)
            println("Perm_cols: ", Perm_cols)
            println("Perm_rows: ", Perm_rows)
            println("row: ", row)
            println("col: ", col)
            println()

            println("d_A:")
            display(@view d_A[1:rows, 1:cols])
            println("d_L:")
            display(@view d_L[1:rows, 1:rows])
            println()
        end
    end

    # NVTX.@range "Init PLUQ d_A, d_L, Perm_rows, Perm_cols" begin
    N = A.N

    A_padded_rows = size(A.data, 1)
    A_padded_cols = size(A.data, 2)

    d_A = copy(A.data)
    d_L = CUDA.zeros(eltype(A.data), (A_padded_rows, A_padded_rows))

    Perm_rows = Array{Tuple{Int,Int}}(undef, 0)
    Perm_cols = Array{Tuple{Int,Int}}(undef, 0)
    # end

    rows, cols = size(A.data)
    rows -= TILE_WIDTH
    cols -= TILE_WIDTH

    row = 1
    col = 1
    Perm_col_idx = cols

    while row <= rows && col <= cols

        pivot_val, pivot_idx = -1, -1

        _print_plup_debug("Iteration $row, $col start")

        # NVTX.@range "Find pivot col" begin
        while true
            pivot_val, pivot_idx = find_pivot(d_A, rows, row, col, Perm_cols, Perm_col_idx)

            if pivot_val > 0
                break
            else
                col += 1
                if col > cols
                    break
                end
            end
        end
        # end

        if col > cols
            break
        end

        # NVTX.@range "Mod Inverse" begin
        pivot_val_inv = mod_inv(pivot_val, N)
        # end

        # NVTX.@range "Swap and mod" begin
        swap_and_mod(
            d_A,
            d_L,
            row,
            pivot_idx+row-1,
            pivot_val_inv,
            rows,
            cols,
            N,
            Perm_rows,
        )
        # end

        _print_plup_debug("Iteration $row, $col swapped and modded")

        # NVTX.@range "Move and zero out" begin
        move_and_zero_out(d_A, d_L, rows, row, col, pivot_val_inv, pivot_val, N)
        # end

        _print_plup_debug("Iteration $row, $col moved and zeroed out")

        # NVTX.@range "Update sub matrix" begin
        @cuda blocks=cld(cols - col + 1, TILE_WIDTH) threads=TILE_WIDTH shmem=TILE_WIDTH*sizeof(
            DEFAULT_TYPE,
        ) update_sub_matrix_kernel(d_A, d_L, row, col, N, rows)
        # end

        _print_plup_debug("Iteration $row, $col updated sub matrix")

        CUDA.synchronize()

        row += 1
        col += 1

    end

    # NVTX.@range "End PLUQ" begin
    U = CuModMatrix(d_A, N; new_size = (rows, cols))
    L = CuModMatrix(d_L, N; new_size = (rows, rows))
    # end

    return (
        U,
        L,
        Perm_rows,
        Perm_cols,
        # perm_array_to_matrix(Perm_rows, N, (rows, rows); perm_stack=true),
        # perm_array_to_matrix(Perm_cols, N, (cols, cols); perm_stack=true)
    )
end

"""
Find the pivot value and index in a column of a CuModMatrix
below the provided row (inclusive).

If the pivot value is 0, we conclude that the column is all zeros,
so we swap the column with the column at Perm_col_idx, and update
Perm_cols. In this case, we return -1, -1.

# Arguments
- `d_A`: The CuModMatrix to find the pivot in.
- `A_rows`: The number of rows in the CuModMatrix.
- `row`: The row to start searching from (inclusive).
- `col`: The column to search in.
- `Perm_cols`: The array of column permutations.
- `Perm_col_idx`: The index of the column to swap with if the pivot is 0.

# Returns
- `pivot_val`: The pivot value.
- `pivot_idx`: The index of the pivot value.
"""
function find_pivot(d_A, A_rows, row, col, Perm_cols, Perm_col_idx)

    col_view = @view d_A[row:A_rows, col]
    pivot_val, pivot_idx = findmax(col_view)

    if pivot_val == 0
        # NVTX.@range "Swap cols" begin
        @cuda blocks=cld(A_rows, TILE_WIDTH) threads=TILE_WIDTH swap_cols(
            d_A,
            col,
            Perm_col_idx,
        )
        # end
        # NVTX.@range "Update Perm_cols" begin
        push!(Perm_cols, (col, Perm_col_idx))
        # end
        return -1, -1
    end

    return pivot_val, pivot_idx
end

"""
Kernel to swap two columns of a CuModMatrix.
It is meant to be called in blocks of 32 threads,
with cld(nrows, 32) blocks in total.

# Arguments
- `d_A`: The CuModMatrix to swap columns in.
- `col1`: The index of the first column to swap.
- `col2`: The index of the second column to swap.

# Returns
- `nothing`
"""
function swap_cols(d_A, col1, col2)

    row = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    temp = d_A[row, col1]
    d_A[row, col1] = d_A[row, col2]
    d_A[row, col2] = temp

    return nothing
end

"""
Swap two rows of a CuModMatrix and mod them.
It is meant to be called in blocks of 32 threads,
with cld(ncols, 32) blocks in total.

# Arguments
- `d_A`: The CuModMatrix to swap rows in.
- `d_L`: The CuModMatrix to store the swapped rows.
- `row`: The index of the row to swap.
- `prow`: The index of the row to swap with.
- `p_inv`: The modular inverse of the pivot value.
- `nrows`: The number of rows in the CuModMatrix.
- `ncols`: The number of columns in the CuModMatrix.
- `N`: The modulus.
- `Perm_rows`: The array of row permutation tuples.

# Returns
- `nothing`
"""
function swap_and_mod(d_A, d_L, row, prow, p_inv, nrows, ncols, N, Perm_rows)

    # NVTX.@range "Swap d_A rows and mod" begin
    @cuda blocks=cld(ncols, TILE_WIDTH) threads=TILE_WIDTH swap_rows_and_mod(
        d_A,
        prow,
        row,
        ncols,
        p_inv,
        N,
    )
    # end

    # NVTX.@range "Swap d_L rows" begin
    @cuda blocks=cld(nrows, TILE_WIDTH) threads=TILE_WIDTH swap_rows(d_L, row, prow, nrows)
    # end

    # NVTX.@range "Update Perm_rows" begin
    if row != prow
        push!(Perm_rows, (row, prow))
    end
    # end

    return nothing
end

"""
Kernel to swap two rows of a CuModMatrix.
It is meant to be called in blocks of 32 threads,
with cld(ncols, 32) blocks in total.

# Arguments
- `matrix`: The CuModMatrix to swap rows in.
- `row1`: The index of the first row to swap.
- `row2`: The index of the second row to swap.
- `ncols`: The number of columns in the CuModMatrix.

# Returns
- `nothing`
"""
function swap_rows(matrix, row1, row2, ncols)

    col = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    temp = matrix[row1, col]
    matrix[row1, col] = matrix[row2, col]
    matrix[row2, col] = temp

    return nothing
end

"""
Kernel to swap two rows of a CuModMatrix and mod them.
It is meant to be called in blocks of 32 threads,
with cld(ncols, 32) blocks in total.

# Arguments
- `matrix`: The CuModMatrix to swap rows in.
- `row1`: The index of the first row to swap.
- `row2`: The index of the second row to swap.
- `ncols`: The number of columns in the CuModMatrix.
- `p_inv`: The modular inverse of the pivot value.
- `N`: The modulus.

# Returns
- `nothing`
"""
function swap_rows_and_mod(matrix, row1, row2, ncols, p_inv, N)

    col = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    while col <= ncols
        temp = matrix[row1, col]
        matrix[row1, col] = matrix[row2, col]
        matrix[row2, col] = mod(temp * p_inv, N)

        col += blockDim().x * gridDim().x
    end

    return nothing
end

"""
Kernel to move a column and zero out the elements below the pivot.
It is meant to be called in blocks of 32 threads,
with cld(A_rows - row + 1, 32) blocks in total.

# Arguments
- `d_A`: The CuModMatrix to move and zero out.
- `d_L`: The CuModMatrix to store the moved column.
- `A_rows`: The number of rows in the CuModMatrix.
- `row`: The row to start moving from.
- `col`: The column to move.
- `p_inv`: The modular inverse of the pivot value.
- `p`: The pivot value.
- `N`: The modulus.

# Returns
- `nothing`
"""
function move_and_zero_out(d_A, d_L, A_rows, row, col, p_inv, p, N)

    # NVTX.@range "Set d_L diag to p" begin
    @cuda blocks=1 threads=1 set_elem_kernel(d_L, row, col, p)
    # end

    # NVTX.@range "Move col and zero out" begin
    @cuda blocks=cld(A_rows - row + 1, TILE_WIDTH) threads=TILE_WIDTH move_col_and_zero_out(
        d_A,
        d_L,
        row,
        col,
    )
    # end

    return nothing
end

"""
Kernel to set an element of a CuModMatrix.
It is meant to be called in blocks of 1 thread.

# Arguments
- `matrix`: The CuModMatrix to set the element in.
- `row`: The index of the row to set the element in.
- `col`: The index of the column to set the element in.
- `value`: The value to set the element to.

# Returns
- `nothing`
"""
function set_elem_kernel(matrix, row, col, value)
    matrix[row, col] = value
    return nothing
end

"""
Kernel to move a column and zero out the elements below the pivot.
It is meant to be called in blocks of 32 threads,
with cld(A_rows - row + 1, 32) blocks in total.

# Arguments
- `d_A`: The CuModMatrix to move and zero out.
- `d_L`: The CuModMatrix to store the moved column.
- `row_start`: The row to start moving from.
- `col`: The column to move.

# Returns
- `nothing`
"""
function move_col_and_zero_out(d_A, d_L, row_start, col)

    row = threadIdx().x + (blockIdx().x - 1) * blockDim().x + row_start

    d_L[row, col] = d_A[row, col]
    d_A[row, col] = 0

    return nothing
end

"""
Kernel to update the sub matrix in pluq/rref.
It is meant to be called in blocks of 32 threads,
with cld(num_rows, 32) blocks in total.

# Arguments
- `d_A`: The CuModMatrix to update.
- `d_L`: The CuModMatrix to store the updated sub matrix.
- `p_row`: The row of the pivot.
- `p_col`: The column of the pivot.
- `N`: The modulus.
- `num_rows`: The number of rows in the CuModMatrix.

# Returns
- `nothing`
"""
function update_sub_matrix_kernel(d_A, d_L, p_row, p_col, N, num_rows)

    tx = threadIdx().x
    bx = blockIdx().x
    bdx = blockDim().x

    t_shift = (bx - 1) * bdx + tx
    b_shift = (bx - 1) * bdx
    row_idx = p_row + tx

    shared_pivot_row = CUDA.CuStaticSharedArray(DEFAULT_TYPE, TILE_WIDTH)

    @inbounds shared_pivot_row[tx] = d_A[p_row, p_col+t_shift]

    CUDA.sync_threads()

    # while row_idx <= num_rows + TILE_WIDTH - p_row + 1
    while row_idx <= num_rows

        multiplier = N - d_L[row_idx, p_col]

        @unroll for col_idx = 1:TILE_WIDTH
            @inbounds d_A[row_idx, col_idx+b_shift+p_col] = mod(
                d_A[row_idx, col_idx+b_shift+p_col] +
                multiplier * shared_pivot_row[col_idx],
                N,
            )
        end

        row_idx += TILE_WIDTH
    end

    return
end
