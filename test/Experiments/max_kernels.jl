using CUDA

"""
Optimized kernel to find maximum value and index in a matrix column subrange.
Based on optimized reduction patterns with shared memory and unrolling.
"""

# Kernel for finding max with index tracking
function max_reduce_kernel(
    matrix,
    result_val,
    result_idx,
    col,
    start_row,
    nrows,
    ::Val{BLOCK_SIZE},
) where {BLOCK_SIZE}
    # Shared memory for values and indices
    sdata_val = @cuDynamicSharedMem(Float64, BLOCK_SIZE)
    sdata_idx = @cuDynamicSharedMem(Int64, BLOCK_SIZE)

    tid = threadIdx().x  # 1-based in Julia
    bid = blockIdx().x   # 1-based in Julia

    # Initialize shared memory with thread's initial value
    local_max = 0
    local_idx = 0

    # Simple grid-stride: each thread handles every gridDim().x * blockDim().x elements
    thread_id = (bid - 1) * BLOCK_SIZE + tid
    stride = BLOCK_SIZE

    # Start from this thread's first element
    i = thread_id
    while i <= nrows
        abs_row = start_row + i - 1
        val = matrix[abs_row, col]

        if val > local_max
            local_max = val
            local_idx = abs_row
        end

        i += stride
    end

    # NOW store the thread's local maximum in shared memory
    sdata_val[tid] = local_max
    sdata_idx[tid] = local_idx

    sync_threads()

    # Unrolled reduction with index tracking
    # Note: Using compile-time constants for efficiency
    if BLOCK_SIZE >= 512
        if tid <= 256
            other_idx = tid + 256
            if other_idx <= BLOCK_SIZE && sdata_val[other_idx] > sdata_val[tid]
                sdata_val[tid] = sdata_val[other_idx]
                sdata_idx[tid] = sdata_idx[other_idx]
            end
        end
        sync_threads()
    end

    if BLOCK_SIZE >= 256
        if tid <= 128
            other_idx = tid + 128
            if other_idx <= BLOCK_SIZE && sdata_val[other_idx] > sdata_val[tid]
                sdata_val[tid] = sdata_val[other_idx]
                sdata_idx[tid] = sdata_idx[other_idx]
            end
        end
        sync_threads()
    end

    if BLOCK_SIZE >= 128
        if tid <= 64
            other_idx = tid + 64
            if other_idx <= BLOCK_SIZE && sdata_val[other_idx] > sdata_val[tid]
                sdata_val[tid] = sdata_val[other_idx]
                sdata_idx[tid] = sdata_idx[other_idx]
            end
        end
        sync_threads()
    end

    # Warp-level reduction (no sync needed within warp)
    if tid <= 32
        if BLOCK_SIZE >= 64
            other_idx = tid + 32
            if other_idx <= BLOCK_SIZE && sdata_val[other_idx] > sdata_val[tid]
                sdata_val[tid] = sdata_val[other_idx]
                sdata_idx[tid] = sdata_idx[other_idx]
            end
        end

        if BLOCK_SIZE >= 32
            other_idx = tid + 16
            if other_idx <= BLOCK_SIZE && sdata_val[other_idx] > sdata_val[tid]
                sdata_val[tid] = sdata_val[other_idx]
                sdata_idx[tid] = sdata_idx[other_idx]
            end
        end

        if BLOCK_SIZE >= 16
            other_idx = tid + 8
            if other_idx <= BLOCK_SIZE && sdata_val[other_idx] > sdata_val[tid]
                sdata_val[tid] = sdata_val[other_idx]
                sdata_idx[tid] = sdata_idx[other_idx]
            end
        end

        if BLOCK_SIZE >= 8
            other_idx = tid + 4
            if other_idx <= BLOCK_SIZE && sdata_val[other_idx] > sdata_val[tid]
                sdata_val[tid] = sdata_val[other_idx]
                sdata_idx[tid] = sdata_idx[other_idx]
            end
        end

        if BLOCK_SIZE >= 4
            other_idx = tid + 2
            if other_idx <= BLOCK_SIZE && sdata_val[other_idx] > sdata_val[tid]
                sdata_val[tid] = sdata_val[other_idx]
                sdata_idx[tid] = sdata_idx[other_idx]
            end
        end

        if BLOCK_SIZE >= 2
            other_idx = tid + 1
            if other_idx <= BLOCK_SIZE && sdata_val[other_idx] > sdata_val[tid]
                sdata_val[tid] = sdata_val[other_idx]
                sdata_idx[tid] = sdata_idx[other_idx]
            end
        end
    end

    # Store result from first thread of each block
    if tid == 1
        result_val[bid] = sdata_val[1]
        result_idx[bid] = sdata_idx[1]
    end

    return nothing
end

# Final reduction kernel for multiple blocks
function final_max_kernel(block_vals, block_indices, result_val, result_idx, num_blocks)

    max_val = block_vals[1]
    max_idx = block_indices[1]

    for i = 2:num_blocks
        if block_vals[i] > max_val
            max_val = block_vals[i]
            max_idx = block_indices[i]
        end
    end

    result_val[1] = max_val
    result_idx[1] = max_idx

    return nothing
end

"""
    find_column_max_optimized(matrix, col, start_row; block_size=256)

Find the maximum value and its absolute row index in a column subrange using 
an optimized CUDA kernel with shared memory reduction.

# Arguments
- `matrix`: CUDA matrix (Float64)
- `col`: Column index (1-based)
- `start_row`: Starting row index (1-based)
- `block_size`: CUDA block size (must be power of 2, default 256)

# Returns
- `(max_value, absolute_row_index)`: Tuple with maximum value and its row index

# Example
```julia
M = CuArray(rand(Float64, 1024, 32))
max_val, row_idx = find_column_max_optimized(M, 5, 100)
```
"""
function find_column_max_optimized(
    matrix::CuMatrix{Float64},
    col::Int,
    start_row::Int;
    block_size::Int = 256,
)
    @assert ispow2(block_size) "block_size must be a power of 2"
    @assert 1 <= col <= size(matrix, 2) "Column index out of bounds"
    @assert 1 <= start_row <= size(matrix, 1) "Start row out of bounds"

    nrows = size(matrix, 1)
    num_blocks = cld(nrows - start_row + 1, block_size * 2)

    # Allocate temporary arrays for block results
    block_vals = CuArray{Float64}(undef, num_blocks)
    block_indices = CuArray{Int64}(undef, num_blocks)

    # Final result arrays
    result_val = CuArray{Float64}(undef, 1)
    result_idx = CuArray{Int64}(undef, 1)

    # Calculate shared memory size
    shared_mem_size = block_size * (sizeof(Float64) + sizeof(Int64))

    # Launch main reduction kernel
    @cuda blocks=num_blocks threads=block_size shmem=shared_mem_size max_reduce_kernel(
        matrix,
        block_vals,
        block_indices,
        col,
        start_row,
        nrows,
        Val(block_size),
    )

    # Final reduction if multiple blocks
    if num_blocks > 1
        @cuda blocks=1 threads=1 final_max_kernel(
            block_vals,
            block_indices,
            result_val,
            result_idx,
            num_blocks,
        )
    end

    # Return results
    max_val = Array(result_val)[1]
    max_idx = Array(result_idx)[1]

    return (max_val, max_idx)
end

# Benchmark and comparison function
function benchmark_max_methods(matrix_size = (10000, 10), col = 1, start_row = 10)
    println("=== Benchmark: Finding Column Maximum ===")
    println("Matrix size: $matrix_size")
    println("Column: $col, Start row: $start_row")
    println()

    # Create test matrix
    M = CuArray(rand(Float64, matrix_size...))

    # Warm up GPU
    for _ = 1:3
        maximum(@view M[start_row:end, col])
        find_column_max_optimized(M, col, start_row)
    end

    # Benchmark built-in method
    println("Built-in findmax:")
    @btime val1, idx1 = findmax(@view $M[($start_row):end, $col])
    println()

    # Benchmark optimized kernel
    println("Optimized kernel:")
    @btime val2, abs_idx2 = find_column_max_optimized($M, $col, $start_row)
    println()

    # # Verify results match
    # println("Results match: $(abs(val1 - val2) < 1e-10 && abs_idx1 == abs_idx2)")
    # println("Speedup: $(builtin_time/kernel_time)x")
end

# Run benchmark
benchmark_max_methods()
