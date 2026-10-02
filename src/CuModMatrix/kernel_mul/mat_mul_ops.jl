"""
    GPU MatMul with counting number of operations,
    so MAX_OPS <= TILE_WIDTH.
    Note that TILE_WDITH must be a constant variable.
"""
function mat_mul_ops(d_A, d_B, d_C, P, width, TYPE, MAX_OPS)

    # Define shared matrices for storage
    sharedA = CUDA.CuStaticSharedArray(TYPE, (TILE_WIDTH, TILE_WIDTH))
    sharedB = CUDA.CuStaticSharedArray(TYPE, (TILE_WIDTH, TILE_WIDTH))

    # Define common values as variables in cache
    bx = blockIdx().x
    by = blockIdx().y
    tx = threadIdx().x
    ty = threadIdx().y

    # Identify row and col
    row = (by-1) * TILE_WIDTH + ty
    col = (bx-1) * TILE_WIDTH + tx

    # Loop setup
    total = 0
    bound = div(width, TILE_WIDTH)
    m = 0

    # Load and compute one entry
    while m < bound

        # Load the rows and cols into shared matrices
        sharedA[ty, tx] = d_A[row, m*TILE_WIDTH+tx]
        sharedB[ty, tx] = d_B[m*TILE_WIDTH+ty, col]

        CUDA.sync_threads()

        # Loop through the row and column and find total
        k = 1
        counter = 0 # Special counter for operations

        while k <= TILE_WIDTH
            # If number of operations has reached limit, then mod
            if counter >= MAX_OPS
                counter = 0
                total = total % P
            end

            total += sharedA[ty, k] * sharedB[k, tx]
            counter += 1
            k += 1
        end

        CUDA.sync_threads()

        m += 1
    end

    # Assign value
    d_C[row, col] = total % P

    return
end
