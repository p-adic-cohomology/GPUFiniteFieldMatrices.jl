function swap_rows_basic(matrix, row1, row2, ncols)

    col = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    temp = matrix[row1, col]
    matrix[row1, col] = matrix[row2, col]
    matrix[row2, col] = temp

    return nothing
end

function swap_rows_runthrough(matrix, row1, row2, ncols)

    col = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    while col <= ncols
        temp = matrix[row1, col]
        matrix[row1, col] = matrix[row2, col]
        matrix[row2, col] = temp

        col += blockDim().x * gridDim().x
    end

    return nothing
end

function benchmark_swap_rows(matrix_size = (10, 1024*16), row1 = 1, row2 = 2)

    matrix = CuArray(rand(Float64, matrix_size...))
    ncols = matrix_size[2]

    # prime the GPU
    @cuda blocks=cld(ncols, 32) threads=32 swap_rows_basic(matrix, row1, row2, ncols)
    @cuda blocks=cld(ncols, 32) threads=32 swap_rows_runthrough(matrix, row1, row2, ncols)
    # benchmark the kernels
    @btime @cuda blocks=cld($ncols, 32) threads=32 swap_rows_basic(
        $matrix,
        $row1,
        $row2,
        $ncols,
    )
    @btime @cuda blocks=cld($ncols, 32) threads=32 swap_rows_runthrough(
        $matrix,
        $row1,
        $row2,
        $ncols,
    )
    @btime $matrix[$row1, :] = $matrix[$row2, :]

end

benchmark_swap_rows()
