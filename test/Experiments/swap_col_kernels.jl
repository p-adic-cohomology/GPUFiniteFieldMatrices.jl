function swap_cols_basic(matrix, col1, col2, nrows)

    row = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    temp = matrix[row, col1]
    matrix[row, col1] = matrix[row, col2]
    matrix[row, col2] = temp

    return nothing
end

function swap_cols_runthrough(matrix, col1, col2, nrows)

    row = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    while row <= nrows
        temp = matrix[row, col1]
        matrix[row, col1] = matrix[row, col2]
        matrix[row, col2] = temp

        row += blockDim().x * gridDim().x
    end

    return nothing
end

function benchmark_swap_cols(matrix_size = (1024*16, 10), col1 = 1, col2 = 2)

    matrix = CuArray(rand(Float64, matrix_size...))
    nrows = matrix_size[1]

    # prime the GPU
    @cuda blocks=cld(nrows, 256) threads=256 swap_cols_basic(matrix, col1, col2, nrows)
    @cuda blocks=cld(nrows, 256) threads=256 swap_cols_runthrough(matrix, col1, col2, nrows)
    # benchmark the kernels
    @btime @cuda blocks=cld($nrows, 256) threads=256 swap_cols_basic(
        $matrix,
        $col1,
        $col2,
        $nrows,
    )
    @btime @cuda blocks=cld($nrows, 256) threads=256 swap_cols_runthrough(
        $matrix,
        $col1,
        $col2,
        $nrows,
    )
    @btime $matrix[$col1, :] = $matrix[$col2, :]

end

benchmark_swap_cols()
