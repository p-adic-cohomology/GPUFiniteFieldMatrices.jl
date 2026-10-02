function set_elem_kernel(matrix, row, col, value)
    matrix[row, col] = value
    return nothing
end

function benchmark_set_elem(matrix_size = (1000, 1000), row = 1, col = 2, value = 10)
    matrix = CuArray(rand(Float64, matrix_size...))
    @cuda blocks=1 threads=1 set_elem_kernel(matrix, row, col, value)
    @btime @cuda blocks=1 threads=1 set_elem_kernel($matrix, $row, $col, $value)
    @btime $matrix[($row):end, $col] .= $value
    @btime CUDA.@allowscalar $matrix[$row, $col] = $value
end

benchmark_set_elem()
