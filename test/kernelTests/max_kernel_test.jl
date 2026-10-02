using CUDA
using CUDA.CUBLAS
using BenchmarkTools

function find_max_cublas(M::CuMatrix{T}, col::Int, start_row::Int) where {T}

    @assert 1 <= col <= size(M, 2) "Column index out of bounds"
    @assert 1 <= start_row <= size(M, 1) "Start row out of bounds"

    # Extract the column subrange from start_row to end
    col_subrange = @view M[start_row:end, col]

    # Create result index storage on GPU
    relative_max_idx = 0
    relative_max_idx = CuRef{Int64}(0)

    # Call CUBLAS iamax to find index of maximum absolute value
    CUBLAS.iamax(length(col_subrange), col_subrange, relative_max_idx)

    # Convert to absolute row index
    absolute_row_idx = relative_max_idx[] + start_row - 1

    # Get the actual maximum value
    # max_value = @view M[absolute_row_idx, col]

    return (absolute_row_idx)
end

for i = 10000:10000:10000
    A = rand(1:11, i, i)
    d_A = CuArray{Float64}(A)

    pivot_row_idx = 10
    pivot_col_idx = 10

    val = -1
    idx = -1

    println("Size of A: ", size(A))

    println("Normal argmax:")
    @btime argmax($d_A[($pivot_row_idx):end, $pivot_col_idx])
    idx = argmax(d_A[pivot_row_idx:end, pivot_col_idx])
    println("idx: ", idx)

    println("Normal argmax with view:")
    @btime argmax(@view($d_A[($pivot_row_idx):end, $pivot_col_idx]))
    idx = argmax(@view(d_A[pivot_row_idx:end, pivot_col_idx]))
    println("idx: ", idx)

    println("isamax wrapper:")
    @btime find_max_cublas($d_A, $pivot_col_idx, $pivot_row_idx)
    idx = find_max_cublas(d_A, pivot_col_idx, pivot_row_idx)
    println("idx: ", idx)

    println("Findmax:")
    @btime findmax($d_A[($pivot_row_idx):end, $pivot_col_idx])
    val, idx = findmax(d_A[pivot_row_idx:end, pivot_col_idx])
    println("val: ", val, " idx: ", idx)

    println("Done with size(A): ", size(A))

    # println("isamax wrapper of all elements:")
    # result = 0
    # result = CuRef{Int64}(0)
    # @btime CUBLAS.iamax(length($d_A), $d_A, $result)

    return
end
