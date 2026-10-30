using CUDA, LinearAlgebra

"""
Swaps row i with row j, and then mods the new row i (old row j) by p.
Does it by broadcasting... though this doesn't actually work
"""
function swap_rows_broadcast(A, startcol, i, j, p)
    s = startcol + 1

    tmp = copy(A[j, s:end])
    #tmp .%= p
    A[j, s:end] .= A[i, s:end]

    @. A[i, s:end] = tmp % p

    nothing
end

const NUM_THREADS_PER_BLOCK = 32

"""
    swap_rows(A,startcol,i,j,p)
Swaps the rows i and j in A, then mods row i by p
Note: only swaps entries that are to the left of
startcol. I.e. startcol and everthing to the
right will remain unchanged.
Note: 
"""
function swap_rows(A, startcol, i, j, p)
    # clal the kernel

    t = NUM_THREADS_PER_BLOCK
    nCols = size(A, 2)
    b = div(nCols, t) # will miss the last few entries, be sure to pad

    @cuda threads=t blocks=b swap_rows_kernel(A, startcol, i, j, p)

    nothing
end

function swap_rows_kernel(A, startcol, i, j, p)
    offset = threadIdx().x + (blockIdx().x - 1) * blockDim().x

    c = startcol + offset

    tmp = A[j, c]
    A[j, c] = A[i, c]
    tmp = tmp % p
    A[i, c] = tmp

    nothing
end

function test_swap_rows()

    A = CUDA.rand(1000, 1000)
    @. A = floor(A * 100)

    benchres1 = @benchmark CUDA.@sync swap_rows(A, 0, 2, 3, 11)
    benchres2 = @benchmark CUDA.@sync swap_rows_broadcast(A, 0, 3, 4, 11)

    (benchres1, benchres2)
end
