"""
    apply_col_perm!(P::Array{Int, 1}, A::CuModMatrix)

Apply a column permutation to a CuModMatrix.
That is, compute A * P in place.

INPUTS:
* "P" -- array of tuples of integers, a permutation stack
* "A" -- CuModMatrix, the matrix to permute
"""
function apply_col_perm!(P::Array{Tuple{Int,Int},1}, A)
    n = size(A.data, 1)
    P_gpu = CuArray(P)

    @cuda threads=TILE_WIDTH blocks=div(n, TILE_WIDTH) _apply_col_perm_kernel!(
        A.data,
        P_gpu,
        size(A, 2),
    )

    return
end

function apply_col_inv_perm!(P::Array{Tuple{Int,Int},1}, A)
    n = size(A.data, 1)
    P_gpu = CuArray(reverse(P))

    @cuda threads=TILE_WIDTH blocks=div(n, TILE_WIDTH) _apply_col_perm_kernel!(
        A.data,
        P_gpu,
        size(A, 2),
    )

    return
end

function apply_col_perm(P::Array{Tuple{Int,Int},1}, A)
    n = size(A.data, 1)
    P_gpu = CuArray(P)
    A_data_copy = copy(A.data)

    @cuda threads=TILE_WIDTH blocks=div(n, TILE_WIDTH) _apply_col_perm_kernel!(
        A_data_copy,
        P_gpu,
        size(A, 2),
    )

    return CuModMatrix(A_data_copy, A.N; new_size = size(A))
end

"""
    _apply_col_perm_kernel!(P::Array{Int, 1}, A::CuModMatrix)

Internal kernel for apply_col_perm!, i.e. A <- A * P.
Each thread is responsible for one row of the matrix.

INPUTS:
* "P" -- array of integers, a permutation
* "A" -- CuModMatrix, the matrix to permute
"""
function _apply_col_perm_kernel!(
    A::CuDeviceMatrix{T},
    P::CuDeviceVector{Tuple{Int,Int},1},
    n::Int,
) where {T}
    tid = threadIdx().x
    bid = blockIdx().x
    row = (bid - 1) * TILE_WIDTH + tid

    for i = 1:length(P)
        col1, col2 = P[i]
        temp = A[row, col1]
        A[row, col1] = A[row, col2]
        A[row, col2] = temp
    end

    return
end

"""
    apply_row_perm!(P::Array{Int, 1}, A::CuModMatrix)

Apply a row permutation to a CuModMatrix.

INPUTS:
* "P" -- array of integers, a permutation
* "A" -- CuModMatrix, the matrix to permute
"""
function apply_row_perm!(P::Array{Tuple{Int,Int},1}, A::CuModMatrix)
    n = size(A.data, 2)
    P_gpu = CuArray(P)

    @cuda threads=TILE_WIDTH blocks=div(n, TILE_WIDTH) _apply_row_perm_kernel!(
        A.data,
        P_gpu,
        size(A, 1),
    )

    return
end

function apply_row_inv_perm!(P::Array{Tuple{Int,Int},1}, A)
    n = size(A.data, 2)
    P_gpu = CuArray(reverse(P))

    @cuda threads=TILE_WIDTH blocks=div(n, TILE_WIDTH) _apply_row_perm_kernel!(
        A.data,
        P_gpu,
        size(A, 1),
    )

    return
end

function apply_row_perm(P::Array{Tuple{Int,Int},1}, A)

    n = size(A.data, 2)
    P_gpu = CuArray(P)
    A_data_copy = copy(A.data)

    @cuda threads=TILE_WIDTH blocks=div(n, TILE_WIDTH) _apply_row_perm_kernel!(
        A_data_copy,
        P_gpu,
        size(A, 1),
    )

    return CuModMatrix(A_data_copy, A.N; new_size = size(A))
end

"""
    _apply_row_perm_kernel!(P::Array{Int, 1}, A::CuModMatrix)

Internal kernel for apply_row_perm.
Each thread is responsible for one column of the matrix.

INPUTS:
* "P" -- array of integers, a permutation
* "A" -- CuModMatrix, the matrix to permute
"""
function _apply_row_perm_kernel!(
    A::CuDeviceMatrix{T},
    P::CuDeviceVector{Tuple{Int,Int},1},
    n::Int,
) where {T}
    tid = threadIdx().x
    bid = blockIdx().x
    col = (bid - 1) * TILE_WIDTH + tid

    for i = 1:length(P)
        row1, row2 = P[i]
        temp = A[row1, col]
        A[row1, col] = A[row2, col]
        A[row2, col] = temp
    end

    return
end

"""
    perm_array_to_matrix(perm::Vector, N::Integer, new_size::Tuple{Int,Int}; perm_stack::Bool=false)

Convert a permutation array or stack of tuples to a permutation matrix.
Returns a CuModMatrix where the column with 1 in the ith column
is at the position of i in the input array or stack of tuples.

# Arguments
- `perm`: A permutation array or stack of tuples
- `N`: The modulus for the CuModMatrix (default: 11)

# Returns
- A CuModMatrix representation of the permutation
"""
function perm_array_to_matrix(
    perm::Vector,
    N::Integer,
    new_size::Tuple{Int,Int};
    perm_stack::Bool = false,
)
    rows, cols = length(perm), length(perm)

    if perm_stack
        P = Matrix{Int}(I, rows, cols)
        for (i, j) in perm
            P[i, :], P[j, :] = P[j, :], P[i, :]
        end
    else
        P = Base.zeros(Int, rows, cols)
        for i = 1:rows
            P[perm[i], i] = 1
        end
    end

    return CuModMatrix(P, N; new_size = new_size)
end
