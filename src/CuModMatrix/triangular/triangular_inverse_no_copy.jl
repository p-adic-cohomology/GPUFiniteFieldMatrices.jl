MAX_INVERSE_SIZE = TILE_WIDTH * 1

"""
    _recursive_upper_triangular_inverse_no_copy(
        A::CuArray{T},
        A_inv::CuArray{T},
        i::Int,
        j::Int,
        N::Int
    )

Computes the inverse of an upper triangular matrix using a recursive algorithm.
Namely, it find the inverse in place for the submatrix A[i:j, i:j].

NOTE: This only accepts square or wide matrices!
(A tall matrix can simply be cut off at the bottom.)
"""
function _recursive_upper_triangular_inverse_no_copy(
    A::CuArray,
    A_inv::CuArray,
    row_lower::Int,
    row_upper::Int,
    col_lower::Int,
    col_upper::Int,
    N::Int;
    debug::Bool = false,
)
    if row_upper - row_lower < MAX_INVERSE_SIZE

        if debug
            println("First branch: Input is already small enough!")
            println("row_lower: $row_lower, row_upper: $row_upper")
            println("col_lower: $col_lower, col_upper: $col_upper")
            println("")
        end

        A_all_inv = CUDA.zeros(eltype(A), TILE_WIDTH, col_upper - col_lower + 1)

        if debug
            println("A_all:")
            display(@view A[row_lower:row_upper, col_lower:col_upper])
        end

        n_active = min(TILE_WIDTH, row_upper - row_lower + 1, col_upper - col_lower + 1)
        @cuda threads=TILE_WIDTH blocks=1 backward_sub_kernel_32(
            A,
            A_all_inv,
            N,
            row_lower-1,
            col_lower-1,
            n_active,
        )

        if debug
            println("A_all_inv:")
            display(A_all_inv)

            println("A_all_inv * A_all:")
            A_all = @view A[row_lower:row_upper, col_lower:col_upper]
            tmp = A_all_inv * A_all
            mod!(tmp, tmp, N)
            display(tmp)
            println("")
        end

        A_inv[row_lower:row_upper, col_lower:col_upper] = A_all_inv

        if debug
            println("A_inv:")
            display(A_inv)
        end

        return

    end

    # See lower for an explanation of the split.

    m = floor((row_upper - row_lower + 1) / (2*TILE_WIDTH) + 1/2)
    m = min(max(m, 1), m)
    row_mid = Int(row_lower - 1 + m * TILE_WIDTH)
    col_mid = row_mid

    if debug
        println("row_mid: $row_mid, col_mid: $col_mid")
        println("row_lower: $row_lower, row_upper: $row_upper")
        println("col_lower: $col_lower, col_upper: $col_upper")

        @assert (row_mid - row_lower + 1) % TILE_WIDTH == 0
        @assert (col_mid - col_lower + 1) % TILE_WIDTH == 0
    end

    if row_mid - row_lower < MAX_INVERSE_SIZE && row_upper - row_mid <= MAX_INVERSE_SIZE

        if debug
            println("Second branch: Each triangle is small enough!")
        end

        A_11_inv = CUDA.zeros(eltype(A), col_mid - col_lower + 1, row_mid - row_lower + 1)

        if debug
            println("A_11:")
            display(@view A[row_lower:row_mid, col_lower:col_mid])
            @assert (row_mid - row_lower + 1) == size(A_11_inv, 1)
            @assert (col_mid - col_lower + 1) == size(A_11_inv, 2)
        end

        n11 = min(TILE_WIDTH, row_mid - row_lower + 1, col_mid - col_lower + 1)
        @cuda threads=TILE_WIDTH blocks=1 backward_sub_kernel_32(
            A,
            A_11_inv,
            N,
            row_lower-1,
            col_lower-1,
            n11,
        )

        if debug
            println("A_11_inv:")
            display(A_11_inv)

            A_11 = @view A[row_lower:row_mid, col_lower:col_mid]
            println("A_11_inv * A_11:")
            tmp = A_11_inv * A_11
            mod!(tmp, tmp, N)
            display(tmp)
            println("")
        end

        A_inv[col_lower:col_mid, row_lower:row_mid] = A_11_inv

        if debug
            println("A_inv:")
            display(A_inv)
        end

        A_22_inv = CUDA.zeros(
            eltype(A),
            max(col_upper - col_mid, TILE_WIDTH),
            max(row_upper - row_mid, TILE_WIDTH),
        )

        if debug
            println("A_22:")
            display(
                @view A[(row_mid+1):(row_mid+TILE_WIDTH), (col_mid+1):(col_mid+TILE_WIDTH)]
            )
        end

        n22 = min(TILE_WIDTH, row_upper - row_mid, col_upper - col_mid)
        @cuda threads=TILE_WIDTH blocks=1 backward_sub_kernel_32(
            A,
            A_22_inv,
            N,
            row_mid,
            col_mid,
            n22,
        )

        if debug
            println("A_22_inv:")
            display(A_22_inv)
            println("A_22_inv * A_22:")
            A_22 =
                @view A[(row_mid+1):(row_mid+TILE_WIDTH), (col_mid+1):(col_mid+TILE_WIDTH)]
            tmp = A_22_inv * A_22
            mod!(tmp, tmp, N)
            display(tmp)
            println("")
        end

        A_inv[(col_mid+1):col_upper, (row_mid+1):row_upper] =
            A_22_inv[1:(col_upper-col_mid), 1:(row_upper-row_mid)]

        if debug
            println("A_inv:")
            display(A_inv)
        end

        A_12 = @view A[row_lower:row_mid, (col_mid+1):col_upper]

        if debug
            println("A_12:")
            display(A_12)
        end

        tmp = _exact_mod_matmul_data(A_11_inv, A_12, N)
        tmp2 = _exact_mod_matmul_data(
            tmp,
            A_22_inv[1:(col_upper-col_mid), 1:(row_upper-row_mid)],
            N,
        )
        rscalar_sub!(tmp2, tmp2, N, N)
        A_inv[col_lower:col_mid, (row_mid+1):row_upper] = tmp2

        if debug
            println("-A_11_inv * A_12 * A_22_inv:")
            display(tmp2)

            println("A_inv:")
            display(A_inv)
        end

        return

    else

        if debug
            println("Third branch: Recursive call!")
        end

        _recursive_upper_triangular_inverse_no_copy(
            A,
            A_inv,
            row_lower,
            row_mid,
            col_lower,
            col_mid,
            N;
            debug = debug,
        )
        _recursive_upper_triangular_inverse_no_copy(
            A,
            A_inv,
            row_mid+1,
            row_upper,
            col_mid+1,
            col_upper,
            N;
            debug = debug,
        )
        A_12 = @view A[row_lower:row_mid, (col_mid+1):col_upper]
        A_11_inv = @view A_inv[col_lower:col_mid, row_lower:row_mid]
        A_22_inv = @view A_inv[(col_mid+1):col_upper, (row_mid+1):row_upper]
        tmp = _exact_mod_matmul_data(A_11_inv, A_12, N)
        tmp2 = _exact_mod_matmul_data(
            tmp,
            A_22_inv[1:(col_upper-col_mid), 1:(row_upper-row_mid)],
            N,
        )
        rscalar_sub!(tmp2, tmp2, N, N)
        A_inv[col_lower:col_mid, (row_mid+1):row_upper] = tmp2

        return
    end
end

"""
    upper_triangular_inverse_no_copy(A::CuModMatrix)

Computes the inverse of an upper triangular matrix.
"""
function upper_triangular_inverse_no_copy(A::CuModMatrix; debug::Bool = false)
    rows, cols = size(A)

    if rows <= MAX_INVERSE_SIZE
        return backward_sub_gpu_type_32(A, 0, 0)
    end

    A_inv = zeros(eltype(A.data), size(A)[2], size(A)[1], A.N)

    if debug
        println("size(A.data): ", size(A.data))
    end

    if rows <= cols #square or wide
        _recursive_upper_triangular_inverse_no_copy(
            A.data,
            A_inv.data,
            1,
            rows,
            1,
            cols,
            A.N;
            debug = debug,
        )
    else #tall
        _recursive_upper_triangular_inverse_no_copy(
            A.data,
            A_inv.data,
            1,
            cols,
            1,
            cols,
            A.N;
            debug = debug,
        )
    end

    if debug
        println("A.data * A_inv.data:")
        tmp = A.data * A_inv.data
        mod!(tmp, tmp, A.N)
        display(tmp)
        println("A_inv.data:")
        display(A_inv.data)
    end

    return A_inv
end

"""
    _recursive_lower_triangular_inverse_no_copy(
        A::CuArray{T},
        A_inv::CuArray{T},
        i::Int,
        j::Int,
        N::Int
    )

Computes the inverse of an lower triangular matrix using a recursive algorithm.
Namely, it find the inverse in place for the submatrix A[i:j, i:j].

NOTE: This only accepts square or tall matrices!
(A wide matrix can simply be cut off at the side.)
"""

function _recursive_lower_triangular_inverse_no_copy(
    A::CuArray,
    A_inv::CuArray,
    row_lower::Int,
    row_upper::Int,
    col_lower::Int,
    col_upper::Int,
    N::Int;
    debug::Bool = false,
)

    if col_upper - col_lower < MAX_INVERSE_SIZE

        if debug
            println("First branch: Input is already small enough!")
            println("row_lower: $row_lower, row_upper: $row_upper")
            println("col_lower: $col_lower, col_upper: $col_upper")
            println("")
        end

        A_all_inv = CUDA.zeros(eltype(A), TILE_WIDTH, TILE_WIDTH)

        if debug
            println("A_all:")
            display(@view A[col_lower:col_upper, row_lower:row_upper])
        end

        n_active = min(TILE_WIDTH, row_upper - row_lower + 1, col_upper - col_lower + 1)
        @cuda threads=TILE_WIDTH blocks=1 forward_sub_kernel_32(
            A,
            A_all_inv,
            N,
            row_lower-1,
            col_lower-1,
            n_active,
        )

        if debug
            println("A_all_inv:")
            display(A_all_inv)

            println("A_all_inv * A_all:")
            A_all = @view A[col_lower:col_upper, row_lower:row_upper]
            tmp = A_all_inv * A_all
            mod!(tmp, tmp, N)
            display(tmp)
            println("")
        end

        A_inv[col_lower:col_upper, row_lower:row_upper] = A_all_inv

        if debug
            println("A_inv:")
            display(A_inv)
        end

        return

    end

    # We want a split such that the top left block A_11is square, such that A_12 is all zeros.
    # We also want the size of A_11 to be a multiple of TILE_WIDTH
    # This way the only "inefficient kernel call" where we find the inverse of a non-full-TILE_WIDTH matrix
    # is the last one; all other calls will be full TILE_WIDTH.

    # To do this, we want a mid as close as possible to (lower + upper) / 2
    # Yet lower - mid + 1 should be a multiple of TILE_WIDTH.

    # In other words, we want to find a coefficient k such that
    # mid = lower - 1 + k * TILE_WIDTH
    # and mid is as close to (and ideally always larger or equal to) (lower + upper) / 2.

    # To do this, notice m = floor( (U - L + 1)/TILE_WIDTH + 1/2)
    # is nearly there. It computes the middle,
    # then finds how many multiple of TILE_WIDTH are needed to get to the middle.

    # We add 1/2 to shift the mid upwards, so that mid is always larger or equal to (lower + upper) / 2.

    # Then we floor to get a discrete multiple of TILE_WIDTH.

    # However, this can break down if this discrete multiple rounds down to 0.
    # To resolve this, we clamp the result using min(max(m, 1),m)

    # Thus middle = (L - 1) + m * TILE_WIDTH

    m = floor((col_upper - col_lower + 1) / (2*TILE_WIDTH) + 1/2)
    m = min(max(m, 1), m)
    col_mid = Int(col_lower - 1 + m * TILE_WIDTH)
    row_mid = col_mid

    if debug
        println("row_mid: $row_mid, col_mid: $col_mid")
        println("row_lower: $row_lower, row_upper: $row_upper")
        println("col_lower: $col_lower, col_upper: $col_upper")

        @assert (row_mid - row_lower + 1) % TILE_WIDTH == 0
        @assert (col_mid - col_lower + 1) % TILE_WIDTH == 0
    end

    if col_mid - col_lower < MAX_INVERSE_SIZE && col_upper - col_mid <= MAX_INVERSE_SIZE

        if debug
            println("Second branch: Each triangle is small enough!")
        end

        A_11_inv = CUDA.zeros(eltype(A), col_mid - col_lower + 1, row_mid - row_lower + 1)

        if debug
            println("A_11:")
            display(@view A[row_lower:row_mid, col_lower:col_mid])
            @assert (row_mid - row_lower + 1) == size(A_11_inv, 1)
            @assert (col_mid - col_lower + 1) == size(A_11_inv, 2)
        end

        n11 = min(TILE_WIDTH, row_mid - row_lower + 1, col_mid - col_lower + 1)
        @cuda threads=TILE_WIDTH blocks=1 forward_sub_kernel_32(
            A,
            A_11_inv,
            N,
            row_lower-1,
            col_lower-1,
            n11,
        )

        if debug
            println("A_11_inv:")
            display(A_11_inv)

            A_11 = @view A[row_lower:row_mid, col_lower:col_mid]
            println("A_11_inv * A_11:")
            tmp = A_11_inv * A_11
            mod!(tmp, tmp, N)
            display(tmp)
            println("")
        end

        A_inv[col_lower:col_mid, row_lower:row_mid] = A_11_inv

        if debug
            println("A_inv:")
            display(A_inv)
        end

        A_22_inv = CUDA.zeros(
            eltype(A),
            max(col_upper - col_mid, TILE_WIDTH),
            max(row_upper - row_mid, TILE_WIDTH),
        )

        if debug
            println("A_22:")
            display(
                @view A[(row_mid+1):(row_mid+TILE_WIDTH), (col_mid+1):(col_mid+TILE_WIDTH)]
            )
        end

        n22 = min(TILE_WIDTH, row_upper - row_mid, col_upper - col_mid)
        @cuda threads=TILE_WIDTH blocks=1 forward_sub_kernel_32(
            A,
            A_22_inv,
            N,
            row_mid,
            col_mid,
            n22,
        )

        if debug
            println("A_22_inv:")
            display(A_22_inv)
            println("A_22_inv * A_22:")
            A_22 =
                @view A[(row_mid+1):(row_mid+TILE_WIDTH), (col_mid+1):(col_mid+TILE_WIDTH)]
            tmp = A_22_inv * A_22
            mod!(tmp, tmp, N)
            display(tmp)
            println("")
        end

        A_inv[(col_mid+1):col_upper, (row_mid+1):row_upper] =
            A_22_inv[1:(row_upper-row_mid), 1:(col_upper-col_mid)]
        # A_inv[col_mid+1:col_upper, row_mid+1:row_upper] = A_22_inv[1:(col_upper - col_mid), 1:(row_upper - row_mid)]

        if debug
            println("A_inv:")
            display(A_inv)
        end

        A_21 = @view A[(row_mid+1):row_upper, col_lower:col_mid]

        if debug
            println("A_21:")
            display(A_21)
        end

        tmp = _exact_mod_matmul_data(
            A_22_inv[1:(row_upper-row_mid), 1:(col_upper-col_mid)],
            A_21,
            N,
        )
        tmp2 = _exact_mod_matmul_data(tmp, A_11_inv, N)
        rscalar_sub!(tmp2, tmp2, N, N)
        A_inv[(col_mid+1):col_upper, row_lower:row_mid] = tmp2

        if debug
            println("A_22_inv * A_21 * A_11_inv:")
            display(tmp2)

            println("A_inv:")
            display(A_inv)
        end

        return

    else

        if debug
            println("Third branch: Recursive call!")
        end

        _recursive_lower_triangular_inverse_no_copy(
            A,
            A_inv,
            row_lower,
            row_mid,
            col_lower,
            col_mid,
            N;
            debug = debug,
        )
        _recursive_lower_triangular_inverse_no_copy(
            A,
            A_inv,
            row_mid+1,
            row_upper,
            col_mid+1,
            col_upper,
            N;
            debug = debug,
        )
        A_21 = @view A[(row_mid+1):row_upper, col_lower:col_mid]
        A_11_inv = @view A_inv[col_lower:col_mid, row_lower:row_mid]
        A_22_inv = @view A_inv[(col_mid+1):col_upper, (row_mid+1):row_upper]
        tmp = _exact_mod_matmul_data(A_22_inv, A_21, N)
        tmp2 = _exact_mod_matmul_data(tmp, A_11_inv, N)
        rscalar_sub!(tmp2, tmp2, N, N)
        A_inv[(col_mid+1):col_upper, row_lower:row_mid] = tmp2

        return
    end
end

"""
    lower_triangular_inverse_no_copy(A::CuModMatrix)

Computes the inverse of an lower triangular matrix.
"""
function lower_triangular_inverse_no_copy(A::CuModMatrix; debug::Bool = false)
    rows, cols = size(A)

    if cols <= MAX_INVERSE_SIZE
        return forward_sub_gpu_type_32(A, 0, 0)
    end

    A_inv = zeros(eltype(A.data), size(A)[2], size(A)[1], A.N)

    if rows <= cols #square or wide
        _recursive_lower_triangular_inverse_no_copy(
            A.data,
            A_inv.data,
            1,
            rows,
            1,
            cols,
            A.N;
            debug = debug,
        )
    else #tall
        throw(
            InverseNotDefinedException(
                "(Right) pseudoinverse not defined for tall lower triangular matrices",
            ),
        )
        # _recursive_lower_triangular_inverse_no_copy(
        #     A.data, A_inv.data, 1, rows, 1, rows, A.N; debug=debug)
    end

    if debug
        println("A.data * A_inv.data:")
        tmp = A.data * A_inv.data
        mod!(tmp, tmp, A.N)
        display(tmp)
        println("A_inv.data:")
        display(A_inv.data)
    end

    return A_inv
end
