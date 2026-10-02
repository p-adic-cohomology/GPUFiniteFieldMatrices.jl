const global TILE_WIDTH = 32

function mat_mul_gpu(A, B, N, REGIME = "⊡", type = Float64, tile_width = 25)
    """
    Hybrid matmul algorithm that incorporates three different regimes:

    if MAX_OPS >= A_cols and MAX_OPS >= B_cols,
        do normal matrix multiplication and broadcast mod
    else if MAX_OPS >= TILE_WIDTH
        call the custom kernel without counting operations
    else
        call the custom kernel with counting operations

    The default is to use the "⊡" regime, which is the most efficient.
    An exception will be thrown if the matrices are too large for this regime.

    The optional argument type determines the datatype used.
    By default, type is Float64 or Int53. 
    For reference, Float32 is Int24; Float16 is Int10.
    """

    A_rows, A_cols = size(A)
    B_rows, B_cols = size(B)

    if A_cols != B_rows
        throw(CuModArraySizeMismatchException("Matrix dimensions do not match.
            A has $A_rows rows and $A_cols cols, 
            B has $B_rows rows and $B_cols cols."))
    end

    A_padded_rows = ceil(Int, A_rows / TILE_WIDTH) * TILE_WIDTH
    A_padded_cols = ceil(Int, A_cols / TILE_WIDTH) * TILE_WIDTH
    B_padded_cols = ceil(Int, B_cols / TILE_WIDTH) * TILE_WIDTH

    # Define indices for moving to CUDA Arrays
    Ainds = CartesianIndices(A)
    d_Ainds = CartesianIndices((1:A_rows, 1:A_cols))
    Binds = CartesianIndices(B)
    d_Binds = CartesianIndices((1:B_rows, 1:B_cols))

    # Define CUDA arrays of appropriate size
    t = eltype(A)
    # Note that undef makes all values default to 0
    d_A = CUDA.CuArray{t}(undef, (A_padded_rows, A_padded_cols))
    d_B = CUDA.CuArray{t}(undef, (A_padded_cols, B_padded_cols))
    d_C = CUDA.CuArray{t}(undef, (A_padded_rows, B_padded_cols))

    # Move the matrices from CPU to GPU CUDA Arrays
    copyto!(d_A, d_Ainds, A, Ainds)
    copyto!(d_B, d_Binds, B, Binds)

    # Hardcode tile width unles inputted
    if tile_width < 1
        error("Invalid tile width")
    end

    # Compute the MAX_OPS
    MAX_OPS = find_max_ops(type, N)

    # Determine regiment to use if not hardcoded
    if REGIME == "⊠"
        if MAX_OPS >= A_cols # equal to B_rows
            REGIME = "⊡"
        elseif MAX_OPS > TILE_WIDTH
            REGIME = "⊟"
        else
            REGIME = "⊞"
        end
    end

    # Compute based on regime
    if REGIME == "⊡"
        return mat_mul_plain(d_A, d_B, N)[1:A_rows, 1:B_cols]

    elseif REGIME == "⊟"
        println("running the algorithm")
        @cuda threads=(TILE_WIDTH, TILE_WIDTH) blocks=(
            div(B_padded_cols, TILE_WIDTH),
            div(A_padded_rows, TILE_WIDTH),
        ) mat_mul_no_ops(d_A, d_B, d_C, N, A_padded_rows, type)
        return d_C[1:A_rows, 1:B_cols]

    elseif REGIME == "⊞"
        @cuda threads=(TILE_WIDTH, TILE_WIDTH) blocks=(
            div(B_padded_cols, TILE_WIDTH),
            div(A_padded_rows, TILE_WIDTH),
        ) mat_mul_ops(d_A, d_B, d_C, N, A_padded_rows, type, MAX_OPS)
        return d_C[1:A_rows, 1:B_cols]

    else
        error("Input regime is invalid.")
    end

    return
end

# """
#     find_max_ops(type, N)

# Returns the maximum number of operations before a modulus is necessary given a datatype type and modulus N.

# Supports all basic Julia Float, Int, and UInt types.
# """
# function find_max_ops(type, N)

#     if occursin("Float", string(type))
#         bits = match(r"\d+", string(type))
#         d = Dict("64"=>51, "32"=>22, "16"=>9)
#         bits = get(d, bits.match, -1)

#     elseif occursin("UInt", string(type))
#         bits = int(match(r"\d+", string(type)).match) - 1

#     elseif occursin("Int", string(type))
#         bits = int(match(r"\d+", string(type)).match)

#     else
#         error("The input type is neither Int, UInt, nor Float.")
#     end

#     if bits == -1
#         error("Input type is not recognized.")
#     end

#     return floor((2^bits-1)/N^2) - 1
# end
