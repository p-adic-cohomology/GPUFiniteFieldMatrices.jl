using Nemo
using GPUFiniteFieldMatrices
using BenchmarkTools

function float_entries!(dest::Matrix{Float64}, src)
    res = dest
    A = src

    for i = 1:size(A, 1)
        for j = 1:size(A, 2)
            lifted = Int64(lift(ZZ, A[i, j]))
            res[i, j] = convert(Float64, lifted)
        end
    end

    res
end

function compute_inverse_from_nemo_lu(rank::Int, P::Perm, L, U)
    # First check if matrix is invertible
    if rank != nrows(U)
        return false, nothing
    end

    # Get the modulus from the base ring
    N = Int(characteristic(base_ring(L)))

    # Convert matrices to regular matrices first

    # Convert to float matrices
    L_float = zeros(Float64, size(L)...)
    U_float = zeros(Float64, size(U)...)
    float_entries!(L_float, L)
    float_entries!(U_float, U)

    # Create CuModMatrix versions
    L_gpu = CuModMatrix(L_float, N; new_size = size(L_float))
    U_gpu = CuModMatrix(U_float, N; new_size = size(U_float))

    # Compute inverses using triangular inverse functions
    U_inv = upper_triangular_inverse(U_gpu)
    L_inv = lower_triangular_inverse(L_gpu)

    # Convert permutation to array format
    # P_arr = Array(P.d)
    # length_P = length(P_arr)
    # P_gpu = perm_array_to_matrix(P_arr, N, (length_P, length_P))

    # Apply permutations and multiply to get final inverse
    A_inv = U_inv * L_inv

    return true, A_inv
end

function test_combined_inv(rows, cols, p)
    R = GF(p)
    MS = matrix_space(R, rows, cols)
    A_nemo = MS([R(x) for x in rand(1:p, rows, cols)])
    rank, P, L, U = Nemo.lu(A_nemo)
    @btime compute_inverse_from_nemo_lu($rank, $P, $L, $U)
end

for i = 1000:1000:20000
    test_combined_inv(i, i, 11)
end
