using GPUFiniteFieldMatrices
using LinearAlgebra
using CUDA

function test_sub(p, m, n, debug::Bool = false, assert::Bool = false)
    # NVTX.@range "Init A p=$p, m=$m, n=$n" begin
    A = rand(1:p, m, n)
    A = Matrix{eltype(A)}(I, m, n)
    if debug
        println("A")
        display(A)
    end
    # end

    # NVTX.@range "Init d_A p=$p, m=$m, n=$n" begin
    d_A = CuModMatrix(A, p)
    # end

    # NVTX.@range "Setup PLUQ p=$p, m=$m, n=$n" begin
    U, L, P, Q = GPUFiniteFieldMatrices._setup_PLUQ(d_A; debug = false)
    if debug
        println("U")
        display(U)
        println("L")
        display(L)
        println("P")
        display(P)
        println("Q")
        display(Q)
        println("")
    end

    if assert
        L_copy = CuModMatrix(copy(L.data), p, new_size = size(L))
        U_copy = CuModMatrix(copy(U.data), p, new_size = size(U))
        GPUFiniteFieldMatrices.apply_row_inv_perm!(P, L_copy)
        GPUFiniteFieldMatrices.apply_row_inv_perm!(Q, U_copy)
        if debug
            println("L permuted")
            display(L_copy)
            println("P * L * U * Q")
            display(L_copy * U_copy)
            println("A")
            display(A)
            println(findall(Array(L_copy * U_copy) .!= Array(d_A)))
        end
        @assert Array(L_copy * U_copy) ≈ Array(d_A)
    end
    # end

    # NVTX.@range "Lower triangular inverse p=$p, m=$m, n=$n" begin
    L_inv = GPUFiniteFieldMatrices.lower_triangular_inverse_no_copy(L, debug = debug)
    if debug
        println("L_inv")
        display(L_inv)
        println("L_inv * L")
        display(L_inv * L)
    end
    # end

    # NVTX.@range "Upper triangular inverse p=$p, m=$m, n=$n" begin
    U_inv = GPUFiniteFieldMatrices.upper_triangular_inverse_no_copy(U, debug = debug)
    if debug
        println("U_inv")
        display(U_inv)
        println("U_inv * U")
        display(U_inv * U)
    end
    # end

    # NVTX.@range "Apply col inv perm p=$p, m=$m, n=$n" begin
    GPUFiniteFieldMatrices.apply_col_inv_perm!(P, L_inv)
    if debug
        println("L_inv after col inv perm")
        display(L_inv)
    end
    # end

    # NVTX.@range "Apply row inv perm p=$p, m=$m, n=$n" begin
    GPUFiniteFieldMatrices.apply_row_inv_perm!(Q, U_inv)
    if debug
        println("U_inv after row inv perm")
        display(U_inv)
    end
    # end

    # NVTX.@range "Multiply p=$p, m=$m, n=$n" begin
    # (PLUQ)^{-1} = Q^{-1} U^{-1} L^{-1} P^{-1}
    #
    A_inv = U_inv * L_inv
    if debug
        println("A_inv")
        display(A_inv)
        println("A_inv * A")
        display(A_inv * d_A)
        println("A * A_inv")
        display(d_A * A_inv)
        println("")
    end
    # end

    if assert
        if m >= n
            @assert Array(A_inv * d_A) ≈ Array(Matrix{eltype(A)}(I, n, n))
        else
            @assert Array(d_A * A_inv) ≈ Array(Matrix{eltype(A)}(I, m, m))
        end
        println("Assertion passed!")
    end
    return A_inv
end

m = 6
n = 6
A_inv = test_sub(11, m, n, false, true);
# B_inv = test_sub(11, n, m, false, false);
