using GPUFiniteFieldMatrices
using NVTX

function test_sub(p, i)
    # NVTX.@range "Init A p=$p, i=$i" begin
    A = rand(1:p, i, i)
    # end

    # NVTX.@range "Init d_A p=$p, i=$i" begin
    d_A = CuModMatrix(A, p)
    # end

    # NVTX.@range "Setup PLUQ p=$p, i=$i" begin
    U, L, P, Q = GPUFiniteFieldMatrices._setup_PLUQ(d_A; debug = false)
    # end

    # NVTX.@range "Lower triangular inverse p=$p, i=$i" begin
    L_inv = forward_sub_gpu_type_32(L, 0, 0)
    # end

    # NVTX.@range "Upper triangular inverse p=$p, i=$i" begin
    U_inv = backward_sub_gpu_type_32(U, 0, 0)
    # end

    println("A")
    display(A)
    println("L")
    display(L)
    println("U")
    display(U)

    println("L_inv")
    display(L_inv)
    println("U_inv")
    display(U_inv)
    println("L_inv * L")
    display(L_inv * L)
    println("U_inv * U")
    display(U_inv * U)
    return
end

i = 33
A = test_sub(11, 32)
# B = test_sub(11, i)
