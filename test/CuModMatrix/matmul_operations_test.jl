#using GPUFiniteFieldMatrices
#using Test
#using CUDA
#using LinearAlgebra

"""
Test matrix multiplication operations on CuModMatrix.
This tests both the standard and direct implementations.
"""
function test_matmul_operations()
    println("Testing matrix multiplication operations on CuModMatrix...")

    # Test matrices
    A_data = [1 2 3; 4 5 6]
    B_data = [7 8; 9 10; 0 1]
    C_data = [58 64; 139 154]  # Expected result A*B mod 11 = [3 9; 7 0]
    modulus = 11  # Prime modulus

    A = CuModMatrix(A_data, modulus)
    B = CuModMatrix(B_data, modulus)

    println("Matrix A = ")
    display(A)
    println()

    println("Matrix B = ")
    display(B)
    println()

    # Test using the standard matrix multiplication
    println("Testing standard matrix multiplication...")
    C = A * B

    println("A * B = ")
    display(C)
    println()

    # Verify the result matches expected calculation (mod 11)
    # TODO: Note we cast just to check the construtor works (though I think this should be removed)
    expected_C = CuModMatrix(C_data .% modulus, modulus)
    @test Array(C) ≈ Array(expected_C)

    # Test using the direct multiplication implementation
    println("Testing mat_mul_gpu_type direct implementation...")
    C_direct = mat_mul_gpu_type(A, B)

    println("mat_mul_gpu_type(A, B) = ")
    display(C_direct)
    println()

    @test Array(C) ≈ Array(C_direct)

    # Test multiplication with modulus override
    override_modulus = 7
    C_mod = mat_mul_gpu_type(A, B, override_modulus)

    println("mat_mul_gpu_type(A, B) with modulus $override_modulus = ")
    display(C_mod)
    println()

    # Verify the result matches expected calculation (mod override_modulus)
    # See previous TODO
    expected_C_mod = (A_data * B_data) .% override_modulus
    println(A_data)
    println(B_data)
    println(expected_C_mod)
    @test Array(C_mod) ≈ expected_C_mod

    # Force the exact modular GEMM to split its inner dimension.  A single
    # Float32 dot product of this length can exceed the exactly represented
    # integer range even though each field representative is exact.
    k = GPUFiniteFieldMatrices.find_max_ops(Float32, 101) + 17
    A_chunk_host = reshape(Float32.(mod.(17 .* (1:(2k)), 101)), 2, k)
    B_chunk_host = reshape(Float32.(mod.(29 .* (1:(2k)), 101)), k, 2)
    A_chunk = CuModMatrix(A_chunk_host, 101; elem_type = Float32)
    B_chunk = CuModMatrix(B_chunk_host, 101; elem_type = Float32)
    C_chunk = mat_mul_gpu_type(A_chunk, B_chunk)
    expected_chunk = mod.(Int64.(A_chunk_host) * Int64.(B_chunk_host), 101)
    @test round.(Int64, Array(C_chunk)) == expected_chunk

    println("All matrix multiplication operations tests passed!")
end

"""
     Testing MMPSingularities tests passed
Test in-place matrix multiplication operations on CuModMatrix.
"""
function test_inplace_matmul_operations()
    println("Testing in-place matrix multiplication operations on CuModMatrix...")

    A_data = [1 2 3; 4 5 6]
    B_data = [7 8; 9 10; 11 12]
    C = GPUFiniteFieldMatrices.zeros(Float32, 2, 2, 9)
    C_data = C.data
    modulus = 9  # Prime modulus

    A = CuModMatrix(A_data, modulus)
    B = CuModMatrix(B_data, modulus)

    println("Matrix A = ")
    display(A)
    println()

    println("Matrix B = ")
    display(B)
    println()

    println("Initial Matrix C = ")
    display(C)
    println()

    # Test using the in-place multiplication implementation
    println("Testing mat_mul_gpu_type...")
    mat_mul_type_inplace!(C, A, B)

    println("After mat_mul_gpu_type(A, B, C):")
    display(C)
    println()

    # Verify the result matches expected calculation (mod 11)
    # See previous TODO
    expected_C = (A_data * B_data) .% modulus
    @test Array(C) ≈ expected_C

    # Test in-place multiplication with modulus override
    override_modulus = 3
    C2 = GPUFiniteFieldMatrices.zeros(Float32, 2, 2, override_modulus)

    mat_mul_type_inplace!(C2, A, B, override_modulus)

    println("In-place multiplication with modulus $override_modulus:")
    display(C2)
    println()

    # Verify the result matches expected calculation (mod override_modulus)
    # see previous TODO
    expected_C2 = mod.(A_data * B_data, override_modulus)
    @test Array(C2) ≈ expected_C2

    println("All in-place matrix multiplication operations tests passed!")
end

function test_matmul()
    test_matmul_operations()
    test_inplace_matmul_operations()

    println("\nAll matrix multiplication tests passed!")
end

# Run the tests if this file is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    test_matmul()
end
