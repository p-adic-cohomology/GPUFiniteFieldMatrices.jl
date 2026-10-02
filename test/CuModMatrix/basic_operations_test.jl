#using GPUFiniteFieldMatrices
#using Test
#using CUDA
#using LinearAlgebra

"""
Test basic operations of the CuModMatrix type.
This includes:
- Initialization and properties
- Basic arithmetic operations
- Scalar operations
- Utility functions
"""
function test_basic_operations()
    println("Testing basic operations of CuModMatrix...")

    # Test initialization
    A_data = [1 2 3; 4 5 6; 7 8 9]
    modulus = 11  # Use a prime modulus
    A = CuModMatrix(A_data, modulus)

    cuA = CuArray(A_data)
    Aprime = CuModMatrix(cuA, modulus)
    A_data = [1 2 3; 4 5 6; 7 8 9]
    println("A = ")
    display(A)
    println()

    # Test basic properties
    @test size(A) == (3, 3)
    A_data = Array(A)
    @test A_data[1, 1] == 1
    @test A_data[3, 3] == 9

    # Test arithmetic operations
    B_data = [9 8 7; 6 5 4; 3 2 1]
    B = CuModMatrix(B_data, modulus)

    println("B = ")
    display(B)
    println()

    # Addition
    C = A + B
    println("A + B = ")
    display(C)
    println()

    # Expected result: (1+9, 2+8, 3+7; 4+6, 5+5, 6+4; 7+3, 8+2, 9+1) mod 11
    expected = mod.([10 10 10; 10 10 10; 10 10 10], modulus)
    @test Array(C) == expected

    # Subtraction
    D = A - B
    println("A - B = ")
    display(D)
    println()

    # Expected result: (1-9, 2-8, 3-7; 4-6, 5-5, 6-4; 7-3, 8-2, 9-1) mod 11
    expected = mod.([1-9 2-8 3-7; 4-6 5-5 6-4; 7-3 8-2 9-1], modulus)
    @test Array(D) == expected

    # Matrix multiplication
    E = A * B
    println("A * B = ")
    display(E)
    println()

    # Expected result: standard matrix multiplication then mod 11
    expected = mod.(A_data * B_data, modulus)
    @test Array(E) == expected

    # Element-wise multiplication
    F = GPUFiniteFieldMatrices.zeros(eltype(A.data), size(A, 1), size(A, 2), modulus)
    elementwise_multiply!(F, A, B)
    println("A .* B = ")
    display(F)
    println()

    # Expected result: element-wise multiplication then mod 11
    expected = mod.(A_data .* B_data, modulus)
    @test Array(F) == expected

    # Test scalar operations
    scalar = 3

    # Scalar addition
    S1 = scalar + A
    println("$scalar + A = ")
    display(S1)
    println()
    expected = mod.(scalar .+ A_data, modulus)
    @test Array(S1) == expected

    # Scalar subtraction
    S2 = A - scalar
    println("A - $scalar = ")
    display(S2)
    println()
    expected = mod.(A_data .- scalar, modulus)
    @test Array(S2) == expected

    # Scalar multiplication
    S3 = scalar * A
    println("$scalar * A = ")
    display(S3)
    println()
    expected = mod.(scalar .* A_data, modulus)
    @test Array(S3) == expected

    # Test unary negation
    S4 = A * -1
    println("-A = ")
    display(S4)
    println()
    expected = mod.(-A_data, modulus)
    @test Array(S4) == expected

    println("All basic operations tests passed!")
end

"""
Test utility functions of the CuModMatrix type.
"""
function test_utility_functions()
    println("Testing utility functions of CuModMatrix...")
    modulus = 11

    # Test identity
    I3 = eye(Int, 3, modulus)
    println("eye(Int, 3, $modulus) = ")
    display(I3)
    println()
    @test Array(I3) == Matrix{Int}(I, 3, 3)

    # Test zeros
    Z = GPUFiniteFieldMatrices.zeros(Int, 2, 3, modulus)
    println("zeros(Int, 2, 3, $modulus) = ")
    display(Z)
    println()
    @test Array(Z) == Base.zeros(Int, 2, 3)

    # Test rand
    R = GPUFiniteFieldMatrices.rand(Int, 2, 2, modulus)
    println("rand(Int, 2, 2, $modulus) = ")
    display(R)
    println()
    @test all(0 .<= Array(R) .< modulus)

    println("All utility functions tests passed!")
end

"""
Test matrix operations of the CuModMatrix type.
"""
function test_matrix_operations()
    println("Testing matrix operations of CuModMatrix...")
    modulus = 11

    # Create an invertible matrix mod 11
    G_data = [1 2 3; 0 1 4; 5 6 0]
    G = CuModMatrix(G_data, modulus)

    println("G = ")
    display(G)
    println()

    # Test invertibility
    # @test GPUFiniteFieldMatrices.is_invertible(G)

    # TODO: FIX INVERSES!
    # # Test inverse
    # G_inv = inverse(G)
    # println("G^(-1) = ")
    # display(G_inv)
    # println()

    # # Test that G * G^(-1) = I (mod 11)
    # H = G * G_inv
    # println("G * G^(-1) = ")
    # display(H)
    # println()

    # # Expected result: identity matrix mod 11
    I_3 = Matrix{Int}(I, 3, 3)
    # @test all(isapprox.(Array(H), I_3))

    # Test matrix power
    G2 = G^2
    println("G^2 = ")
    display(G2)
    println()
    expected = mod.(G_data * G_data, modulus)
    @test Array(G2) == expected

    G0 = G^0
    println("G^0 = ")
    display(G0)
    println()
    @test Array(G0) == Matrix{Int}(I, 3, 3)

    # TODO: fix
    # Test non-invertible matrix
    N_data = [1 2 3; 2 4 6; 3 6 9]  # Linearly dependent rows
    N = CuModMatrix(N_data, modulus)

    println("N = ")
    display(N)
    println()

    # @test !GPUFiniteFieldMatrices.is_invertible(N)
end

"""
Test modulus-changing functions of the CuModMatrix type.
"""
function test_modulus_functions()
    println("Testing modulus-changing functions of CuModMatrix...")

    # Test initialization
    A_data = [1 2 3; 4 5 6; 7 8 9]
    modulus1 = 11  # First modulus
    modulus2 = 7   # Second modulus

    A = CuModMatrix(A_data, modulus1)

    println("A (mod $modulus1) = ")
    display(A)
    println()

    # Test change_modulus (creates new matrix)
    B = change_modulus(A, modulus2)

    println("B = change_modulus(A, $modulus2) = ")
    display(B)
    println()

    # Check properties and values
    @test B.N == modulus2
    @test size(B) == size(A)
    @test Array(B) == mod.(A_data, modulus2)

    # Test that A is unchanged
    @test A.N == modulus1

    # Test change_modulus! (in-place)
    C = CuModMatrix(A_data, modulus1)
    Cprime = change_modulus_no_alloc!(C, modulus2)

    println("C after change_modulus_no_alloc!(C, $modulus2) = ")
    display(C)
    println("The output of change_modulus_no_alloc!(C, $modulus2) = ")
    display(Cprime)
    println()

    # Check properties and values
    @test Cprime.N == modulus2
    @test Array(C) == mod.(A_data, modulus2)

    println("All modulus-changing function tests passed!")
end

# Run all tests
function test_gpu_mat()
    test_basic_operations()
    test_utility_functions()
    test_matrix_operations()
    test_modulus_functions()

    println("\nAll CuModMatrix tests passed!")
end

# Run the tests if this file is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    test_gpu_mat()
end
