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
    A = CuModMatrix(A_data; modulus = modulus)

    cuA = CuArray(A_data)
    Aprime = CuModMatrix(cuA; modulus = modulus)
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
    B = CuModMatrix(B_data; modulus = modulus)

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
    F = GPUFiniteFieldMatrices.zeros(
        GPUFiniteFieldMatrices.CuModMatrix{eltype(parent(A))},
        size(A, 1),
        size(A, 2);
        modulus = modulus,
    )
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
    I3 = identity_matrix(CuModMatrix{Int}, 3; modulus = modulus)
    println("identity_matrix(CuModMatrix{Int}, 3; modulus = $modulus) = ")
    display(I3)
    println()
    @test Array(I3) == Matrix{Int}(I, 3, 3)

    # Test zeros
    Z = GPUFiniteFieldMatrices.zeros(
        GPUFiniteFieldMatrices.CuModMatrix{Int},
        2,
        3;
        modulus = modulus,
    )
    println("zeros(Int, 2, 3, $modulus) = ")
    display(Z)
    println()
    @test Array(Z) == Base.zeros(Int, 2, 3)

    # Test rand
    R = GPUFiniteFieldMatrices.rand(CuModMatrix{Int}, 2, 2; modulus = modulus)
    println("rand(Int, 2, 2, $modulus) = ")
    display(R)
    println()
    @test all(0 .<= Array(R) .< modulus)
    @test all(isinteger, Array(R))

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
    G = CuModMatrix(G_data; modulus = modulus)

    println("G = ")
    display(G)
    println()

    # Test invertibility
    # @test GPUFiniteFieldMatrices.is_invertible(G)

    # TODO: FIX INVERSES!
    # # Test inverse
    # G_inv = inv(G)
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
    N = CuModMatrix(N_data; modulus = modulus)

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

    A = CuModMatrix(A_data; modulus = modulus1)

    println("A (mod $modulus1) = ")
    display(A)
    println()

    # Test change_modulus (creates new matrix)
    B = change_modulus(A, modulus2)

    println("B = change_modulus(A, $modulus2) = ")
    display(B)
    println()

    # Check properties and values
    @test GPUFiniteFieldMatrices.modulus(B) == modulus2
    @test size(B) == size(A)
    @test Array(B) == mod.(A_data, modulus2)

    # Test that A is unchanged
    @test GPUFiniteFieldMatrices.modulus(A) == modulus1

    # Shrink consumes the input wrapper and retains its GPU storage.
    C = CuModMatrix(A_data; modulus = modulus1)
    C_parent = parent(C)
    C = GPUFiniteFieldMatrices.change_modulus!!(C, modulus2)
    @test GPUFiniteFieldMatrices.modulus(C) == modulus2
    @test size(C) == size(A_data)
    @test parent(C) === C_parent
    @test Array(C) == mod.(A_data, modulus2)

    # Grow reuses the same storage without changing values.
    D = CuModMatrix(A_data; modulus = modulus2)
    D_parent = parent(D)
    D = GPUFiniteFieldMatrices.change_modulus!!(D, 13)
    @test GPUFiniteFieldMatrices.modulus(D) == 13
    @test size(D) == size(A_data)
    @test parent(D) === D_parent
    @test Array(D) == mod.(A_data, modulus2)

    # Reject nonpositive moduli before changing the input storage.
    invalid_zero = CuModMatrix(A_data; modulus = modulus1)
    invalid_zero_parent = parent(invalid_zero)
    invalid_zero_values = Array(invalid_zero_parent)
    @test_throws ArgumentError GPUFiniteFieldMatrices.change_modulus!!(invalid_zero, 0)
    @test parent(invalid_zero) === invalid_zero_parent
    @test GPUFiniteFieldMatrices.modulus(invalid_zero) == modulus1
    @test Array(invalid_zero_parent) == invalid_zero_values

    invalid_negative = CuModMatrix(A_data; modulus = modulus1)
    invalid_negative_parent = parent(invalid_negative)
    invalid_negative_values = Array(invalid_negative_parent)
    @test_throws ArgumentError GPUFiniteFieldMatrices.change_modulus!!(invalid_negative, -1)
    @test parent(invalid_negative) === invalid_negative_parent
    @test GPUFiniteFieldMatrices.modulus(invalid_negative) == modulus1
    @test Array(invalid_negative_parent) == invalid_negative_values

    function gpu_allocated_bytes_after_warmup(values, current_modulus, new_modulus)
        warmup = CuModMatrix(values; modulus = current_modulus)
        warmup = GPUFiniteFieldMatrices.change_modulus!!(warmup, new_modulus)
        CUDA.synchronize()

        input = CuModMatrix(values; modulus = current_modulus)
        bytes = CUDA.@allocated begin
            GPUFiniteFieldMatrices.change_modulus!!(input, new_modulus)
            CUDA.synchronize()
        end
        return bytes
    end

    # Warm both branches before checking device-memory allocation.
    gpu_allocated_bytes_after_warmup(A_data, modulus1, modulus2)
    gpu_allocated_bytes_after_warmup(A_data, modulus2, 13)
    @test gpu_allocated_bytes_after_warmup(A_data, modulus1, modulus2) == 0
    @test gpu_allocated_bytes_after_warmup(A_data, modulus2, 13) == 0

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
