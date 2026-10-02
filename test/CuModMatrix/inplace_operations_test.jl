#using GPUFiniteFieldMatrices
#using Test
#using CUDA
#using LinearAlgebra

"""
Test in-place operations of the CuModMatrix type.
This includes:
- Basic in-place arithmetic operations
- In-place scalar operations
- In-place operations with modulus override
"""
function test_inplace_basic_operations()
    println("Testing in-place operations of CuModMatrix...")

    A_data = [1 2 3; 4 5 6; 7 8 9]
    B_data = [9 8 7; 6 5 4; 3 2 1]
    modulus = 11  # Use a prime modulus

    A = CuModMatrix(A_data, modulus)
    B = CuModMatrix(B_data, modulus)

    C = GPUFiniteFieldMatrices.zeros(Int, 3, 3, modulus)
    D = GPUFiniteFieldMatrices.zeros(Int, 3, 3, modulus)

    println("A = ")
    display(A)
    println()

    println("B = ")
    display(B)
    println()

    # Test add!
    GPUFiniteFieldMatrices.add!(C, A, B)
    println("C = A + B (in-place) = ")
    display(C)
    println()

    expected = mod.(A_data + B_data, modulus)
    @test Array(C) == expected

    # Test sub!
    GPUFiniteFieldMatrices.sub!(D, A, B)
    println("D = A - B (in-place) = ")
    display(D)
    println()

    expected = mod.(A_data - B_data, modulus)
    @test Array(D) == expected

    # Test elementwise_multiply!
    elementwise_multiply!(C, A, B)
    println("C = A .* B (in-place) = ")
    display(C)
    println()

    expected = mod.(A_data .* B_data, modulus)
    @test Array(C) == expected

    # Test negate!
    negate!(D, A)
    println("D = -A (in-place) = ")
    display(D)
    println()

    expected = mod.(-A_data, modulus)
    @test Array(D) == expected

    # Test scalar operations
    scalar = 3

    # Test scalar_add!
    scalar_add!(C, A, scalar)
    println("C = A + $scalar (in-place) = ")
    display(C)
    println()

    expected = mod.(A_data .+ scalar, modulus)
    @test Array(C) == expected

    scalar_sub!(D, A, scalar)
    println("D = A - $scalar (in-place) = ")
    display(D)
    println()

    expected = mod.(A_data .- scalar, modulus)
    @test Array(D) == expected

    # Test scalar_multiply!
    mul!(C, A, scalar)
    println("C = A * $scalar (in-place) = ")
    display(C)
    println()

    expected = mod.(A_data .* scalar, modulus)
    @test Array(C) == expected

    # Test matrix multiplication
    E = GPUFiniteFieldMatrices.zeros(Float32, 3, 3, modulus)
    mul!(E, A, B)
    println("E = A * B (in-place) = ")
    display(E)
    println()

    expected = mod.(A_data * B_data, modulus)
    @test Array(E) == convert.(Float32, expected)

    # Test copyto!
    F_data = convert.(Float32, rand(1:(modulus-1), 100, 100))
    F = GPUFiniteFieldMatrices.zeros(Float32, 100, 100, modulus)

    copyto!(F, F_data)
    @test Array(F) == F_data

    println("All in-place operations tests passed!")
end

"""
Test in-place operations with modulus override.
"""
function test_inplace_modulus_override()
    println("Testing in-place operations with modulus override...")

    A_data = [1 5 10; 15 20 25; 30 35 40]
    modulus1 = 27
    modulus2 = 9

    A = CuModMatrix(A_data, modulus1)
    B = CuModMatrix(A_data, modulus2)
    C = GPUFiniteFieldMatrices.zeros(Int, 3, 3, modulus1)

    # Test add! with modulus override
    override_modulus = 3
    GPUFiniteFieldMatrices.add!(C, A, B; mod_N = override_modulus)

    println("C = A + B (in-place with modulus override $override_modulus) = ")
    display(C)
    println()

    expected = mod.(A_data + A_data, override_modulus)
    @test Array(C) == expected

    # Test sub! with modulus override
    GPUFiniteFieldMatrices.sub!(C, A, B; mod_N = override_modulus)

    println("C = A - B (in-place with modulus override $override_modulus) = ")
    display(C)
    println()

    expected = mod.(A_data - A_data, override_modulus)
    @test Array(C) == expected

    # Test elementwise_multiply! with modulus override
    elementwise_multiply!(C, A, B, override_modulus)

    println("C = A .* B (in-place with modulus override $override_modulus) = ")
    display(C)
    println()

    expected = mod.(A_data .* A_data, override_modulus)
    @test Array(C) == expected

    scalar = 3

    # Test scalar_add! with modulus override
    scalar_add!(C, A, scalar, override_modulus)
    println("C = A + $scalar (in-place with modulus override $override_modulus) = ")
    display(C)
    println()

    expected = mod.(A_data .+ scalar, override_modulus)
    @test Array(C) == expected

    # Test scalar_sub! with modulus override
    scalar_sub!(C, A, scalar, override_modulus)
    println("C = A - $scalar (in-place with modulus override $override_modulus) = ")
    display(C)
    println()

    expected = mod.(A_data .- scalar, override_modulus)
    @test Array(C) == expected

    # Test scalar_multiply! with modulus override
    mul!(C, A, scalar, override_modulus)
    println("C = A * $scalar (in-place with modulus override $override_modulus) = ")
    display(C)
    println()

    expected = mod.(A_data .* scalar, override_modulus)
    @test Array(C) == expected

    # Test matrix multiplication with modulus override
    #
    #NOTE: this feature was removed, but here's the test preserved in this comment
    #if want to put it back.
    #
    #D = GPUFiniteFieldMatrices.zeros(Float32, 3, 3, modulus1)
    #mul!(D, A, B, override_modulus)
    #println("D = A * B (in-place with modulus override $override_modulus) = ")
    #display(D)
    #println()
    #
    #expected = mod.(A_data * A_data, override_modulus)
    #@test Array(D) == expected

    println("All in-place operations with modulus override tests passed!")
end

function test_inplace_copy_and_mod()
    println("Testing in-place copy and modulus operations...")

    A_data = [1 2 3; 4 5 6; 7 8 9]
    modulus = 11

    A = CuModMatrix(A_data, modulus)
    B = GPUFiniteFieldMatrices.zeros(Float32, 3, 3, modulus)

    # Test copy!
    copy!(B, A)
    println("B = copy of A (in-place) = ")
    display(B)
    println()

    @test Array(B) == A_data

    # Test mod_elements!
    F = CuModMatrix(A_data * 2, modulus)
    mod_elements!(F)
    println("F with modulus applied in-place = ")
    display(F)
    println()

    expected = mod.(A_data * 2, modulus)
    @test Array(F) == expected

    # Test mod_elements! with modulus override
    override_modulus = 5
    mod_elements!(F, override_modulus)
    println("F with modulus $override_modulus applied in-place = ")
    display(F)
    println()

    expected = mod.(expected, override_modulus)
    @test Array(F) == expected

    # Test filling (and modding)
    fill!(F, 122.0)
    expected = ones(Float32, 3, 3)
    @test Array(F) == expected

    # Test zeroing
    GPUFiniteFieldMatrices.zero!(A)
    expected = zeros(Float32, 3, 3)
    @test Array(A) == expected

    println("All in-place copy and modulus operations tests passed!")
end

function test_inplace()
    test_inplace_basic_operations()
    test_inplace_modulus_override()
    test_inplace_copy_and_mod()

    println("\nAll in-place operation tests passed!")
end

# Run the tests if this file is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    test_inplace()
end
