
function test_allocations()

    A = GPUFiniteFieldMatrices.CuModMatrix([1 2; 3 4]; modulus = 100)
    B = GPUFiniteFieldMatrices.CuModMatrix([5 6; 7 8]; modulus = 100)
    C = GPUFiniteFieldMatrices.CuModMatrix([0 0; 0 0]; modulus = 100)
    #D = GPUFiniteFieldMatrices.CuModMatrix([0 0; 0 0]; modulus = 100)
    AK = GPUFiniteFieldMatrices.to_karatsuba(A; modulus = 100)
    BK = GPUFiniteFieldMatrices.to_karatsuba(B; modulus = 100)
    CK = GPUFiniteFieldMatrices.to_karatsuba(C; modulus = 100)
    GPUFiniteFieldMatrices.initialize_plan!(AK)
    GPUFiniteFieldMatrices.initialize_plan!(BK)
    GPUFiniteFieldMatrices.initialize_plan!(CK)
    #DK = to_karatsuba(D; modulus = 100)

    @testset "Allocation tests" begin
        result = CUDA.@timed GPUFiniteFieldMatrices.add!(CK, AK, BK)
        @test result[:gpu_bytes] == 0

        result = CUDA.@timed GPUFiniteFieldMatrices.sub!(CK, AK, BK)
        @test result[:gpu_bytes] == 0

        #MatMul allocates a few views
        result = CUDA.@timed GPUFiniteFieldMatrices.LinearAlgebra.mul!(CK, AK, BK)
        @test result[:gpu_bytes] < 30

    end
end
