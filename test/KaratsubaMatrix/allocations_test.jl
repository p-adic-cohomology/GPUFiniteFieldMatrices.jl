
function test_allocations()

    A = GPUFiniteFieldMatrices.CuModMatrix([1 2; 3 4], 100)
    B = GPUFiniteFieldMatrices.CuModMatrix([5 6; 7 8], 100)
    C = GPUFiniteFieldMatrices.CuModMatrix([0 0; 0 0], 100)
    #D = GPUFiniteFieldMatrices.CuModMatrix([0 0; 0 0],100)
    AK = GPUFiniteFieldMatrices.to_karatsuba(A, 100)
    BK = GPUFiniteFieldMatrices.to_karatsuba(B, 100)
    CK = GPUFiniteFieldMatrices.to_karatsuba(C, 100)
    GPUFiniteFieldMatrices.initialize_plan!(AK)
    GPUFiniteFieldMatrices.initialize_plan!(BK)
    GPUFiniteFieldMatrices.initialize_plan!(CK)
    #DK = to_karatsuba(D, 100)

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
