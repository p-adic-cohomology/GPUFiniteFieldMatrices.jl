
function test_allocations()

    A = GPUFiniteFieldMatrices.CuModMatrix([1 2; 3 4], 100)
    B = GPUFiniteFieldMatrices.CuModMatrix([5 6; 7 8], 100)
    C = GPUFiniteFieldMatrices.CuModMatrix([0 0; 0 0], 100)
    #D = GPUFiniteFieldMatrices.CuModMatrix([0 0; 0 0],100)
    AK = GPUFiniteFieldMatrices.MatToKMat(A, 100)
    BK = GPUFiniteFieldMatrices.MatToKMat(B, 100)
    CK = GPUFiniteFieldMatrices.MatToKMat(C, 100)
    GPUFiniteFieldMatrices.initialize_plan!(AK)
    GPUFiniteFieldMatrices.initialize_plan!(BK)
    GPUFiniteFieldMatrices.initialize_plan!(CK)
    #DK = MatToKMat(D, 100)

    @testset "Allocation tests" begin
        result = CUDA.@timed GPUFiniteFieldMatrices.add!(CK, AK, BK)
        @test result[:gpu_bytes] == 0

        result = CUDA.@timed GPUFiniteFieldMatrices.sub!(CK, AK, BK)
        @test result[:gpu_bytes] == 0

        #MatMul allocates a few views
        result = CUDA.@timed GPUFiniteFieldMatrices.KMatMul!(CK, AK, BK)
        @test result[:gpu_bytes] < 30

    end
end
