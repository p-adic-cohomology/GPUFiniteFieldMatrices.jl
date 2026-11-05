using Test, CUDA, LinearAlgebra, GPUFiniteFieldMatrices

function test_karatsuba_allocations()

    A = GPUFiniteFieldMatrices.CuModMatrix{Float64}([1 2; 3 4]; modulus = 100)
    B = GPUFiniteFieldMatrices.CuModMatrix{Float64}([5 6; 7 8]; modulus = 100)
    C = GPUFiniteFieldMatrices.CuModMatrix{Float64}([0 0; 0 0]; modulus = 100)
    #D = GPUFiniteFieldMatrices.CuModMatrix{Float64}([0 0; 0 0]; modulus = 100)
    AK = GPUFiniteFieldMatrices.to_karatsuba(A; modulus = 100)
    BK = GPUFiniteFieldMatrices.to_karatsuba(B; modulus = 100)
    CK = GPUFiniteFieldMatrices.to_karatsuba(C; modulus = 100)
    GPUFiniteFieldMatrices.initialize_plan!(AK)
    GPUFiniteFieldMatrices.initialize_plan!(BK)
    GPUFiniteFieldMatrices.initialize_plan!(CK)
    #DK = to_karatsuba(D; modulus = 100)

    @testset "Allocation tests" begin
        CUDA.@sync GPUFiniteFieldMatrices.add!(CK, AK, BK)
        result = CUDA.@timed GPUFiniteFieldMatrices.add!(CK, AK, BK)
        CUDA.synchronize()
        @test result[:gpu_bytes] == 0

        CUDA.@sync GPUFiniteFieldMatrices.sub!(CK, AK, BK)
        result = CUDA.@timed GPUFiniteFieldMatrices.sub!(CK, AK, BK)
        CUDA.synchronize()
        @test result[:gpu_bytes] == 0

        #MatMul allocates a few views
        CUDA.@sync LinearAlgebra.mul!(CK, AK, BK)
        result = CUDA.@timed LinearAlgebra.mul!(CK, AK, BK)
        CUDA.synchronize()
        @test result[:gpu_bytes] < 30

    end
end
