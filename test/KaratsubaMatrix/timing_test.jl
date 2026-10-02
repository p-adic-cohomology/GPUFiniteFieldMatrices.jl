
function test_timings()
    println("Begin testing timing.")

    n = 5000
    A_data1 = Base.rand(0:10, n, n)
    B_data1 = Base.rand(0:10, n, n)
    x_data1 = Base.rand(0:10, n)

    A_data2 = Base.rand(0:10, n, n)
    B_data2 = Base.rand(0:10, n, n)
    x_data2 = Base.rand(0:10, n)

    A1 = GPUFiniteFieldMatrices.CuModMatrix(A_data1, 11^2)
    B1 = GPUFiniteFieldMatrices.CuModMatrix(B_data1, 11^2)
    x1 = GPUFiniteFieldMatrices.CuModVector(x_data1, 11^2)

    A2 = GPUFiniteFieldMatrices.CuModMatrix(A_data2, 11^2)
    B2 = GPUFiniteFieldMatrices.CuModMatrix(B_data2, 11^2)
    x2 = GPUFiniteFieldMatrices.CuModVector(x_data2, 11^2)

    A = GPUFiniteFieldMatrices.KaratsubaMatrix(A1, A2, 11, 11, 11^2)
    B = GPUFiniteFieldMatrices.KaratsubaMatrix(B1, B2, 11, 11, 11^2)
    x = GPUFiniteFieldMatrices.KaratsubaVector(x1, x2, 11, 11, 11^2)

    C = GPUFiniteFieldMatrices.KaratsubaZeros(Float32, n, n, 11, 11, 11^2, true)
    z = GPUFiniteFieldMatrices.KaratsubaZeros(Float32, n, 11, 11, 11^2, true)

    plan1 = GPUFiniteFieldMatrices.KaratsubaZeros(Float32, n, n, 11, 11, 11^2, true)
    plan2 = GPUFiniteFieldMatrices.KaratsubaZeros(Float32, n, 11, 11, 11^2, true)

    result = @btimed CUDA.@sync GPUFiniteFieldMatrices.add!($C, $A, $B)
    @test result[:time] < 0.015

    result = @btimed CUDA.@sync GPUFiniteFieldMatrices.scalar_multiply!($C, $A, 2)
    @test result[:time] < 0.01

    result = @btimed CUDA.@sync GPUFiniteFieldMatrices.KMatMul!($C, $A, $B, $plan1, $plan1)
    @test result[:time] < 1

    result = @btimed CUDA.@sync GPUFiniteFieldMatrices.KMatMul!($z, $A, $x, $plan2, $plan1)
    @test result[:time] < 0.01

    A1 = GPUFiniteFieldMatrices.CuModMatrix(A_data1, 11^2, elem_type = Float64)
    B1 = GPUFiniteFieldMatrices.CuModMatrix(B_data1, 11^2, elem_type = Float64)
    x1 = GPUFiniteFieldMatrices.CuModVector(x_data1, 11^2, elem_type = Float64)

    A2 = GPUFiniteFieldMatrices.CuModMatrix(A_data2, 11^2, elem_type = Float64)
    B2 = GPUFiniteFieldMatrices.CuModMatrix(B_data2, 11^2, elem_type = Float64)
    x2 = GPUFiniteFieldMatrices.CuModVector(x_data2, 11^2, elem_type = Float64)

    A = GPUFiniteFieldMatrices.KaratsubaMatrix(A1, A2, 11, 11, 11^2)
    B = GPUFiniteFieldMatrices.KaratsubaMatrix(B1, B2, 11, 11, 11^2)
    x = GPUFiniteFieldMatrices.KaratsubaVector(x1, x2, 11, 11, 11^2)

    C = GPUFiniteFieldMatrices.KaratsubaZeros(Float64, n, n, 11, 11, 11^2, true)
    z = GPUFiniteFieldMatrices.KaratsubaZeros(Float64, n, 11, 11, 11^2, true)

    plan1 = GPUFiniteFieldMatrices.KaratsubaZeros(Float64, n, n, 11, 11, 11^2, true)
    plan2 = GPUFiniteFieldMatrices.KaratsubaZeros(Float64, n, 11, 11, 11^2, true)

    result = @btimed CUDA.@sync GPUFiniteFieldMatrices.add!($C, $A, $B)
    @test result[:time] < 0.04

    result = @btimed CUDA.@sync GPUFiniteFieldMatrices.scalar_multiply!($C, $A, 2)
    @test result[:time] < 0.025

    result = @btimed CUDA.@sync GPUFiniteFieldMatrices.KMatMul!($C, $A, $B, $plan1, $plan1)
    @test result[:time] < 4

    result = @btimed CUDA.@sync GPUFiniteFieldMatrices.KMatMul!($z, $A, $x, $plan2, $plan1)
    @test result[:time] < 0.01

    CUDA.@sync println("Done testing timing.")
end
