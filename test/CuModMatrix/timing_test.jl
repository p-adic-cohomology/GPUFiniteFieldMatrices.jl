
function test_timings()

    println("Begin testing timing.")

    n = 5000
    A_data = Base.rand(0:10, n, n)
    B_data = Base.rand(0:10, n, n)
    x_data = Base.rand(0:10, n)

    A = CuModMatrix(A_data, 11)
    B = CuModMatrix(B_data, 11)
    x = CuModVector(x_data, 11)

    C = GPUFiniteFieldMatrices.zeros(Float32, n, n, 11)
    z = GPUFiniteFieldMatrices.zeros(Float32, n, 11)

    #These are mostly meant to be sanity checks to detect bugs,
    #rather than minor performance regressions

    result = @btimed CUDA.@sync GPUFiniteFieldMatrices.add!($C, $A, $B)
    @test result[:time] < 0.01 # on a 3070, I can get less than 0.001

    result = @btimed CUDA.@sync mul!($C, $A, 2)
    @test result[:time] < 0.01 # on a 3070, I can get less than 0.001

    result = @btimed CUDA.@sync mul!($C, $A, $B)
    @test result[:time] < 2 # on a 3070, I can get less than 0.2

    result = @btimed CUDA.@sync mul!($z, $A, $x)
    @test result[:time] < 0.01 # on a 3070, I can get less than 0.001

    A = CuModMatrix(A_data, 11, elem_type = Float64)
    B = CuModMatrix(B_data, 11, elem_type = Float64)
    x = CuModVector(x_data, 11, elem_type = Float64)

    C = GPUFiniteFieldMatrices.zeros(Float64, n, n, 11)
    z = GPUFiniteFieldMatrices.zeros(Float64, n, 11)

    #These are mostly meant to be sanity checks, rather than

    result = @btimed CUDA.@sync GPUFiniteFieldMatrices.add!($C, $A, $B)
    @test result[:time] < 0.01 # on a 3070, I can get less than 0.002

    result = @btimed CUDA.@sync mul!($C, $A, 2)
    @test result[:time] < 0.01 # on a 3070, I can get less than 0.002

    result = @btimed CUDA.@sync mul!($C, $A, $B)
    @test result[:time] < 5 # on a 3070, I can get less than 1

    result = @btimed CUDA.@sync mul!($z, $A, $x)
    @test result[:time] < 0.01 # on a 3070, I can get less than 0.001

    CUDA.@sync println("Done testing timing.")
end
