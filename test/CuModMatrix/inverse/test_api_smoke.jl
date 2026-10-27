function test_api_smoke()
    N = 101
    Ahost = [
        3 5 7
        2 4 9
        1 8 6
    ]
    A = CuModMatrix(Ahost, N)
    F = pluq(A)
    @test length(F.p) == 3
    @test length(F.q) == 3
    @test F.rank == 3
    @test :L in propertynames(F)
    @test :U in propertynames(F)
    @test F.L isa CuModMatrix
    @test F.U isa CuModMatrix
    @test GPUFiniteFieldMatrices.pluq_check_identity(F, A)

    A_inplace = CuModMatrix(copy(Ahost), N)
    F_inplace = pluq!(A_inplace)
    @test F_inplace.rank == 3
    @test F_inplace.LU.data === A_inplace.data
    @test GPUFiniteFieldMatrices.pluq_check_identity(F_inplace, A)

    Ainv = inv(A)
    I1 = mod.(Array(A * Ainv), N)
    expected = Matrix{eltype(I1)}([1 0 0; 0 1 0; 0 0 1])
    @test I1 == expected

    opts = PLUQOptions(lazy_q = true, nftb = 8)
    F2 = pluq(A, options = opts)
    @test F2.rank == 3
    @test GPUFiniteFieldMatrices.pluq_check_identity(F2, A)

    batch = [A, A]
    invs = GPUFiniteFieldMatrices.inverse_batched(batch, options = opts)
    @test length(invs) == 2
    I2 = mod.(Array(A * invs[1]), N)
    @test I2 == expected
end

function test_batch_fallbacks()
    N = 101
    square = [CuModMatrix(Matrix{Float32}(I, 33, 33), N) for _ = 1:2]
    Fs = GPUFiniteFieldMatrices.pluq_batched(
        square,
        options = PLUQOptions(batch_streams = 2),
    )
    @test all(F -> F.rank == 33, Fs)
    Xs = GPUFiniteFieldMatrices.inverse_batched(
        square,
        options = PLUQOptions(batch_streams = 2),
    )
    @test all(X -> mod.(round.(Int, Array(square[1] * X)), N) == Matrix{Int}(I, 33, 33), Xs)

    wide = CuModMatrix(Float32[1 0 0; 0 1 0], N)
    tall = CuModMatrix(Float32[1 0; 0 1; 0 0], N)
    mixed = GPUFiniteFieldMatrices.inverse_batched(
        [wide, tall],
        options = PLUQOptions(batch_streams = 2),
    )
    @test mod.(round.(Int, Array(wide * mixed[1])), N) == Matrix{Int}(I, 2, 2)
    @test mod.(round.(Int, Array(mixed[2] * tall)), N) == Matrix{Int}(I, 2, 2)
end
