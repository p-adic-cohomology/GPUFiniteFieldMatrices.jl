function test_basecase_pluq()
    N = 101
    Ahost = [
        0 2 4 1
        0 0 5 3
        1 7 6 4
        2 8 9 10
    ]
    A = CuModMatrix(Ahost; modulus = N)
    p = GPUFiniteFieldMatrices.pluq_init_perm(4)
    q = GPUFiniteFieldMatrices.pluq_init_perm(4)
    rank = GPUFiniteFieldMatrices.pluq_basecase_gpu!(parent(A), N, p, q, 1, 4, 4)
    @test rank == 4
end
