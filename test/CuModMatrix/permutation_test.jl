function test_permutations()

    A = [
        1.0 2.0 3.0;
        4.0 5.0 6.0;
        7.0 8.0 9.0
    ]

    A_gpu = CuModMatrix(A, 11)

    @test size(A_gpu) == (3, 3)

    P = [(2, 3)]

    GPUFiniteFieldMatrices.apply_col_perm!(P, A_gpu)

    @test size(A_gpu) == (3, 3)

    @test Array(A_gpu) ≈ [
        1.0 3.0 2.0;
        4.0 6.0 5.0;
        7.0 9.0 8.0
    ]

    GPUFiniteFieldMatrices.apply_col_inv_perm!(P, A_gpu)

    @test Array(A_gpu) ≈ [
        1.0 2.0 3.0;
        4.0 5.0 6.0;
        7.0 8.0 9.0
    ]

    A = [
        1.0 2.0 3.0;
        4.0 5.0 6.0;
        7.0 8.0 9.0
    ]

    A_gpu = CuModMatrix(A, 11)

    GPUFiniteFieldMatrices.apply_row_perm!(P, A_gpu)

    @test Array(A_gpu) ≈ [
        1.0 2.0 3.0;
        7.0 8.0 9.0;
        4.0 5.0 6.0
    ]

    GPUFiniteFieldMatrices.apply_row_inv_perm!(P, A_gpu)

    @test Array(A_gpu) ≈ [
        1.0 2.0 3.0;
        4.0 5.0 6.0;
        7.0 8.0 9.0
    ]

end
