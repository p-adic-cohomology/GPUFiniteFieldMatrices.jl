function test_de_rham()

    A = [
        0 0 0 0 0 0 0 -3 0 0;
        0 0 0 0 0 0 0 0 -3 0;
        0 1 0 0 0 0 0 0 0 0;
        0 0 1 0 0 0 0 0 0 0;
        0 -2 0 0 0 0 0 0 0 -3;
        0 0 -2 0 2 0 0 0 0 0;
        0 0 0 1 0 2 0 0 0 0;
        0 -3 0 -2 0 0 0 -1 0 0;
        0 0 -3 0 0 0 2 0 -1 0;
        1 0 0 -3 0 0 0 0 0 -1
    ]

    A_gpu = CuModMatrix(A, 7)

    flag, B_gpu = GPUFiniteFieldMatrices.is_invertible_with_inverse(A_gpu, debug = true)

    # CUDA.sync()

    @test flag == true

    res = [
        1 0 0 0 0 0 0 0 0 0;
        0 1 0 0 0 0 0 0 0 0;
        0 0 1 0 0 0 0 0 0 0;
        0 0 0 1 0 0 0 0 0 0;
        0 0 0 0 1 0 0 0 0 0;
        0 0 0 0 0 1 0 0 0 0;
        0 0 0 0 0 0 1 0 0 0;
        0 0 0 0 0 0 0 1 0 0;
        0 0 0 0 0 0 0 0 1 0;
        0 0 0 0 0 0 0 0 0 1;
    ]

    @test Array(A_gpu*B_gpu) ≈ res

end
