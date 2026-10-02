function test_upper_triangular_inverse(DEBUG::Bool = false)

    function unit_test_upper_triangular_inverse(A::Matrix, N::Int; debug::Bool = false)
        println("Running upper triangular inverse test with size $(size(A)) and N=$N")
        A_gpu = CuModMatrix(A, N; new_size = size(A))
        A_inv = upper_triangular_inverse_no_copy(A_gpu; debug = debug)
        if debug
            println("A:")
            display(A_gpu)
            println("A_inv:")
            display(A_inv)
            println("A * A_inv:")
            display(A_gpu * A_inv)
        end
        res = Array(A_gpu * A_inv)

        I_ref = Matrix{eltype(A)}(I, (size(A_gpu)[1], size(A_gpu)[1]))
        @test isapprox(res, I_ref, atol = 1e-10)
    end

    A = [
        1.0 3.0;
        0.0 2.0
    ]

    unit_test_upper_triangular_inverse(A, 7; debug = DEBUG)

    A = Matrix(I, 1000, 1000)
    unit_test_upper_triangular_inverse(A, 2; debug = DEBUG)

    # for N in [3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97]
    #     for i in 33:65
    #         A = triu(rand(1:(N-1), i, i))

    #         unit_test_upper_triangular_inverse(A, N; debug=DEBUG)
    #     end
    # end

    A = triu(rand(1:12, 1000, 1000))

    unit_test_upper_triangular_inverse(A, 13; debug = DEBUG)

    A = triu(rand(1:12, 1000, 5000))

    unit_test_upper_triangular_inverse(A, 13; debug = DEBUG)

    return nothing
end

function test_lower_triangular_inverse(DEBUG::Bool = false)

    function unit_test_lower_triangular_inverse(A::Matrix, N::Int; debug::Bool = false)
        println("Running lower triangular inverse test with size $(size(A)) and N=$N")
        A_gpu = CuModMatrix(A, N)
        A_inv = lower_triangular_inverse_no_copy(A_gpu; debug = debug)

        res = Array(A_gpu * A_inv)
        I_ref = Matrix{eltype(A)}(I, (size(A_gpu)[1], size(A_gpu)[1]))

        if debug
            println("A:")
            display(A)
            println("A_inv:")
            display(A_inv)
            println("A * A_inv:")
            display(res)
        end

        @test isapprox(res, I_ref, atol = 1e-10)
    end

    A = [
        1.0 0.0;
        3.0 2.0
    ]

    unit_test_lower_triangular_inverse(A, 7; debug = DEBUG)

    A = Matrix(I, 1000, 1000)
    unit_test_lower_triangular_inverse(A, 2; debug = DEBUG)

    for N in [
        3,
        5,
        7,
        11,
        13,
        17,
        19,
        23,
        29,
        31,
        37,
        41,
        43,
        47,
        53,
        59,
        61,
        67,
        71,
        73,
        79,
        83,
        89,
        97,
    ]
        for i = 33:65
            A = tril(rand(1:(N-1), i, i))

            unit_test_lower_triangular_inverse(A, N; debug = DEBUG)
        end
    end

    A = tril(rand(1:12, 1000, 1000))

    unit_test_lower_triangular_inverse(A, 13; debug = DEBUG)

    return nothing
end
