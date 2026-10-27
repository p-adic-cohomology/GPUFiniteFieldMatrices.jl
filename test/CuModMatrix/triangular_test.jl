function test_upper_triangular_inverse(DEBUG::Bool = false)

    function unit_test_upper_triangular_inverse(A::Matrix, N::Int; debug::Bool = false)
        println("Running upper triangular inverse test with size $(size(A)) and N=$N")
        A_gpu = CuModMatrix(A; modulus = N, new_size = size(A))
        A_inv =
            GPUFiniteFieldMatrices.upper_triangular_inverse_no_copy(A_gpu; debug = debug)
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
        A_gpu = CuModMatrix(A; modulus = N)
        A_inv =
            GPUFiniteFieldMatrices.lower_triangular_inverse_no_copy(A_gpu; debug = debug)

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

function _triangular_inverse_integer_oracle(
    A::AbstractMatrix{<:Integer},
    N::Int,
    lower::Bool,
)
    n = size(A, 1)
    result = zeros(Int, n, n)
    row_order = lower ? (1:n) : (n:-1:1)

    for col = 1:n
        for row in row_order
            product_sum = 0
            if lower
                for k = 1:(row-1)
                    product_sum += A[row, k] * result[k, col]
                end
            else
                for k = (row+1):n
                    product_sum += A[row, k] * result[k, col]
                end
            end
            rhs = row == col ? 1 : 0
            result[row, col] = mod(invmod(A[row, row], N) * (rhs - product_sum), N)
        end
    end

    return result
end

function test_triangular_destination_helpers()
    N = 101
    n = 5
    lower = Matrix{Int}(undef, n, n)
    upper = Matrix{Int}(undef, n, n)
    for row = 1:n, col = 1:n
        lower[row, col] = row == col ? row + 1 : row > col ? mod(17row + 3col, N) : 0
        upper[row, col] = row == col ? row + 1 : row < col ? mod(11row + 5col, N) : 0
    end

    @testset "destination-first triangular32 helpers" begin
        for (A, is_lower) in ((lower, true), (upper, false))
            source_parent = CUDA.zeros(Float32, n + 4, n + 5)
            source_outer = @view source_parent[1:(n+3), 1:(n+4)]
            source_view = @view source_outer[2:(n+1), 3:(n+2)]
            copyto!(source_view, CUDA.CuArray(Float32.(A)))

            destination_parent = CUDA.fill(-7.0f0, n + 4, n + 5)
            destination_outer = @view destination_parent[1:(n+3), 1:(n+4)]
            destination_view = @view destination_outer[2:(n+1), 3:(n+2)]

            if is_lower
                GPUFiniteFieldMatrices._lower_triangular_inv_32!(
                    destination_view,
                    source_view,
                    N,
                    0,
                    0,
                    n,
                )
            else
                GPUFiniteFieldMatrices._upper_triangular_inv_32!(
                    destination_view,
                    source_view,
                    N,
                    0,
                    0,
                    n,
                )
            end

            expected = _triangular_inverse_integer_oracle(A, N, is_lower)
            @test round.(Int, Array(destination_view)) == expected
            @test Array(source_view) == Float32.(A)

            destination = Array(destination_parent)
            @test all(==(-7.0f0), destination[1, :])
            @test all(==(-7.0f0), destination[:, 1:2])
            @test all(==(-7.0f0), destination[(n+2):end, :])
            @test all(==(-7.0f0), destination[:, (n+3):end])
        end
    end

    @testset "recursive triangular inverse agrees with integer oracle" begin
        recursive_n = TILE_WIDTH + 3
        recursive_lower = Matrix{Int}(undef, recursive_n, recursive_n)
        recursive_upper = Matrix{Int}(undef, recursive_n, recursive_n)
        for row = 1:recursive_n, col = 1:recursive_n
            recursive_lower[row, col] =
                row == col ? row + 1 : row > col ? mod(17row + 3col, N) : 0
            recursive_upper[row, col] =
                row == col ? row + 1 : row < col ? mod(11row + 5col, N) : 0
        end

        lower_gpu = CuModMatrix(recursive_lower; modulus = N)
        lower_inverse = GPUFiniteFieldMatrices.lower_triangular_inverse_no_copy(lower_gpu)
        @test round.(Int, Array(parent(lower_inverse))[1:recursive_n, 1:recursive_n]) ==
              _triangular_inverse_integer_oracle(recursive_lower, N, true)

        upper_gpu = CuModMatrix(recursive_upper; modulus = N)
        upper_inverse = GPUFiniteFieldMatrices.upper_triangular_inverse_no_copy(upper_gpu)
        @test round.(Int, Array(parent(upper_inverse))[1:recursive_n, 1:recursive_n]) ==
              _triangular_inverse_integer_oracle(recursive_upper, N, false)
    end

    return nothing
end
