function _batch_stream_test_host(k::Int, n::Int, N::Int)
    A = Matrix{Float32}(I, n, n)
    for j = 2:n
        A[1, j] = Float32(mod(3j + 7k, N))
    end
    return A
end

function _batch_stream_test_inputs(hosts, N::Int)
    return [CuModMatrix{Float32}(copy(A); modulus = N) for A in hosts]
end

function test_batched_streams()
    n = 64
    N = 101
    # pluq_batched handles the first item synchronously; three inputs exercise both streams.
    hosts = [_batch_stream_test_host(k, n, N) for k = 1:3]
    single_stream = PLUQOptions(batch_streams = 1)
    two_streams = PLUQOptions(batch_streams = 2)

    pluq_inputs_single = _batch_stream_test_inputs(hosts, N)
    pluq_inputs_two = _batch_stream_test_inputs(hosts, N)
    factors_single =
        GPUFiniteFieldMatrices.pluq_batched(pluq_inputs_single; options = single_stream)
    factors_two =
        GPUFiniteFieldMatrices.pluq_batched(pluq_inputs_two; options = two_streams)
    @test length(factors_single) == 3
    @test length(factors_two) == 3
    for i in eachindex(factors_single)
        @test factors_two[i].rank == factors_single[i].rank == n
        @test factors_two[i].p == factors_single[i].p
        @test factors_two[i].q == factors_single[i].q
        @test Array(factors_two[i].LU) == Array(factors_single[i].LU)
        @test GPUFiniteFieldMatrices.pluq_check_identity(factors_two[i], pluq_inputs_two[i])
    end

    inverse_inputs_single = _batch_stream_test_inputs(hosts, N)
    inverse_inputs_two = _batch_stream_test_inputs(hosts, N)
    inverses_single = GPUFiniteFieldMatrices.inverse_batched(
        inverse_inputs_single;
        options = single_stream,
    )
    inverses_two =
        GPUFiniteFieldMatrices.inverse_batched(inverse_inputs_two; options = two_streams)
    @test length(inverses_single) == 3
    @test length(inverses_two) == 3
    for i in eachindex(inverses_single)
        @test Array(inverses_two[i]) == Array(inverses_single[i])
        got = mod.(round.(Int, Array(inverse_inputs_two[i] * inverses_two[i])), N)
        @test got == Matrix{Int}(I, n, n)
    end
end
