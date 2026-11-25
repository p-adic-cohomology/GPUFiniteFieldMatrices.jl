@testset "CUDA modular wrapper methods and native batch streams" begin
    G = GPUFiniteFieldMatrices
    ext = Base.get_extension(G, :GPUFiniteFieldMatricesCUDAExt)
    device = CUDA.device()

    source = G.CuModArray{Float32,2}(
        Float32[1 2 91; 3 4 92; 93 94 95];
        modulus = 7,
        new_size = (2, 2),
    )
    copied = copy(source)
    @test copied isa G.CuModArray{Float32,2}
    @test copied !== source
    @test parent(copied) !== parent(source)
    @test size(copied) == size(source)
    @test G.modulus(copied) == G.modulus(source)
    @test CUDA.device(parent(copied)) == device
    @test Array(copied) == Float32[1 2; 3 4]
    @test all(iszero, Array(parent(copied))[3:end, :])
    @test all(iszero, Array(parent(copied))[:, 3:end])

    destination = similar(source)
    @test destination isa G.CuModArray{Float32,2}
    @test CUDA.device(parent(destination)) == device
    @test all(iszero, Array(destination))
    @test fill!(destination, 5) === destination
    @test Array(destination) == fill(5.0f0, 2, 2)
    @test G.modulus(destination) == 7
    @test copyto!(destination, Float32[6 0; 2 1]) === destination
    @test Array(destination) == Float32[6 0; 2 1]
    @test Array(parent(destination))[3, 1] == 0

    random = rand(G.CuModVector{Float32}, 33; modulus = 7)
    values = Array(random)
    @test CUDA.device(parent(random)) == device
    @test all(value -> 0 <= value < 7 && isinteger(value), values)

    one_stream_context = Ref(:unset)
    one_stream_result = ext._with_cuda_streams(parent(source), 1) do context
        one_stream_context[] = context
        ext._with_cuda_stream(context, 1) do
            :current_queue
        end
    end
    @test one_stream_context[] === nothing
    @test one_stream_result === :current_queue

    stream_result = ext._with_cuda_streams(parent(source), 2) do context
        foreach(1:2) do index
            ext._with_cuda_stream(context, index) do
                @test CUDA.stream() == context.streams[index]
            end
        end
        :streams_finished
    end
    @test stream_result === :streams_finished

    @test_throws ErrorException ext._with_cuda_streams(parent(source), 2) do context
        ext._with_cuda_stream(context, 1) do
            error("exercise CUDA batch stream cleanup")
        end
    end
    @test CUDA.device() == device
end
