using Random

@testset "CUDA native runtime edge contracts" begin
    G = GPUFiniteFieldMatrices
    CUDA.allowscalar(false)
    try
        current_device = CUDA.device()
        current_stream = CUDA.stream()
        backing = CUDA.CuArray(fill(9.0f0, 4))
        selected = SubArray(backing, (2:3,))
        memory_type = typeof(backing).parameters[3]
        adapted = G.CuStorage{Float32,1,memory_type,typeof(selected)}(selected)

        @test G.fill_storage!(selected, 6.0f0) === selected
        @test Array(backing) == Float32[9, 6, 6, 9]
        @test G.copy_storage!(selected, Float32[2, 3]) === selected
        @test Array(backing) == Float32[9, 2, 3, 9]

        for unsupported_destination in (backing, selected, adapted)
            copyto!(backing, fill(9.0f0, 4))
            before = Array(backing)
            @test_throws ArgumentError G.rand_storage!(
                unsupported_destination,
                Random.MersenneTwister(19),
            )
            @test Array(backing) == before
        end

        @test G.rand_storage!(backing) === backing
        @test all(value -> 0.0f0 <= value < 1.0f0, Array(backing))
        copyto!(backing, fill(9.0f0, 4))
        @test G.rand_storage!(selected, CUDA.default_rng()) === selected
        after = Array(backing)
        @test after[1] == 9.0f0
        @test after[4] == 9.0f0
        @test all(value -> 0.0f0 <= value < 1.0f0, after[2:3])
        @test G.rand_storage!(adapted) === adapted
        @test all(value -> 0.0f0 <= value < 1.0f0, Array(backing)[2:3])

        copyto!(backing, Float32[1, 2, 3, 4])
        scalar_view = SubArray(backing, (2:2,))
        scalar_wrappers = (
            scalar_view,
            reshape(scalar_view, (1,)),
            transpose(scalar_view),
            adjoint(scalar_view),
        )
        for scalar_wrapper in scalar_wrappers
            @test G.host_scalar(scalar_wrapper) === 2.0f0
        end
        adapted_scalar_view =
            G.CuStorage{Float32,1,memory_type,typeof(scalar_view)}(scalar_view)
        @test G.host_scalar(adapted_scalar_view) === 2.0f0
        nonsingleton_view = SubArray(backing, (1:2,))
        @test_throws ArgumentError G.host_scalar(nonsingleton_view)

        # Copy failures happen inside the selected-device scope and must restore it.
        before_copy_error = Array(backing)
        @test_throws DimensionMismatch G.copy_storage!(selected, Float32[1])
        @test CUDA.device() == current_device
        @test Array(backing) == before_copy_error

        @test G.with_backend_streams(backing, 1) do context
            @test context === nothing
            G.with_backend_stream(context, 1) do
                @test CUDA.stream() == current_stream
                :current_stream
            end
        end === :current_stream

        @test_throws ErrorException G.with_backend_streams(backing, 2) do context
            G.with_backend_stream(context, 1) do
                error("CUDA stream callback cleanup")
            end
        end
        @test CUDA.device() == current_device
        @test CUDA.stream() == current_stream
    finally
        CUDA.allowscalar(true)
    end
end
