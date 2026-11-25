using Random

@testset "Metal modular storage and backend selection" begin
    G = GPUFiniteFieldMatrices
    metal_ext = Base.get_extension(G, :GPUFiniteFieldMatricesMetalExt)
    @test metal_ext !== nothing
    @test G.MtlModMatrix{Float32} === G.MtlModArray{Float32,2}
    @test G.MtlModVector{Float32} === G.MtlModArray{Float32,1}
    @test G._auto_backend(true, true) === :cuda
    @test G._auto_backend(false, true) === :metal
    @test_throws ArgumentError G._auto_backend(false, false)
    metal_precision = BackendTestSupport.precision_fixture(:metal)
    @test metal_precision.storage_types == (Float32,)
    @test only(last.(metal_precision.exact_bits)) == 24
    @test BackendTestSupport.modulus_fixture(:metal).first_unsupported_float32_modulus ==
          BigInt(2)^24 + 1

    if !Metal.functional()
        @test_throws ArgumentError G.MtlModArray{Float32,2}(
            ones(Float32, 2, 3);
            modulus = 7,
        )
    else
        host = Float32[1 2 3; 4 5 6]
        device = Metal.device()
        @info "Metal storage/factory fixture" device storage_modes=(
            Metal.SharedStorage,
            Metal.PrivateStorage,
        ) scalar_indexing=false
        typed_alias = @inferred G.MtlModMatrix{Float32}(host; modulus = 7)
        typed_zeros = @inferred zeros(G.MtlModMatrix{Float32}, 2, 3; modulus = 7)
        typed_rand = @inferred rand(G.MtlModVector{Float32}, 17; modulus = 7)
        @test isconcretetype(typeof(typed_alias))
        @test isconcretetype(typeof(typed_zeros))
        @test isconcretetype(typeof(typed_rand))
        Metal.allowscalar(false)
        try
            for mode in (Metal.SharedStorage, Metal.PrivateStorage)
                array = BackendTestSupport.modular_array(
                    :metal,
                    host;
                    modulus = 7,
                    storage = mode,
                    device,
                )
                backing = parent(array)
                @test array isa G.MtlModMatrix{Float32}
                @test isconcretetype(typeof(array))
                @test size(array) == size(host)
                @test size(backing) == (32, 32)
                @test Metal.storagemode(backing) === mode
                @test Metal.device(backing) == device
                @test G.backendof(array) == G.backendof(backing)
                @test G.deviceof(array) == G.deviceof(backing)
                @test G.deviceof(array)[1] === typeof(G.backendof(backing))

                backing_view = view(backing, 1:2, 1:3)
                storage_wrappers = (
                    backing,
                    backing_view,
                    reshape(backing, length(backing)),
                    transpose(backing),
                    adjoint(backing),
                )
                for storage_wrapper in storage_wrappers
                    @test G.backendof(storage_wrapper) == G.backendof(backing)
                    @test G.deviceof(storage_wrapper) == G.deviceof(backing)
                    similar_wrapper = G.similar_storage(storage_wrapper, Float32, (2, 3))
                    @test Metal.storagemode(similar_wrapper) === mode
                    @test Metal.device(similar_wrapper) == device
                end
                BackendTestSupport.synchronize_array!(array)
                @test Array(array) == BackendTestSupport.exact_modular_reference(host, 7)

                allocated = G.allocate_storage(
                    Metal.MetalBackend(),
                    Float32,
                    (2, 3);
                    storage = mode,
                    device,
                )
                allocated_storage =
                    G.MtlStorage{Float32,2,mode,typeof(allocated)}(allocated)
                @test G.fill_storage!(allocated_storage, 2.0f0) === allocated_storage
                copied = G.similar_storage(allocated, Float32, (2, 3))
                copied_storage = G.MtlStorage{Float32,2,mode,typeof(copied)}(copied)
                @test G.copy_storage!(copied_storage, allocated_storage) === copied_storage
                BackendTestSupport.synchronize_array!(copied_storage)
                @test G.deviceof(copied_storage) == G.deviceof(allocated)
                @test Metal.storagemode(copied) === mode
                @test Array(copied) ==
                      BackendTestSupport.exact_modular_reference(fill(2.0f0, 2, 3), 7)

                padded_destination = Metal.MtlArray{Float32,2,mode}(undef, (3, 4))
                copyto!(padded_destination, fill(8.0f0, 3, 4))
                source_region = view(allocated, 1:2, 1:3)
                destination_region = view(padded_destination, 2:3, 2:4)
                source_region_storage =
                    G.MtlStorage{Float32,2,mode,typeof(source_region)}(source_region)
                destination_region_storage =
                    G.MtlStorage{Float32,2,mode,typeof(destination_region)}(
                        destination_region,
                    )
                @test G.copy_storage!(destination_region_storage, source_region_storage) ===
                      destination_region_storage
                BackendTestSupport.synchronize_array!(destination_region_storage)
                padded_host = Array(padded_destination)
                @test padded_host[1, :] == fill(8.0f0, 4)
                @test padded_host[2:3, 1] == fill(8.0f0, 2)
                @test padded_host[2:3, 2:4] == Array(allocated)
                @test G.fill_storage!(destination_region_storage, 5.0f0) ===
                      destination_region_storage
                BackendTestSupport.synchronize_array!(destination_region_storage)
                padded_host = Array(padded_destination)
                @test padded_host[1, :] == fill(8.0f0, 4)
                @test padded_host[2:3, 1] == fill(8.0f0, 2)
                @test padded_host[2:3, 2:4] == fill(5.0f0, 2, 3)

                similar_array = G.similar_with_modulus(array, 11)
                @test similar_array isa G.MtlModArray{Float32,2,mode}
                @test isconcretetype(typeof(similar_array))
                @test Metal.storagemode(parent(similar_array)) === mode
                @test Metal.device(parent(similar_array)) == device
                @test all(iszero, Array(similar_array))

                base_similar = similar(array)
                @test base_similar isa G.MtlModArray{Float32,2,mode}
                @test isconcretetype(typeof(base_similar))
                @test Metal.storagemode(parent(base_similar)) === mode
                @test Metal.device(parent(base_similar)) == device
                @test all(iszero, Array(base_similar))

                base = Metal.MtlArray{Float32,2,mode}(undef, (4, 5))
                copyto!(
                    base,
                    Float32[1 2 3 4 5; 6 7 8 9 10; 11 12 13 14 15; 16 17 18 19 20],
                )
                wrapped = G.GPUModArray(base; modulus = 23, new_size = (3, 4))
                @test parent(wrapped) === base
                @test isconcretetype(typeof(wrapped))
                @test size(wrapped) == (3, 4)
                @test size(parent(wrapped)) == (4, 5)
                @test Array(wrapped) == Float32[1 2 3 4; 6 7 8 9; 11 12 13 14]
                @test Metal.storagemode(parent(wrapped)) === mode

                backing_view = view(base, 2:4, 2:5)
                wrapped_view = G.GPUModArray(backing_view; modulus = 23)
                @test parent(wrapped_view) === backing_view
                @test isconcretetype(typeof(wrapped_view))
                @test Array(wrapped_view) == Float32[7 8 9 10; 12 13 14 15; 17 18 19 20]
                @test G.deviceof(wrapped_view) == G.deviceof(base)

                transformed_wrappers =
                    (reshape(base, length(base)), transpose(base), adjoint(base))
                for transformed in transformed_wrappers
                    wrapped_transformed =
                        G.MtlModArray{Float32,ndims(transformed)}(transformed; modulus = 23)
                    @test parent(wrapped_transformed) === transformed
                    @test G.deviceof(wrapped_transformed) == G.deviceof(base)
                    native_transformed = metal_ext._metal_root(parent(wrapped_transformed))
                    @test Metal.storagemode(native_transformed) === mode
                    @test Metal.device(native_transformed) == device
                end

                half = Metal.MtlArray{Float16,2,mode}(undef, (3, 4))
                copyto!(half, Float16[1 2 3 4; 5 6 7 8; 9 10 11 12])
                half_view = view(half, 1:2, 2:4)
                wrapped_half = G.MtlModMatrix(half_view; modulus = 13)
                @test eltype(wrapped_half) === Float16
                @test isconcretetype(typeof(wrapped_half))
                @test parent(wrapped_half) === half_view
                @test Array(wrapped_half) == Float16[2 3 4; 6 7 8]

                zeros_array = zeros(
                    G.MtlModMatrix{Float32},
                    2,
                    3;
                    modulus = 7,
                    storage = mode,
                    device,
                )
                @test zeros_array isa G.MtlModArray{Float32,2,mode}
                @test isconcretetype(typeof(zeros_array))
                @test Metal.storagemode(parent(zeros_array)) === mode
                @test Metal.device(parent(zeros_array)) == device
                @test Array(zeros_array) ==
                      BackendTestSupport.exact_modular_reference(zeros(Float32, 2, 3), 7)
                random_array =
                    rand(G.MtlModVector{Float32}, 17; modulus = 7, storage = mode, device)
                random_host = Array(random_array)
                @test random_array isa G.MtlModVector{Float32}
                @test isconcretetype(typeof(random_array))
                @test Metal.storagemode(parent(random_array)) === mode
                @test Metal.device(parent(random_array)) == device
                @test all(x -> 0 <= x < 7, random_host)
                @test all(iszero, Array(parent(random_array))[18:end])

                random_storage = G.MtlStorage{Float32,2,mode,typeof(allocated)}(allocated)
                G.fill_storage!(random_storage, 9.0f0)
                before_unsupported_rng = Array(allocated)
                @test_throws ArgumentError G.rand_storage!(
                    random_storage,
                    Random.MersenneTwister(5),
                )
                @test Array(allocated) == before_unsupported_rng
                @test G.rand_storage!(random_storage) === random_storage
                BackendTestSupport.synchronize_array!(random_storage)
                random_values = Array(allocated)
                @test all(value -> 0.0f0 <= value < 1.0f0, random_values)

                scalar_storage = Metal.MtlArray{Float32,1,mode}(undef, (1,))
                copyto!(scalar_storage, Float32[3.25])
                @test G.host_scalar(scalar_storage) === 3.25f0
                staging = G.pinned_host_buffer(Metal.MetalBackend(), Float32, (2, 3))
                @test staging isa Array{Float32,2}
                @test size(staging) == (2, 3)
            end
        finally
            Metal.allowscalar(true)
        end

        # Private storage must use device operations; host reads below are bulk conversions.
        Metal.allowscalar(false)
        try
            private =
                G.MtlModArray{Float32,2}(host; modulus = 7, storage = Metal.PrivateStorage)
            BackendTestSupport.synchronize_array!(private)
            @test Array(private) == BackendTestSupport.exact_modular_reference(host, 7)
            @test isconcretetype(typeof(private))
            queue_callback_ran = Ref(false)
            @test G.with_backend_streams(private, 1) do stream_context
                @test stream_context === nothing
                G.with_backend_stream(stream_context, 1) do
                    queue_callback_ran[] = true
                    :current_queue
                end
            end === :current_queue
            @test queue_callback_ran[]
            queue_callback_ran[] = false
            @test_throws ArgumentError G.with_backend_streams(private, 2) do _
                queue_callback_ran[] = true
            end
            @test !queue_callback_ran[]
            @test_throws ArgumentError G.MtlModArray{Int32,2}(Int32[1 2; 3 4]; modulus = 7)
            @test_throws ArgumentError G.MtlModArray{Float64,2}(
                ones(Float64, 1, 1);
                modulus = 7,
            )
            @test_throws DomainError G.MtlModArray{Float32,2}(host; modulus = 2^24 + 1)
            @test_throws ArgumentError G.MtlModArray{Float32,2}(
                parent(private);
                modulus = 7,
                storage = Metal.SharedStorage,
            )
        finally
            Metal.allowscalar(true)
        end

        previous_backend = G.set_default_backend!(:metal)
        try
            @test G.default_backend() === :metal
            generic = G.GPUModArray{Float32,2}(host; modulus = 7)
            explicit = G.GPUModArray{Float32,2}(host; modulus = 7, backend = :metal)
            @test generic isa G.MtlModMatrix{Float32}
            @test isconcretetype(typeof(generic))
            @test explicit isa G.MtlModMatrix{Float32}
            @test isconcretetype(typeof(explicit))

            G.set_default_backend!(:cuda)
            explicit_metal_alias = G.MtlModArray{Float32,2}(host; modulus = 7)
            @test explicit_metal_alias isa G.MtlModMatrix{Float32}
            @test isconcretetype(typeof(explicit_metal_alias))
            G.set_default_backend!(:metal)

            nested = G.with_backend(:cuda) do
                @test G.default_backend() === :cuda
                G.with_backend(:metal) do
                    @test G.default_backend() === :metal
                end
                G.default_backend()
            end
            @test nested === :cuda
            @test G.default_backend() === :metal

            @sync begin
                gate = Channel{Nothing}(1)
                task = Threads.@spawn G.with_backend(:cuda) do
                    put!(gate, nothing)
                    yield()
                    G.default_backend()
                end
                take!(gate)
                @test G.default_backend() === :metal
                @test fetch(task) === :cuda
            end

            @test_throws ErrorException G.with_backend(:cuda) do
                @test G.default_backend() === :cuda
                error("scoped selection restoration")
            end
            @test G.default_backend() === :metal

            automatic = G.set_default_backend!(:auto)
            @test automatic === :auto
            @test G.default_backend() === (GFM_TEST_CUDA_FUNCTIONAL ? :cuda : :metal)
        finally
            G.set_default_backend!(:auto)
        end
    end
end
