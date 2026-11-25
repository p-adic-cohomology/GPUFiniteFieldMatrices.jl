using Random

@testset "Backend runtime hook contracts" begin
    G = GPUFiniteFieldMatrices

    host = reshape(Float32.(1:12), 3, 4)
    host_wrappers =
        (host, view(host, 1:2, 2:4), reshape(host, 4, 3), transpose(host), adjoint(host))
    expected_backend = G.backendof(host)
    expected_device = G.deviceof(host)
    for wrapped in host_wrappers
        @test G.backendof(wrapped) == expected_backend
        @test G.deviceof(wrapped) == expected_device
        result = G.similar_storage(wrapped, Float64, (2, 3))
        @test result isa Matrix{Float64}
        @test size(result) == (2, 3)
        @test G.synchronize_backend!(wrapped) === nothing
    end

    zero_a = StorageContractArray(StorageContractBackendA(), 0)
    zero_b = StorageContractArray(StorageContractBackendB(), 0)
    @test G.deviceof(zero_a) == (StorageContractBackendA, 0)
    @test G.deviceof(zero_b) == (StorageContractBackendB, 0)
    @test G.deviceof(zero_a) != G.deviceof(zero_b)
    @test G.deviceof(StorageContractArray(StorageContractBackendA(), 1)) !=
          G.deviceof(StorageContractArray(StorageContractBackendA(), 2))

    backing = fill(-9.0f0, 8)
    destination = view(backing, 2:5)
    source = Float32[1, 2, 3, 4]
    @test G.copy_storage!(destination, source) === destination
    @test backing == Float32[-9, 1, 2, 3, 4, -9, -9, -9]
    @test G.fill_storage!(destination, 7.0f0) === destination
    @test backing == Float32[-9, 7, 7, 7, 7, -9, -9, -9]

    rng_destination = zeros(Float32, 32)
    rng = Random.MersenneTwister(13)
    @test G.rand_storage!(rng_destination, rng) === rng_destination
    @test any(value -> !iszero(value), rng_destination)
    @test G.host_scalar(3.0f0) === 3.0f0
    @test G.host_scalar(host) === host
    pinned = G.pinned_host_buffer(expected_backend, UInt8, (2, 3))
    @test pinned isa Array{UInt8,2}
    @test size(pinned) == (2, 3)

    callback_ran = Ref(false)
    result = G.with_backend_streams(host, 1) do context
        @test context === nothing
        G.with_backend_stream(context, 1) do
            callback_ran[] = true
            :same_task_queue
        end
    end
    @test result === :same_task_queue
    @test callback_ran[]

    for invalid_count in (0, -1, 2, true, big(typemax(Int)) + 1)
        callback_ran[] = false
        @test_throws ArgumentError G.with_backend_streams(host, invalid_count) do _
            callback_ran[] = true
        end
        @test !callback_ran[]
    end
    for invalid_index in (0, -1, 2, true, big(typemax(Int)) + 1)
        callback_ran[] = false
        @test_throws ArgumentError G.with_backend_stream(nothing, invalid_index) do
            callback_ran[] = true
        end
        @test !callback_ran[]
    end
end
