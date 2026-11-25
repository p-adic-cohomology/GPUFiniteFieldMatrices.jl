@testset "Generic storage wrapper and CUDA extension boundary" begin
    mod = GPUFiniteFieldMatrices
    cuda_ext = Base.get_extension(mod, :GPUFiniteFieldMatricesCUDAExt)
    metal_ext = Base.get_extension(mod, :GPUFiniteFieldMatricesMetalExt)
    @test (cuda_ext !== nothing) == GFM_TEST_CUDA_LOADED
    @test (metal_ext !== nothing) == GFM_TEST_METAL_LOADED

    if GFM_TEST_CUDA_LOADED
        @testset "CUDA device keyword policy" begin
            @test cuda_ext._cuda_device_index(nothing) === nothing
            @test cuda_ext._cuda_device_index(1) == 1
            @test cuda_ext._cuda_device_index(big(7)) == 7
            @test_throws ArgumentError cuda_ext._cuda_device_index(0)
            @test_throws ArgumentError cuda_ext._cuda_device_index(-1)
            @test_throws ArgumentError cuda_ext._cuda_device_index(true)
            @test_throws ArgumentError cuda_ext._cuda_device_index(1.0)
            @test_throws ArgumentError cuda_ext._cuda_device_index("cuda:0")
            @test_throws ArgumentError cuda_ext._cuda_device_index(CUDA.DEVICE_CPU)
            @test_throws ArgumentError cuda_ext._cuda_device_index(CUDA.DEVICE_INVALID)

            # Synthetic native CuDevice handle: this only checks the zero-based
            # deviceid -> one-based KernelInterface normalization, not CUDA runtime.
            synthetic_device = CUDA.CUDACore._CuDevice(CUDA.CUDACore.CUdevice(0))
            @test cuda_ext._cuda_device_index(synthetic_device) == 1

            allocation_ran = Ref(false)
            @test_throws ArgumentError cuda_ext._with_cuda_device("cuda:0") do
                allocation_ran[] = true
            end
            @test !allocation_ran[]
            @test_throws ArgumentError mod.CuModVector{Float32}(
                Float32[1, 2];
                modulus = 7,
                device = 0,
            )
            @test_throws ArgumentError mod.CuModVector{Float32}(
                Float32[1, 2];
                modulus = 7,
                device = "cuda:0",
            )
            @test_throws ArgumentError mod.allocate_storage(
                CUDA.CUDABackend(),
                Float32,
                (1,);
                device = "cuda:0",
            )
        end

        if GFM_TEST_CUDA_FUNCTIONAL
            @testset "CUDA device-aware wrapping and allocation" begin
                devices = collect(CUDA.devices())
                @test !isempty(devices)
                target = first(devices)
                target_index = CUDA.deviceid(target) + 1
                caller = CUDA.device()
                raw_cuda = CUDA.device!(target) do
                    CUDA.CuArray(Float32[1, 2, 3])
                end

                native_wrapped = mod.CuModVector(raw_cuda; modulus = 7, device = target)
                integer_wrapped =
                    mod.CuModVector(raw_cuda; modulus = 7, device = target_index)
                @test parent(native_wrapped) === raw_cuda
                @test parent(integer_wrapped) === raw_cuda

                raw_view = @view raw_cuda[1:2]
                native_view_wrapped =
                    mod.CuModVector(raw_view; modulus = 7, device = target)
                integer_view_wrapped =
                    mod.CuModVector(raw_view; modulus = 7, device = target_index)
                @test parent(native_view_wrapped) === raw_view
                @test parent(integer_view_wrapped) === raw_view
                @test CUDA.device() == caller

                host_wrapped =
                    mod.CuModVector(Float32[1, 2, 3]; modulus = 7, device = target)
                @test CUDA.device(parent(host_wrapped)) == target
                @test CUDA.device() == caller

                @test_throws ErrorException cuda_ext._with_cuda_device(target) do
                    error("exercise CUDA device restoration after an exception")
                end
                @test CUDA.device() == caller

                factory_zero =
                    zeros(mod.CuModVector{Float32}, 3; modulus = 7, device = target_index)
                @test CUDA.device(parent(factory_zero)) == target
                @test CUDA.device() == caller

                allocated =
                    mod.allocate_storage(CUDA.CUDABackend(), Float32, (3,); device = target)
                @test CUDA.device(allocated) == target
                @test CUDA.device() == caller

                if length(devices) >= 2
                    other = devices[2]
                    other_index = CUDA.deviceid(other) + 1
                    @test_throws ArgumentError mod.CuModVector(
                        raw_cuda;
                        modulus = 7,
                        device = other,
                    )
                    @test_throws ArgumentError mod.CuModVector(
                        raw_cuda;
                        modulus = 7,
                        device = other_index,
                    )
                    @test CUDA.device() == caller
                end
            end
        end
    end

    raw = reshape(Float32[1, 2, 3, 4], 2, 2)
    wrapped = mod.GPUModArray{Float32,2,typeof(raw)}(raw, size(raw), 7)
    @test parent(wrapped) === raw
    @test size(wrapped) == size(raw)
    @test axes(wrapped) == axes(raw)
    @test mod.modulus(wrapped) == 7
    @test wrapped.data === raw
    @test wrapped.size == size(raw)
    @test Array(wrapped) == raw

    @test mod.CuModArray{Float32,2} <: mod.GPUModArray{Float32,2}
    @test mod.MtlModArray{Float32,2} <: mod.GPUModArray{Float32,2}
    @test mod.CuModMatrix{Float32} === mod.CuModArray{Float32,2}
    @test mod.MtlModMatrix{Float32} === mod.MtlModArray{Float32,2}
    if GFM_TEST_CUDA_LOADED
        @test applicable(mod.CuModMatrix{Float32}, raw)
        @test applicable(mod.CuModVector{Float32}, Float32[1, 2])
        @test applicable(mod.CuModArray{Float32,2}, raw)
        @test applicable(mod.CuModArray{Float64,2}, raw)
    end

    @test mod._modular_host_values(Float32, reshape(Int[2^24+1], 1, 1), 7, true) ==
          reshape(Float32[2], 1, 1)
    if GFM_TEST_CUDA_LOADED
        @test_throws DomainError mod.CuModMatrix{Float32}(raw; modulus = 2^24 + 1)
        @test_throws DomainError mod.CuModVector{Float64}(Float64[1]; modulus = 2^53 + 1)
    end
    if GFM_TEST_METAL_LOADED
        @test_throws ArgumentError mod._validate_backend_storage(Val(:metal), Float64, 7)
        @test_throws DomainError mod._validate_backend_storage(
            Val(:metal),
            Float32,
            2^24 + 1,
        )
    end

    if !GFM_TEST_CUDA_FUNCTIONAL && !GFM_TEST_METAL_FUNCTIONAL
        @test_throws ArgumentError mod.default_backend()
    else
        @test mod.default_backend() in GFM_TEST_FUNCTIONAL_BACKENDS
    end
    @test GPUFiniteFieldMatrices.with_backend(:metal) do
        GPUFiniteFieldMatrices.default_backend()
    end == :metal
end
