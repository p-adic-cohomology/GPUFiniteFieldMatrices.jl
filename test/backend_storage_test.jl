@testset "Generic storage wrapper and CUDA extension boundary" begin
    mod = GPUFiniteFieldMatrices
    cuda_ext = Base.get_extension(mod, :GPUFiniteFieldMatricesCUDAExt)
    metal_ext = Base.get_extension(mod, :GPUFiniteFieldMatricesMetalExt)
    @test cuda_ext !== nothing
    @test metal_ext !== nothing

    raw = reshape(Float32[1, 2, 3, 4], 2, 2)
    wrapped = mod.GPUModArray(raw; modulus = 7)
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
    @test applicable(mod.CuModMatrix{Float32}, raw)
    @test applicable(mod.CuModVector{Float32}, Float32[1, 2])
    @test applicable(mod.CuModArray{Float32,2}, raw)
    @test applicable(mod.CuModArray{Float64,2}, raw)

    @test mod._modular_host_values(Float32, reshape(Int[2^24+1], 1, 1), 7, true) ==
          reshape(Float32[2], 1, 1)
    @test_throws DomainError mod.CuModMatrix{Float32}(raw; modulus = 2^24 + 1)
    @test_throws DomainError mod.CuModVector{Float64}(Float64[1]; modulus = 2^53 + 1)
    @test_throws ArgumentError mod._validate_backend_storage(Val(:metal), Float64, 7)
    @test_throws DomainError mod._validate_backend_storage(Val(:metal), Float32, 2^24 + 1)

    if !CUDA.functional() && !Metal.functional()
        @test_throws ArgumentError mod.default_backend()
    else
        @test mod.default_backend() in (:cuda, :metal)
    end
    @test GPUFiniteFieldMatrices.with_backend(:metal) do
        GPUFiniteFieldMatrices.default_backend()
    end == :metal
end
