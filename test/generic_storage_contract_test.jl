const STORAGE_CONTRACT_G = GPUFiniteFieldMatrices

@testset "Generic modular storage contracts" begin
    G = STORAGE_CONTRACT_G

    @test isdefined(G, :GPUModMatrix)
    @test isdefined(G, :GPUModVector)
    @test Base.isexported(G, :GPUModMatrix)
    @test Base.isexported(G, :GPUModVector)
    @test G.GPUModMatrix{Float32} <: G.GPUModArray{Float32,2}
    @test G.GPUModVector{Float32} <: G.GPUModArray{Float32,1}
    selected_backend = try
        G.default_backend()
    catch err
        @test err isa ArgumentError
        :none
    end
    if selected_backend === :none
        @test_throws ArgumentError G.GPUModMatrix{Float32}(Float32[1 2; 3 4]; modulus = 7)
    else
        generic_matrix = G.GPUModMatrix{Float32}(Float32[1 2; 3 4]; modulus = 7)
        generic_vector = G.GPUModVector(Float32[1, 2]; modulus = 7)
        @test generic_matrix isa G.GPUModMatrix{Float32}
        @test generic_vector isa G.GPUModVector{Float32}
        @test G._storage_backend(parent(generic_matrix)) === Val(selected_backend)
        @test Array(generic_matrix) == Float32[1 2; 3 4]
        @test Array(generic_vector) == Float32[1, 2]

        exact_integer = G.GPUModArray{Float32,1}(Int[2^24+1]; modulus = 7, mod = true)
        @test Array(exact_integer) == Float32[2]
    end

    raw = reshape(Float32.(1:16), 4, 4)
    backing = @view raw[2:3, 2:3]
    viewed = G.GPUModArray{Float32,2,typeof(backing)}(backing, size(backing), 97)
    @test parent(viewed) === backing
    @test Array(viewed) == Array(backing)

    padded = fill(-99.0f0, 4, 4)
    padded[1:2, 1:2] = Float32[1 3; 2 4]
    wrapped = G.GPUModArray{Float32,2,typeof(padded)}(padded, (2, 2), 7)
    @test collect(wrapped) == Float32[1 3; 2 4]
    @test wrapped[3] == 3.0f0
    @test wrapped[2, 2] == 4.0f0
    @test_throws BoundsError wrapped[5]
    @test size(wrapped, 3) == 1
    wrapped[3] = 8.0f0
    @test padded[1, 2] == 1.0f0
    @test padded[3, 1] == -99.0f0
    wrapped[2, 2] = -1.0f0
    @test padded[2, 2] == 6.0f0

    modulus_five = G.GPUModArray{Float32,1,Vector{Float32}}(Float32[1, 2], (2,), 5)
    modulus_seven = G.GPUModArray{Float32,1,Vector{Float32}}(Float32[1, 2], (2,), 7)
    @test modulus_five != modulus_seven

    backends = Symbol[]
    GFM_TEST_CUDA_LOADED && push!(backends, :cuda)
    GFM_TEST_METAL_LOADED && push!(backends, :metal)
    for backend in backends
        backend_name = backend === :cuda ? "CUDA" : "Metal"
        err = try
            G._validate_backend_storage(Val(backend), Int32, 7)
            nothing
        catch exception
            exception
        end
        @test err isa ArgumentError
        if err isa Exception
            @test occursin(backend_name, sprint(showerror, err))
            @test occursin("16777216", sprint(showerror, err))
        end
    end
    if GFM_TEST_METAL_LOADED
        @test G._backend_instance(Val(:metal)) isa Metal.MetalBackend
        @test G._backend_available(Val(:metal)) === GFM_TEST_METAL_FUNCTIONAL
        @test_throws ArgumentError G._validate_backend_storage(Val(:metal), Float64, 7)
    end

    if GFM_TEST_CUDA_LOADED
        @test G._backend_instance(Val(:cuda)) isa CUDA.CUDABackend
        @test G._backend_available(Val(:cuda)) === GFM_TEST_CUDA_FUNCTIONAL
        cuda_modulus_error = try
            G._validate_backend_storage(Val(:cuda), Float32, 2^24 + 1)
            nothing
        catch exception
            exception
        end
        @test cuda_modulus_error isa DomainError
        if cuda_modulus_error isa Exception
            @test occursin("CUDA", sprint(showerror, cuda_modulus_error))
            @test occursin("16777216", sprint(showerror, cuda_modulus_error))
        end

        # These host wrappers exercise CUDA-specific factory preflight without
        # constructing a CuArray or claiming CUDA device execution.
        fake_cuda_storage = G.CuStorage{Float32,2,Nothing,Matrix{Float32}}(
            Float32[1 3 0 0; 2 4 0 0; 0 0 0 0; 0 0 0 0],
        )
        fake_cuda =
            G.GPUModArray{Float32,2,typeof(fake_cuda_storage)}(fake_cuda_storage, (2, 2), 7)
        @test_throws DomainError G.similar_with_modulus(fake_cuda, 2^24 + 1)
        @test_throws DomainError G.change_modulus(fake_cuda, 2^24 + 1)
        before = copy(parent(fake_cuda))
        @test_throws DomainError G.change_modulus!!(fake_cuda, 2^24 + 1)
        @test parent(fake_cuda) == before

        indexed = fake_cuda[[2, 1], [1, 2]]
        row = fake_cuda[[2, 1], 1]
        column = fake_cuda[1, [2, 1]]
        @test indexed isa G.CuModMatrix{Float32}
        @test row isa G.CuModMatrix{Float32}
        @test column isa G.CuModMatrix{Float32}
        @test size(indexed) == (2, 2)
        @test size(row) == (2, 1)
        @test size(column) == (1, 2)
        @test modulus(indexed) == modulus(row) == modulus(column) == modulus(fake_cuda)
        @test Array(indexed) == Float32[2 4; 1 3]
        @test Array(row) == reshape(Float32[2, 1], 2, 1)
        @test Array(column) == Float32[3 1]
    end

    @test which(LinearAlgebra.rank, Tuple{G.GPUModMatrix{Float32}}).module === G
    if GFM_TEST_CUDA_LOADED
        @test fieldtype(G.PLUQ{Float32,typeof(fake_cuda)}, :LU) === typeof(fake_cuda)
    end

    if GFM_TEST_CUDA_LOADED && !GFM_TEST_CUDA_FUNCTIONAL
        @test_throws DomainError zeros(G.CuModMatrix{Float32}, 1, 1; modulus = 2^24 + 1)
        @test_throws DomainError zeros(G.CuModVector{Float32}, 1; modulus = 2^24 + 1)
        @test_throws DomainError rand(G.CuModMatrix{Float32}, 1, 1; modulus = 2^24 + 1)
        @test_throws DomainError rand(G.CuModVector{Float32}, 1; modulus = 2^24 + 1)
        @test_throws ArgumentError zeros(G.CuModMatrix{Int32}, 1, 1; modulus = 7)
        @test_throws DimensionMismatch zeros(
            G.CuModMatrix{Float32},
            1,
            1;
            modulus = 7,
            new_size = (2, 2),
        )
    end
end
