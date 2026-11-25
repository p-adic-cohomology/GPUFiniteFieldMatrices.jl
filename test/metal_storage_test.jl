using Random
using SparseArrays

_metal_exact(values, modulus) = map(x -> mod(BigInt(x), BigInt(modulus)), values)
_metal_exact2(f, left, right, modulus) = map(
    (x, y) -> mod(f(BigInt(x), BigInt(y)), BigInt(modulus)),
    left,
    right,
)

@testset "Metal modular wrapper and elementwise contracts" begin
    G = GPUFiniteFieldMatrices
    ext = Base.get_extension(G, :GPUFiniteFieldMatricesMetalExt)
    @test ext !== nothing
    @test G.MtlModMatrix{Float32} === G.MtlModArray{Float32,2}
    @test G.MtlModVector{Float32} === G.MtlModArray{Float32,1}
    @test G._auto_backend(true, true) === :cuda
    @test G._auto_backend(false, true) === :metal
    @test_throws ArgumentError G._auto_backend(false, false)
    @test_throws ArgumentError G._validate_backend_storage(Val(:metal), Float64, 7)
    @test_throws ArgumentError G._validate_backend_storage(Val(:metal), Int32, 7)
    @test_throws DomainError G._validate_backend_storage(Val(:metal), Float32, 2^24 + 1)

    if !Metal.functional()
        @test_throws ArgumentError G.MtlModArray{Float32,2}(
            ones(Float32, 2, 3);
            modulus = 7,
        )
    else
        device = Metal.device()
        Metal.allowscalar(false)
        try
        for mode in (Metal.SharedStorage, Metal.PrivateStorage)
            @testset "$(nameof(mode)) wrapper methods" begin
                host = Float32[1 2 3; 4 5 6]
                A = G.MtlModArray{Float32,2}(
                    host;
                    modulus = 7,
                    storage = mode,
                    device,
                )
                @test A isa G.MtlModArray{Float32,2,mode}
                @test Metal.storagemode(parent(A)) === mode
                @test Metal.device(parent(A)) == device
                @test Array(A) == _metal_exact(host, 7)
                @test all(iszero, Array(parent(A))[3:end, :])

                copied = copy(A)
                @test copied isa G.MtlModArray{Float32,2,mode}
                @test parent(copied) !== parent(A)
                @test size(copied) == size(A)
                @test G.modulus(copied) == G.modulus(A)
                @test Metal.storagemode(parent(copied)) === mode
                @test Metal.device(parent(copied)) == device
                @test Array(copied) == _metal_exact(host, 7)
                @test all(iszero, Array(parent(copied))[3:end, :])

                similar_A = similar(A)
                @test similar_A isa G.MtlModArray{Float32,2,mode}
                @test all(iszero, Array(similar_A))
                @test Metal.storagemode(parent(similar_A)) === mode
                @test Metal.device(parent(similar_A)) == device
                changed = G.similar_with_modulus(A, 11)
                @test G.modulus(changed) == 11
                @test Metal.storagemode(parent(changed)) === mode
                @test all(iszero, Array(changed))

                destination = G.similar_with_modulus(A, 7)
                identity = parent(destination)
                result = G.add!(destination, A, copied)
                @test result === destination
                @test parent(destination) === identity
                @test Array(destination) == _metal_exact2(+, host, host, 7)

                before_padding = Array(parent(destination))
                @test G.sub!(destination, A, copied) === destination
                @test Array(destination) == _metal_exact2(-, host, host, 7)
                @test Array(parent(destination))[3:end, :] == before_padding[3:end, :]
                @test G.mul_elementwise!(destination, A, copied) === destination
                @test Array(destination) == _metal_exact2(*, host, host, 7)
                @test G.elementwise_multiply!(destination, A, copied) === destination
                @test Array(destination) == _metal_exact2(*, host, host, 7)

                @test G.neg!(destination, A) === destination
                @test Array(destination) == map(x -> mod(-BigInt(x), 7), host)
                @test G.scalar_add!(destination, A, -9) === destination
                @test Array(destination) == map(x -> mod(BigInt(x) - 9, 7), host)
                @test G.scalar_sub!(destination, A, 11) === destination
                @test Array(destination) == map(x -> mod(BigInt(x) - 11, 7), host)
                @test G.rscalar_sub!(destination, A, 11) === destination
                @test Array(destination) == map(x -> mod(11 - BigInt(x), 7), host)
                @test LinearAlgebra.mul!(destination, A, 13) === destination
                @test Array(destination) == map(x -> mod(BigInt(x) * 13, 7), host)
                @test G.div!(destination, A, 3) === destination
                @test Array(destination) == map(x -> mod(BigInt(x) * invmod(3, 7), 7), host)
                @test G.add!(A, A, copied; threads = 1) === A
                @test Array(A) == _metal_exact2(+, host, host, 7)

                @test Array(A + copied) == _metal_exact2(+, host, host, 7)
                @test Array(A - copied) == _metal_exact2(-, host, host, 7)
                @test Array(-A) == map(x -> mod(-BigInt(x), 7), host)
                @test Array(A + 11) == map(x -> mod(BigInt(x) + 11, 7), host)
                @test Array(11 - A) == map(x -> mod(11 - BigInt(x), 7), host)
                @test Array(A * 13) == map(x -> mod(BigInt(x) * 13, 7), host)
                @test Array(A / 3) == map(x -> mod(BigInt(x) * invmod(3, 7), 7), host)

                fill!(destination, -1)
                @test Array(destination) == fill(6, 2, 3)
                @test Array(parent(destination))[3:end, :] == before_padding[3:end, :]
                @test copyto!(destination, Float32[6 5 4; 3 2 1]) === destination
                @test Array(destination) == Float32[6 5 4; 3 2 1]
                @test copyto!(destination, A) === destination
                @test Array(destination) == Array(A)
                @test G.mod_elements!(destination) === destination
                @test Array(destination) == Array(A)
                @test Array(G.change_modulus(A, 5)) == _metal_exact(host, 5)
                @test Array(G.change_modulus!!(copy(A), 5)) == _metal_exact(host, 5)
                @test Array(mod(A, 5)) == _metal_exact(host, 5)

                random = rand(G.MtlModVector{Float32}, 33; modulus = 7, storage = mode, device)
                random_values = Array(random)
                @test Metal.storagemode(parent(random)) === mode
                @test Metal.device(parent(random)) == device
                @test all(x -> 0 <= x < 7 && isinteger(x), random_values)
                @test all(iszero, Array(parent(random))[34:end])

                edge_modulus = 2^24
                edge_values = Float32[edge_modulus - 1, edge_modulus - 2, 1, 9]
                edge_other = Float32[edge_modulus - 2, edge_modulus - 1, 3, 7]
                edge_A = G.MtlModArray{Float32,1}(
                    edge_values;
                    modulus = edge_modulus,
                    storage = mode,
                    device,
                )
                edge_B = G.MtlModArray{Float32,1}(
                    edge_other;
                    modulus = edge_modulus,
                    storage = mode,
                    device,
                )
                edge_C = similar(edge_A)
                @test G.add!(edge_C, edge_A, edge_B) === edge_C
                @test Array(edge_C) == _metal_exact2(+, edge_values, edge_other, edge_modulus)
                @test G.mul_elementwise!(edge_C, edge_A, edge_B) === edge_C
                @test Array(edge_C) == _metal_exact2(*, edge_values, edge_other, edge_modulus)
                large_scalar = BigInt(edge_modulus) + 3
                @test G.scalar_add!(edge_C, edge_A, large_scalar) === edge_C
                @test Array(edge_C) == map(
                    x -> mod(BigInt(x) + large_scalar, BigInt(edge_modulus)),
                    edge_values,
                )
                @test G.div!(edge_C, edge_A, 3; threads = 1) === edge_C
                @test Array(edge_C) == map(
                    x -> mod(BigInt(x) * invmod(BigInt(3), BigInt(edge_modulus)), BigInt(edge_modulus)),
                    edge_values,
                )
                @test Array(edge_A + large_scalar) == map(
                    x -> mod(BigInt(x) + large_scalar, BigInt(edge_modulus)),
                    edge_values,
                )
                exact_alias = copy(edge_A)
                @test G.add!(exact_alias, exact_alias, edge_B) === exact_alias
                @test Array(exact_alias) == _metal_exact2(+, edge_values, edge_other, edge_modulus)

                modulus_one = G.MtlModArray{Float32,1}(
                    zeros(Float32, 3);
                    modulus = 1,
                    storage = mode,
                    device,
                )
                @test G.neg!(modulus_one, modulus_one) === modulus_one
                @test Array(modulus_one) == zeros(Float32, 3)
            end
        end

        @testset "Elementwise shapes, exact modular arithmetic, and bounds" begin
            shapes = ((0, 3), (1,), (1, 1), (7,), (3, 5), (31, 33), (33, 31),
                      (65, 67), (257, 255))
            for (shape_index, shape) in enumerate(shapes)
                values = reshape(Float32.(mod.(collect(0:prod(shape)-1), 97)), shape)
                other = reshape(Float32.(mod.(collect(13:12+prod(shape)), 97)), shape)
                A = G.MtlModArray{Float32,length(shape)}(
                    values;
                    modulus = 97,
                    storage = Metal.PrivateStorage,
                    device,
                )
                B = G.MtlModArray{Float32,length(shape)}(
                    other;
                    modulus = 97,
                    storage = Metal.PrivateStorage,
                    device,
                )
                C = similar(A)
                @test G.add!(C, A, B) === C
                @test Array(C) == _metal_exact2(+, values, other, 97)
                physical = Array(parent(C))
                padding_mask = trues(size(physical))
                padding_mask[ntuple(d -> 1:size(C, d), length(shape))...] .= false
                @test all(iszero, physical[padding_mask])
                if shape_index == length(shapes)
                    @test G.mul_elementwise!(C, A, B) === C
                    @test Array(C) == _metal_exact2(*, values, other, 97)
                end
            end
        end

        @testset "Padded views, aliasing, and preflight no-write behavior" begin
            raw_a = Metal.MtlArray{Float32,2,Metal.PrivateStorage}(undef, (4, 5))
            raw_b = Metal.MtlArray{Float32,2,Metal.PrivateStorage}(undef, (4, 5))
            raw_c = Metal.MtlArray{Float32,2,Metal.PrivateStorage}(undef, (4, 5))
            copyto!(raw_a, fill(91.0f0, 4, 5))
            copyto!(raw_b, fill(92.0f0, 4, 5))
            copyto!(raw_c, fill(77.0f0, 4, 5))
            logical_a = Float32[1 2 3 4; 5 6 0 1; 2 3 4 5]
            logical_b = Float32[4 3 2 1; 0 6 5 4; 3 2 1 0]
            copyto!(view(raw_a, 1:3, 1:4), logical_a)
            copyto!(view(raw_b, 1:3, 1:4), logical_b)
            A = G.GPUModArray(view(raw_a, 1:3, 1:4); modulus = 7)
            B = G.GPUModArray(view(raw_b, 1:3, 1:4); modulus = 7)
            C = G.GPUModArray(view(raw_c, 1:3, 1:4); modulus = 7)
            before = Array(raw_c)
            identity = parent(C)
            @test G.add!(C, A, B) === C
            @test parent(C) === identity
            @test Array(C) == _metal_exact2(+, logical_a, logical_b, 7)
            after = Array(raw_c)
            @test after[4, :] == before[4, :]
            @test after[:, 5] == before[:, 5]

            alias_backing = Metal.MtlArray{Float32,2,Metal.PrivateStorage}(undef, (4, 4))
            copyto!(alias_backing, reshape(Float32.(1:16), 4, 4))
            alias_C = G.GPUModArray(view(alias_backing, 1:2, 1:2); modulus = 17)
            alias_A = G.GPUModArray(view(alias_backing, 2:3, 2:3); modulus = 17)
            alias_b_native = Metal.MtlArray{Float32,2,Metal.PrivateStorage}(undef, (2, 2))
            copyto!(alias_b_native, Float32[1 1; 1 1])
            alias_B = G.GPUModArray(alias_b_native; modulus = 17)
            alias_before = Array(alias_backing)
            @test_throws ArgumentError G.add!(alias_C, alias_A, alias_B)
            @test Array(alias_backing) == alias_before

            mismatch = G.MtlModArray{Float32,2}(
                logical_b;
                modulus = 101,
                storage = Metal.PrivateStorage,
            )
            before_C = Array(parent(C))
            @test_throws G.ModulusMismatchError G.add!(C, A, mismatch)
            @test Array(parent(C)) == before_C
            wrong_shape = G.MtlModArray{Float32,2}(
                logical_b[1:2, :];
                modulus = 7,
                storage = Metal.PrivateStorage,
            )
            @test_throws DimensionMismatch G.add!(C, A, wrong_shape)
            @test Array(parent(C)) == before_C

            invalid_parent = Metal.MtlArray{Float32,2,Metal.PrivateStorage}(undef, (3, 4))
            invalid_values = zeros(Float32, 3, 4)
            invalid_values[1, 1] = NaN32
            copyto!(invalid_parent, invalid_values)
            invalid = G.GPUModArray(invalid_parent; modulus = 7, new_size = (3, 4))
            @test_throws ArgumentError G.add!(C, A, invalid)
            @test Array(parent(C)) == before_C
            @test_throws ArgumentError G.scalar_add!(C, A, Inf)
            @test Array(parent(C)) == before_C
            @test_throws LinearAlgebra.SingularException G.div!(C, A, 7)
            @test Array(parent(C)) == before_C

            @test_throws ArgumentError G.add!(C, A, B; modulus_override = 0)
            @test Array(parent(C)) == before_C
            @test_throws ArgumentError G.add!(C, A, B; threads = 0)
            @test Array(parent(C)) == before_C

            raw_backing = Metal.MtlArray{Float32,1,Metal.PrivateStorage}(undef, (8,))
            copyto!(raw_backing, Float32[1, 2, 3, 4, 5, 6, 7, 8])
            raw_C = view(raw_backing, 2:5)
            raw_A = view(raw_backing, 4:7)
            raw_before = Array(raw_backing)
            @test_throws ArgumentError G.mod!(raw_C, raw_A, 7)
            @test Array(raw_backing) == raw_before
            @test_throws ArgumentError G.rscalar_sub!(raw_C, raw_A, 3, 7)
            @test Array(raw_backing) == raw_before
            @test G.mod!(raw_backing, raw_backing, 7) === raw_backing
            @test Array(raw_backing) == _metal_exact(Float32[1, 2, 3, 4, 5, 6, 7, 8], 7)
        end

        @testset "Metal factory and backend selection" begin
            host = Float32[1 2; 3 4]
            previous = G.set_default_backend!(:metal)
            try
                generic = G.GPUModArray{Float32,2}(host; modulus = 7)
                @test generic isa G.MtlModMatrix{Float32}
                @test Metal.device(parent(generic)) == device
                @test G.default_backend() === :metal
                zeros_array = zeros(
                    G.MtlModMatrix{Float32},
                    2,
                    2;
                    modulus = 7,
                    storage = Metal.PrivateStorage,
                    device,
                )
                @test all(iszero, Array(zeros_array))
                @test Metal.storagemode(parent(zeros_array)) === Metal.PrivateStorage
            finally
                G.set_default_backend!(:auto)
            end
        end

        @testset "Host sparse and CHOLMOD copy intersections" begin
            for mode in (Metal.SharedStorage, Metal.PrivateStorage)
                original = Float32[1 0; 3 4]
                destination = G.MtlModArray{Float32,2}(
                    zeros(Float32, 2, 2);
                    modulus = 17,
                    storage = mode,
                    device,
                )
                native = parent(destination)
                physical = Array(native)
                padding_mask = trues(size(physical))
                padding_mask[1:size(destination, 1), 1:size(destination, 2)] .= false
                physical[padding_mask] .= 91.0f0
                copyto!(native, physical)
                sources = (
                    sparse(original),
                    SparseArrays.CHOLMOD.Dense(original),
                    SparseArrays.CHOLMOD.Dense(Float64.(original)),
                )

                for source in sources
                    @test copyto!(destination, source) === destination
                    @test Array(destination) == original
                    @test parent(destination) === native
                    @test G.modulus(destination) == 17
                    @test all(==(91.0f0), Array(parent(destination))[padding_mask])
                end

                saved = Array(parent(destination))
                @test_throws DimensionMismatch copyto!(
                    destination,
                    sparse(ones(Float32, 3, 2)),
                )
                @test Array(parent(destination)) == saved
            end
        end
        finally
            Metal.allowscalar(true)
        end
    end
end
