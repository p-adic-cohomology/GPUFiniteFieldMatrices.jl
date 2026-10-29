const _ELEMENTWISE_PREFLIGHT = GPUFiniteFieldMatrices

@testset "Elementwise host preflight metadata" begin
    mod = _ELEMENTWISE_PREFLIGHT
    size2 = (2, 2)
    n = 101

    @test mod._check_elementwise_metadata(size2, size2, size2, n, n, n, 12, 12, 12) ===
          nothing
    @test mod._check_elementwise_metadata(size2, size2, n, n, 12, 12) === nothing
    @test mod._check_elementwise_storage(size2, size2, 12, 12) === nothing

    @test_throws mod.CuModArraySizeMismatchException mod._check_elementwise_metadata(
        size2,
        (2, 1),
        size2,
        n,
        n,
        n,
        12,
        12,
        12,
    )
    @test_throws mod.CuModArrayModulusMismatchException mod._check_elementwise_metadata(
        size2,
        size2,
        size2,
        103,
        n,
        n,
        12,
        12,
        12,
    )
    @test_throws mod.CuModArrayModulusMismatchException mod._check_elementwise_metadata(
        size2,
        size2,
        n,
        103,
        12,
        12,
    )
    @test_throws mod.CuModArraySizeMismatchException mod._check_elementwise_metadata(
        size2,
        size2,
        size2,
        n,
        n,
        n,
        12,
        12,
        16,
    )
    @test_throws mod.CuModArraySizeMismatchException mod._check_elementwise_storage(
        size2,
        size2,
        12,
        16,
    )

    # Allocating methods require the source storage to match the package's
    # existing rounded geometry before similar/zeros can launch or clear output.
    @test mod._check_elementwise_allocating_metadata(size2, size2, n, n, 1024, 1024) ===
          nothing
    @test_throws mod.CuModArraySizeMismatchException mod._check_elementwise_allocating_metadata(
        size2,
        size2,
        n,
        n,
        12,
        12,
    )
    @test mod._check_elementwise_allocating_input(size2, 1024) === nothing
    @test_throws mod.CuModArraySizeMismatchException mod._check_elementwise_allocating_input(
        size2,
        12,
    )

    @test mod._validate_positive_modulus_override(nothing) === nothing
    @test mod._validate_positive_modulus_override(7) == 7
    @test_throws ArgumentError mod._validate_positive_modulus_override(0)
    @test_throws ArgumentError mod._validate_positive_modulus_override(-1)
    @test_throws ArgumentError (@inbounds mod._validate_positive_modulus_override(0))
    @test_throws ArgumentError (@inbounds mod._validate_positive_modulus_override(-1))
end

function _elementwise_preflight_wrap(
    values::AbstractArray{Float64,D},
    backing_dims::Dims{D},
    n,
    padding,
) where {D}
    mod = _ELEMENTWISE_PREFLIGHT
    storage = CUDA.fill(Float64(padding), backing_dims...)
    logical_ranges = ntuple(d -> 1:size(values, d), D)
    storage[logical_ranges...] .= values
    return mod.CuModArray{Float64,D}(
        storage;
        modulus = Int(n),
        mod = false,
        new_size = size(values),
    )
end

function _elementwise_preflight_arrays(;
    output_modulus = 101,
    left_modulus = 101,
    right_modulus = 101,
    output_size = (2, 2),
    output_backing = (2, 6),
    left_backing = (3, 4),
    right_backing = (4, 3),
)
    output_values = reshape(Float64.(1:prod(output_size)), output_size)
    left_values = [1.0 2.0; 3.0 4.0]
    right_values = [5.0 7.0; 6.0 8.0]
    output =
        _elementwise_preflight_wrap(output_values, output_backing, output_modulus, -3333)
    left = _elementwise_preflight_wrap(left_values, left_backing, left_modulus, -1111)
    right = _elementwise_preflight_wrap(right_values, right_backing, right_modulus, -2222)
    return output, left, right
end

function _elementwise_assert_size_no_write(call, arrays...)
    before = map(A -> Array(parent(A)), arrays)
    @test_throws GPUFiniteFieldMatrices.CuModArraySizeMismatchException call()
    for (A, snapshot) in zip(arrays, before)
        @test Array(parent(A)) == snapshot
    end
end

function _elementwise_assert_modulus_no_write(call, arrays...)
    before = map(A -> Array(parent(A)), arrays)
    @test_throws GPUFiniteFieldMatrices.CuModArrayModulusMismatchException call()
    for (A, snapshot) in zip(arrays, before)
        @test Array(parent(A)) == snapshot
    end
end

if CUDA.functional()
    @testset "Elementwise binary entrypoint preflight and no-write behavior" begin
        mod = _ELEMENTWISE_PREFLIGHT
        binary_ops = (
            ("add!", (C, A, B, m) -> mod.add!(C, A, B; modulus_override = m)),
            ("sub!", (C, A, B, m) -> mod.sub!(C, A, B; modulus_override = m)),
            (
                "mul_elementwise!",
                (C, A, B, m) -> mod.mul_elementwise!(C, A, B; modulus_override = m),
            ),
            (
                "elementwise_multiply!",
                (C, A, B, m) ->
                    mod.elementwise_multiply!(C, A, B; modulus_override = m),
            ),
        )

        for (name, operation) in binary_ops
            @testset "$name" begin
                C, A, B = _elementwise_preflight_arrays(output_size = (2, 1))
                _elementwise_assert_size_no_write(() -> operation(C, A, B, 7), C, A, B)

                C, A, B = _elementwise_preflight_arrays(output_modulus = 103)
                _elementwise_assert_modulus_no_write(() -> operation(C, A, B, 7), C, A, B)

                C, A, B = _elementwise_preflight_arrays(right_modulus = 103)
                _elementwise_assert_modulus_no_write(() -> operation(C, A, B, 7), C, A, B)

                C, A, B = _elementwise_preflight_arrays(right_backing = (4, 4))
                _elementwise_assert_size_no_write(() -> operation(C, A, B, 7), C, A, B)
            end
        end
    end

    @testset "Elementwise scalar and unary entrypoint preflight and no-write behavior" begin
        mod = _ELEMENTWISE_PREFLIGHT
        unary_ops = (
            ("scalar_add!", (C, A, m) -> mod.scalar_add!(C, A, 5; modulus_override = m)),
            ("scalar_sub!", (C, A, m) -> mod.scalar_sub!(C, A, 5; modulus_override = m)),
            ("rscalar_sub!", (C, A, m) -> mod.rscalar_sub!(C, A, 5; modulus_override = m)),
            (
                "mul!(scalar)",
                (C, A, m) -> LinearAlgebra.mul!(C, A, 5; modulus_override = m),
            ),
            ("div!", (C, A, m) -> mod.div!(C, A, 2; modulus_override = m)),
            ("neg!", (C, A, m) -> mod.neg!(C, A; modulus_override = m)),
        )

        for (name, operation) in unary_ops
            @testset "$name" begin
                C, A, _ = _elementwise_preflight_arrays(output_size = (2, 1))
                _elementwise_assert_size_no_write(() -> operation(C, A, 7), C, A)

                C, A, _ = _elementwise_preflight_arrays(output_modulus = 103)
                _elementwise_assert_modulus_no_write(() -> operation(C, A, 7), C, A)

                C, A, _ = _elementwise_preflight_arrays(left_modulus = 103)
                _elementwise_assert_modulus_no_write(() -> operation(C, A, 7), C, A)

                C, A, _ = _elementwise_preflight_arrays(left_backing = (4, 4))
                _elementwise_assert_size_no_write(() -> operation(C, A, 7), C, A)
            end
        end
    end

    @testset "Explicit mod! and allocating elementwise routes" begin
        mod = _ELEMENTWISE_PREFLIGHT
        C, A, _ = _elementwise_preflight_arrays(output_modulus = 103)
        @test mod.mod!(C, A, 13) === C
        @test Array(C) == Float64.(Base.mod.(Array(A), 13))
        @test mod.modulus(C) == 103

        C, A, B = _elementwise_preflight_arrays(output_size = (2, 1))
        _elementwise_assert_size_no_write(() -> mod.mod!(C, A, 13), C, A)
        C, A, B = _elementwise_preflight_arrays(output_backing = (4, 4))
        _elementwise_assert_size_no_write(() -> mod.mod!(C, A, 13), C, A)

        # Allocating arithmetic uses its current standard rounded output size.
        # These valid logical views have only 12 elements of backing storage.
        C, A, B = _elementwise_preflight_arrays()
        before_a, before_b = Array(parent(A)), Array(parent(B))
        @test_throws mod.CuModArraySizeMismatchException A + B
        @test_throws mod.CuModArraySizeMismatchException A - B
        @test_throws mod.CuModArraySizeMismatchException A * 5
        @test_throws mod.CuModArraySizeMismatchException 5 * A
        @test_throws mod.CuModArraySizeMismatchException A / 2
        @test Array(parent(A)) == before_a
        @test Array(parent(B)) == before_b
        @test_throws mod.CuModArraySizeMismatchException A + 3
        @test_throws mod.CuModArraySizeMismatchException 3 + A
        @test_throws mod.CuModArraySizeMismatchException A - 3
        @test_throws mod.CuModArraySizeMismatchException 3 - A
        @test_throws mod.CuModArraySizeMismatchException mod.mod(A, 13)

        wrong_modulus = _elementwise_preflight_wrap([5.0 7.0; 6.0 8.0], (3, 4), 103, -2222)
        @test_throws mod.CuModArrayModulusMismatchException A + wrong_modulus

        # A canonical source geometry still allocates and computes normally.
        values_a = [1.0 2.0; 3.0 4.0]
        values_b = [5.0 7.0; 6.0 8.0]
        canonical_a = mod.CuModMatrix{Float64}(values_a; modulus = 101)
        canonical_b = mod.CuModMatrix{Float64}(values_b; modulus = 101)
        @test Array(canonical_a + canonical_b) == Base.mod.(values_a .+ values_b, 101)
        @test Array(canonical_a - canonical_b) == Base.mod.(values_a .- values_b, 101)
        @test Array(canonical_a * 5) == Base.mod.(values_a .* 5, 101)
        @test Array(canonical_a / 2) == Base.mod.(values_a .* invmod(2, 101), 101)
    end

    @testset "Override validity is unconditional and preserves zero-work behavior" begin
        mod = _ELEMENTWISE_PREFLIGHT
        empty_data = CUDA.CuArray(Float64[])
        empty = mod.CuModVector{Float64}(
            empty_data;
            modulus = 101,
            mod = false,
            new_size = (0,),
        )
        @test_throws ArgumentError mod.add!(empty, empty, empty; modulus_override = 0)
        @test_throws ArgumentError (@inbounds mod.add!(
            empty,
            empty,
            empty;
            modulus_override = -1,
        ))
        @test isempty(parent(empty))
    end
end
