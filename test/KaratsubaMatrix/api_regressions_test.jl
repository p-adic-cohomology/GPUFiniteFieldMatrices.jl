function test_karatsuba_api_regressions()
    @testset "Karatsuba required GPU API regressions" begin
        mod = GPUFiniteFieldMatrices
        q = 100
        a_values = Float64[20 30; 40 50]
        b_values = Float64[2 1; 0 3]
        a = mod.CuModMatrix{Float64}(a_values; modulus = q)
        @test mod.modulus(a) == q
        @test size(a) == size(a_values)

        exact_input_bound = 2^25 + 1
        exact_modulus = 2^53 - 2^50
        exact_a_values = Float64[1 2 3; 4 5 6]
        exact_b_values = Float64[2 1; 3 4; 5 6]
        exact_a = mod.CuModMatrix{Float64}(exact_a_values; modulus = 2^52)
        exact_b = mod.CuModMatrix{Float64}(exact_b_values; modulus = 2^52)
        exact_c = mod.CuModMatrix{Float64}(fill(9.0, 2, 2); modulus = 37)
        exact_c_before = Array(parent(exact_c))
        exact_budget =
            mod.find_max_stripe_ops(Float64, exact_input_bound; modulus = exact_modulus)
        @test exact_budget == 1
        @test_throws ArgumentError LinearAlgebra.mul!(
            exact_c,
            exact_a,
            exact_b;
            input_bound = exact_input_bound,
            modulus_override = exact_modulus,
            max_ops = exact_budget + 1,
        )
        @test Array(parent(exact_c)) == exact_c_before
        LinearAlgebra.mul!(
            exact_c,
            exact_a,
            exact_b;
            input_bound = exact_input_bound,
            modulus_override = exact_modulus,
            max_ops = exact_budget,
        )
        exact_expected = Float64.(
            Base.mod.(
                BigInt.(exact_a_values) * BigInt.(exact_b_values),
                BigInt(exact_modulus),
            ),
        )
        @test Array(exact_c) == exact_expected
        @test mod.modulus(exact_c) == 37

        # Keep the public logical-size wrapper contract over padded CUDA storage.
        backing = CUDA.CuArray(zeros(Float64, 5, 6))
        wrapped_backing = mod.CuModMatrix{Float64}(backing; modulus = q, new_size = (2, 3))
        @test Base.parent(wrapped_backing) === backing
        @test size(wrapped_backing) == (2, 3)
        @test size(Base.parent(wrapped_backing)) == (5, 6)
        @test mod.modulus(wrapped_backing) == q

        explicit = mod.to_karatsuba(a; modulus = q)
        automatic = mod.to_karatsuba(a)
        @test Array(mod.from_karatsuba(Float64, explicit)) == a_values
        @test Array(mod.from_karatsuba(Float64, automatic)) == a_values

        typed_explicit = mod.to_karatsuba(Float64, a; modulus = q)
        typed_automatic = mod.to_karatsuba(Float64, a)
        @test Array(mod.from_karatsuba(Float64, typed_explicit)) == a_values
        @test Array(mod.from_karatsuba(Float64, typed_automatic)) == a_values

        different_source_values = Float32[20 30; 40 50]
        different_source = mod.CuModMatrix{Float32}(different_source_values; modulus = q)
        typed_different_source = mod.to_karatsuba(Float64, different_source; modulus = q)
        @test low_part(typed_different_source) isa mod.CuModMatrix{Float64}
        @test high_part(typed_different_source) isa mod.CuModMatrix{Float64}
        @test parent(low_part(typed_different_source)) !== parent(different_source)
        @test parent(high_part(typed_different_source)) !== parent(different_source)
        typed_different_roundtrip = mod.from_karatsuba(Float64, typed_different_source)
        @test typed_different_roundtrip isa CUDA.CuArray{Float64,2}
        @test Array(typed_different_roundtrip) == Float64.(different_source_values)

        gpu_low = CUDA.CuArray(Int64[1 0])
        gpu_high = CUDA.CuArray(Int64[37 0])
        gpu_wrapped = mod.KaratsubaMatrix(
            gpu_low,
            gpu_high;
            low_modulus = 2,
            high_modulus = 7,
            modulus = 19,
        )
        gpu_reconstructed = @inferred mod.from_karatsuba(Int64, gpu_wrapped)
        @test gpu_reconstructed isa CUDA.CuArray{Int64,2}
        @test Array(gpu_reconstructed) == Int64[75 0]

        gpu_overflow = mod.KaratsubaVector(
            CUDA.CuArray(Int64[0]),
            CUDA.CuArray(Int64[typemax(Int64)]);
            low_modulus = 2,
            high_modulus = 7,
            modulus = 19,
        )
        @test_throws InexactError mod.from_karatsuba(Int64, gpu_overflow)

        gpu_float_exact = mod.KaratsubaVector(
            CUDA.CuArray(Float64[0]),
            CUDA.CuArray(Float64[2^52]);
            low_modulus = 2,
            high_modulus = 7,
            modulus = 19,
        )
        gpu_float_result = mod.from_karatsuba(Float64, gpu_float_exact)
        @test gpu_float_result isa CUDA.CuArray{Float64,1}
        @test Array(gpu_float_result) == Float64[2^53]

        gpu_float_inexact = mod.KaratsubaVector(
            CUDA.CuArray(Float64[1]),
            CUDA.CuArray(Float64[2^53]);
            low_modulus = 2,
            high_modulus = 7,
            modulus = 19,
        )
        @test_throws InexactError mod.from_karatsuba(Float64, gpu_float_inexact)

        gpu_component_inexact = mod.KaratsubaVector(
            CUDA.CuArray(Float64[0]),
            CUDA.CuArray(Float64[2^24+1]);
            low_modulus = 2,
            high_modulus = 7,
            modulus = 19,
        )
        @test_throws InexactError mod.from_karatsuba(Float32, gpu_component_inexact)

        # Legacy copy recomputes M as N1*N2. Keep the square fixture coherent
        # with that established behavior while checking an independent oracle.
        zero_component() = mod.CuModMatrix{Float64}(zeros(Float64, 2, 2); modulus = q)
        ak = mod.KaratsubaMatrix(
            a,
            zero_component();
            low_modulus = q,
            high_modulus = q,
            modulus = q^2,
        )
        bk = mod.KaratsubaMatrix(
            mod.CuModMatrix{Float64}(b_values; modulus = q),
            zero_component();
            low_modulus = q,
            high_modulus = q,
            modulus = q^2,
        )
        ck = mod.KaratsubaMatrix(
            zero_component(),
            zero_component();
            low_modulus = q,
            high_modulus = q,
            modulus = q^2,
        )
        @test (@inferred mod.initialize_plan!(ak)) === ak
        foreach(mod.initialize_plan!, (bk, ck))

        ak_copy = copy(ak)
        @test ak_copy.plan !== nothing
        @test ak_copy.plan !== ak.plan
        @test parent(ak_copy.plan) !== parent(ak.plan)
        square_result = LinearAlgebra.mul!(ck, ak_copy, ak_copy)
        @test square_result === ck
        square_expected =
            Float64.(Base.mod.(BigInt.(a_values) * BigInt.(a_values), BigInt(q)^2))
        @test Array(ck) == square_expected

        # The output radix is independent from the common input radix, and M
        # remains metadata rather than the reduction modulus.
        discriminator_a = mod.KaratsubaMatrix(
            mod.CuModMatrix{Float64}(reshape([2.0], 1, 1); modulus = 3),
            mod.CuModMatrix{Float64}(reshape([4.0], 1, 1); modulus = 5);
            low_modulus = 3,
            high_modulus = 5,
            modulus = 7,
        )
        discriminator_b = mod.KaratsubaMatrix(
            mod.CuModMatrix{Float64}(reshape([1.0], 1, 1); modulus = 3),
            mod.CuModMatrix{Float64}(reshape([2.0], 1, 1); modulus = 3);
            low_modulus = 3,
            high_modulus = 3,
            modulus = 7,
        )
        discriminator_c_small = mod.KaratsubaMatrix(
            mod.CuModMatrix{Float64}(zeros(Float64, 1, 1); modulus = 2),
            mod.CuModMatrix{Float64}(zeros(Float64, 1, 1); modulus = 5);
            low_modulus = 2,
            high_modulus = 5,
            modulus = 7,
        )
        discriminator_c_large = mod.KaratsubaMatrix(
            mod.CuModMatrix{Float64}(zeros(Float64, 1, 1); modulus = 5),
            mod.CuModMatrix{Float64}(zeros(Float64, 1, 1); modulus = 7);
            low_modulus = 5,
            high_modulus = 7,
            modulus = 7,
        )
        foreach(
            mod.initialize_plan!,
            (
                discriminator_a,
                discriminator_b,
                discriminator_c_small,
                discriminator_c_large,
            ),
        )
        @test LinearAlgebra.mul!(
            discriminator_c_small,
            discriminator_a,
            discriminator_b,
        ) === discriminator_c_small
        @test (
            Array(mod.low_part(discriminator_c_small))[1],
            Array(mod.high_part(discriminator_c_small))[1],
        ) == (0.0, 4.0)
        @test Array(mod.from_karatsuba(Float64, discriminator_c_small)) ==
              reshape([8.0], 1, 1)
        @test mod.modulus(discriminator_c_small) == 7
        @test LinearAlgebra.mul!(
            discriminator_c_large,
            discriminator_a,
            discriminator_b,
        ) === discriminator_c_large
        @test (
            Array(mod.low_part(discriminator_c_large))[1],
            Array(mod.high_part(discriminator_c_large))[1],
        ) == (3.0, 5.0)
        @test Array(mod.from_karatsuba(Float64, discriminator_c_large)) ==
              reshape([28.0], 1, 1)

        rect_a_values = Float64[1 2 3; 4 5 6]
        rect_b_values = Float64[2 1 0 3; 1 2 3 4; 0 1 2 3]
        rect_a = mod.CuModMatrix{Float64}(rect_a_values; modulus = q)
        rect_b = mod.CuModMatrix{Float64}(rect_b_values; modulus = q)
        rect_ak = mod.to_karatsuba(rect_a; modulus = q)
        rect_bk = mod.to_karatsuba(rect_b; modulus = q)
        rect_output = mod.CuModMatrix{Float64}(zeros(Float64, 2, 4); modulus = q)
        rect_ck = mod.to_karatsuba(rect_output; modulus = q)
        foreach(mod.initialize_plan!, (rect_ak, rect_bk, rect_ck))

        rectangular_result = LinearAlgebra.mul!(rect_ck, rect_ak, rect_bk)
        @test rectangular_result === rect_ck
        @test size(rect_ck) == (2, 4)
        rectangular_expected = Float64.(
            Base.mod.(BigInt.(rect_a_values) * BigInt.(rect_b_values), BigInt(q)^2),
        )
        @test Array(rect_ck) == rectangular_expected

        # Three-product GEMM with nonzero high planes, different high bounds,
        # rectangular logical and physical shapes, and M metadata distinct
        # from the output component capacity.
        radix = 2^26 - 3
        a_low_values = Float64[radix-1 radix-2 1; radix-3 radix-1 5]
        a_high_values = Float64[1 0 1; 0 1 1]
        b_low_values =
            Float64[radix-1 2 radix-3 1; 5 radix-2 4 radix-4; radix-5 0 radix-1 2]
        b_high_values = Float64[2 1 0 2; 1 2 0 1; 2 1 2 2]
        function padded_matrix(values, rows, cols, N; sentinel = 0.0)
            data = fill(Float64(sentinel), rows, cols)
            data[1:size(values, 1), 1:size(values, 2)] .= values
            return mod.CuModMatrix{Float64}(
                CUDA.CuArray(data);
                modulus = N,
                mod = false,
                new_size = size(values),
            )
        end
        function padding_values(matrix)
            data = Array(parent(matrix))
            rows, cols = size(matrix)
            mask = trues(size(data))
            mask[1:rows, 1:cols] .= false
            return data[mask]
        end

        a_low = padded_matrix(a_low_values, 5, 7, radix; sentinel = 901.0)
        a_high = padded_matrix(a_high_values, 4, 6, 2; sentinel = 902.0)
        b_low = padded_matrix(b_low_values, 6, 8, radix; sentinel = 903.0)
        b_high = padded_matrix(b_high_values, 7, 5, 3; sentinel = 904.0)
        c_low = padded_matrix(zeros(Float64, 2, 4), 4, 7, 7; sentinel = 911.0)
        c_high = padded_matrix(zeros(Float64, 2, 4), 5, 9, 5; sentinel = 912.0)
        ak = mod.KaratsubaMatrix(
            a_low,
            a_high;
            low_modulus = radix,
            high_modulus = 2,
            modulus = 211,
        )
        bk = mod.KaratsubaMatrix(
            b_low,
            b_high;
            low_modulus = radix,
            high_modulus = 3,
            modulus = 211,
        )
        ck = mod.KaratsubaMatrix(
            c_low,
            c_high;
            low_modulus = 7,
            high_modulus = 5,
            modulus = 211,
        )
        foreach(mod.initialize_plan!, (ak, bk, ck))
        ak.plan = padded_matrix(zeros(Float64, size(ak)...), 5, 8, radix; sentinel = 905.0)
        bk.plan = padded_matrix(zeros(Float64, size(bk)...), 7, 9, radix; sentinel = 906.0)
        ck.plan = padded_matrix(zeros(Float64, size(ck)...), 4, 9, 7; sentinel = 913.0)
        @test size(parent(a_low)) != size(parent(b_low))
        @test size(parent(b_low)) != size(parent(c_low))
        @test mod.low_modulus(ak) != mod.high_modulus(ak)
        @test mod.low_modulus(bk) != mod.high_modulus(bk)
        product_plan = mod._karatsuba_matmul_product_plan(
            mod.low_modulus(ak),
            mod.high_modulus(ak),
            mod.low_modulus(bk),
            mod.high_modulus(bk),
            mod.low_modulus(ck),
            mod.high_modulus(ck),
        )
        @test product_plan.low.max_ops == 1
        @test product_plan.middle.max_ops == 1
        @test product_plan.output_capacity == 35
        @test product_plan.low.modulus == 35
        @test product_plan.middle.modulus == 35
        @test product_plan.high.modulus == 35
        @test size(mod.low_part(ak), 2) > product_plan.low.max_ops
        @test size(mod.low_part(ak), 2) > product_plan.middle.max_ops

        inputs_before = (
            Array(parent(mod.low_part(ak))),
            Array(parent(mod.high_part(ak))),
            Array(parent(mod.low_part(bk))),
            Array(parent(mod.high_part(bk))),
        )
        padding_sentinels = (
            (a_low, 901.0),
            (a_high, 902.0),
            (ak.plan, 905.0),
            (b_low, 903.0),
            (b_high, 904.0),
            (bk.plan, 906.0),
            (c_low, 911.0),
            (c_high, 912.0),
            (ck.plan, 913.0),
        )
        @test all(plane -> all(==(plane[2]), padding_values(plane[1])), padding_sentinels)
        a_reconstructed = BigInt.(a_low_values) .+ BigInt(radix) .* BigInt.(a_high_values)
        b_reconstructed = BigInt.(b_low_values) .+ BigInt(radix) .* BigInt.(b_high_values)
        product_modulus = BigInt(mod.low_modulus(ck)) * BigInt(mod.high_modulus(ck))
        product_expected = Base.mod.(a_reconstructed * b_reconstructed, product_modulus)
        metadata_modulus_expected =
            Base.mod.(a_reconstructed * b_reconstructed, BigInt(mod.modulus(ck)))
        @test product_modulus == BigInt(7) * 5
        @test product_modulus != BigInt(mod.modulus(ck))
        @test product_expected != metadata_modulus_expected

        @test LinearAlgebra.mul!(ck, ak, bk) === ck
        @test Array(ck) == product_expected
        @test mod.modulus(ck) == 211
        @test mod.low_modulus(ck) == 7
        @test mod.high_modulus(ck) == 5
        @test BigInt.(Array(mod.low_part(ck))) ==
              Base.mod.(a_reconstructed * b_reconstructed, BigInt(7))
        @test BigInt.(Array(mod.high_part(ck))) == div.(product_expected, BigInt(7))
        @test Array(parent(mod.low_part(ak))) == inputs_before[1]
        @test Array(parent(mod.high_part(ak))) == inputs_before[2]
        @test Array(parent(mod.low_part(bk))) == inputs_before[3]
        @test Array(parent(mod.high_part(bk))) == inputs_before[4]
        @test all(plane -> all(==(plane[2]), padding_values(plane[1])), padding_sentinels)

        mismatched_radix = mod.KaratsubaMatrix(
            mod.CuModMatrix{Float64}(a_low_values; modulus = 5),
            mod.CuModMatrix{Float64}(a_high_values; modulus = 2);
            low_modulus = 5,
            high_modulus = 2,
            modulus = 211,
        )
        mod.initialize_plan!(mismatched_radix)
        storage_snapshot =
            K -> (
                Array(parent(mod.low_part(K))),
                Array(parent(mod.high_part(K))),
                Array(parent(K.plan)),
            )
        mismatch_before =
            (storage_snapshot(mismatched_radix), storage_snapshot(bk), storage_snapshot(ck))
        @test_throws ArgumentError LinearAlgebra.mul!(ck, mismatched_radix, bk)
        mismatch_after =
            (storage_snapshot(mismatched_radix), storage_snapshot(bk), storage_snapshot(ck))
        @test mismatch_after == mismatch_before

        # Build direct wrappers over known CuArray parents. CuModMatrix's GPU
        # wrapper constructor preserves the exact parent, allowing these
        # fixtures to prove that each intended alias reaches the storage guard.
        function alias_fixture_storage(seed, rows, cols; logical_value = 1.0)
            physical = fill(Float64(seed), rows + 2, cols + 3)
            physical[1:rows, 1:cols] .= Float64(logical_value)
            return CUDA.CuArray(physical)
        end

        function alias_fixture_plane(storage, plane_modulus, rows, cols)
            plane = mod.CuModMatrix{Float64}(
                storage;
                modulus = plane_modulus,
                mod = false,
                new_size = (rows, cols),
            )
            @test parent(plane) === storage
            return plane
        end

        function alias_fixture_matrix(
            rows,
            cols;
            low_modulus,
            high_modulus,
            seed,
            low_storage = nothing,
            high_storage = nothing,
        )
            low_parent =
                low_storage === nothing ?
                alias_fixture_storage(seed, rows, cols; logical_value = 1.0) : low_storage
            high_parent =
                high_storage === nothing ?
                alias_fixture_storage(seed + 10, rows, cols; logical_value = 1.0) :
                high_storage
            low = alias_fixture_plane(low_parent, low_modulus, rows, cols)
            high = alias_fixture_plane(high_parent, high_modulus, rows, cols)
            matrix = mod.KaratsubaMatrix(low, high; low_modulus, high_modulus, modulus = 37)
            mod.initialize_plan!(matrix)
            plan_parent = parent(matrix.plan)
            plan_values = fill(Float64(seed + 20), size(plan_parent))
            plan_values[1:rows, 1:cols] .= 2.0
            copyto!(plan_parent, plan_values)
            @test size(matrix.plan) == (rows, cols)
            return matrix
        end

        function alias_fixture_snapshot(matrix)
            return (
                Array(parent(mod.low_part(matrix))),
                Array(parent(mod.high_part(matrix))),
                Array(parent(matrix.plan)),
            )
        end

        function pad_initialized_plan!(matrix, seed)
            rows, cols = size(matrix)
            storage = alias_fixture_storage(seed, rows, cols; logical_value = 2.0)
            matrix.plan = alias_fixture_plane(storage, mod.low_modulus(matrix), rows, cols)
            return matrix
        end

        function assert_alias_rejection_without_writes(A, B, C, expected_message)
            @test A !== B
            @test A !== C
            @test B !== C
            before = (
                alias_fixture_snapshot(A),
                alias_fixture_snapshot(B),
                alias_fixture_snapshot(C),
            )
            @test all(
                plane -> all(value -> !iszero(value), plane),
                Iterators.flatten(before),
            )
            caught = try
                LinearAlgebra.mul!(C, A, B)
                nothing
            catch error
                error
            end
            @test caught isa ArgumentError
            @test caught isa ArgumentError && occursin(expected_message, caught.msg)
            after = (
                alias_fixture_snapshot(A),
                alias_fixture_snapshot(B),
                alias_fixture_snapshot(C),
            )
            @test after == before
            return nothing
        end

        @testset "Karatsuba matrix alias rejection has no writes" begin
            # Output/input: C.low directly wraps A.low's parent.
            alias_a =
                alias_fixture_matrix(2, 2; low_modulus = 13, high_modulus = 3, seed = 101)
            alias_b =
                alias_fixture_matrix(2, 2; low_modulus = 13, high_modulus = 5, seed = 201)
            alias_c = alias_fixture_matrix(
                2,
                2;
                low_modulus = 7,
                high_modulus = 5,
                seed = 301,
                low_storage = parent(mod.low_part(alias_a)),
            )
            @test parent(mod.low_part(alias_c)) === parent(mod.low_part(alias_a))
            @test mod._stripe_mightalias(
                parent(mod.low_part(alias_c)),
                parent(mod.low_part(alias_a)),
            )
            assert_alias_rejection_without_writes(
                alias_a,
                alias_b,
                alias_c,
                "must not alias input components",
            )

            # Output/output: C.low and C.high wrap one shared CuArray parent.
            output_shared = alias_fixture_storage(401, 2, 2; logical_value = 1.0)
            alias_a =
                alias_fixture_matrix(2, 2; low_modulus = 13, high_modulus = 3, seed = 411)
            alias_b =
                alias_fixture_matrix(2, 2; low_modulus = 13, high_modulus = 5, seed = 421)
            alias_c = alias_fixture_matrix(
                2,
                2;
                low_modulus = 7,
                high_modulus = 5,
                seed = 431,
                low_storage = output_shared,
                high_storage = output_shared,
            )
            @test parent(mod.low_part(alias_c)) === output_shared
            @test parent(mod.high_part(alias_c)) === output_shared
            @test mod._stripe_mightalias(
                parent(mod.low_part(alias_c)),
                parent(mod.high_part(alias_c)),
            )
            assert_alias_rejection_without_writes(
                alias_a,
                alias_b,
                alias_c,
                "output components must not alias each other",
            )

            # Scratch/input: A.plan directly wraps A.low's parent.
            alias_a =
                alias_fixture_matrix(2, 2; low_modulus = 13, high_modulus = 3, seed = 501)
            alias_b =
                alias_fixture_matrix(2, 2; low_modulus = 13, high_modulus = 5, seed = 511)
            alias_c =
                alias_fixture_matrix(2, 2; low_modulus = 7, high_modulus = 5, seed = 521)
            alias_a.plan = alias_fixture_plane(parent(mod.low_part(alias_a)), 13, 2, 2)
            @test parent(alias_a.plan) === parent(mod.low_part(alias_a))
            @test mod._stripe_mightalias(
                parent(alias_a.plan),
                parent(mod.low_part(alias_a)),
            )
            assert_alias_rejection_without_writes(
                alias_a,
                alias_b,
                alias_c,
                "scratch must not alias input components",
            )

            # Scratch/output: A.plan directly wraps C.low's parent.
            alias_a =
                alias_fixture_matrix(2, 2; low_modulus = 13, high_modulus = 3, seed = 601)
            alias_b =
                alias_fixture_matrix(2, 2; low_modulus = 13, high_modulus = 5, seed = 611)
            alias_c =
                alias_fixture_matrix(2, 2; low_modulus = 7, high_modulus = 5, seed = 621)
            alias_a.plan = alias_fixture_plane(parent(mod.low_part(alias_c)), 13, 2, 2)
            @test parent(alias_a.plan) === parent(mod.low_part(alias_c))
            @test mod._stripe_mightalias(
                parent(alias_a.plan),
                parent(mod.low_part(alias_c)),
            )
            assert_alias_rejection_without_writes(
                alias_a,
                alias_b,
                alias_c,
                "scratch must not alias output components",
            )

            # Distinct input objects sharing a plan remain rejected; the
            # positive A === B square regression above is preserved.
            alias_a =
                alias_fixture_matrix(2, 2; low_modulus = 13, high_modulus = 3, seed = 701)
            alias_b =
                alias_fixture_matrix(2, 2; low_modulus = 13, high_modulus = 5, seed = 711)
            alias_c =
                alias_fixture_matrix(2, 2; low_modulus = 7, high_modulus = 5, seed = 721)
            alias_b.plan = alias_fixture_plane(parent(alias_a.plan), 13, 2, 2)
            @test alias_a !== alias_b
            @test parent(alias_a.plan) === parent(alias_b.plan)
            @test mod._stripe_mightalias(parent(alias_a.plan), parent(alias_b.plan))
            assert_alias_rejection_without_writes(
                alias_a,
                alias_b,
                alias_c,
                "input plan scratch must not alias each other",
            )
        end

        @testset "Karatsuba matrix invalid shapes have no writes" begin
            # The public GEMM inner-dimension guard must reject before any
            # component or initialized, nonzero scratch parent is modified.
            inner_a =
                alias_fixture_matrix(2, 3; low_modulus = 13, high_modulus = 3, seed = 801)
            inner_b =
                alias_fixture_matrix(2, 2; low_modulus = 13, high_modulus = 5, seed = 811)
            inner_c =
                alias_fixture_matrix(2, 2; low_modulus = 7, high_modulus = 5, seed = 821)
            @test size(inner_a) == (2, 3)
            @test size(inner_b) == (2, 2)
            @test size(inner_c) == (2, 2)
            inner_before = (
                alias_fixture_snapshot(inner_a),
                alias_fixture_snapshot(inner_b),
                alias_fixture_snapshot(inner_c),
            )
            @test all(
                plane -> all(value -> !iszero(value), plane),
                Iterators.flatten(inner_before),
            )
            inner_error = try
                LinearAlgebra.mul!(inner_c, inner_a, inner_b)
                nothing
            catch error
                error
            end
            @test inner_error isa ErrorException
            @test inner_error isa ErrorException && occursin(
                "Matrix dimensions don't work for multiplication",
                inner_error.msg,
            )
            inner_after = (
                alias_fixture_snapshot(inner_a),
                alias_fixture_snapshot(inner_b),
                alias_fixture_snapshot(inner_c),
            )
            @test inner_after == inner_before

            # Each wrong logical C shape isolates one destination axis and
            # reaches the stripe guard after the plans have been initialized.
            destination_a =
                alias_fixture_matrix(2, 2; low_modulus = 13, high_modulus = 3, seed = 901)
            destination_b =
                alias_fixture_matrix(2, 2; low_modulus = 13, high_modulus = 5, seed = 911)
            pad_initialized_plan!(destination_a, 941)
            pad_initialized_plan!(destination_b, 951)
            @test size(destination_a, 2) == size(destination_b, 1)
            expected_destination_shape = (size(destination_a, 1), size(destination_b, 2))
            for (destination_shape, mismatch_axis, seed) in
                (((2, 3), :column, 921), ((3, 2), :row, 931))
                @testset "wrong $mismatch_axis destination dimension" begin
                    destination_c = alias_fixture_matrix(
                        destination_shape...;
                        low_modulus = 7,
                        high_modulus = 5,
                        seed,
                    )
                    pad_initialized_plan!(destination_c, seed + 20)
                    @test size(destination_c) == destination_shape
                    if mismatch_axis === :column
                        @test size(destination_c, 1) == expected_destination_shape[1]
                        @test size(destination_c, 2) != expected_destination_shape[2]
                    else
                        @test size(destination_c, 1) != expected_destination_shape[1]
                        @test size(destination_c, 2) == expected_destination_shape[2]
                    end
                    for matrix in (destination_a, destination_b, destination_c)
                        for plane in
                            (mod.low_part(matrix), mod.high_part(matrix), matrix.plan)
                            @test size(parent(plane), 1) > size(plane, 1)
                            @test size(parent(plane), 2) > size(plane, 2)
                        end
                    end
                    destination_before = (
                        alias_fixture_snapshot(destination_a),
                        alias_fixture_snapshot(destination_b),
                        alias_fixture_snapshot(destination_c),
                    )
                    @test all(
                        plane -> all(value -> !iszero(value), plane),
                        Iterators.flatten(destination_before),
                    )
                    destination_error = try
                        LinearAlgebra.mul!(destination_c, destination_a, destination_b)
                        nothing
                    catch error
                        error
                    end
                    @test destination_error isa mod.CuModArraySizeMismatchException
                    @test destination_error isa mod.CuModArraySizeMismatchException &&
                          destination_error.message ==
                          "Output matrix C has incorrect dimensions"
                    destination_after = (
                        alias_fixture_snapshot(destination_a),
                        alias_fixture_snapshot(destination_b),
                        alias_fixture_snapshot(destination_c),
                    )
                    @test destination_after == destination_before
                end
            end
        end
    end
end
