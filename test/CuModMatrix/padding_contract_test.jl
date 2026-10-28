@testset "Padding geometry helpers" begin
    mod = GPUFiniteFieldMatrices

    dimensions = (0, 1, 31, 32, 33, 63, 64, 65)
    @test map(mod._padded_extent, dimensions) == (0, 32, 32, 32, 64, 64, 64, 96)
    @test mod._padded_size((31, 32, 33, 63, 64, 65)) == (32, 32, 64, 64, 64, 96)

    cpu_low = reshape(Float64.(1:6), 2, 3)
    cpu_high = fill(2.0, 2, 3)
    wrapped = mod.KaratsubaMatrix(
        cpu_low,
        cpu_high;
        low_modulus = 7,
        high_modulus = 5,
        modulus = 35,
    )
    @test mod.low_part(wrapped) === cpu_low
    @test mod.high_part(wrapped) === cpu_high
    @test size(wrapped) == (2, 3)

    cpu_storage = Core.apply_type(mod.KaratsubaArray{Float64,2}, Matrix{Float64})
    cpu_zeros =
        mod.zeros(cpu_storage, 5, 7; low_modulus = 7, high_modulus = 5, modulus = 35)
    @test mod.low_part(cpu_zeros) isa Matrix{Float64}
    @test mod.high_part(cpu_zeros) isa Matrix{Float64}
    @test size(mod.low_part(cpu_zeros)) == (5, 7)
    @test size(mod.high_part(cpu_zeros)) == (5, 7)
    @test cpu_zeros.plan === nothing
end

function _padding_values_for_test(A::GPUFiniteFieldMatrices.CuModArray)
    backing = Array(parent(A))
    mask = trues(size(backing))
    logical_ranges = ntuple(d -> 1:size(A, d), ndims(A))
    mask[logical_ranges...] .= false
    return backing[mask]
end

function _check_new_cumod_padding(A, dims, expected_modulus)
    mod = GPUFiniteFieldMatrices
    @test size(A) == dims
    @test size(parent(A)) == map(mod._padded_extent, dims)
    @test mod.modulus(A) == expected_modulus
    @test all(iszero, _padding_values_for_test(A))
end

function _dirty_padding_for_test!(A::GPUFiniteFieldMatrices.CuModArray, value)
    if ndims(A) == 1
        logical_length = size(A, 1)
        backing_length = length(parent(A))
        logical_length < backing_length &&
            (parent(A)[(logical_length+1):backing_length] .= value)
    elseif ndims(A) == 2
        rows, cols = size(A)
        backing_rows, backing_cols = size(parent(A))
        rows < backing_rows && (parent(A)[(rows+1):backing_rows, :] .= value)
        cols < backing_cols && (parent(A)[1:rows, (cols+1):backing_cols] .= value)
    else
        error("test padding helper supports vectors and matrices")
    end
    return A
end

function _raw_karatsuba_vector(low_values, high_values)
    mod = GPUFiniteFieldMatrices
    low = CUDA.CuArray(Float64.(low_values))
    high = CUDA.CuArray(Float64.(high_values))
    return mod.KaratsubaVector(
        low,
        high;
        low_modulus = 11,
        high_modulus = 13,
        modulus = 143,
    )
end

if CUDA.functional()
    @testset "CuMod allocation, copy, and modulus-change padding" begin
        mod = GPUFiniteFieldMatrices
        q = 101

        for dims in ((32, 32), (64, 64), (33, 31), (31, 63), (65, 33), (0, 33))
            values = zeros(Float64, dims...)
            source = mod.CuModMatrix{Float64}(values; modulus = q)
            _check_new_cumod_padding(source, dims, q)

            zero_matrix = mod.zeros(mod.CuModMatrix{Float64}, dims...; modulus = q)
            _check_new_cumod_padding(zero_matrix, dims, q)

            random_matrix = rand(mod.CuModMatrix{Float64}, dims...; modulus = q)
            _check_new_cumod_padding(random_matrix, dims, q)

            similar_matrix = similar(source)
            _check_new_cumod_padding(similar_matrix, dims, q)

            similar_modulus = mod.similar_with_modulus(source, 103)
            _check_new_cumod_padding(similar_modulus, dims, 103)

            copied = copy(source)
            _check_new_cumod_padding(copied, dims, q)
            @test Array(copied) == values

            changed = mod.change_modulus(source, 103)
            _check_new_cumod_padding(changed, dims, 103)
            @test Array(changed) == values
        end

        for n in (0, 31, 32, 33, 63, 64, 65)
            values = Float64.(1:n)
            source = mod.CuModVector{Float64}(values; modulus = q)
            _check_new_cumod_padding(source, (n,), q)

            zero_vector = mod.zeros(mod.CuModVector{Float64}, n; modulus = q)
            _check_new_cumod_padding(zero_vector, (n,), q)

            random_vector = rand(mod.CuModVector{Float64}, n; modulus = q)
            _check_new_cumod_padding(random_vector, (n,), q)

            similar_vector = similar(source)
            _check_new_cumod_padding(similar_vector, (n,), q)

            similar_modulus = mod.similar_with_modulus(source, 103)
            _check_new_cumod_padding(similar_modulus, (n,), 103)

            copied = copy(source)
            _check_new_cumod_padding(copied, (n,), q)
            @test Array(copied) == values

            changed = mod.change_modulus(source, 103)
            _check_new_cumod_padding(changed, (n,), 103)
            @test Array(changed) == values
        end

        for n in (32, 33, 64)
            identity = mod.identity_matrix(mod.CuModMatrix{Float64}, n; modulus = q)
            _check_new_cumod_padding(identity, (n, n), q)
            @test Array(identity) == Matrix{Float64}(I, n, n)
        end

        source_values = reshape(Float64.(1:35), 5, 7)
        source = mod.CuModMatrix{Float64}(source_values; modulus = q)
        _dirty_padding_for_test!(source, 777.0)
        dirty_before = Array(parent(source))

        copied = copy(source)
        _check_new_cumod_padding(copied, (5, 7), q)
        @test Array(copied) == source_values
        @test Array(parent(source)) == dirty_before

        changed = mod.change_modulus(source, 43)
        _check_new_cumod_padding(changed, (5, 7), 43)
        @test Array(changed) == Base.mod.(source_values, 43)
        @test Array(parent(source)) == dirty_before

        raw_storage = CUDA.fill(777.0, 5, 7)
        raw_wrapper = mod.CuModMatrix{Float64}(raw_storage; modulus = q, new_size = (3, 4))
        @test parent(raw_wrapper) === raw_storage
        @test size(raw_wrapper) == (3, 4)
        @test all(==(777.0), Array(parent(raw_wrapper)))
        raw_copy = copy(raw_wrapper)
        _check_new_cumod_padding(raw_copy, (3, 4), q)
        @test Array(raw_copy) == fill(777.0, 3, 4)
        @test all(==(777.0), Array(parent(raw_wrapper)))

        empty_src_data = CUDA.zeros(Float64, 0)
        empty_dst_data = CUDA.fill(-1.0, 0)
        empty_src = mod.CuModVector{Float64}(empty_src_data; modulus = q)
        empty_dst = mod.CuModVector{Float64}(empty_dst_data; modulus = q)
        @test copy!(empty_dst, empty_src) === empty_dst
        @test isempty(parent(empty_dst))

        raw_copy_values = Float64.(1:64)
        raw_copy_source = mod.CuModVector{Float64}(
            CUDA.CuArray(raw_copy_values);
            modulus = q,
            new_size = (5,),
        )
        raw_copy_destination =
            mod.CuModVector{Float64}(CUDA.zeros(Float64, 35); modulus = q, new_size = (5,))
        @test copy!(raw_copy_destination, raw_copy_source) === raw_copy_destination
        @test Array(parent(raw_copy_destination)) == raw_copy_values[1:35]
        @test Array(raw_copy_destination) == raw_copy_values[1:5]

        raw_low_source = mod.CuModVector{Float64}(
            CUDA.CuArray(Float64.(Base.mod.(0:34, 11)));
            modulus = 11,
        )
        raw_high_source = mod.CuModVector{Float64}(
            CUDA.CuArray(Float64.(Base.mod.(3:37, 13)));
            modulus = 13,
        )
        raw_low_destination =
            mod.CuModVector{Float64}(CUDA.zeros(Float64, 35); modulus = 11)
        raw_high_destination =
            mod.CuModVector{Float64}(CUDA.zeros(Float64, 35); modulus = 13)
        raw_karatsuba_source = mod.KaratsubaVector(
            raw_low_source,
            raw_high_source;
            low_modulus = 11,
            high_modulus = 13,
            modulus = 143,
        )
        raw_karatsuba_destination = mod.KaratsubaVector(
            raw_low_destination,
            raw_high_destination;
            low_modulus = 11,
            high_modulus = 13,
            modulus = 143,
        )
        @test copy!(raw_karatsuba_destination, raw_karatsuba_source) ===
              raw_karatsuba_destination
        @test Array(parent(mod.low_part(raw_karatsuba_destination))) ==
              Array(parent(raw_low_source))
        @test Array(parent(mod.high_part(raw_karatsuba_destination))) ==
              Array(parent(raw_high_source))
    end

    @testset "Rand product ignores dirty padding" begin
        mod = GPUFiniteFieldMatrices
        n = 33
        q = 101
        A = rand(mod.CuModMatrix{Float64}, n, n; modulus = q)
        B = rand(mod.CuModMatrix{Float64}, n, n; modulus = q)
        _check_new_cumod_padding(A, (n, n), q)
        _check_new_cumod_padding(B, (n, n), q)

        # Canonicalize the sampled logical cells in the same rand-created
        # objects for the independent exact-integer product oracle.
        A_values = Base.mod.(round.(Array(A)), q)
        B_values = Base.mod.(round.(Array(B)), q)
        parent(A)[1:n, 1:n] .= A_values
        parent(B)[1:n, 1:n] .= B_values
        expected = Float64.(Base.mod.(BigInt.(A_values) * BigInt.(B_values), BigInt(q)))

        clean_result = mod.zeros(mod.CuModMatrix{Float64}, n, n; modulus = q)
        CUDA.@sync LinearAlgebra.mul!(clean_result, A, B)
        @test Array(clean_result) == expected

        _dirty_padding_for_test!(A, 9001.0)
        _dirty_padding_for_test!(B, 9002.0)
        dirty_result = mod.zeros(mod.CuModMatrix{Float64}, n, n; modulus = q)
        CUDA.@sync LinearAlgebra.mul!(dirty_result, A, B)
        @test Array(dirty_result) == expected
    end

    @testset "Karatsuba factories, raw tails, and dirty-padding GEMV" begin
        mod = GPUFiniteFieldMatrices
        N1, N2, M = 11, 13, 143

        K = mod.zeros(
            mod.KaratsubaMatrix{Float64},
            33,
            31;
            low_modulus = N1,
            high_modulus = N2,
            modulus = M,
        )
        @test size(K) == (33, 31)
        for component in (mod.low_part(K), mod.high_part(K))
            @test size(component) == (33, 31)
            @test size(parent(component)) == (64, 32)
            @test all(iszero, _padding_values_for_test(component))
        end
        @test K.plan === nothing
        mod.initialize_plan!(K)
        @test size(K.plan) == (33, 31)
        @test size(parent(K.plan)) == (64, 32)
        @test all(iszero, _padding_values_for_test(K.plan))

        cpu_storage = Core.apply_type(mod.KaratsubaArray{Float64,2}, Matrix{Float64})
        cpu_K =
            mod.zeros(cpu_storage, 33, 31; low_modulus = N1, high_modulus = N2, modulus = M)
        @test size(mod.low_part(cpu_K)) == (33, 31)
        @test size(mod.high_part(cpu_K)) == (33, 31)
        @test mod.low_part(cpu_K) isa Matrix{Float64}

        raw_prototype = CUDA.zeros(Float64, 2, 3)
        raw_storage = typeof(raw_prototype)
        raw_storage_type = Core.apply_type(mod.KaratsubaArray{Float64,2}, raw_storage)
        raw_K = mod.zeros(
            raw_storage_type,
            33,
            31;
            low_modulus = N1,
            high_modulus = N2,
            modulus = M,
        )
        @test typeof(mod.low_part(raw_K)) === raw_storage
        @test typeof(mod.high_part(raw_K)) === raw_storage
        @test size(mod.low_part(raw_K)) == (33, 31)
        @test size(parent(mod.low_part(raw_K))) == (33, 31)
        @test raw_K.plan === nothing
        mod.initialize_plan!(raw_K)
        @test raw_K.plan isa mod.CuModArray{Float64,2}
        @test size(parent(raw_K.plan)) == (64, 32)

        raw_wrapper_low = CUDA.fill(777.0, 5, 7)
        raw_wrapper_high = CUDA.fill(888.0, 5, 7)
        raw_wrapper = mod.KaratsubaMatrix(
            raw_wrapper_low,
            raw_wrapper_high;
            low_modulus = N1,
            high_modulus = N2,
            modulus = M,
        )
        @test mod.low_part(raw_wrapper) === raw_wrapper_low
        @test mod.high_part(raw_wrapper) === raw_wrapper_high
        @test size(raw_wrapper) == (5, 7)
        @test all(==(777.0), Array(raw_wrapper_low))
        @test all(==(888.0), Array(raw_wrapper_high))

        values = reshape(Float64.(1:(33*31)), 33, 31)
        source = mod.CuModMatrix{Float64}(values; modulus = M)
        converted = mod.to_karatsuba(
            Float64,
            source;
            low_modulus = N1,
            high_modulus = N2,
            modulus = M,
        )
        for component in (mod.low_part(converted), mod.high_part(converted))
            @test size(parent(component)) == (64, 32)
            @test all(iszero, _padding_values_for_test(component))
        end
        @test Array(mod.from_karatsuba(Float64, converted)) == values

        copied = copy(converted)
        @test size(parent(mod.low_part(copied))) == (64, 32)
        @test size(parent(mod.high_part(copied))) == (64, 32)
        @test Array(mod.from_karatsuba(Float64, copied)) == values
        @test all(iszero, _padding_values_for_test(mod.low_part(copied)))
        @test all(iszero, _padding_values_for_test(mod.high_part(copied)))

        tail = 35
        raw_low_a = Float64.(Base.mod.(0:(tail-1), N1))
        raw_high_a = Float64.(Base.mod.(3:(tail+2), N2))
        raw_low_b = Float64.(Base.mod.(7:(tail+6), N1))
        raw_high_b = Float64.(Base.mod.(8:(tail+7), N2))
        raw_a = _raw_karatsuba_vector(raw_low_a, raw_high_a)
        raw_b = _raw_karatsuba_vector(raw_low_b, raw_high_b)
        raw_sum = _raw_karatsuba_vector(zeros(tail), zeros(tail))
        raw_difference = _raw_karatsuba_vector(zeros(tail), zeros(tail))
        raw_scaled = _raw_karatsuba_vector(zeros(tail), zeros(tail))
        raw_negative = _raw_karatsuba_vector(zeros(tail), zeros(tail))

        value_a = BigInt.(raw_low_a) .+ BigInt(N1) .* BigInt.(raw_high_a)
        value_b = BigInt.(raw_low_b) .+ BigInt(N1) .* BigInt.(raw_high_b)
        @test size(parent(mod.low_part(raw_a))) == (tail,)
        @test add!(raw_sum, raw_a, raw_b) === raw_sum
        @test BigInt.(Array(raw_sum)) == Base.mod.(value_a .+ value_b, BigInt(M))
        @test sub!(raw_difference, raw_a, raw_b) === raw_difference
        @test BigInt.(Array(raw_difference)) == Base.mod.(value_a .- value_b, BigInt(M))
        @test mod.scalar_multiply!(raw_scaled, raw_a, 3) === raw_scaled
        @test BigInt.(Array(raw_scaled)) == Base.mod.(3 .* value_a, BigInt(M))
        @test mod.negate!(raw_negative, raw_a) === raw_negative
        @test BigInt.(Array(raw_negative)) == Base.mod.(-value_a, BigInt(M))

        empty = _raw_karatsuba_vector(Float64[], Float64[])
        empty_out = _raw_karatsuba_vector(Float64[], Float64[])
        @test add!(empty_out, empty, empty) === empty_out

        function make_gemv_case(matrix_low, matrix_high, vector_low, vector_high)
            A1 = mod.CuModMatrix{Float64}(matrix_low; modulus = N1, mod = false)
            A2 = mod.CuModMatrix{Float64}(matrix_high; modulus = N2, mod = false)
            x1 = mod.CuModVector{Float64}(vector_low; modulus = N1, mod = false)
            x2 = mod.CuModVector{Float64}(vector_high; modulus = N2, mod = false)
            A = mod.KaratsubaMatrix(
                A1,
                A2;
                low_modulus = N1,
                high_modulus = N2,
                modulus = M,
            )
            x = mod.KaratsubaVector(
                x1,
                x2;
                low_modulus = N1,
                high_modulus = N2,
                modulus = M,
            )
            y = mod.zeros(
                mod.KaratsubaVector{Float64},
                size(matrix_low, 1);
                low_modulus = N1,
                high_modulus = N2,
                modulus = M,
            )
            foreach(mod.initialize_plan!, (A, x, y))
            return (A = A, x = x, y = y)
        end

        for (rows, cols) in ((33, 33), (33, 31))
            matrix_low = reshape(Float64.(Base.mod.(0:(rows*cols-1), N1)), rows, cols)
            matrix_high =
                reshape(Float64.(Base.mod.(3 .* (0:(rows*cols-1)) .+ 1, N2)), rows, cols)
            vector_low = Float64.(Base.mod.(1:cols, N1))
            vector_high = Float64.(Base.mod.(3 .* (1:cols) .+ 1, N2))
            expected_matrix = BigInt.(matrix_low) .+ BigInt(N1) .* BigInt.(matrix_high)
            expected_vector = BigInt.(vector_low) .+ BigInt(N1) .* BigInt.(vector_high)
            expected_gemv =
                Float64.(Base.mod.(expected_matrix * expected_vector, BigInt(M)))

            clean = make_gemv_case(matrix_low, matrix_high, vector_low, vector_high)
            CUDA.@sync mod.karatsuba_gemv!(clean.y, clean.A, clean.x)
            @test Array(clean.y) == expected_gemv

            dirty = make_gemv_case(matrix_low, matrix_high, vector_low, vector_high)
            sentinel = 9007.0
            for K in (dirty.A, dirty.x, dirty.y)
                _dirty_padding_for_test!(mod.low_part(K), sentinel)
                _dirty_padding_for_test!(mod.high_part(K), sentinel)
                _dirty_padding_for_test!(K.plan, sentinel)
            end
            CUDA.@sync mod.karatsuba_gemv!(dirty.y, dirty.A, dirty.x)
            @test Array(dirty.y) == expected_gemv
            @test all(==(sentinel), _padding_values_for_test(dirty.A.plan))
            @test all(==(sentinel), _padding_values_for_test(dirty.x.plan))
            @test all(==(sentinel), _padding_values_for_test(mod.low_part(dirty.y)))
            @test all(==(sentinel), _padding_values_for_test(mod.high_part(dirty.y)))
        end
    end
end
