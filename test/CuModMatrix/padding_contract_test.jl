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

function _wrapped_cumod_for_test(
    values::Array{Float64,D},
    backing_dims::Dims{D},
    q,
    padding,
) where {D}
    mod = GPUFiniteFieldMatrices
    storage = CUDA.fill(Float64(padding), backing_dims...)
    logical_ranges = ntuple(d -> 1:size(values, d), D)
    storage[logical_ranges...] .= values
    return mod.CuModArray{Float64,D}(
        storage;
        modulus = Int(q),
        mod = false,
        new_size = size(values),
    )
end

function _wrapped_karatsuba_matrix_for_test(
    low_values,
    high_values,
    low_backing_dims,
    high_backing_dims;
    low_modulus = 11,
    high_modulus = 13,
    modulus = 143,
)
    mod = GPUFiniteFieldMatrices
    low = _wrapped_cumod_for_test(low_values, low_backing_dims, low_modulus, -901.0)
    high = _wrapped_cumod_for_test(high_values, high_backing_dims, high_modulus, -902.0)
    return mod.KaratsubaMatrix(
        low,
        high;
        low_modulus = low_modulus,
        high_modulus = high_modulus,
        modulus = modulus,
    )
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
        raw_copy_destination =
            mod.CuModVector{Float64}(CUDA.fill(-444.0, 40); modulus = q, new_size = (35,))
        raw_copy_source = mod.CuModVector{Float64}(
            CUDA.CuArray(raw_copy_values);
            modulus = q,
            new_size = (35,),
        )
        @test copy!(raw_copy_destination, raw_copy_source) === raw_copy_destination
        @test Array(parent(raw_copy_destination))[1:35] == raw_copy_values[1:35]
        @test all(==(-444.0), Array(parent(raw_copy_destination))[36:40])
        @test Array(raw_copy_destination) == raw_copy_values[1:35]

        matrix_copy_source_storage = CUDA.fill(991.0, 3, 3)
        matrix_copy_source_storage[1:2, 1:2] .= [1.0 2.0; 3.0 4.0]
        matrix_copy_destination_storage = CUDA.fill(-555.0, 4, 4)
        matrix_copy_source = mod.CuModMatrix{Float64}(
            matrix_copy_source_storage;
            modulus = q,
            new_size = (2, 2),
        )
        matrix_copy_destination = mod.CuModMatrix{Float64}(
            matrix_copy_destination_storage;
            modulus = q,
            new_size = (2, 2),
        )
        @test copy!(matrix_copy_destination, matrix_copy_source) === matrix_copy_destination
        @test Array(matrix_copy_destination) == [1.0 2.0; 3.0 4.0]
        @test all(==(-555.0), _padding_values_for_test(matrix_copy_destination))
        @test all(==(991.0), _padding_values_for_test(matrix_copy_source))

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

    @testset "CuMod logical operations across backing pitches" begin
        mod = GPUFiniteFieldMatrices
        q = 101
        left_values = [1.0 2.0; 3.0 4.0]
        right_values = [5.0 7.0; 6.0 8.0]
        left = _wrapped_cumod_for_test(left_values, (64, 64), q, -1111.0)
        right = _wrapped_cumod_for_test(right_values, (35, 41), q, -2222.0)
        expected_sum = Float64.(Base.mod.(left_values .+ right_values, q))
        expected_difference = Float64.(Base.mod.(left_values .- right_values, q))

        @test parent(left) !== parent(right)
        @test size(left) == size(right) == (2, 2)
        @test size(parent(left)) == (64, 64)
        @test size(parent(right)) == (35, 41)

        destination = _wrapped_cumod_for_test(zeros(2, 2), (4, 5), q, -3333.0)
        @test mod.add!(destination, left, right) === destination
        @test Array(destination) == expected_sum
        @test all(==(-3333.0), _padding_values_for_test(destination))
        @test Array(left + right) == expected_sum
        @test size(parent(left + right)) == (32, 32)

        destination = _wrapped_cumod_for_test(zeros(2, 2), (5, 4), q, -4444.0)
        @test mod.sub!(destination, left, right) === destination
        @test Array(destination) == expected_difference
        @test Array(left - right) == expected_difference

        expected_scalar_add = Float64.(Base.mod.(left_values .+ 9, q))
        expected_scalar_sub = Float64.(Base.mod.(left_values .- 9, q))
        expected_reverse_sub = Float64.(Base.mod.(9 .- left_values, q))
        destination = similar(left)
        @test mod.scalar_add!(destination, left, 9) === destination
        @test Array(destination) == expected_scalar_add
        @test Array(left + 9) == expected_scalar_add
        @test Array(9 + left) == expected_scalar_add

        @test mod.scalar_sub!(destination, left, 9) === destination
        @test Array(destination) == expected_scalar_sub
        @test Array(left - 9) == expected_scalar_sub
        @test mod.rscalar_sub!(destination, left, 9) === destination
        @test Array(destination) == expected_reverse_sub
        @test Array(9 - left) == expected_reverse_sub

        expected_elementwise = Float64.(Base.mod.(left_values .* right_values, q))
        @test mod.mul_elementwise!(destination, left, right) === destination
        @test Array(destination) == expected_elementwise
        expected_scaled = Float64.(Base.mod.(left_values .* 7, q))
        @test mod.LinearAlgebra.mul!(destination, left, 7) === destination
        @test Array(destination) == expected_scaled
        @test Array(left * 7) == expected_scaled
        @test Array(7 * left) == expected_scaled

        expected_divided =
            Float64.(Base.mod.(BigInt.(left_values) .* invmod(3, q), BigInt(q)))
        @test mod.div!(destination, left, 3) === destination
        @test Array(destination) == expected_divided
        @test Array(left / 3) == expected_divided

        @test mod.mod!(destination, left, 13) === destination
        @test Array(destination) == Float64.(Base.mod.(left_values, 13))

        vector_left_values = Float64.(1:35)
        vector_right_values = Float64.(2:36)
        vector_left = _wrapped_cumod_for_test(vector_left_values, (64,), q, -5555.0)
        vector_right = _wrapped_cumod_for_test(vector_right_values, (40,), q, -6666.0)
        vector_sum = similar(vector_left)
        @test mod.add!(vector_sum, vector_left, vector_right) === vector_sum
        @test Array(vector_sum) ==
              Float64.(Base.mod.(vector_left_values .+ vector_right_values, q))
        @test Array(vector_sum)[end] ==
              Base.mod(vector_left_values[end] + vector_right_values[end], q)
    end

    @testset "Karatsuba logical operations across component pitches" begin
        mod = GPUFiniteFieldMatrices
        N1, N2, M = 11, 13, 143
        a_low = [1.0 2.0; 3.0 4.0]
        a_high = [2.0 1.0; 4.0 3.0]
        b_low = [5.0 4.0; 3.0 2.0]
        b_high = [1.0 3.0; 2.0 4.0]
        A = _wrapped_karatsuba_matrix_for_test(
            a_low,
            a_high,
            (3, 3),
            (4, 5);
            low_modulus = N1,
            high_modulus = N2,
            modulus = M,
        )
        B = _wrapped_karatsuba_matrix_for_test(
            b_low,
            b_high,
            (5, 4),
            (3, 4);
            low_modulus = N1,
            high_modulus = N2,
            modulus = M,
        )
        value_a = BigInt.(a_low) .+ BigInt(N1) .* BigInt.(a_high)
        value_b = BigInt.(b_low) .+ BigInt(N1) .* BigInt.(b_high)
        represented(K) =
            BigInt.(Array(mod.low_part(K))) .+
            BigInt(N1) .* BigInt.(Array(mod.high_part(K)))

        sum_result = _wrapped_karatsuba_matrix_for_test(
            zeros(2, 2),
            zeros(2, 2),
            (4, 5),
            (5, 4);
            low_modulus = N1,
            high_modulus = N2,
            modulus = M,
        )
        @test mod.add!(sum_result, A, B) === sum_result
        @test represented(sum_result) == Base.mod.(value_a .+ value_b, BigInt(M))
        @test all(==(-901.0), _padding_values_for_test(mod.low_part(sum_result)))
        @test all(==(-902.0), _padding_values_for_test(mod.high_part(sum_result)))

        difference_result = _wrapped_karatsuba_matrix_for_test(
            zeros(2, 2),
            zeros(2, 2),
            (5, 4),
            (4, 5);
            low_modulus = N1,
            high_modulus = N2,
            modulus = M,
        )
        @test mod.sub!(difference_result, A, B) === difference_result
        @test represented(difference_result) == Base.mod.(value_a .- value_b, BigInt(M))

        scaled_result = _wrapped_karatsuba_matrix_for_test(
            zeros(2, 2),
            zeros(2, 2),
            (4, 4),
            (5, 5);
            low_modulus = N1,
            high_modulus = N2,
            modulus = M,
        )
        @test mod.scalar_multiply!(scaled_result, A, 3) === scaled_result
        @test represented(scaled_result) == Base.mod.(3 .* value_a, BigInt(M))

        negative_result = _wrapped_karatsuba_matrix_for_test(
            zeros(2, 2),
            zeros(2, 2),
            (5, 5),
            (4, 4);
            low_modulus = N1,
            high_modulus = N2,
            modulus = M,
        )
        @test mod.negate!(negative_result, A) === negative_result
        @test represented(negative_result) == Base.mod.(-value_a, BigInt(M))

        copy_result = _wrapped_karatsuba_matrix_for_test(
            zeros(2, 2),
            zeros(2, 2),
            (4, 5),
            (5, 4);
            low_modulus = N1,
            high_modulus = N2,
            modulus = M,
        )
        @test copy!(copy_result, A) === copy_result
        @test represented(copy_result) == value_a
        @test all(==(-901.0), _padding_values_for_test(mod.low_part(copy_result)))
        @test all(==(-902.0), _padding_values_for_test(mod.high_part(copy_result)))
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

        input_values = reshape(Float64.(1:(33*31)), 33, 31)
        source = mod.CuModMatrix{Float64}(input_values; modulus = M)
        values = Float64.(Base.mod.(input_values, M))
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

        function make_gemv_case(
            matrix_low,
            matrix_high,
            vector_low,
            vector_high;
            raw_components = false,
            low_radix = 11,
            high_radix = 11,
            represented_modulus = 121,
        )
            if raw_components
                A1 = CUDA.CuArray(matrix_low)
                A2 = CUDA.CuArray(matrix_high)
                x1 = CUDA.CuArray(vector_low)
                x2 = CUDA.CuArray(vector_high)
            else
                A1 = mod.CuModMatrix{Float64}(matrix_low; modulus = low_radix, mod = false)
                A2 =
                    mod.CuModMatrix{Float64}(matrix_high; modulus = high_radix, mod = false)
                x1 = mod.CuModVector{Float64}(vector_low; modulus = low_radix, mod = false)
                x2 =
                    mod.CuModVector{Float64}(vector_high; modulus = high_radix, mod = false)
            end
            A = mod.KaratsubaMatrix(
                A1,
                A2;
                low_modulus = low_radix,
                high_modulus = high_radix,
                modulus = represented_modulus,
            )
            x = mod.KaratsubaVector(
                x1,
                x2;
                low_modulus = low_radix,
                high_modulus = high_radix,
                modulus = represented_modulus,
            )
            y = mod.zeros(
                mod.KaratsubaVector{Float64},
                size(matrix_low, 1);
                low_modulus = low_radix,
                high_modulus = high_radix,
                modulus = represented_modulus,
            )
            foreach(mod.initialize_plan!, (A, x, y))
            return (A = A, x = x, y = y)
        end

        for (rows, cols) in ((33, 33), (33, 31), (33, 35)), active_col in (1, 2)
            matrix_low = zeros(Float64, rows, cols)
            matrix_high = zeros(Float64, rows, cols)
            matrix_low[:, active_col] .= 1.0
            matrix_high[:, active_col] .= 1.0
            vector_low = zeros(Float64, cols)
            vector_high = zeros(Float64, cols)
            vector_low[active_col] = 1.0
            vector_high[active_col] = 9.0
            gemv_radix = 11
            gemv_modulus = 121
            expected_matrix =
                BigInt.(matrix_low) .+ BigInt(gemv_radix) .* BigInt.(matrix_high)
            expected_vector =
                BigInt.(vector_low) .+ BigInt(gemv_radix) .* BigInt.(vector_high)
            expected_gemv =
                Float64.(Base.mod.(expected_matrix * expected_vector, BigInt(gemv_modulus)))
            @test all(==(111.0), expected_gemv)

            clean = make_gemv_case(matrix_low, matrix_high, vector_low, vector_high)
            @test stride(parent(mod.low_part(clean.A)), 2) == 64
            @test stride(parent(clean.A.plan), 2) == 64
            @test parent(clean.x.plan) !== parent(clean.y.plan)
            @test !mod._stripe_mightalias(parent(clean.x.plan), parent(clean.y.plan))
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
            @test all(==(sentinel), _padding_values_for_test(dirty.y.plan))

            raw_clean = make_gemv_case(
                matrix_low,
                matrix_high,
                vector_low,
                vector_high;
                raw_components = true,
            )
            @test stride(mod.low_part(raw_clean.A), 2) == rows
            @test stride(parent(raw_clean.A.plan), 2) == 64
            @test parent(raw_clean.x.plan) !== parent(raw_clean.y.plan)
            @test !mod._stripe_mightalias(
                parent(raw_clean.x.plan),
                parent(raw_clean.y.plan),
            )
            CUDA.@sync mod.karatsuba_gemv!(raw_clean.y, raw_clean.A, raw_clean.x)
            @test Array(raw_clean.y) == expected_gemv

            raw_dirty = make_gemv_case(
                matrix_low,
                matrix_high,
                vector_low,
                vector_high;
                raw_components = true,
            )
            # Preserve the original first-column sentinel; 9007 distinguishes
            # the later-column wrong-lda read from the logical plan value.
            sentinel = active_col == 1 ? 9011.0 : 9007.0
            for K in (raw_dirty.A, raw_dirty.x, raw_dirty.y)
                for component in (mod.low_part(K), mod.high_part(K))
                    component isa mod.CuModArray &&
                        _dirty_padding_for_test!(component, sentinel)
                end
                _dirty_padding_for_test!(K.plan, sentinel)
            end
            CUDA.@sync mod.karatsuba_gemv!(raw_dirty.y, raw_dirty.A, raw_dirty.x)
            @test Array(raw_dirty.y) == expected_gemv
            @test all(==(sentinel), _padding_values_for_test(raw_dirty.A.plan))
            @test all(==(sentinel), _padding_values_for_test(raw_dirty.x.plan))
            @test all(==(sentinel), _padding_values_for_test(raw_dirty.y.plan))
        end

        aligned_rows, aligned_cols = 32, 32
        aligned_column = 2
        aligned_matrix_low = zeros(Float64, aligned_rows, aligned_cols)
        aligned_matrix_high = zeros(Float64, aligned_rows, aligned_cols)
        aligned_matrix_low[:, aligned_column] .= 1.0
        aligned_matrix_high[:, aligned_column] .= 1.0
        aligned_vector_low = zeros(Float64, aligned_cols)
        aligned_vector_high = zeros(Float64, aligned_cols)
        aligned_vector_low[aligned_column] = 1.0
        aligned_vector_high[aligned_column] = 9.0
        aligned_expected_matrix =
            BigInt.(aligned_matrix_low) .+ BigInt(11) .* BigInt.(aligned_matrix_high)
        aligned_expected_vector =
            BigInt.(aligned_vector_low) .+ BigInt(11) .* BigInt.(aligned_vector_high)
        aligned_expected = Float64.(
            Base.mod.(aligned_expected_matrix * aligned_expected_vector, BigInt(121)),
        )
        @test all(==(111.0), aligned_expected)

        aligned = make_gemv_case(
            aligned_matrix_low,
            aligned_matrix_high,
            aligned_vector_low,
            aligned_vector_high,
        )
        aligned_matrix_values =
            (mod.low_part(aligned.A), aligned.A.plan, mod.high_part(aligned.A))
        aligned_vector_values =
            (mod.low_part(aligned.x), aligned.x.plan, mod.high_part(aligned.x))
        aligned_target_values =
            (mod.low_part(aligned.y), mod.high_part(aligned.y), aligned.y.plan)
        aligned_mats = [map(parent, aligned_matrix_values)...]
        aligned_vecs = [map(parent, aligned_vector_values)...]
        aligned_targets = [map(parent, aligned_target_values)...]
        @test all(
            size(aligned_mats[i]) == size(aligned_matrix_values[i]) for
            i in eachindex(aligned_mats)
        )
        @test all(
            size(aligned_vecs[i]) == size(aligned_vector_values[i]) for
            i in eachindex(aligned_vecs)
        )
        @test all(
            size(aligned_targets[i]) == size(aligned_target_values[i]) for
            i in eachindex(aligned_targets)
        )
        @test all(
            aligned_mats[i] === parent(aligned_matrix_values[i]) for
            i in eachindex(aligned_mats)
        )
        @test all(
            aligned_vecs[i] === parent(aligned_vector_values[i]) for
            i in eachindex(aligned_vecs)
        )
        @test all(
            aligned_targets[i] === parent(aligned_target_values[i]) for
            i in eachindex(aligned_targets)
        )
        @test mod._karatsuba_gemv_batch_containers_supported(
            aligned_mats,
            aligned_vecs,
            aligned_targets,
        )
        aligned_lda = stride(aligned_mats[1], 2)
        aligned_incx = stride(aligned_vecs[1], 1)
        aligned_incy = stride(aligned_targets[1], 1)
        @test all(stride(matrix, 2) == aligned_lda for matrix in aligned_mats)
        @test all(stride(vector, 1) == aligned_incx for vector in aligned_vecs)
        @test all(stride(target, 1) == aligned_incy for target in aligned_targets)
        CUDA.@sync mod.karatsuba_gemv!(aligned.y, aligned.A, aligned.x)
        @test Array(aligned.y) == aligned_expected
    end
end
