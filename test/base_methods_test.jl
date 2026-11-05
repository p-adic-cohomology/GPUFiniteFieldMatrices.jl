@testset "Base method CPU dispatch" begin
    mod = GPUFiniteFieldMatrices
    matrix_type = mod.CuModMatrix{Float32}
    karatsuba_type = mod.KaratsubaArray{Float64,1}

    @test which(Base.:-, Tuple{matrix_type}).module === mod
    @test which(Base.:(==), Tuple{matrix_type,matrix_type}).module === mod
    @test which(Base.:-, Tuple{karatsuba_type}).module === mod
    @test which(LinearAlgebra.rank, Tuple{mod.PLUQ{Float32}}).module === mod
    @test which(LinearAlgebra.rank, Tuple{matrix_type}).module === mod

    rank_keywords = Base.kwarg_decl(which(LinearAlgebra.rank, Tuple{matrix_type}))
    @test :options in rank_keywords
end

function _base_methods_test_wrapped_matrix(
    logical_values::Matrix{Float32},
    backing_size::Tuple{Int,Int},
    logical_size::Tuple{Int,Int},
    field_modulus::Int,
    padding::Float32,
)
    backing = fill(padding, backing_size...)
    rows, cols = size(logical_values)
    backing[1:rows, 1:cols] = logical_values
    return GPUFiniteFieldMatrices.CuModMatrix(
        CUDA.CuArray(backing);
        modulus = field_modulus,
        mod = false,
        new_size = logical_size,
    )
end

function _base_methods_test_rank_mod_prime(A::AbstractMatrix{<:Integer}, p::Integer)
    field = BigInt(p)
    work = Base.mod.(BigInt.(A), field)
    pivot_rows = 0

    for col in axes(work, 2)
        pivot = 0
        for row = (pivot_rows+1):size(work, 1)
            if !iszero(work[row, col])
                pivot = row
                break
            end
        end
        pivot == 0 && continue

        next_pivot_row = pivot_rows + 1
        if pivot != next_pivot_row
            old_pivot_row = copy(work[next_pivot_row, :])
            work[next_pivot_row, :] = work[pivot, :]
            work[pivot, :] = old_pivot_row
        end

        pivot_inverse = Base.invmod(work[next_pivot_row, col], field)
        for row = (next_pivot_row+1):size(work, 1)
            factor = Base.mod(work[row, col] * pivot_inverse, field)
            iszero(factor) && continue
            for trailing_col = col:size(work, 2)
                work[row, trailing_col] = Base.mod(
                    work[row, trailing_col] - factor * work[next_pivot_row, trailing_col],
                    field,
                )
            end
        end

        pivot_rows = next_pivot_row
        pivot_rows == size(work, 1) && break
    end

    return pivot_rows
end

function test_base_methods_gpu()
    mod = GPUFiniteFieldMatrices

    @testset "Allocating unary minus preserves its CuModArray input" begin
        field_modulus = 101
        logical_values = Float32[1 2 3; 4 5 6]
        backing = fill(Float32(777), 64, 16)
        backing[1:2, 1:3] = logical_values
        A = mod.CuModMatrix(
            CUDA.CuArray(backing);
            modulus = field_modulus,
            mod = false,
            new_size = (2, 3),
        )
        source_parent = parent(A)
        source_before = Array(source_parent)
        source_device = CUDA.device(source_parent)
        caller_device = CUDA.device()
        source_size = size(A)
        source_modulus = modulus(A)

        negative = -A
        @test CUDA.device() == caller_device
        expected = Float32[
            Base.mod(-BigInt(value), BigInt(field_modulus)) for value in logical_values
        ]

        @test negative isa typeof(A)
        @test negative !== A
        @test parent(negative) !== source_parent
        @test size(negative) == source_size
        @test modulus(negative) == source_modulus
        @test CUDA.device(parent(negative)) == source_device
        @test Array(negative) == expected
        @test parent(A) === source_parent
        @test CUDA.device(parent(A)) == source_device
        @test Array(source_parent) == source_before
        @test size(A) == source_size
        @test modulus(A) == source_modulus

        devices = collect(CUDA.devices())
        if length(devices) >= 2
            input_device, different_caller_device = devices[1], devices[2]
            @test input_device != different_caller_device

            multi_backing = copy(backing)
            multi_A = CUDA.device!(input_device) do
                mod.CuModMatrix(
                    CUDA.CuArray(multi_backing);
                    modulus = field_modulus,
                    mod = false,
                    new_size = (2, 3),
                )
            end
            multi_parent = parent(multi_A)
            multi_before = CUDA.device!(input_device) do
                Array(multi_parent)
            end

            multi_negative = CUDA.device!(different_caller_device) do
                result = -multi_A
                @test CUDA.device() == different_caller_device
                result
            end
            multi_result = CUDA.device!(input_device) do
                Array(multi_negative)
            end
            multi_after = CUDA.device!(input_device) do
                Array(multi_parent)
            end

            @test CUDA.device(parent(multi_A)) == input_device
            @test CUDA.device(parent(multi_negative)) == input_device
            @test multi_result == expected
            @test parent(multi_A) === multi_parent
            @test multi_after == multi_before
            @test size(multi_negative) == size(multi_A)
            @test modulus(multi_negative) == field_modulus
        end

        empty_input =
            mod.CuModVector(CUDA.CuArray(Float32[]); modulus = field_modulus, mod = false)
        empty_negative = -empty_input
        @test size(empty_negative) == size(empty_input) == (0,)
        @test modulus(empty_negative) == field_modulus
        @test isempty(empty_negative)
        @test CUDA.device(parent(empty_negative)) == CUDA.device(parent(empty_input))
    end

    @testset "CuModArray equality uses logical entries" begin
        logical_values = Float32[1 2; 3 4]
        A = _base_methods_test_wrapped_matrix(
            logical_values,
            (6, 2),
            (2, 2),
            101,
            Float32(777),
        )
        B = _base_methods_test_wrapped_matrix(
            logical_values,
            (3, 4),
            (2, 2),
            101,
            Float32(-50),
        )
        C = _base_methods_test_wrapped_matrix(
            logical_values,
            (4, 4),
            (2, 2),
            101,
            Float32(1234),
        )
        A_parent = parent(A)
        B_parent = parent(B)
        A_before = Array(A_parent)
        B_before = Array(B_parent)

        @test A == B
        @test A == C

        changed_values = copy(logical_values)
        changed_values[2, 2] = 9
        entry_mismatch = _base_methods_test_wrapped_matrix(
            changed_values,
            (3, 4),
            (2, 2),
            101,
            Float32(999),
        )
        shape_mismatch = _base_methods_test_wrapped_matrix(
            logical_values,
            (4, 4),
            (2, 1),
            101,
            Float32(321),
        )
        modulus_mismatch = _base_methods_test_wrapped_matrix(
            logical_values,
            (3, 4),
            (2, 2),
            103,
            Float32(654),
        )

        @test !(A == entry_mismatch)
        @test !(A == shape_mismatch)
        @test !(A == modulus_mismatch)
        @test parent(A) === A_parent
        @test parent(B) === B_parent
        @test Array(A_parent) == A_before
        @test Array(B_parent) == B_before

        empty_a = mod.CuModMatrix(
            CUDA.zeros(Float32, 0, 32);
            modulus = 101,
            mod = false,
            new_size = (0, 2),
        )
        empty_b = mod.CuModMatrix(
            CUDA.zeros(Float32, 0, 16);
            modulus = 101,
            mod = false,
            new_size = (0, 2),
        )
        empty_shape_mismatch = mod.CuModMatrix(
            CUDA.zeros(Float32, 0, 32);
            modulus = 101,
            mod = false,
            new_size = (0, 1),
        )
        @test empty_a == empty_b
        @test !(empty_a == empty_shape_mismatch)
    end

    @testset "Karatsuba unary minus preserves its input and plan" begin
        low_modulus = 10
        high_modulus = 10
        represented_modulus = low_modulus * high_modulus
        low_values = Float64[0, 1, 9, 4]
        high_values = Float64[0, 9, 3, 5]
        low = mod.CuModVector{Float64}(
            CUDA.CuArray(low_values);
            modulus = low_modulus,
            mod = false,
        )
        high = mod.CuModVector{Float64}(
            CUDA.CuArray(high_values);
            modulus = high_modulus,
            mod = false,
        )
        A = mod.KaratsubaVector{Float64}(
            low,
            high;
            low_modulus = low_modulus,
            high_modulus = high_modulus,
            modulus = represented_modulus,
        )
        mod.initialize_plan!(A)

        low_parent = parent(mod.low_part(A))
        high_parent = parent(mod.high_part(A))
        source_device = CUDA.device(low_parent)
        caller_device = CUDA.device()
        source_plan = A.plan
        @test CUDA.device(parent(source_plan)) == source_device
        fill!(parent(source_plan), 37)
        low_before = Array(low_parent)
        high_before = Array(high_parent)
        plan_before = Array(parent(source_plan))
        low_component_before = mod.low_part(A)
        high_component_before = mod.high_part(A)

        negative = -A
        @test CUDA.device() == caller_device
        expected = BigInt[
            Base.mod(
                -(BigInt(low_values[i]) + BigInt(low_modulus) * BigInt(high_values[i])),
                BigInt(represented_modulus),
            ) for i in eachindex(low_values)
        ]

        @test negative isa typeof(A)
        @test negative !== A
        @test Array(negative) == expected
        @test mod.low_modulus(negative) == low_modulus
        @test mod.high_modulus(negative) == high_modulus
        @test modulus(negative) == represented_modulus
        @test negative.plan !== nothing
        @test negative.plan !== source_plan
        @test CUDA.device() == caller_device
        @test CUDA.device(parent(mod.low_part(negative))) == source_device
        @test CUDA.device(parent(mod.high_part(negative))) == source_device
        @test CUDA.device(parent(negative.plan)) == source_device
        @test mod.low_part(A) === low_component_before
        @test mod.high_part(A) === high_component_before
        @test parent(mod.low_part(A)) === low_parent
        @test parent(mod.high_part(A)) === high_parent
        @test CUDA.device(parent(mod.low_part(A))) == source_device
        @test CUDA.device(parent(mod.high_part(A))) == source_device
        @test A.plan === source_plan
        @test Array(low_parent) == low_before
        @test Array(high_parent) == high_before
        @test Array(parent(source_plan)) == plan_before
        @test (mod.low_modulus(A), mod.high_modulus(A), modulus(A)) ==
              (low_modulus, high_modulus, represented_modulus)
        devices = collect(CUDA.devices())
        if length(devices) >= 2
            input_device, different_caller_device = devices[1], devices[2]
            @test input_device != different_caller_device

            multi_A = CUDA.device!(input_device) do
                low = mod.CuModVector{Float64}(
                    CUDA.CuArray(low_values);
                    modulus = low_modulus,
                    mod = false,
                )
                high = mod.CuModVector{Float64}(
                    CUDA.CuArray(high_values);
                    modulus = high_modulus,
                    mod = false,
                )
                A = mod.KaratsubaVector{Float64}(
                    low,
                    high;
                    low_modulus = low_modulus,
                    high_modulus = high_modulus,
                    modulus = represented_modulus,
                )
                mod.initialize_plan!(A)
                A
            end

            multi_low_component = mod.low_part(multi_A)
            multi_high_component = mod.high_part(multi_A)
            multi_low_parent = parent(multi_low_component)
            multi_high_parent = parent(multi_high_component)
            multi_source_plan = multi_A.plan
            multi_low_before, multi_high_before, multi_plan_before =
                CUDA.device!(input_device) do
                    fill!(parent(multi_source_plan), 37)
                    (
                        Array(multi_low_parent),
                        Array(multi_high_parent),
                        Array(parent(multi_source_plan)),
                    )
                end

            multi_negative = CUDA.device!(different_caller_device) do
                result = -multi_A
                @test CUDA.device() == different_caller_device
                result
            end
            multi_result = CUDA.device!(input_device) do
                Array(multi_negative)
            end
            multi_low_after, multi_high_after, multi_plan_after =
                CUDA.device!(input_device) do
                    (
                        Array(multi_low_parent),
                        Array(multi_high_parent),
                        Array(parent(multi_source_plan)),
                    )
                end
            multi_expected = BigInt[
                Base.mod(
                    -(BigInt(low_values[i]) + BigInt(low_modulus) * BigInt(high_values[i])),
                    BigInt(represented_modulus),
                ) for i in eachindex(low_values)
            ]

            @test CUDA.device(parent(multi_low_component)) == input_device
            @test CUDA.device(parent(multi_high_component)) == input_device
            @test CUDA.device(parent(multi_source_plan)) == input_device
            @test CUDA.device(parent(mod.low_part(multi_negative))) == input_device
            @test CUDA.device(parent(mod.high_part(multi_negative))) == input_device
            @test CUDA.device(parent(multi_negative.plan)) == input_device
            @test multi_result == multi_expected
            @test multi_A.plan === multi_source_plan
            @test mod.low_part(multi_A) === multi_low_component
            @test mod.high_part(multi_A) === multi_high_component
            @test multi_low_after == multi_low_before
            @test multi_high_after == multi_high_before
            @test multi_plan_after == multi_plan_before
            @test (
                mod.low_modulus(multi_negative),
                mod.high_modulus(multi_negative),
                modulus(multi_negative),
            ) == (low_modulus, high_modulus, represented_modulus)
        end
    end

    @testset "PLUQ rank methods reuse the copy path and forward options" begin
        field_modulus = 101
        host_values = [
            1 2 3
            2 4 6
            1 1 0
        ]
        A = mod.CuModMatrix(host_values; modulus = field_modulus)
        source_parent = parent(A)
        source_before = Array(source_parent)
        source_size = size(A)
        source_modulus = modulus(A)
        options = mod.PLUQOptions(
            blocksize = 32,
            basecase = 32,
            lazy_q = false,
            nftb = 4,
            check_prime = true,
        )
        oracle_rank = _base_methods_test_rank_mod_prime(host_values, field_modulus)

        factorization = mod.pluq(A; options = options)
        @test LinearAlgebra.rank(factorization) == factorization.rank
        @test factorization.rank == oracle_rank
        @test factorization.LU !== A
        @test Array(parent(A)) == source_before

        @test LinearAlgebra.rank(A; options = options) == oracle_rank
        @test parent(A) === source_parent
        @test Array(parent(A)) == source_before
        @test size(A) == source_size
        @test modulus(A) == source_modulus

        composite = mod.CuModMatrix(Float32[1 0; 0 1]; modulus = 15)
        composite_parent = parent(composite)
        composite_before = Array(composite_parent)
        @test_throws mod.CuModMatrixModulusNotPrimeException LinearAlgebra.rank(
            composite;
            options = mod.PLUQOptions(check_prime = true),
        )
        @test parent(composite) === composite_parent
        @test Array(composite_parent) == composite_before

        for (case_name, host_values, expected_rank) in
            (("zero", zeros(Int, 3, 3), 0), ("full", [1 0 0; 0 1 0; 0 0 1], 3))
            @testset "rank boundary: $case_name" begin
                A = mod.CuModMatrix(host_values; modulus = field_modulus)
                source_parent = parent(A)
                source_before = Array(source_parent)
                source_size = size(A)
                source_modulus = modulus(A)
                oracle_rank = _base_methods_test_rank_mod_prime(host_values, field_modulus)

                @test oracle_rank == expected_rank

                factorization = mod.pluq(A; options = options)
                @test LinearAlgebra.rank(factorization) == expected_rank
                @test factorization.rank == oracle_rank
                @test factorization.LU !== A
                @test Array(parent(A)) == source_before

                @test LinearAlgebra.rank(A; options = options) == expected_rank
                @test parent(A) === source_parent
                @test Array(parent(A)) == source_before
                @test size(A) == source_size
                @test modulus(A) == source_modulus
            end
        end
    end
end

if CUDA.functional()
    test_base_methods_gpu()
end
