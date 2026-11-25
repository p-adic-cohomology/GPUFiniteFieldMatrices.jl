using Test
BitIntegers.@define_integers 24

const GFM_TEST_CUDA_EXT =
    GFM_TEST_CUDA_LOADED ?
    Base.get_extension(GPUFiniteFieldMatrices, :GPUFiniteFieldMatricesCUDAExt) : nothing

# Host-only precision, conversion, ownership, and inference contracts.
# CUDA-private helper/dispatch assertions are guarded by the selected vendor.

@testset "CPU Karatsuba Float32 and Float64 product precision plans" begin
    mod = GPUFiniteFieldMatrices

    for (T, bits) in ((Float32, 24), (Float64, 53))
        exact_limit = BigInt(2)^bits
        exact_max = BigInt(2)^bits - 1
        @test mod._karatsuba_checked_working_modulus(exact_limit, "endpoint", T) ==
              Int(exact_limit)
        @test_throws ArgumentError mod._karatsuba_checked_working_modulus(
            exact_limit + 1,
            "past endpoint",
            T,
        )
        @test mod._karatsuba_checked_sum_bound(
            Int(exact_limit - 1),
            1,
            "exact sum endpoint",
            T,
        ) == Int(exact_limit)
        @test_throws ArgumentError mod._karatsuba_checked_sum_bound(
            Int(exact_limit),
            1,
            "past exact sum endpoint",
            T,
        )
        near_root = isqrt(exact_max)
        near_dot_bound = mod._karatsuba_gemv_dot_bound(
            T,
            Int(exact_limit),
            2,
            2,
            "exact-endpoint GEMV dot",
        )
        @test near_dot_bound == exact_limit
        @test exact_limit - near_dot_bound < 4 * near_root + 4
        @test_throws ArgumentError mod._karatsuba_gemv_dot_bound(
            T,
            1,
            Int(near_root + 3),
            Int(near_root + 3),
            "past-limit GEMV dot",
        )

        product_plan = mod._karatsuba_matmul_product_plan(T, 13, 7, 13, 5, 5, 11)
        @test product_plan.output_capacity == 55
        @test product_plan.low.max_ops > 0
        @test product_plan.middle.max_ops > 0
        @test product_plan.high.max_ops > 0
        for product in (product_plan.low, product_plan.middle, product_plan.high)
            @test BigInt(product.modulus - 1) +
                  BigInt(product.max_ops) * BigInt(product.input_bound - 1)^2 <= exact_max
        end

        gemv_plan = mod._karatsuba_gemv_precision_plan(T, 2, 13, 7, 13, 5, 13, 13)
        @test gemv_plan.low_dot_bound == BigInt(2) * 12 * 12
        @test gemv_plan.middle_dot_bound == BigInt(2) * 18 * 16
        @test gemv_plan.high_dot_bound == BigInt(2) * 6 * 4
        @test gemv_plan.output_capacity == 169
        @test gemv_plan.high_total_bound == gemv_plan.middle_dot_bound

        if GFM_TEST_CUDA_LOADED
            # Evaluate the three-product identity independently with BigInt, then
            # compare the type-specific device helper's exact residue and split.
            radix = BigInt(13)
            out_radix = BigInt(5)
            capacity = BigInt(55)
            a_low, a_high, b_low, b_high = BigInt(12), BigInt(4), BigInt(3), BigInt(2)
            p0 = a_low * b_low
            p1 = (a_low + a_high) * (b_low + b_high)
            p2 = a_high * b_high
            expected =
                Base.mod((a_low + radix * a_high) * (b_low + radix * b_high), capacity)
            low, high = GFM_TEST_CUDA_EXT._karatsuba_recombine(
                T(Base.mod(p0, capacity)),
                T(Base.mod(p1, capacity)),
                T(Base.mod(p2, capacity)),
                T(Base.mod(radix, capacity)),
                T(Base.mod(radix^2, capacity)),
                T(out_radix),
                T(capacity),
            )
            @test typeof((low, high)) === Tuple{T,T}
            @test BigInt(low) + out_radix * BigInt(high) == expected
            @test BigInt(low) == Base.mod(expected, out_radix)

            # Exercise recombination at T's exact-integer endpoint with an output
            # split that forces both a low digit and a high quotient.
            endpoint_capacity = exact_limit
            endpoint_radix = BigInt(1) << (bits ÷ 2)
            endpoint_high = div(endpoint_capacity, endpoint_radix)
            endpoint_products = (exact_max, exact_max - 1, exact_max - 2)
            endpoint_expected = Base.mod(
                endpoint_products[1] +
                radix *
                (endpoint_products[2] - endpoint_products[1] - endpoint_products[3]) +
                radix^2 * endpoint_products[3],
                endpoint_capacity,
            )
            endpoint_low, endpoint_high_digit = GFM_TEST_CUDA_EXT._karatsuba_recombine(
                T(endpoint_products[1]),
                T(endpoint_products[2]),
                T(endpoint_products[3]),
                T(Base.mod(radix, endpoint_capacity)),
                T(Base.mod(radix^2, endpoint_capacity)),
                T(endpoint_radix),
                T(endpoint_capacity),
            )
            @test BigInt(endpoint_low) + endpoint_radix * BigInt(endpoint_high_digit) ==
                  endpoint_expected
            @test 0 <= BigInt(endpoint_low) < endpoint_radix
            @test 0 <= BigInt(endpoint_high_digit) < endpoint_high
        end

        # The GEMV budget rejects a dot product whose worst-case accumulator
        # reaches the first non-exact integer for the selected precision.
        if T === Float32
            @test_throws ArgumentError mod._karatsuba_gemv_precision_plan(
                T,
                2,
                3000,
                2,
                3000,
                2,
                3000,
                1,
            )
            @test_throws ArgumentError mod._karatsuba_matmul_product_plan(
                T,
                1,
                1,
                1,
                1,
                4097,
                4097,
            )
        end
    end

    if GFM_TEST_CUDA_LOADED
        cpu_matrix = mod.KaratsubaMatrix(
            zeros(Float32, 2, 3),
            zeros(Float32, 2, 3);
            low_modulus = 5,
            high_modulus = 4,
            modulus = 20,
        )
        cpu_vector_shape = GFM_TEST_CUDA_EXT._karatsuba_output_like(cpu_matrix, (2,))
        @test cpu_vector_shape isa mod.KaratsubaVector{Float32}
        @test size(cpu_vector_shape) == (2,)
        matrix_dispatch_type = mod.KaratsubaArray{Float32,2}
        vector_dispatch_type = mod.KaratsubaArray{Float32,1}
        @test which(
            LinearAlgebra.mul!,
            Tuple{vector_dispatch_type,matrix_dispatch_type,vector_dispatch_type},
        ).module === GFM_TEST_CUDA_EXT
        @test which(Base.:*, Tuple{matrix_dispatch_type,vector_dispatch_type}).module ===
              GFM_TEST_CUDA_EXT
        @test which(Base.:*, Tuple{matrix_dispatch_type,matrix_dispatch_type}).module ===
              GFM_TEST_CUDA_EXT
        @test_throws ArgumentError cpu_matrix * mod.KaratsubaVector(
            zeros(Float32, 3),
            zeros(Float32, 3);
            low_modulus = 5,
            high_modulus = 4,
            modulus = 20,
        )

        wrong_shape_a = mod.KaratsubaMatrix(
            zeros(Float32, 2, 3),
            zeros(Float32, 2, 3);
            low_modulus = 5,
            high_modulus = 4,
            modulus = 20,
        )
        wrong_shape_x = mod.KaratsubaVector(
            zeros(Float32, 2),
            zeros(Float32, 2);
            low_modulus = 5,
            high_modulus = 4,
            modulus = 20,
        )
        wrong_shape_y = mod.KaratsubaVector(
            fill(Float32(7), 2),
            fill(Float32(9), 2);
            low_modulus = 5,
            high_modulus = 4,
            modulus = 20,
        )
        @test_throws DimensionMismatch LinearAlgebra.mul!(
            wrong_shape_y,
            wrong_shape_a,
            wrong_shape_x,
        )
        @test low_part(wrong_shape_y) == fill(Float32(7), 2)
        @test high_part(wrong_shape_y) == fill(Float32(9), 2)

        destination = mod.KaratsubaMatrix(
            zeros(Float64, 1, 1),
            zeros(Float64, 1, 1);
            low_modulus = 2,
            high_modulus = 3,
            modulus = 6,
        )
        left32 = mod.KaratsubaMatrix(
            zeros(Float32, 1, 1),
            zeros(Float32, 1, 1);
            low_modulus = 2,
            high_modulus = 3,
            modulus = 6,
        )
        right64 = mod.KaratsubaMatrix(
            zeros(Float64, 1, 1),
            zeros(Float64, 1, 1);
            low_modulus = 2,
            high_modulus = 3,
            modulus = 6,
        )
        @test_throws ArgumentError LinearAlgebra.mul!(destination, left32, right64)
        @test Array(low_part(destination)) == zeros(Float64, 1, 1)
        @test Array(high_part(destination)) == zeros(Float64, 1, 1)
    end
end

@testset "CPU constructor and exact Karatsuba conversion" begin
    mod = GPUFiniteFieldMatrices

    if GFM_TEST_CUDA_LOADED
        identity_ctor =
            which(mod.CuModMatrix{Int}, Tuple{LinearAlgebra.UniformScaling,Int,Int})
        @test :modulus in Base.kwarg_decl(identity_ctor)
        @test which(mod.identity_matrix, Tuple{Type{mod.CuModMatrix{Int}},Int}).module ===
              GFM_TEST_CUDA_EXT
        @test :modulus in
              Base.kwarg_decl(which(Base.zeros, Tuple{Type{mod.CuModMatrix{Int}},Int,Int}))
        @test :modulus in
              Base.kwarg_decl(which(mod.rand, Tuple{Type{mod.CuModVector{Int}},Int}))
    end
    @test !isdefined(mod, :eye)
    @test all(
        name in names(mod) for
        name in (:modulus, :low_modulus, :high_modulus, :low_part, :high_part)
    )

    low = [1 2; 0 1]
    high = [3 4; 5 6]
    wrapped =
        mod.KaratsubaMatrix(low, high; low_modulus = 2, high_modulus = 7, modulus = 19)
    @test low_part(wrapped) === low
    @test high_part(wrapped) === high
    @test (
        low_modulus(wrapped),
        high_modulus(wrapped),
        GPUFiniteFieldMatrices.modulus(wrapped),
    ) == (2, 7, 19)
    @test isconcretetype(typeof(wrapped))
    @test typeof(wrapped).parameters[3] === typeof(low)
    expected_plan_type = mod._karatsuba_plan_type(typeof(low))
    @test typeof(wrapped).parameters[4] === expected_plan_type
    @test fieldtype(typeof(wrapped), :data1) === Matrix{Int}
    @test fieldtype(typeof(wrapped), :data2) === Matrix{Int}
    @test fieldtype(typeof(wrapped), :plan) === Union{Nothing,expected_plan_type}
    @test fieldtype(typeof(wrapped), :N1) === Int
    @test fieldtype(typeof(wrapped), :N2) === Int
    @test fieldtype(typeof(wrapped), :M) === Int
    @test mod.from_karatsuba(Int, wrapped) == low + 2 .* high
    shorthand = mod.KaratsubaMatrix(low, high; modulus = 19)
    @test (
        low_modulus(shorthand),
        high_modulus(shorthand),
        GPUFiniteFieldMatrices.modulus(shorthand),
    ) == (19, 19, 19)
    inferred = mod.KaratsubaVector(Int[1, 2], Int[0, 1])
    @test (
        low_modulus(inferred),
        high_modulus(inferred),
        GPUFiniteFieldMatrices.modulus(inferred),
    ) == (
        mod.find_max_ops_karatsuba(Int, 2),
        mod.find_max_ops_karatsuba(Int, 2),
        mod.find_max_ops_karatsuba(Int, 2),
    )
    auto_split = mod.to_karatsuba(Float64, Float64[1 2; 3 4])
    @test GPUFiniteFieldMatrices.modulus(auto_split) < 2^51
    @test mod.from_karatsuba(Float64, auto_split) == Float64[1 2; 3 4]

    wide_source = reshape(Int128[Int128(2)^100+17, Int128(2)^99+3], 1, 2)
    wide_split = mod.to_karatsuba(
        Int64,
        wide_source;
        low_modulus = 2^50,
        high_modulus = 2^50,
        modulus = 19,
    )
    @test low_part(wide_split) isa Matrix{Int64}
    @test high_part(wide_split) isa Matrix{Int64}
    @test mod.from_karatsuba(Int128, wide_split) == wide_source
    @test_throws InexactError mod.from_karatsuba(Float64, wide_split)

    typed_float_split = mod.to_karatsuba(
        Float32,
        Int128[2^60+123];
        low_modulus = 2^40,
        high_modulus = 2^20,
        modulus = 97,
    )
    @test low_part(typed_float_split) isa Vector{Float32}
    @test high_part(typed_float_split) isa Vector{Float32}
    @test mod.from_karatsuba(Int128, typed_float_split) == Int128[2^60+123]
    @test_throws InexactError mod.from_karatsuba(Float32, typed_float_split)
    @test_throws InexactError mod.to_karatsuba(
        Int8,
        BigInt[256];
        low_modulus = 1,
        high_modulus = 1,
        modulus = 97,
    )

    zero_matrix = Base.zeros(
        mod.KaratsubaMatrix{Float32,Matrix{Float32}},
        2,
        3;
        low_modulus = 3,
        high_modulus = 5,
        modulus = 19,
    )
    zero_vector = Base.zeros(
        mod.KaratsubaVector{Float32,Vector{Float32}},
        4;
        low_modulus = 3,
        high_modulus = 5,
        modulus = 19,
    )
    matrix_plan_type = mod._karatsuba_plan_type(Matrix{Float32})
    vector_plan_type = mod._karatsuba_plan_type(Vector{Float32})
    @test typeof(zero_matrix) ===
          mod.KaratsubaMatrix{Float32,Matrix{Float32},matrix_plan_type}
    @test typeof(zero_vector) ===
          mod.KaratsubaVector{Float32,Vector{Float32},vector_plan_type}
    @test zero_matrix.plan === nothing
    @test zero_vector.plan === nothing
    @test low_part(zero_matrix) isa Matrix{Float32}
    @test high_part(zero_matrix) isa Matrix{Float32}
    @test low_part(zero_vector) isa Vector{Float32}
    @test mod.from_karatsuba(Float32, zero_matrix) == zeros(Float32, 2, 3)
    @test mod.from_karatsuba(Float32, zero_vector) == zeros(Float32, 4)
    @test (
        low_modulus(zero_matrix),
        high_modulus(zero_matrix),
        GPUFiniteFieldMatrices.modulus(zero_matrix),
    ) == (3, 5, 19)
    zero_shorthand = Base.zeros(mod.KaratsubaVector{Int,Vector{Int}}, 3; modulus = 17)
    @test (
        low_modulus(zero_shorthand),
        high_modulus(zero_shorthand),
        GPUFiniteFieldMatrices.modulus(zero_shorthand),
    ) == (17, 17, 17)
    @test_throws ArgumentError Base.zeros(mod.KaratsubaVector{Int,Vector{Int}}, 3)
    @test_throws ArgumentError Base.zeros(
        mod.KaratsubaMatrix{Int,Matrix{Int}},
        2,
        2;
        low_modulus = 3,
    )
    explicit_plan_zero = @inferred Base.zeros(
        mod.KaratsubaMatrix{Float32,Matrix{Float32},mod.CuModMatrix{Float32}},
        2,
        2;
        modulus = 19,
    )
    @test typeof(explicit_plan_zero) ===
          mod.KaratsubaMatrix{Float32,Matrix{Float32},matrix_plan_type}
    @test explicit_plan_zero.plan === nothing
    if GFM_TEST_CUDA_LOADED
        @test_throws ArgumentError Base.zeros(
            mod.KaratsubaMatrix{
                Float32,
                Matrix{Float32},
                CUDA.CuArray{Float32,2,CUDA.DeviceMemory},
            },
            2,
            2;
            modulus = 19,
        )
    end
    @test_throws MethodError Base.zeros(
        mod.KaratsubaMatrix{Int,Matrix{Int}},
        2,
        2;
        low_modulus = 3,
        high_modulus = 5,
        modulus = 19,
        use_gpu = false,
    )
    @test !hasmethod(Base.zeros, Tuple{Type{mod.KaratsubaMatrix},Type{Float32},Int,Int})
    @test !isdefined(mod, :KaratsubaZeros)
end

@testset "Fixed-width precision budgets" begin
    mod = GPUFiniteFieldMatrices
    type_budgets = (
        (Float16, 11, 9),
        (Float32, 24, 22),
        (Float64, 53, 51),
        (Int8, 7, 7),
        (Int16, 15, 15),
        (Int32, 31, 31),
        (Int64, 63, 63),
        (Int128, 127, 127),
        (UInt8, 8, 8),
        (UInt16, 16, 16),
        (UInt32, 32, 32),
        (UInt64, 64, 64),
        (UInt128, 128, 128),
        (BitIntegers.Int256, 255, 255),
        (BitIntegers.UInt256, 256, 256),
        (Int24, 23, 23),
        (UInt24, 24, 24),
    )

    expected_karatsuba =
        (bits, n) -> begin
            bound = ((BigInt(1) << bits) - 1) ÷ BigInt(n)^2 - 1
            Int(max(1, min(bound, BigInt(typemax(Int)))))
        end
    expected_cumod = (bits, n) -> begin
        maxterm = BigInt(max(n - 1, 1))^2
        value = ((BigInt(1) << bits) - 1) ÷ maxterm - 1
        Int(max(0, min(value, BigInt(typemax(Int)))))
    end

    for (T, shared_bits, karatsuba_bits) in type_budgets
        @test mod.get_bits(T) == shared_bits
        @test mod.find_max_ops_karatsuba(T, 4) == expected_karatsuba(karatsuba_bits, 4)
        @test mod.find_max_ops(T, 5) == expected_cumod(shared_bits, 5)

        low = fill(zero(T), 1, 1)
        high = fill(one(T), 1, 1)
        wrapped =
            mod.KaratsubaMatrix(low, high; low_modulus = 2, high_modulus = 3, modulus = 5)
        @test low_part(wrapped) === low
        @test high_part(wrapped) === high
        @test wrapped.plan === nothing
        @test typeof(wrapped).parameters[3] === Matrix{T}
        @test typeof(wrapped).parameters[4] === mod._karatsuba_plan_type(Matrix{T})
    end

    @test mod.get_bits(Int) == 8 * sizeof(Int) - 1
    @test mod.get_bits(UInt) == 8 * sizeof(UInt)
    @test mod.find_max_ops_karatsuba(Int64, 2) == expected_karatsuba(63, 2)
    @test mod.find_max_ops_karatsuba(UInt64, 2) == expected_karatsuba(64, 2)
    @test mod.find_max_ops(Int64, 2) == expected_cumod(63, 2)
    @test mod.find_max_ops(UInt64, 2) == expected_cumod(64, 2)
    @test mod.find_max_ops_karatsuba(Int64, 2) == 2^61 - 2
    @test mod.find_max_ops_karatsuba(UInt64, 2) == 2^62 - 2
    @test mod.find_max_ops(Int64, 2) == typemax(Int) - 1
    @test mod.find_max_ops(UInt64, 2) == typemax(Int)
    @test mod.find_max_ops_karatsuba(Int128, 1) == typemax(Int)
    @test mod.find_max_ops_karatsuba(UInt128, 1) == typemax(Int)
    @test mod.find_max_ops(Int128, 2) == typemax(Int)
    @test mod.find_max_ops(UInt128, 2) == typemax(Int)
    @test mod.find_max_ops_karatsuba(BitIntegers.Int256, 1) == typemax(Int)
    @test mod.find_max_ops_karatsuba(BitIntegers.UInt256, 1) == typemax(Int)
    @test mod.find_max_ops(BitIntegers.Int256, 2) == typemax(Int)
    @test mod.find_max_ops(BitIntegers.UInt256, 2) == typemax(Int)

    wide_budget_dimension = BigInt(1) << 100
    @test mod.find_max_ops_karatsuba(BitIntegers.Int256, wide_budget_dimension) ==
          expected_karatsuba(255, wide_budget_dimension)
    @test mod.find_max_ops_karatsuba(BitIntegers.UInt256, wide_budget_dimension) ==
          expected_karatsuba(256, wide_budget_dimension)
    @test mod.find_max_ops(BitIntegers.Int256, wide_budget_dimension) ==
          expected_cumod(255, wide_budget_dimension)
    @test mod.find_max_ops(BitIntegers.UInt256, wide_budget_dimension) ==
          expected_cumod(256, wide_budget_dimension)

    factory_zero = Base.zeros(
        mod.KaratsubaMatrix{UInt64,Matrix{UInt64}},
        1,
        1;
        low_modulus = 2,
        high_modulus = 3,
        modulus = 5,
    )
    factory_plan_type = mod._karatsuba_plan_type(Matrix{UInt64})
    @test typeof(factory_zero) ===
          mod.KaratsubaMatrix{UInt64,Matrix{UInt64},factory_plan_type}
    @test factory_zero.plan === nothing

    for unsupported in
        (BigInt, BigFloat, Bool, ComplexF64, AbstractFloat, Integer, Signed, Unsigned)
        unsupported_error = try
            mod.get_bits(unsupported)
            nothing
        catch err
            err
        end
        @test unsupported_error isa ArgumentError
        if unsupported_error isa Exception
            @test occursin(string(unsupported), sprint(showerror, unsupported_error))
        end
    end
    @test_throws ArgumentError mod.find_max_ops_karatsuba(BigInt, 2)
    @test_throws ArgumentError mod.find_max_ops(BigFloat, 2)
    @test_throws ArgumentError mod.get_bits(1)
end

@testset "Strict CPU Karatsuba storage and inference" begin
    mod = GPUFiniteFieldMatrices
    low_matrix = Int[1 2; 3 4]
    high_matrix = Int[5 6; 7 8]
    matrix = @inferred mod.KaratsubaArray{Int,2}(
        low_matrix,
        high_matrix;
        low_modulus = 2,
        high_modulus = 3,
        modulus = 11,
    )
    concrete_matrix = @inferred mod.KaratsubaMatrix{Int,Matrix{Int},mod.CuModMatrix{Int}}(
        low_matrix,
        high_matrix;
        low_modulus = 2,
        high_modulus = 3,
        modulus = 11,
    )
    @test typeof(concrete_matrix) === typeof(matrix)
    @test_throws ArgumentError mod.KaratsubaMatrix{Int,AbstractArray{Int,2}}(
        low_matrix,
        high_matrix;
        low_modulus = 2,
        high_modulus = 3,
        modulus = 11,
    )
    plan_typed_matrix = @inferred mod.KaratsubaMatrix{Int,Matrix{Int},mod.CuModMatrix{Int}}(
        low_matrix,
        high_matrix;
        low_modulus = 2,
        high_modulus = 3,
        modulus = 11,
    )
    int_plan_type = mod._karatsuba_plan_type(Matrix{Int})
    @test typeof(plan_typed_matrix) === mod.KaratsubaMatrix{Int,Matrix{Int},int_plan_type}
    @test plan_typed_matrix.plan === nothing
    typed_plan_matrix = mod.KaratsubaMatrix{Int,Matrix{Int},Matrix{Int}}(
        low_matrix,
        high_matrix;
        low_modulus = 2,
        high_modulus = 3,
        modulus = 11,
    )
    @test typeof(typed_plan_matrix) === mod.KaratsubaMatrix{Int,Matrix{Int},Matrix{Int}}
    @test (@inferred mod.initialize_plan!(typed_plan_matrix)) === typed_plan_matrix
    @test typed_plan_matrix.plan isa Matrix{Int}
    matrix_value = @inferred mod.from_karatsuba(Int, matrix)
    @test matrix_value == low_matrix + 2 .* high_matrix
    @test matrix_value isa Matrix{Int}

    low_vector = Int[1, 2, 3]
    high_vector = Int[4, 5, 6]
    vector = @inferred mod.KaratsubaArray{Int,1}(
        low_vector,
        high_vector;
        low_modulus = 2,
        high_modulus = 3,
        modulus = 11,
    )
    vector_value = @inferred mod.from_karatsuba(Int, vector)
    @test vector_value == low_vector + 2 .* high_vector
    @test vector_value isa Vector{Int}

    cpu_zero_matrix = @inferred Base.zeros(
        mod.KaratsubaMatrix{Float32,Matrix{Float32}},
        2,
        3;
        modulus = 19,
    )
    cpu_zero_vector = @inferred Base.zeros(
        mod.KaratsubaVector{Float32,Vector{Float32}},
        4;
        low_modulus = 3,
        high_modulus = 5,
        modulus = 19,
    )
    matrix_plan_type = mod._karatsuba_plan_type(Matrix{Float32})
    vector_plan_type = mod._karatsuba_plan_type(Vector{Float32})
    @test typeof(cpu_zero_matrix) ===
          mod.KaratsubaMatrix{Float32,Matrix{Float32},matrix_plan_type}
    @test typeof(cpu_zero_vector) ===
          mod.KaratsubaVector{Float32,Vector{Float32},vector_plan_type}
    @test cpu_zero_matrix.plan === nothing
    @test cpu_zero_vector.plan === nothing

    if GFM_TEST_CUDA_LOADED
        gpu_kwargs_type = NamedTuple{(:modulus,),Tuple{Int}}
        gpu_matrix_type = mod.KaratsubaMatrix{Float32,mod.CuModMatrix{Float32}}
        gpu_vector_type = mod.KaratsubaVector{Float32,mod.CuModVector{Float32}}
        default_matrix_storage =
            GFM_TEST_CUDA_EXT._default_cumod_storage_type(Float32, Val(2))
        default_vector_storage =
            GFM_TEST_CUDA_EXT._default_cumod_storage_type(Float32, Val(1))
        gpu_matrix_return = Core.Compiler.return_type(
            Core.kwcall,
            Tuple{gpu_kwargs_type,typeof(Base.zeros),Type{gpu_matrix_type},Int,Int},
        )
        gpu_vector_return = Core.Compiler.return_type(
            Core.kwcall,
            Tuple{gpu_kwargs_type,typeof(Base.zeros),Type{gpu_vector_type},Int},
        )
        @test gpu_matrix_return ===
              mod.KaratsubaMatrix{Float32,default_matrix_storage,default_matrix_storage}
        @test gpu_vector_return ===
              mod.KaratsubaVector{Float32,default_vector_storage,default_vector_storage}
        # CUDA's untyped default alias selects its concrete backend storage at
        # runtime; strict inference for that default is intentionally deferred.

        raw_matrix_storage = CUDA.CuArray{Float32,2,CUDA.DeviceMemory}
        raw_vector_storage = CUDA.CuArray{Float32,1,CUDA.DeviceMemory}
        raw_matrix_type = mod.KaratsubaMatrix{Float32,raw_matrix_storage}
        raw_vector_type = mod.KaratsubaVector{Float32,raw_vector_storage}
        raw_matrix_return = Core.Compiler.return_type(
            Core.kwcall,
            Tuple{gpu_kwargs_type,typeof(Base.zeros),Type{raw_matrix_type},Int,Int},
        )
        raw_vector_return = Core.Compiler.return_type(
            Core.kwcall,
            Tuple{gpu_kwargs_type,typeof(Base.zeros),Type{raw_vector_type},Int},
        )
        @test raw_matrix_return ===
              mod.KaratsubaMatrix{Float32,raw_matrix_storage,default_matrix_storage}
        @test raw_vector_return ===
              mod.KaratsubaVector{Float32,raw_vector_storage,default_vector_storage}
    end

    too_large_for_int = BigInt(typemax(Int)) + 1
    @test_throws InexactError mod.KaratsubaMatrix(
        low_matrix,
        high_matrix;
        low_modulus = too_large_for_int,
        high_modulus = 3,
        modulus = 11,
    )
    @test_throws InexactError mod.KaratsubaMatrix(
        low_matrix,
        high_matrix;
        low_modulus = 2,
        high_modulus = too_large_for_int,
        modulus = 11,
    )
    @test_throws InexactError mod.KaratsubaMatrix(
        low_matrix,
        high_matrix;
        low_modulus = 2,
        high_modulus = 3,
        modulus = too_large_for_int,
    )
    @test_throws InexactError Base.zeros(
        mod.KaratsubaMatrix{Int,Matrix{Int}},
        1,
        1;
        low_modulus = too_large_for_int,
        high_modulus = 3,
        modulus = 11,
    )
    if GFM_TEST_CUDA_LOADED
        unsupported_component_type = typeof(view(zeros(Float32, 2, 2), :, :))
        unsupported_factory_type =
            Core.apply_type(mod.KaratsubaMatrix{Float32}, unsupported_component_type)
        @test_throws ArgumentError Base.zeros(unsupported_factory_type, 2, 2; modulus = 19)
    end

    @test_throws ArgumentError mod.KaratsubaMatrix(
        low_matrix,
        view(high_matrix, :, :);
        low_modulus = 2,
        high_modulus = 3,
        modulus = 11,
    )
end
