@testset "CPU API surface and dispatch" begin
    mod = GPUFiniteFieldMatrices
    matrix_type = mod.CuModMatrix{Float32}
    karatsuba_type = mod.KaratsubaArray{Int,2}

    exported = (
        :CuModArray,
        :CuModMatrix,
        :CuModVector,
        :KaratsubaArray,
        :KaratsubaMatrix,
        :KaratsubaVector,
        :to_karatsuba,
        :from_karatsuba,
        :modulus,
        :low_modulus,
        :high_modulus,
        :low_part,
        :high_part,
        :PLUQOptions,
        :PLUQ,
        :pluq,
        :pluq!,
        :pluq_batched!,
        :pluq_batched,
        :inverse_batched,
        :is_invertible,
        :is_invertible_with_inverse,
        :right_inverse,
        :left_inverse,
        :change_modulus,
        :identity_matrix,
        :elementwise_multiply!,
        :scalar_add!,
        :scalar_sub!,
        :mod_elements!,
    )
    @test all(name -> Base.isexported(mod, name), exported)
    module_names = filter(!=(nameof(mod)), names(mod; all = true, imported = false))
    @test Set(filter(name -> Base.isexported(mod, name), module_names)) == Set(exported)

    public_names = (
        :add!,
        :sub!,
        :neg!,
        :zero!,
        :zeros,
        :rand,
        :similar_with_modulus,
        :change_modulus_no_alloc!,
        :mod_inv,
        :karatsuba_gemv!,
        :inverse_pluq,
        :pluq_batched_4x4!,
        :pluq_batched_8x8!,
        :pluq_batched_16x16!,
        :pluq_batched_32x32!,
        :pluq_batched_NxN!,
        :inverse_batched_4x4!,
        :inverse_batched_8x8!,
        :inverse_batched_16x16!,
        :inverse_batched_32x32!,
        :upper_triangular_inverse_no_copy,
        :lower_triangular_inverse_no_copy,
        :stripe_mul,
        :stripe_mul!,
        :TILE_WIDTH,
    )
    @test all(name -> isdefined(mod, name), public_names)
    @test all(name -> Base.ispublic(mod, name), public_names)
    @test all(name -> !Base.isexported(mod, name), public_names)
    tiny_inverse_functions = (
        :inverse_batched_4x4!,
        :inverse_batched_8x8!,
        :inverse_batched_16x16!,
        :inverse_batched_32x32!,
    )
    empty_batch = mod.CuModMatrix[]
    @test all(
        name ->
            applicable(getfield(mod, name), empty_batch) &&
            applicable(getfield(mod, name), empty_batch, empty_batch),
        tiny_inverse_functions,
    )
    @test all(
        name ->
            getfield(mod, name)(empty_batch) === empty_batch &&
            getfield(mod, name)(empty_batch, empty_batch) === empty_batch,
        tiny_inverse_functions,
    )
    @test !isdefined(mod, :inverse_batched!)
    internal_triangular_helpers = (
        :_lower_triangular_inv_32!,
        :_upper_triangular_inv_32!,
        :_lower_triangular_inv_32_kernel!,
        :_upper_triangular_inv_32_kernel!,
    )
    @test all(name -> isdefined(mod, name), internal_triangular_helpers)
    @test all(name -> !Base.ispublic(mod, name), internal_triangular_helpers)
    @test all(name -> !Base.isexported(mod, name), internal_triangular_helpers)
    expected_public = union(Set(exported), Set(public_names))
    @test Set(filter(name -> Base.ispublic(mod, name), module_names)) == expected_public
    @test !Base.ispublic(mod, :pluq_check_identity)
    @test !Base.isexported(mod, :pluq_check_identity)

    removed = (
        :CuModcopy,
        :forward_sub_gpu_type_32,
        :backward_sub_gpu_type_32,
        :forward_sub_kernel_32,
        :backward_sub_kernel_32,
        :mulN!,
        :inverse,
        :inverse_new,
        :is_invertible_new,
        :inverse_pluq_new,
        :right_inverse_new,
        :left_inverse_new,
        :pluq_new,
        :pluq_new!,
        :pluq_new_batch,
        :inverse_new_batch,
        :PLUQFactorization,
        :pluq_extract_L,
        :pluq_extract_U,
        :KMatMul!,
        :KMatMul_gemv!,
        :Karatsubacopy,
        :MatToKMat,
        :KMatToMat,
    )
    @test all(name -> !isdefined(mod, name), removed)

    @test !hasmethod(mod.CuModMatrix, Tuple{Matrix{Int},Int})
    @test !hasmethod(mod.CuModMatrix{Float64}, Tuple{Matrix{Int},Int})
    @test !hasmethod(mod.KaratsubaMatrix, Tuple{Matrix{Int},Matrix{Int},Int})
    @test !hasmethod(mod.KaratsubaMatrix, Tuple{Matrix{Int},Matrix{Int},Int,Int,Int})
    @test !hasmethod(mod.KaratsubaVector, Tuple{Vector{Int},Vector{Int},Int,Int,Int})
    @test which(Base.copy, Tuple{matrix_type}).module === mod
    @test which(Base.inv, Tuple{matrix_type}).module === mod
    @test which(mod.pluq, Tuple{matrix_type}).module === mod
    @test which(mod.is_invertible, Tuple{matrix_type}).module === mod
    @test which(getproperty, Tuple{mod.PLUQ{Float32},Symbol}).module === mod
    @test which(propertynames, Tuple{mod.PLUQ{Float32}}).module === mod
    @test which(mod.similar_with_modulus, Tuple{matrix_type,Int}).module === mod
    @test which(Base.similar, Tuple{matrix_type,Int}).module !== mod
    @test which(LinearAlgebra.mul!, Tuple{matrix_type,matrix_type,matrix_type}).module ===
          mod
    @test which(LinearAlgebra.mul!, Tuple{karatsuba_type,karatsuba_type,karatsuba_type}).module ===
          mod
    allocating_karatsuba_type =
        mod.KaratsubaArray{Float64,2,mod.CuModMatrix{Float64},mod.CuModMatrix{Float64}}
    allocating_product_method =
        which(Base.:*, Tuple{allocating_karatsuba_type,allocating_karatsuba_type})
    @test allocating_product_method.module === mod
    @test allocating_product_method ===
          which(Base.:*, Tuple{mod.KaratsubaArray{Float64,2},mod.KaratsubaArray{Float64,2}})
    cpu_karatsuba_a = mod.KaratsubaMatrix(
        zeros(Float64, 1, 1),
        zeros(Float64, 1, 1);
        low_modulus = 2,
        high_modulus = 3,
        modulus = 5,
    )
    cpu_karatsuba_b = mod.KaratsubaMatrix(
        zeros(Float64, 1, 1),
        zeros(Float64, 1, 1);
        low_modulus = 2,
        high_modulus = 3,
        modulus = 5,
    )
    @test_throws ArgumentError cpu_karatsuba_a * cpu_karatsuba_b

    cpu_karatsuba_float32_a = mod.KaratsubaMatrix(
        reshape(Float32[0], 1, 1),
        reshape(Float32[2], 1, 1);
        low_modulus = 2,
        high_modulus = 3,
        modulus = 5,
    )
    cpu_karatsuba_float32_b = mod.KaratsubaMatrix(
        reshape(Float32[0], 1, 1),
        reshape(Float32[2], 1, 1);
        low_modulus = 2,
        high_modulus = 3,
        modulus = 5,
    )
    cpu_karatsuba_int64_a = mod.KaratsubaMatrix(
        zeros(Int64, 1, 1),
        zeros(Int64, 1, 1);
        low_modulus = 2,
        high_modulus = 3,
        modulus = 5,
    )
    cpu_karatsuba_int64_b = mod.KaratsubaMatrix(
        zeros(Int64, 1, 1),
        zeros(Int64, 1, 1);
        low_modulus = 2,
        high_modulus = 3,
        modulus = 5,
    )
    @test which(
        Base.:*,
        Tuple{typeof(cpu_karatsuba_float32_a),typeof(cpu_karatsuba_float32_b)},
    ).module === mod
    @test which(
        Base.:*,
        Tuple{typeof(cpu_karatsuba_int64_a),typeof(cpu_karatsuba_int64_b)},
    ).module === mod
    @test which(Base.:*, Tuple{typeof(cpu_karatsuba_float32_a),typeof(cpu_karatsuba_a)}).module ===
          mod
    @test which(Base.:*, Tuple{typeof(cpu_karatsuba_a),typeof(cpu_karatsuba_float32_a)}).module ===
          mod
    @test_throws ArgumentError cpu_karatsuba_float32_a * cpu_karatsuba_float32_b
    @test_throws ArgumentError cpu_karatsuba_int64_a * cpu_karatsuba_int64_b
    @test_throws ArgumentError cpu_karatsuba_float32_a * cpu_karatsuba_a
    @test_throws ArgumentError cpu_karatsuba_a * cpu_karatsuba_float32_a
    @test which(Base.copy, Tuple{karatsuba_type}).module === mod
    @test which(
        mod.karatsuba_gemv!,
        Tuple{karatsuba_type,karatsuba_type,karatsuba_type},
    ).module === mod

    # Resolve the raw launch signatures without allocating CuArrays or invoking
    # a kernel. GEMV uses matrix/vector storage and the baseline 7/5 arities;
    # GEMM keeps its matrix-only launch signatures.
    raw_matrix_storage = CUDA.CuArray{Float64,2,CUDA.DeviceMemory}
    raw_vector_storage = CUDA.CuArray{Float64,1,CUDA.DeviceMemory}
    kernel_dispatch_cases = (
        (
            mod.karatsuba_matmul_kernel_1!,
            Tuple{
                raw_matrix_storage,
                raw_matrix_storage,
                raw_matrix_storage,
                Int,
                Int,
                Int,
                Int,
                Int,
                Int,
                Int,
                Int,
                raw_matrix_storage,
                raw_matrix_storage,
                raw_matrix_storage,
                Int,
                Int,
                Int,
                Int,
                Int,
                Int,
                Int,
                Int,
            },
            22,
        ),
        (
            mod.karatsuba_matmul_kernel_2!,
            Tuple{
                raw_matrix_storage,
                raw_matrix_storage,
                raw_matrix_storage,
                Int,
                Int,
                Int,
                Int,
                Int,
                Int,
                Int,
                Int,
                Float64,
                Float64,
                Float64,
                Float64,
            },
            15,
        ),
        (
            mod.karatsuba_matmul_kernel_1!,
            Tuple{
                raw_matrix_storage,
                raw_matrix_storage,
                raw_matrix_storage,
                raw_vector_storage,
                raw_vector_storage,
                raw_vector_storage,
                Int,
            },
            7,
        ),
        (
            mod.karatsuba_matmul_kernel_2!,
            Tuple{raw_vector_storage,raw_vector_storage,raw_vector_storage,Int,Int},
            5,
        ),
    )
    for (kernel, signature, positional_arity) in kernel_dispatch_cases
        selected_method = hasmethod(kernel, signature) ? which(kernel, signature) : nothing
        @test selected_method !== nothing
        if selected_method !== nothing
            @test selected_method.module === mod
            @test selected_method.nargs == positional_arity + 1
        end
    end

    @test which(mod.to_karatsuba, Tuple{Matrix{Int}}).module === mod
    @test which(mod.to_karatsuba, Tuple{Type{Int},Matrix{Int}}).module === mod
    @test !hasmethod(mod.to_karatsuba, Tuple{Matrix{Int},Int})
    @test !hasmethod(mod.to_karatsuba, Tuple{Type{Int},Matrix{Int},Int})
    @test which(mod.from_karatsuba, Tuple{Type{Int},karatsuba_type}).module === mod
    @test which(LinearAlgebra.rmul!, Tuple{matrix_type,Float64}).module === mod
    @test which(LinearAlgebra.lmul!, Tuple{Float64,matrix_type}).module === mod

    # Keyword signatures stay CPU-inspectable, so the public names and removed
    # positional overrides are checked even when no CUDA device is available.
    vector_type = mod.CuModVector{Float32}
    keyword_decl(f, signature) = Base.kwarg_decl(which(f, signature))
    keyword_signatures = (
        (mod.add!, Tuple{mod.CuModArray,mod.CuModArray,mod.CuModArray}),
        (mod.sub!, Tuple{mod.CuModArray,mod.CuModArray,mod.CuModArray}),
        (mod.neg!, Tuple{mod.CuModArray,mod.CuModArray}),
        (mod.elementwise_multiply!, Tuple{mod.CuModArray,mod.CuModArray,mod.CuModArray}),
        (mod.scalar_add!, Tuple{mod.CuModArray,mod.CuModArray,Int}),
        (mod.scalar_sub!, Tuple{mod.CuModArray,mod.CuModArray,Int}),
        (mod.rscalar_sub!, Tuple{mod.CuModArray,mod.CuModArray,Int}),
        (mod.mul_elementwise!, Tuple{mod.CuModArray,mod.CuModArray,mod.CuModArray}),
        (mod.div!, Tuple{mod.CuModArray,mod.CuModArray,Int}),
        (mod.mod_elements!, Tuple{mod.CuModArray}),
    )
    for (f, signature) in keyword_signatures
        @test :modulus_override in keyword_decl(f, signature)
        @test :mod_N ∉ keyword_decl(f, signature)
    end
    @test !hasmethod(mod.neg!, Tuple{mod.CuModArray,mod.CuModArray,Int})
    @test !hasmethod(mod.mod_elements!, Tuple{mod.CuModArray,Int})
    @test !hasmethod(
        mod.elementwise_multiply!,
        Tuple{mod.CuModArray,mod.CuModArray,mod.CuModArray,Int},
    )
    @test !hasmethod(mod.scalar_add!, Tuple{mod.CuModArray,mod.CuModArray,Int,Int})
    @test !hasmethod(mod.scalar_sub!, Tuple{mod.CuModArray,mod.CuModArray,Int,Int})
    @test !hasmethod(mod.rscalar_sub!, Tuple{mod.CuModArray,mod.CuModArray,Int,Int})
    @test !hasmethod(
        mod.mul_elementwise!,
        Tuple{mod.CuModArray,mod.CuModArray,mod.CuModArray,Int},
    )
    @test !hasmethod(mod.div!, Tuple{mod.CuModArray,mod.CuModArray,Int,Int})
    @test :modulus_override in
          keyword_decl(LinearAlgebra.mul!, Tuple{matrix_type,matrix_type,Float64})
    scalar_mul_keywords =
        keyword_decl(LinearAlgebra.mul!, Tuple{matrix_type,matrix_type,Int})
    @test :mod_N ∉ scalar_mul_keywords
    @test !hasmethod(LinearAlgebra.mul!, Tuple{mod.CuModArray,mod.CuModArray,Int,Int})
    @test :input_bound in
          keyword_decl(LinearAlgebra.mul!, Tuple{matrix_type,matrix_type,matrix_type})
    @test :max_ops in
          keyword_decl(LinearAlgebra.mul!, Tuple{matrix_type,matrix_type,matrix_type})
    @test :modulus_override in
          keyword_decl(LinearAlgebra.mul!, Tuple{matrix_type,matrix_type,matrix_type})
    @test keyword_decl(LinearAlgebra.mul!, Tuple{vector_type,matrix_type,vector_type}) ==
          [:input_bound, :modulus_override]
    @test :maxopsOverride ∉
          keyword_decl(LinearAlgebra.mul!, Tuple{vector_type,matrix_type,vector_type})
    @test Set(keyword_decl(mod.stripe_mul!, Tuple{vector_type,matrix_type,vector_type})) ==
          Set([:max_ops, :input_bound, :modulus_override, :respect_max_ops])
    @test Set(keyword_decl(mod.stripe_mul!, Tuple{matrix_type,matrix_type,matrix_type})) ==
          Set([:max_ops, :input_bound, :modulus_override])
    @test Set(keyword_decl(mod.stripe_mul, Tuple{matrix_type,matrix_type})) ==
          Set([:max_ops, :input_bound, :modulus_override])
    @test Set(keyword_decl(mod.stripe_mul, Tuple{matrix_type,vector_type})) ==
          Set([:max_ops, :input_bound, :modulus_override])
    @test !hasmethod(mod.stripe_mul, Tuple{matrix_type,matrix_type,Int})
    @test !hasmethod(mod.stripe_mul!, Tuple{matrix_type,matrix_type,matrix_type,Int})
    @test !isdefined(mod, :mat_mul_gpu_type)
    @test !isdefined(mod, :mat_mul_type_inplace!)
    @test which(mod.stripe_mul, Tuple{matrix_type,matrix_type}).module === mod
    @test which(mod.stripe_mul, Tuple{matrix_type,vector_type}).module === mod
    @test which(mod.stripe_mul!, Tuple{matrix_type,matrix_type,matrix_type}).module === mod
    @test which(mod.stripe_mul!, Tuple{vector_type,matrix_type,vector_type}).module === mod

    empty_batch = mod.CuModMatrix{Float32}[]
    @test isempty(mod.pluq_batched!(empty_batch))
    @test isempty(mod.pluq_batched(empty_batch))
    @test isempty(mod.pluq_batched!(empty_batch, empty_batch))
    empty_abstract_batch = mod.CuModMatrix[]
    @test isempty(mod.pluq_batched!(empty_abstract_batch))
    @test isempty(mod.pluq_batched(empty_abstract_batch))
    @test isempty(mod.pluq_batched!(empty_abstract_batch, empty_abstract_batch))

    # Budget and modulus resolution are CPU-only helpers, so safety and metadata
    # contracts remain checked when CUDA is unavailable.
    stripe_bound = 100
    stripe_modulus = 1_000_000
    stripe_budget = mod.find_max_stripe_ops(Float32, stripe_bound; modulus = stripe_modulus)
    @test stripe_budget < mod.find_max_stripe_ops(Float32, stripe_bound)
    @test BigInt(stripe_modulus - 1) + BigInt(stripe_budget) * BigInt(stripe_bound - 1)^2 <=
          BigInt(2)^24 - 1

    # The product bound and reduction modulus both affect the exact budget.
    # An explicit safe caller budget is retained, and a larger one is rejected.
    stripe_input_bound = 2_048
    stripe_reduction_modulus = 5_000_000
    stripe_safe_budget = mod.find_max_stripe_ops(
        Float32,
        stripe_input_bound;
        modulus = stripe_reduction_modulus,
    )
    @test stripe_safe_budget == 2
    @test stripe_safe_budget < mod.find_max_stripe_ops(Float32, stripe_input_bound)
    @test mod._stripe_safe_chunk(
        Float32,
        stripe_reduction_modulus,
        stripe_input_bound,
        nothing,
    ) == stripe_safe_budget
    @test mod._stripe_safe_chunk(
        Float32,
        stripe_reduction_modulus,
        stripe_input_bound,
        1,
    ) == 1
    @test_throws ArgumentError mod._stripe_safe_chunk(
        Float32,
        stripe_reduction_modulus,
        stripe_input_bound,
        stripe_safe_budget + 1,
    )
    @test_throws ArgumentError mod._stripe_safe_chunk(
        Float32,
        stripe_reduction_modulus,
        stripe_input_bound,
        big(2)^100,
    )
    stripe_default_modulus = 101
    @test mod._stripe_safe_chunk(Float32, stripe_default_modulus, nothing, nothing) ==
          mod.find_max_stripe_ops(Float32, stripe_default_modulus)
    @test BigInt(stripe_reduction_modulus - 1) +
          BigInt(stripe_safe_budget) * BigInt(stripe_input_bound - 1)^2 <= BigInt(2)^24 - 1

    @test mod._stripe_effective_modulus(101, nothing) == 101
    @test mod._stripe_effective_modulus(101, 7) == 7
    @test_throws ArgumentError mod._stripe_effective_modulus(101, 0)

    @testset "Karatsuba GEMM host bounds and radix audit" begin
        same_radix = mod._karatsuba_matmul_product_plan(13, 7, 13, 5, 13, 11)
        @test same_radix.output_capacity == 143
        @test same_radix.low.input_bound == 13
        @test same_radix.low.modulus == 143
        @test same_radix.middle.input_bound == 20
        @test same_radix.middle.modulus == 143
        @test same_radix.high.input_bound == 7
        @test same_radix.high.modulus == 143
        for product in (same_radix.low, same_radix.middle, same_radix.high)
            @test BigInt(product.modulus - 1) +
                  BigInt(product.max_ops) * BigInt(product.input_bound - 1)^2 <=
                  BigInt(2)^53 - 1
        end

        independent_small = mod._karatsuba_matmul_product_plan(13, 7, 13, 5, 5, 11)
        independent_large = mod._karatsuba_matmul_product_plan(13, 7, 13, 5, 17, 11)
        @test independent_small.output_capacity == 55
        @test independent_large.output_capacity == 187
        @test independent_small.output_radix < independent_small.input_radix
        @test independent_large.output_radix > independent_large.input_radix
        @test independent_small.radix_mod_capacity == 13
        @test independent_small.radix_squared_mod_capacity == 4
        @test all(
            product -> product.modulus == independent_small.output_capacity,
            (independent_small.low, independent_small.middle, independent_small.high),
        )

        multi_stripe = mod._karatsuba_matmul_product_plan(2^26 - 3, 2, 2^26 - 3, 3, 7, 5)
        @test multi_stripe.output_capacity == 35
        @test multi_stripe.low.max_ops == 1
        @test multi_stripe.middle.max_ops == 1
        @test multi_stripe.high.max_ops > 1
        for product in (multi_stripe.low, multi_stripe.middle, multi_stripe.high)
            @test BigInt(product.modulus - 1) +
                  BigInt(product.max_ops) * BigInt(product.input_bound - 1)^2 <=
                  BigInt(2)^53 - 1
        end

        @test_throws ArgumentError mod._karatsuba_matmul_product_plan(13, 7, 12, 5, 17, 11)

        oracle_radix = 13
        oracle_high_a = 7
        oracle_high_b = 5
        oracle_high_c = 11
        oracle_low_a = BigInt[12 10 1; 11 12 5]
        oracle_plane_a = BigInt[6 5 3; 4 2 1]
        oracle_low_b = BigInt[12 2 3 1; 5 11 4 7; 8 0 9 2]
        oracle_plane_b = BigInt[4 3 1 2; 2 4 0 3; 3 1 2 4]
        p0 = oracle_low_a * oracle_low_b
        p1 = (oracle_low_a + oracle_plane_a) * (oracle_low_b + oracle_plane_b)
        p2 = oracle_plane_a * oracle_plane_b
        exact_a = oracle_low_a + oracle_radix * oracle_plane_a
        exact_b = oracle_low_b + oracle_radix * oracle_plane_b
        exact_product = exact_a * exact_b
        @test length(Set([oracle_radix, oracle_high_a, oracle_high_b, oracle_high_c])) == 4
        for oracle_output_radix in (5, oracle_radix, 17)
            output_capacity = BigInt(oracle_output_radix) * BigInt(oracle_high_c)
            expected = Base.mod.(exact_product, output_capacity)
            low_components = similar(expected)
            high_components = similar(expected)
            recombined_components = similar(expected)
            for i in eachindex(p0)
                low, high = mod._karatsuba_recombine(
                    Float64(Base.mod(p0[i], output_capacity)),
                    Float64(Base.mod(p1[i], output_capacity)),
                    Float64(Base.mod(p2[i], output_capacity)),
                    Float64(Base.mod(oracle_radix, output_capacity)),
                    Float64(Base.mod(BigInt(oracle_radix)^2, output_capacity)),
                    Float64(oracle_output_radix),
                    Float64(output_capacity),
                )
                low_components[i] = BigInt(low)
                high_components[i] = BigInt(high)
                recombined_components[i] =
                    BigInt(low) + BigInt(oracle_output_radix) * BigInt(high)
            end
            @test low_components == Base.mod.(exact_product, BigInt(oracle_output_radix))
            @test high_components == div.(expected, BigInt(oracle_output_radix))
            @test recombined_components == expected
        end

        # Distinguish output-capacity reduction from matching metadata M.
        discriminator_p0 = BigInt(2)
        discriminator_p1 = BigInt(18)
        discriminator_p2 = BigInt(8)
        discriminator_low, discriminator_high =
            mod._karatsuba_recombine(2.0, 8.0, 8.0, 3.0, 9.0, 2.0, 10.0)
        discriminator_exact = BigInt(14) * BigInt(7)
        discriminator_formula =
            discriminator_p0 +
            3 * (discriminator_p1 - discriminator_p0 - discriminator_p2) +
            BigInt(3)^2 * discriminator_p2
        @test discriminator_formula == discriminator_exact
        @test discriminator_exact == 98
        @test Base.mod(discriminator_exact, 10) == 8
        @test Base.mod(discriminator_exact, 7) == 0
        @test (discriminator_low, discriminator_high) == (0.0, 4.0)
        @test discriminator_low + 2 * discriminator_high ==
              Base.mod(discriminator_exact, 10)

        # The three-product identity needs a common input radix. With
        # x = 1 + 3*2 and y = 2 + 5*1, the unweighted cross term is 5,
        # while the exact weighted cross contribution is 17.
        x = BigInt(1) + BigInt(3) * 2
        y = BigInt(2) + BigInt(5) * 1
        p0 = BigInt(1) * 2
        p1 = BigInt(1 + 2) * (BigInt(2 + 1))
        p2 = BigInt(2) * 1
        @test x == 7 && y == 7 && x * y == 49
        @test p1 - p0 - p2 == 5
        @test BigInt(5) * 1 * 1 + BigInt(3) * 2 * 2 == 17
        @test p0 + BigInt(3) * (p1 - p0 - p2) + BigInt(3)^2 * p2 == 35

        first_case = (a_low = 1, a_high = 0, b_low = 0, b_high = 1)
        second_case = (a_low = 0, a_high = 1, b_low = 1, b_high = 0)
        first_products = (
            first_case.a_low * first_case.b_low,
            (first_case.a_low + first_case.a_high) * (first_case.b_low + first_case.b_high),
            first_case.a_high * first_case.b_high,
        )
        second_products = (
            second_case.a_low * second_case.b_low,
            (second_case.a_low + second_case.a_high) *
            (second_case.b_low + second_case.b_high),
            second_case.a_high * second_case.b_high,
        )
        first_exact = first_case.a_low + 2 * first_case.a_high
        first_exact *= first_case.b_low + 3 * first_case.b_high
        second_exact = second_case.a_low + 2 * second_case.a_high
        second_exact *= second_case.b_low + 3 * second_case.b_high
        @test first_products == second_products == (0, 1, 0)
        @test (first_exact, second_exact) == (3, 2)
        @test_throws ArgumentError mod._karatsuba_matmul_product_plan(3, 5, 5, 7, 2, 11)

        # The former unchecked R*p2 boundary exceeds Float64's exact integer
        # range. Double-and-add modular multiplication and recombination stay
        # within exact residues modulo Q instead.
        extreme_plan = mod._karatsuba_matmul_product_plan(5, 2^26 - 5, 5, 2^26 - 5, 1, 2^51)
        extreme_capacity = BigInt(extreme_plan.output_capacity)
        extreme_p2 = BigInt(2251799008378916)
        @test extreme_capacity == BigInt(2)^51
        @test all(
            product -> product.modulus == extreme_capacity && product.max_ops > 0,
            (extreme_plan.low, extreme_plan.middle, extreme_plan.high),
        )
        for product in (extreme_plan.low, extreme_plan.middle, extreme_plan.high)
            @test BigInt(product.modulus - 1) +
                  BigInt(product.max_ops) * BigInt(product.input_bound - 1)^2 <=
                  BigInt(2)^53 - 1
        end
        @test 5 * extreme_p2 > BigInt(2)^53 - 1
        extreme_product = mod._karatsuba_mod_mul(
            5.0,
            Float64(Base.mod(extreme_p2, extreme_capacity)),
            Float64(extreme_capacity),
        )
        @test BigInt(extreme_product) == Base.mod(5 * extreme_p2, extreme_capacity)
        extreme_low, extreme_high = mod._karatsuba_recombine(
            0.0,
            0.0,
            Float64(Base.mod(extreme_p2, extreme_capacity)),
            Float64(extreme_plan.radix_mod_capacity),
            Float64(extreme_plan.radix_squared_mod_capacity),
            1.0,
            Float64(extreme_capacity),
        )
        extreme_expected = Base.mod(
            BigInt(5) * (extreme_capacity - extreme_p2) + BigInt(25) * extreme_p2,
            extreme_capacity,
        )
        @test extreme_low + extreme_high == Float64(extreme_expected)

        # Directly exercise the recombination helper at both working-capacity
        # endpoints. The oracle evaluates the three-product identity with
        # BigInt and reduces once; it does not repeat the helper's modular
        # double-and-add implementation. These helper cases do not imply that
        # GEMM planning admits a nonzero product at Q = 2^53.
        function check_recombination_endpoint(
            p0::BigInt,
            p1::BigInt,
            p2::BigInt,
            input_radix::BigInt,
            output_radix::BigInt,
            output_high::BigInt,
            capacity::BigInt,
        )
            @test all(product -> 0 <= product < capacity, (p0, p1, p2))
            expected =
                Base.mod(p0 + input_radix * (p1 - p0 - p2) + input_radix^2 * p2, capacity)
            low, high = mod._karatsuba_recombine(
                Float64(p0),
                Float64(p1),
                Float64(p2),
                Float64(Base.mod(input_radix, capacity)),
                Float64(Base.mod(input_radix^2, capacity)),
                Float64(output_radix),
                Float64(capacity),
            )
            low_integer = BigInt(low)
            high_integer = BigInt(high)
            reconstructed = low_integer + output_radix * high_integer
            @test 0 <= low_integer < output_radix
            @test 0 <= high_integer < output_high
            @test reconstructed == expected
            @test Base.mod(reconstructed, capacity) == expected
            return nothing
        end

        unit_capacity = BigInt(1)
        check_recombination_endpoint(
            BigInt(0),
            BigInt(0),
            BigInt(0),
            BigInt(7),
            BigInt(1),
            BigInt(1),
            unit_capacity,
        )

        endpoint_capacity = BigInt(2)^53
        endpoint_radices =
            (BigInt(1), BigInt(2), BigInt(2)^26, BigInt(2)^52, endpoint_capacity)
        @test all(radix -> rem(endpoint_capacity, radix) == 0, endpoint_radices)
        endpoint_products = (
            (endpoint_capacity - 1, BigInt(0), BigInt(0)),
            (BigInt(0), endpoint_capacity - 2, BigInt(0)),
            (BigInt(0), BigInt(0), endpoint_capacity - 1),
            (endpoint_capacity - 1, endpoint_capacity - 2, endpoint_capacity - 3),
        )
        for output_radix in endpoint_radices
            output_high = div(endpoint_capacity, output_radix)

            # Isolate integer div/rem split boundaries while retaining exact
            # Q-sized inputs to the shared helper.
            split_targets = unique(
                BigInt[
                    target for target in (
                        BigInt(0),
                        BigInt(1),
                        output_radix - 1,
                        output_radix,
                        endpoint_capacity - 1,
                    ) if 0 <= target < endpoint_capacity
                ],
            )
            for target in split_targets
                check_recombination_endpoint(
                    BigInt(0),
                    target,
                    BigInt(0),
                    BigInt(1),
                    output_radix,
                    output_high,
                    endpoint_capacity,
                )
            end

            # The input radix is independent from the destination split.
            # Q-3 and Q-2 exercise large odd/even modular coefficients, while
            # the canonical product residues include Q-1 and Q-2.
            for input_radix in (BigInt(5), endpoint_capacity - 3, endpoint_capacity - 2)
                for (p0_endpoint, p1_endpoint, p2_endpoint) in endpoint_products
                    check_recombination_endpoint(
                        p0_endpoint,
                        p1_endpoint,
                        p2_endpoint,
                        input_radix,
                        output_radix,
                        output_high,
                        endpoint_capacity,
                    )
                end
            end
        end

        # to_karatsuba retains the exact high quotient even when it exceeds
        # the supplied high-plane metadata. Such values violate the canonical
        # input_bound promise required by stripe GEMM; conversion is preserved.
        high_quotient =
            mod.to_karatsuba([14]; low_modulus = 2, high_modulus = 3, modulus = 101)
        @test mod.low_part(high_quotient) == [0]
        @test mod.high_part(high_quotient) == [7]
        @test mod.high_modulus(high_quotient) == 3
        @test mod.from_karatsuba(Int, high_quotient) == [14]

        exact_input_bound = 2^25 + 1
        exact_modulus = 2^53 - 2^50
        exact_budget =
            mod.find_max_stripe_ops(Float64, exact_input_bound; modulus = exact_modulus)
        @test exact_budget == 1
        @test BigInt(exact_modulus - 1) +
              BigInt(exact_budget) * BigInt(exact_input_bound - 1)^2 == BigInt(2)^53 - 1
        @test BigInt(exact_modulus - 1) +
              BigInt(exact_budget + 1) * BigInt(exact_input_bound - 1)^2 > BigInt(2)^53 - 1
    end

    # Unsupported elem_type keywords fail in keyword dispatch before the
    # constructor body can allocate a GPU array.
    @test_throws MethodError mod.CuModMatrix(
        Float32[1 2; 3 4];
        modulus = 101,
        elem_type = Float32,
    )

    K = karatsuba_type(
        [1 2; 3 4],
        [5 6; 7 8];
        low_modulus = 3,
        high_modulus = 5,
        modulus = 15,
    )
    Kcopy = copy(K)
    @test Kcopy !== K
    @test low_part(Kcopy) == low_part(K)
    @test high_part(Kcopy) == high_part(K)
    @test low_part(Kcopy) !== low_part(K)

    # Characterize the documented legacy changed-M behavior; this does not
    # assert metadata preservation or repair Base.copy.
    legacy_data1 = [1 2; 0 1]
    legacy_data2 = zeros(Int, 2, 2)
    expected_legacy_data1 = copy(legacy_data1)
    expected_legacy_data2 = copy(legacy_data2)
    legacy_source = karatsuba_type(
        legacy_data1,
        legacy_data2;
        low_modulus = 3,
        high_modulus = 3,
        modulus = 3,
    )
    legacy_copy = copy(legacy_source)
    @test GPUFiniteFieldMatrices.modulus(legacy_copy) == 9
    @test GPUFiniteFieldMatrices.modulus(legacy_copy) !=
          GPUFiniteFieldMatrices.modulus(legacy_source)
    @test GPUFiniteFieldMatrices.modulus(legacy_source) == 3
    @test low_part(legacy_source) == expected_legacy_data1
    @test high_part(legacy_source) == expected_legacy_data2
    @test low_part(legacy_copy) == expected_legacy_data1
    @test high_part(legacy_copy) == expected_legacy_data2
    @test low_part(legacy_copy) !== low_part(legacy_source)
    @test high_part(legacy_copy) !== high_part(legacy_source)
    low_part(legacy_source)[1, 1] = -1
    high_part(legacy_source)[1, 1] = -1
    @test low_part(legacy_copy) == expected_legacy_data1
    @test high_part(legacy_copy) == expected_legacy_data2

    # The explicit mul! checks run before CUDA work and can be checked on CPU.
    assert_guard_error_message = function (operation, expected_message)
        err = try
            operation()
            nothing
        catch caught
            caught
        end
        @test err isa ErrorException
        if err isa ErrorException
            @test sprint(showerror, err) == expected_message
        end
    end
    mul_left = mod.KaratsubaArray{Float64,2}(
        zeros(Float64, 2, 2),
        zeros(Float64, 2, 2);
        low_modulus = 3,
        high_modulus = 5,
        modulus = 15,
    )
    mul_right_wrong_modulus = mod.KaratsubaArray{Float64,2}(
        zeros(Float64, 2, 2),
        zeros(Float64, 2, 2);
        low_modulus = 3,
        high_modulus = 5,
        modulus = 16,
    )
    mul_dest = mod.KaratsubaArray{Float64,2}(
        zeros(Float64, 2, 2),
        zeros(Float64, 2, 2);
        low_modulus = 3,
        high_modulus = 5,
        modulus = 15,
    )
    assert_guard_error_message(
        () -> LinearAlgebra.mul!(mul_dest, mul_left, mul_right_wrong_modulus),
        "Matrices must have the same modulus m",
    )

    mul_left_wrong_shape = mod.KaratsubaArray{Float64,2}(
        zeros(Float64, 2, 3),
        zeros(Float64, 2, 3);
        low_modulus = 3,
        high_modulus = 5,
        modulus = 15,
    )
    mul_right_wrong_shape = mod.KaratsubaArray{Float64,2}(
        zeros(Float64, 2, 2),
        zeros(Float64, 2, 2);
        low_modulus = 3,
        high_modulus = 5,
        modulus = 15,
    )
    assert_guard_error_message(
        () -> LinearAlgebra.mul!(mul_dest, mul_left_wrong_shape, mul_right_wrong_shape),
        "Matrix dimensions don't work for multiplication",
    )

    mul_unplanned_right = mod.KaratsubaArray{Float64,2}(
        zeros(Float64, 2, 2),
        zeros(Float64, 2, 2);
        low_modulus = 3,
        high_modulus = 5,
        modulus = 15,
    )
    assert_guard_error_message(
        () -> LinearAlgebra.mul!(mul_dest, mul_left, mul_unplanned_right),
        "Must initialize plans first",
    )
end

@testset "CPU typed scalar Karatsuba negation helper" begin
    mod = GPUFiniteFieldMatrices

    function assert_negation_matches_oracle(
        ::Type{T},
        low::Int,
        high::Int,
        N1::Int,
        N2::Int,
        M::Int,
    ) where {T<:AbstractFloat}
        packed = BigInt(low) + BigInt(N1) * BigInt(high)
        expected = Base.mod(-packed, BigInt(N1) * BigInt(N2))
        actual = mod.karatsuba_negate_helper(T(low), T(high), N1, N2, M)

        @test actual[1] isa T
        @test actual[2] isa T
        @test BigInt(actual[1]) == Base.mod(expected, BigInt(N1))
        @test BigInt(actual[2]) == div(expected, BigInt(N1))
    end

    @test typeof(
        @inferred(mod.karatsuba_negate_helper(Float32(0), Float32(0), 10, 10, 100)),
    ) === Tuple{Float32,Float32}
    @test typeof(
        @inferred(mod.karatsuba_negate_helper(Float64(0), Float64(0), 10, 10, 100)),
    ) === Tuple{Float64,Float64}

    for T in (Float32, Float64)
        N1 = 10
        N2 = 10
        M = 100

        # Exhaust the canonical two-digit base-10 representation, including
        # zero, low-digit borrows, high-digit wrap, and the final value 99.
        for high = 0:(N2-1), low = 0:(N1-1)
            assert_negation_matches_oracle(T, low, high, N1, N2, M)
        end

        # BigInt bounds are checked before converting any boundary fixture to
        # Int or T. M^2 is the helper's largest modulus: it lands exactly on
        # Float32's 2^24 integer limit and within 2^27 of Float64's 2^53 limit.
        if T === Float32
            N1_big = BigInt(64)
            N2_big = BigInt(64)
            exact_limit = BigInt(2)^24
        else
            N1_big = BigInt(5)
            N2_big = BigInt(18_981_253)
            exact_limit = BigInt(2)^53
        end
        M_big = N1_big * N2_big
        M_squared_big = M_big^2
        @test M_squared_big <= exact_limit
        @test M_squared_big <= BigInt(typemax(Int))
        if T === Float32
            @test M_squared_big == exact_limit
        else
            @test exact_limit - M_squared_big < BigInt(2)^27
        end

        N1 = Int(N1_big)
        N2 = Int(N2_big)
        M = Int(M_big)
        @test BigInt(M)^2 == M_squared_big
        boundary_cases = (
            (0, 0),
            (1, 0),
            (0, 1),
            (N1 - 1, 0),
            (0, N2 - 1),
            (N1 - 1, N2 - 1),
            (N1 - 1, 1),
            (1, N2 - 1),
        )
        for (low, high) in boundary_cases
            assert_negation_matches_oracle(T, low, high, N1, N2, M)
        end
    end
end

@testset "CPU constructor and exact Karatsuba conversion" begin
    mod = GPUFiniteFieldMatrices

    identity_ctor = which(mod.CuModMatrix{Int}, Tuple{LinearAlgebra.UniformScaling,Int,Int})
    @test :modulus in Base.kwarg_decl(identity_ctor)
    @test which(mod.identity_matrix, Tuple{Type{mod.CuModMatrix{Int}},Int}).module === mod
    @test :modulus in
          Base.kwarg_decl(which(Base.zeros, Tuple{Type{mod.CuModMatrix{Int}},Int,Int}))
    @test :modulus in
          Base.kwarg_decl(which(mod.rand, Tuple{Type{mod.CuModVector{Int}},Int}))
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
    @test typeof(wrapped).parameters[4] === mod.CuModArray{Int,2}
    @test fieldtype(typeof(wrapped), :data1) === Matrix{Int}
    @test fieldtype(typeof(wrapped), :data2) === Matrix{Int}
    @test fieldtype(typeof(wrapped), :plan) === Union{Nothing,mod.CuModArray{Int,2}}
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
    @test typeof(zero_matrix) ===
          mod.KaratsubaMatrix{Float32,Matrix{Float32},mod.CuModMatrix{Float32}}
    @test typeof(zero_vector) ===
          mod.KaratsubaVector{Float32,Vector{Float32},mod.CuModVector{Float32}}
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
          mod.KaratsubaMatrix{Float32,Matrix{Float32},mod.CuModMatrix{Float32}}
    @test explicit_plan_zero.plan === nothing
    @test_throws ArgumentError Base.zeros(
        mod.KaratsubaMatrix{Float32,Matrix{Float32},Matrix{Float32}},
        2,
        2;
        modulus = 19,
    )
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
        @test typeof(wrapped).parameters[4] === mod.CuModArray{T,2}
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

    factory_zero = Base.zeros(
        mod.KaratsubaMatrix{UInt64,Matrix{UInt64}},
        1,
        1;
        low_modulus = 2,
        high_modulus = 3,
        modulus = 5,
    )
    @test typeof(factory_zero) ===
          mod.KaratsubaMatrix{UInt64,Matrix{UInt64},mod.CuModArray{UInt64,2}}
    @test factory_zero.plan === nothing

    for unsupported in (BigInt, BigFloat, Bool, ComplexF64, AbstractFloat, Signed, Unsigned)
        unsupported_error = try
            mod.get_bits(unsupported)
            nothing
        catch err
            err
        end
        @test unsupported_error isa ErrorException
        if unsupported_error isa Exception
            @test occursin(string(unsupported), sprint(showerror, unsupported_error))
        end
    end
    @test_throws ErrorException mod.find_max_ops_karatsuba(BigInt, 2)
    @test_throws ErrorException mod.find_max_ops(BigFloat, 2)
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
    @test typeof(plan_typed_matrix) ===
          mod.KaratsubaMatrix{Int,Matrix{Int},mod.CuModMatrix{Int}}
    @test plan_typed_matrix.plan === nothing
    unsupported_plan_matrix = mod.KaratsubaMatrix{Int,Matrix{Int},Matrix{Int}}(
        low_matrix,
        high_matrix;
        low_modulus = 2,
        high_modulus = 3,
        modulus = 11,
    )
    @test typeof(unsupported_plan_matrix) ===
          mod.KaratsubaMatrix{Int,Matrix{Int},Matrix{Int}}
    @test_throws ArgumentError mod.initialize_plan!(unsupported_plan_matrix)
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
    @test typeof(cpu_zero_matrix) ===
          mod.KaratsubaMatrix{Float32,Matrix{Float32},mod.CuModMatrix{Float32}}
    @test typeof(cpu_zero_vector) ===
          mod.KaratsubaVector{Float32,Vector{Float32},mod.CuModVector{Float32}}
    @test cpu_zero_matrix.plan === nothing
    @test cpu_zero_vector.plan === nothing

    gpu_kwargs_type = NamedTuple{(:modulus,),Tuple{Int}}
    gpu_matrix_type = mod.KaratsubaMatrix{Float32,mod.CuModMatrix{Float32}}
    gpu_vector_type = mod.KaratsubaVector{Float32,mod.CuModVector{Float32}}
    gpu_matrix_return = Core.Compiler.return_type(
        Core.kwcall,
        Tuple{gpu_kwargs_type,typeof(Base.zeros),Type{gpu_matrix_type},Int,Int},
    )
    gpu_vector_return = Core.Compiler.return_type(
        Core.kwcall,
        Tuple{gpu_kwargs_type,typeof(Base.zeros),Type{gpu_vector_type},Int},
    )
    gpu_default_return = Core.Compiler.return_type(
        Core.kwcall,
        Tuple{
            gpu_kwargs_type,
            typeof(Base.zeros),
            Type{mod.KaratsubaMatrix{Float32}},
            Int,
            Int,
        },
    )
    @test gpu_matrix_return ===
          mod.KaratsubaMatrix{Float32,mod.CuModMatrix{Float32},mod.CuModMatrix{Float32}}
    @test gpu_vector_return ===
          mod.KaratsubaVector{Float32,mod.CuModVector{Float32},mod.CuModVector{Float32}}
    @test gpu_default_return === gpu_matrix_return

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
          mod.KaratsubaMatrix{Float32,raw_matrix_storage,mod.CuModMatrix{Float32}}
    @test raw_vector_return ===
          mod.KaratsubaVector{Float32,raw_vector_storage,mod.CuModVector{Float32}}

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
    unsupported_component_type = typeof(view(zeros(Float32, 2, 2), :, :))
    unsupported_factory_type =
        Core.apply_type(mod.KaratsubaMatrix{Float32}, unsupported_component_type)
    @test_throws ArgumentError Base.zeros(unsupported_factory_type, 2, 2; modulus = 19)

    @test_throws ArgumentError mod.KaratsubaMatrix(
        low_matrix,
        view(high_matrix, :, :);
        low_modulus = 2,
        high_modulus = 3,
        modulus = 11,
    )
end

if CUDA.functional()
    @testset "CUDA-backed Karatsuba plan lifecycle across component storage" begin
        mod = GPUFiniteFieldMatrices

        function check_plan(K, dims)
            plan = K.plan
            @test typeof(plan) === mod.CuModArray{eltype(K),length(dims)}
            @test size(plan) == dims
            @test size(parent(plan)) == map(d -> d + mod.TILE_WIDTH, dims)
            @test modulus(plan) == low_modulus(K)
            @test parent(plan) isa CUDA.CuArray
            @test CUDA.device(parent(plan)) == CUDA.device()
        end

        function check_lifecycle(K, dims)
            @test K.plan === nothing
            @test (@inferred mod.initialize_plan!(K)) === K
            check_plan(K, dims)

            first_plan = K.plan
            first_storage = parent(first_plan)
            @test (@inferred mod.initialize_plan!(K)) === K
            @test K.plan !== first_plan
            @test parent(K.plan) !== first_storage
            check_plan(K, dims)

            copied = copy(K)
            @test typeof(copied) === typeof(K)
            @test low_part(copied) !== low_part(K)
            @test high_part(copied) !== high_part(K)
            @test copied.plan !== K.plan
            @test parent(copied.plan) !== parent(K.plan)
            check_plan(copied, dims)
        end

        function matrix_zeros(::Type{S}) where {S}
            return @inferred Base.zeros(
                mod.KaratsubaMatrix{Float32,S},
                2,
                3;
                low_modulus = 3,
                high_modulus = 5,
                modulus = 19,
            )
        end

        function vector_zeros(::Type{S}) where {S}
            return @inferred Base.zeros(
                mod.KaratsubaVector{Float32,S},
                4;
                low_modulus = 3,
                high_modulus = 5,
                modulus = 19,
            )
        end

        # Keep the original strict 0wjq call shapes. CPU components still
        # allocate the original CuModArray workspace and therefore require CUDA.
        low_matrix = Int[1 2; 3 4]
        high_matrix = Int[5 6; 7 8]
        strict_matrix = mod.KaratsubaArray{Int,2}(
            low_matrix,
            high_matrix;
            low_modulus = 2,
            high_modulus = 3,
            modulus = 11,
        )
        @test (@inferred mod.initialize_plan!(strict_matrix)) === strict_matrix
        check_plan(strict_matrix, (2, 2))

        strict_vector = mod.KaratsubaArray{Int,1}(
            Int[1, 2, 3],
            Int[4, 5, 6];
            low_modulus = 2,
            high_modulus = 3,
            modulus = 11,
        )
        @test (@inferred mod.initialize_plan!(strict_vector)) === strict_vector
        check_plan(strict_vector, (3,))

        matrix_storage = (
            Matrix{Float32},
            CUDA.CuArray{Float32,2,CUDA.DeviceMemory},
            mod.CuModMatrix{Float32},
        )
        vector_storage = (
            Vector{Float32},
            CUDA.CuArray{Float32,1,CUDA.DeviceMemory},
            mod.CuModVector{Float32},
        )
        matrices = map(matrix_zeros, matrix_storage)
        vectors = map(vector_zeros, vector_storage)
        @test all(K -> K.plan === nothing, matrices)
        @test all(K -> K.plan === nothing, vectors)
        foreach(K -> check_lifecycle(K, (2, 3)), matrices)
        foreach(K -> check_lifecycle(K, (4,)), vectors)
    end
end
