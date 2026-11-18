# Flip the package's `@stable` wrap (src/GPUFiniteFieldMatrices.jl) from its
# downstream-safe `disable` default to an enforcing mode for the test run only.
# This MUST run before `using GPUFiniteFieldMatrices` so the package compiles in
# with the gate active. `codegen_level="min"` keeps the `@stable` precompile
# overhead low.
using Preferences: set_preferences!
set_preferences!(
    (Base.UUID("465031f1-b347-4f72-85f6-baf667cdf853"), "GPUFiniteFieldMatrices"),
    "dispatch_doctor_mode" => "warn",
    "dispatch_doctor_codegen_level" => "min";
    force = true,
)

using Test
using CUDA
using LinearAlgebra
using BenchmarkTools
using BitIntegers
using Suppressor
using Unroll
using Aqua
using ExplicitImports
using GPUFiniteFieldMatrices

include("api_surface_test.jl")
include("base_methods_test.jl")
include("exceptions_test.jl")
include("KaratsubaMatrix/api_regressions_test.jl")
include("KaratsubaMatrix/basic_operations_test.jl")
include("KaratsubaMatrix/allocations_test.jl")
include("KaratsubaMatrix/raw_unary_test.jl")

@testset "Stripe budget input validation" begin
    for input_bound in (1, 0, -1)
        @test_throws ArgumentError GPUFiniteFieldMatrices.find_max_stripe_ops(
            Float32,
            input_bound,
        )
    end
end

# Exercise the return type check on CPU, even when CUDA is unavailable.
@testset "DispatchDoctor — return type stability" begin
    @test GPUFiniteFieldMatrices.mod_inv(3, 7) == 5
    @test GPUFiniteFieldMatrices.inverse_permutation([2, 3, 1]) == [3, 1, 2]
    @test GPUFiniteFieldMatrices.gcd(12, 18) == 6
end

# Aqua quality gate — CPU-runnable, so it runs unconditionally (outside the
# CUDA.functional() guard below).
@testset "Aqua" begin
    Aqua.test_all(GPUFiniteFieldMatrices; stale_deps = false, deps_compat = false)
end

@testset "CUDA BLAS binding" begin
    expected_blas_name = isdefined(CUDA, :cuBLAS) ? :cuBLAS : :CUBLAS
    @test nameof(GPUFiniteFieldMatrices._CUDA_BLAS) == expected_blas_name
end

include("cuda_kernel_bindings_test.jl")
include("CuModMatrix/display_test.jl")

include("CuModMatrix/basic_operations_test.jl")
include("CuModMatrix/modulus_caller_bindings_test.jl")
include("CuModMatrix/inplace_operations_test.jl")
include("CuModMatrix/matmul_operations_test.jl")
include("CuModMatrix/benchmark_test.jl")
include("CuModMatrix/stripe_mul_test.jl")
include("CuModMatrix/empty_stripe_test.jl")
include("CuModMatrix/cuda_blas_compat_test.jl")
include("CuModMatrix/padding_contract_test.jl")
include("CuModMatrix/elementwise_preflight_test.jl")
include("CuModMatrix/allocations_test.jl")
include("CuModMatrix/timing_test.jl")
include("CuModMatrix/de_rham_test.jl")
include("CuModMatrix/triangular_test.jl")
include("CuModMatrix/inverse/runtests.jl")

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(GPUFiniteFieldMatrices) === nothing
    @test check_no_stale_explicit_imports(GPUFiniteFieldMatrices) === nothing
    @test check_all_explicit_imports_via_owners(GPUFiniteFieldMatrices) === nothing
    @test check_all_explicit_imports_are_public(GPUFiniteFieldMatrices) === nothing
    @test check_no_self_qualified_accesses(GPUFiniteFieldMatrices) === nothing

    @test check_all_qualified_accesses_via_owners(GPUFiniteFieldMatrices) === nothing

    # The ignored names are non-public bindings required by supported APIs:
    #   CHOLMOD                          – SparseArrays.CHOLMOD.Dense in a copyto! signature
    #   gemm!, gemv!, gemv_batched!      – CUDA cuBLAS low-level BLAS entry points
    #   @atomic, FAST_MATH, fill, math_mode, pin, rand, zeros
    #                                    – CUDA 5.11.3 owns these names in CUDA
    #                                      without public markers. CUDA 6 moves
    #                                      @atomic, FAST_MATH, fill, math_mode,
    #                                      pin, and zeros to CUDACore and marks
    #                                      them public; CUDA 6 also publicly
    #                                      forwards rand from cuRAND.
    #                                      Keep the precise CUDA 5 names ignored
    #                                      to preserve the supported 5.11 line.
    # The separate owner check remains enabled for all of these accesses.
    @test check_all_qualified_accesses_are_public(
        GPUFiniteFieldMatrices;
        ignore = (
            :CHOLMOD,
            :gemm!,
            :gemv!,
            :gemv_batched!,
            Symbol("@atomic"),
            :FAST_MATH,
            :fill,
            :math_mode,
            :pin,
            :rand,
            :zeros,
        ),
    ) === nothing
end

if CUDA.functional()
    @testset "CuModMatrix.jl" begin
        @testset "Triangular Inverse" begin
            println(stderr, "START test_triangular_destination_helpers")
            flush(stderr)
            test_triangular_destination_helpers()
            println(stderr, "FINISH test_triangular_destination_helpers")
            flush(stderr)
            println(stderr, "START test_upper_triangular_inverse")
            flush(stderr)
            test_upper_triangular_inverse()
            println(stderr, "FINISH test_upper_triangular_inverse")
            flush(stderr)
            println(stderr, "START test_upper_wide_triangular_inverse")
            flush(stderr)
            test_upper_wide_triangular_inverse()
            println(stderr, "FINISH test_upper_wide_triangular_inverse")
            flush(stderr)
            println(stderr, "START test_lower_triangular_inverse")
            flush(stderr)
            test_lower_triangular_inverse()
            println(stderr, "FINISH test_lower_triangular_inverse")
            flush(stderr)
            println(stderr, "START test_lower_wide_triangular_inverse")
            flush(stderr)
            test_lower_wide_triangular_inverse()
            println(stderr, "FINISH test_lower_wide_triangular_inverse")
            flush(stderr)
        end
        @testset "De Rham" begin
            println(stderr, "START test_de_rham")
            flush(stderr)
            test_de_rham()
            println(stderr, "FINISH test_de_rham")
            flush(stderr)
        end
        @testset "Inverse Rewrite" begin
            println(stderr, "START test_inverse_rewrite")
            flush(stderr)
            test_inverse_rewrite()
            println(stderr, "FINISH test_inverse_rewrite")
            flush(stderr)
        end
        @testset "GPU Matrix Type" begin
            println(stderr, "START test_gpu_mat")
            flush(stderr)
            test_gpu_mat()
            println(stderr, "FINISH test_gpu_mat")
            flush(stderr)
        end
        @testset "CuModArray display" begin
            println(stderr, "START test_cumod_display")
            flush(stderr)
            test_cumod_display()
            println(stderr, "FINISH test_cumod_display")
            flush(stderr)
        end
        @testset "In-place Operations" begin
            println(stderr, "START test_inplace")
            flush(stderr)
            test_inplace()
            println(stderr, "FINISH test_inplace")
            flush(stderr)
        end
        @testset "Matrix Multiplication" begin
            println(stderr, "START test_matmul")
            flush(stderr)
            test_matmul()
            println(stderr, "FINISH test_matmul")
            flush(stderr)
            println(stderr, "START test_cuda_blas_compatibility")
            flush(stderr)
            test_cuda_blas_compatibility()
            println(stderr, "FINISH test_cuda_blas_compatibility")
            flush(stderr)
            println(stderr, "START test_stripe_mul")
            flush(stderr)
            test_stripe_mul()
            println(stderr, "FINISH test_stripe_mul")
            flush(stderr)
            println(stderr, "START test_empty_stripe_products")
            flush(stderr)
            test_empty_stripe_products()
            println(stderr, "FINISH test_empty_stripe_products")
            flush(stderr)
        end
        println(stderr, "START test_karatsuba_api_regressions")
        flush(stderr)
        test_karatsuba_api_regressions()
        println(stderr, "FINISH test_karatsuba_api_regressions")
        flush(stderr)
        println(stderr, "START test_karatsuba_float32_float64_products")
        flush(stderr)
        test_karatsuba_float32_float64_products()
        println(stderr, "FINISH test_karatsuba_float32_float64_products")
        flush(stderr)
        @testset "Karatsuba arithmetic" begin
            println(stderr, "START test_karatsuba_basic_operations")
            flush(stderr)
            test_karatsuba_basic_operations()
            println(stderr, "FINISH test_karatsuba_basic_operations")
            flush(stderr)
            println(stderr, "START test_karatsuba_operations")
            flush(stderr)
            test_karatsuba_operations()
            println(stderr, "FINISH test_karatsuba_operations")
            flush(stderr)
            println(stderr, "START test_karatsuba_raw_unary")
            flush(stderr)
            test_karatsuba_raw_unary()
            println(stderr, "FINISH test_karatsuba_raw_unary")
            flush(stderr)
        end
        @testset "Allocations" begin
            println(stderr, "START test_allocations")
            flush(stderr)
            test_allocations()
            println(stderr, "FINISH test_allocations")
            flush(stderr)
            println(stderr, "START test_karatsuba_allocations")
            flush(stderr)
            test_karatsuba_allocations()
            println(stderr, "FINISH test_karatsuba_allocations")
            flush(stderr)
        end
        @testset "Timings" begin
            println(stderr, "START test_timings")
            flush(stderr)
            test_timings()
            println(stderr, "FINISH test_timings")
            flush(stderr)
        end
    end
else
    @testset "CuModMatrix.jl" begin
        @info "CUDA is not functional on this machine; skipping GPU-dependent tests."
    end
end
