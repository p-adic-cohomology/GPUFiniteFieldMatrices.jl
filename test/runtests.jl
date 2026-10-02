# Flip the package's `@stable` wrap (src/GPUFiniteFieldMatrices.jl) from its
# downstream-safe `disable` default to an enforcing mode for the test run only.
# This MUST run before `using GPUFiniteFieldMatrices` so the package compiles in
# with the gate active. `codegen_level="min"` keeps the `@stable` precompile
# overhead low.
using Preferences: set_preferences!
set_preferences!(
    "GPUFiniteFieldMatrices",
    "dispatch_doctor_mode" => "error",
    "dispatch_doctor_codegen_level" => "min";
    force = true,
)

using Test
using CUDA
using LinearAlgebra
using BenchmarkTools
using Suppressor
using Unroll
using Aqua
using ExplicitImports
using GPUFiniteFieldMatrices

# Exercise the return type check on CPU, even when CUDA is unavailable.
@testset "DispatchDoctor — return type stability" begin
    @test mod_inv(3, 7) == 5
    @test GPUFiniteFieldMatrices.inverse_permutation([2, 3, 1]) == [3, 1, 2]
    @test GPUFiniteFieldMatrices.gcd(12, 18) == 6
end

# Aqua quality gate — CPU-runnable, so it runs unconditionally (outside the
# CUDA.functional() guard below).
@testset "Aqua" begin
    Aqua.test_all(GPUFiniteFieldMatrices; stale_deps = false, deps_compat = false)
end

include("CuModMatrix/basic_operations_test.jl")
include("CuModMatrix/inplace_operations_test.jl")
include("CuModMatrix/matmul_operations_test.jl")
include("CuModMatrix/benchmark_test.jl")
include("CuModMatrix/stripe_mul_test.jl")
include("CuModMatrix/allocations_test.jl")
include("CuModMatrix/timing_test.jl")
include("CuModMatrix/de_rham_test.jl")
include("CuModMatrix/permutation_test.jl")
include("CuModMatrix/triangular_test.jl")
include("CuModMatrix/inverse/runtests.jl")

@testset "ExplicitImports" begin
    @test check_no_implicit_imports(GPUFiniteFieldMatrices) === nothing
    @test check_no_stale_explicit_imports(GPUFiniteFieldMatrices) === nothing
    @test check_all_explicit_imports_via_owners(GPUFiniteFieldMatrices) === nothing
    @test check_all_explicit_imports_are_public(GPUFiniteFieldMatrices) === nothing
    @test check_no_self_qualified_accesses(GPUFiniteFieldMatrices) === nothing

    # `CuRef` is owned by CUDACore but re-exported through CUDA's CUBLAS
    # submodule; accessing it via CUBLAS is a re-export false positive.
    @test check_all_qualified_accesses_via_owners(
        GPUFiniteFieldMatrices;
        ignore = (:CuRef,),
    ) === nothing

    # The ignored names are non-public internals that the GPU code genuinely
    # needs and for which there is no public alternative:
    #   CHOLMOD                          – SparseArrays.CHOLMOD.Dense, used in a copyto! signature
    #   CuRef, gemm!, gemv!, gemv_batched! – CUDA.CUBLAS low-level BLAS entry points
    @test check_all_qualified_accesses_are_public(
        GPUFiniteFieldMatrices;
        ignore = (:CHOLMOD, :CuRef, :gemm!, :gemv!, :gemv_batched!),
    ) === nothing
end

if CUDA.functional()
    @testset "CuModMatrix.jl" begin
        @testset "Triangular Inverse" begin
            test_upper_triangular_inverse()
            test_lower_triangular_inverse()
        end
        @testset "De Rham" begin
            test_de_rham()
        end
        @testset "Inverse Rewrite" begin
            test_inverse_rewrite()
        end
        @testset "Permutations" begin
            test_permutations()
        end
        @testset "GPU Matrix Type" begin
            test_gpu_mat()
        end
        @testset "In-place Operations" begin
            test_inplace()
        end
        @testset "Matrix Multiplication" begin
            test_matmul()
            test_stripe_mul()
        end
        @testset "Allocations" begin
            test_allocations()
        end
        @testset "Timings" begin
            test_timings()
        end
    end
else
    @testset "CuModMatrix.jl" begin
        @info "CUDA is not functional on this machine; skipping GPU-dependent tests."
    end
end
