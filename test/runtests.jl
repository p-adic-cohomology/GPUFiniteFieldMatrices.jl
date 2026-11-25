using Test
using TOML
using Pkg

include("backend_test_driver.jl")
const GFM_TEST_BACKEND_DEPENDENCIES =
    TOML.parsefile(joinpath(@__DIR__, "backend_optional_dependencies.toml"))
const GFM_TEST_BACKEND_REQUEST = BackendTestDriver.parse_backend_selector(ARGS)

function _gfm_test_provision_backend!(backend::Symbol)
    dependency = GFM_TEST_BACKEND_DEPENDENCIES[string(backend)]
    Base.find_package(dependency["name"]) === nothing || return nothing

    version = VersionNumber(dependency["version"])
    @info "Provisioning selected optional backend" backend package=dependency["name"] version project=Base.active_project()
    Pkg.add(Pkg.PackageSpec(name = dependency["name"], version = version))
    return nothing
end

const GFM_TEST_BACKENDS_TO_PROVISION =
    BackendTestDriver.backends_to_provision(GFM_TEST_BACKEND_REQUEST)
for backend in GFM_TEST_BACKENDS_TO_PROVISION
    _gfm_test_provision_backend!(backend)
end
const GFM_TEST_OPTIONAL_BACKENDS =
    BackendTestDriver.optional_backends(GFM_TEST_BACKEND_REQUEST)

for backend in GFM_TEST_OPTIONAL_BACKENDS
    if backend === :cuda
        @eval import CUDA
    else
        @eval import Metal
    end
    dependency = GFM_TEST_BACKEND_DEPENDENCIES[string(backend)]
    actual_version = Base.pkgversion(getfield(Main, Symbol(dependency["name"])))
    expected_version = VersionNumber(dependency["version"])
    actual_version == expected_version || error(
        "Selected $backend test vendor version is $actual_version; expected $expected_version",
    )
end

_gfm_test_backend_functional(::Val{:cuda}) = CUDA.functional()
_gfm_test_backend_functional(::Val{:metal}) = Metal.functional()
const GFM_TEST_BACKEND_FUNCTIONAL = Dict(
    backend => _gfm_test_backend_functional(Val(backend)) for
    backend in GFM_TEST_OPTIONAL_BACKENDS
)

if GFM_TEST_BACKEND_REQUEST !== nothing &&
   !GFM_TEST_BACKEND_FUNCTIONAL[GFM_TEST_BACKEND_REQUEST]
    error(
        "Requested backend $(GFM_TEST_BACKEND_REQUEST) is installed but not functional on this host; " *
        "no backend tests were run.",
    )
end

const GFM_TEST_CUDA_LOADED = :cuda in GFM_TEST_OPTIONAL_BACKENDS
const GFM_TEST_METAL_LOADED = :metal in GFM_TEST_OPTIONAL_BACKENDS
const GFM_TEST_CUDA_FUNCTIONAL = get(GFM_TEST_BACKEND_FUNCTIONAL, :cuda, false)
const GFM_TEST_METAL_FUNCTIONAL = get(GFM_TEST_BACKEND_FUNCTIONAL, :metal, false)
const GFM_TEST_FUNCTIONAL_BACKENDS = Tuple(
    backend for
    backend in GFM_TEST_OPTIONAL_BACKENDS if GFM_TEST_BACKEND_FUNCTIONAL[backend]
)
const GFM_TEST_PENDING_METAL_FAMILIES = Tuple(
    family["name"] for
    family in TOML.parsefile(joinpath(@__DIR__, "backend_coverage.toml"))["families"] if
    "metal" in family["applies_to"] && occursin("pending", family["status"])
)
@info "Backend coverage selection" selected=GFM_TEST_OPTIONAL_BACKENDS functional=GFM_TEST_FUNCTIONAL_BACKENDS pending_metal_families=GFM_TEST_PENDING_METAL_FAMILIES

# Flip the package's `@stable` wrap to an enforcing mode before loading it.
using Preferences: set_preferences!
set_preferences!(
    (Base.UUID("465031f1-b347-4f72-85f6-baf667cdf853"), "GPUFiniteFieldMatrices"),
    "dispatch_doctor_mode" => "warn",
    "dispatch_doctor_codegen_level" => "min";
    force = true,
)

using LinearAlgebra
using BenchmarkTools
using BitIntegers
using Suppressor
using Unroll
using Aqua
using ExplicitImports
using GPUFiniteFieldMatrices
include("backend_test_support.jl")
const GFM_TEST_CAPABILITIES = BackendTestSupport.capability_fixture(;
    cuda_loaded = GFM_TEST_CUDA_LOADED,
    cuda_functional = GFM_TEST_CUDA_FUNCTIONAL,
    metal_loaded = GFM_TEST_METAL_LOADED,
    metal_functional = GFM_TEST_METAL_FUNCTIONAL,
)

include("core_api_surface_test.jl")
include("core_karatsuba_host_test.jl")
include("backend_test_driver_test.jl")
@testset "Only requested optional vendor is imported" begin
    if GFM_TEST_BACKEND_REQUEST === :metal
        @test !isdefined(Main, :CUDA)
    elseif GFM_TEST_BACKEND_REQUEST === :cuda
        @test !isdefined(Main, :Metal)
    end
end
include("backend_storage_test.jl")
include("generic_storage_contract_test.jl")
include("base_methods_test.jl")
include("exceptions_test.jl")

if GFM_TEST_METAL_LOADED
    include("metal_storage_test.jl")
end

if GFM_TEST_CUDA_LOADED
    GFM_TEST_CUDA_FUNCTIONAL && include("cuda_runtime_contract_test.jl")
    include("api_surface_test.jl")
    include("cuda_kernel_bindings_test.jl")
    include("KaratsubaMatrix/api_regressions_test.jl")
    include("KaratsubaMatrix/basic_operations_test.jl")
    include("KaratsubaMatrix/allocations_test.jl")
    include("KaratsubaMatrix/raw_unary_test.jl")
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
end

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
# Functional CUDA tests below).
@testset "Aqua" begin
    Aqua.test_all(GPUFiniteFieldMatrices; stale_deps = false, deps_compat = false)
end

if GFM_TEST_CUDA_LOADED
    @testset "CUDA BLAS binding" begin
        cuda_ext =
            Base.get_extension(GPUFiniteFieldMatrices, :GPUFiniteFieldMatricesCUDAExt)
        expected_blas_name = isdefined(CUDA, :cuBLAS) ? :cuBLAS : :CUBLAS
        @test nameof(cuda_ext._CUDA_BLAS) == expected_blas_name
    end
end

@testset "ExplicitImports" begin
    extensions = filter(
        !isnothing,
        (
            Base.get_extension(GPUFiniteFieldMatrices, :GPUFiniteFieldMatricesCUDAExt),
            Base.get_extension(GPUFiniteFieldMatrices, :GPUFiniteFieldMatricesMetalExt),
        ),
    )

    @test check_no_implicit_imports(GPUFiniteFieldMatrices) === nothing
    @test check_no_stale_explicit_imports(GPUFiniteFieldMatrices) === nothing
    @test check_all_explicit_imports_via_owners(GPUFiniteFieldMatrices) === nothing
    @test check_no_self_qualified_accesses(GPUFiniteFieldMatrices) === nothing
    @test check_all_qualified_accesses_via_owners(GPUFiniteFieldMatrices) === nothing

    # Extensions still need explicit, used imports from the modules that own
    # each name. Their internal package hooks are private by design.
    for extension in extensions
        @test check_no_implicit_imports(extension) === nothing
        @test check_no_stale_explicit_imports(extension) === nothing
        @test check_all_explicit_imports_via_owners(extension) === nothing
        @test check_no_self_qualified_accesses(extension) === nothing
        @test check_all_qualified_accesses_via_owners(extension) === nothing
    end

    # These names are implementation hooks shared across the package and its
    # extensions. Their ownership is checked above; they are intentionally not
    # part of the stable public API.
    private_extension_bindings = (
        :_backend_available,
        :_backend_instance,
        :_backend_default_component_type,
        :_backend_default_plan_type,
        :_backend_symbol,
        :_karatsuba_component,
        :_karatsuba_gemv_precision_plan,
        :_karatsuba_gemv_validate_radices,
        :_karatsuba_host_array,
        :_karatsuba_matmul_product_plan,
        :_karatsuba_plan_type,
        :_karatsuba_zero_component,
        :_logical_view,
        :_normalize_logical_size,
        :_padded_extent,
        :_padded_size,
        :_pluq_extract_L,
        :_pluq_extract_U,
        :_storage_modulus_limit,
        :_storage_backend,
        :_validate_backend_storage,
        :_validate_storage_modulus,
        :CuStorage,
        :MtlStorage,
        :_modular_host_values,
        :divide_elements!,
        :find_max_ops,
        :find_max_stripe_ops,
        :inverse_permutation,
        :negate!,
        :scalar_multiply!,
    )
    @test check_all_explicit_imports_are_public(
        GPUFiniteFieldMatrices;
        ignore = private_extension_bindings,
    ) === nothing

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
            private_extension_bindings...,
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

if GFM_TEST_CUDA_FUNCTIONAL
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
elseif GFM_TEST_CUDA_LOADED
    @testset "CuModMatrix.jl" begin
        @info "CUDA is not functional on this machine; skipping GPU-dependent tests."
    end
end
