include("test_mod_arith.jl")
include("test_perm_vectors.jl")
include("test_basecase_pluq.jl")
include("test_api_smoke.jl")
include("test_rank_and_q.jl")
include("test_extensive.jl")
include("test_rectangular.jl")
include("test_rectangular_additional.jl")
include("test_phase0_regimes.jl")
include("test_batched_tiny.jl")
include("test_batched_streams.jl")

function test_inverse_rewrite()
    @testset "Mod Arithmetic" begin
        test_mod_arith()
    end
    @testset "Permutation Vectors" begin
        test_perm_vectors()
    end
    @testset "Basecase PLUQ" begin
        test_basecase_pluq()
    end
    @testset "API Smoke" begin
        test_api_smoke()
    end
    @testset "Rank And Q" begin
        test_rank_and_q()
    end
    @testset "Identity Range" begin
        test_identity_range()
    end
    @testset "Square Inverse Strategies" begin
        test_square_inverse_strategies()
    end
    @testset "Random Invertible Batch" begin
        test_random_invertible_batch()
    end
    @testset "Dense Random Blocked Inverse" begin
        test_dense_random_blocked_inverse()
    end
    @testset "Random Singular Batch" begin
        test_random_singular_batch()
    end
    @testset "Rank Deficient Blocked PLUQ" begin
        test_rank_deficient_blocked_pluq()
    end
    @testset "Modulus Contracts" begin
        test_modulus_contracts()
    end
    @testset "Modulus Diversity" begin
        test_modulus_diversity()
    end
    @testset "API Rejections" begin
        test_api_rejections()
    end
    @testset "Non-involutory Permutation Inverse" begin
        test_noninvolutory_permutation_inverse()
    end
    @testset "Batch Fallbacks" begin
        test_batch_fallbacks()
    end
    @testset "Padding Sensitive Sizes" begin
        test_padding_sensitive_sizes()
    end
    @testset "Rectangular Inverses" begin
        test_rectangular_inverses()
    end
    @testset "Rectangular Rank And Failures" begin
        test_rectangular_rank_and_failures()
    end
    @testset "Rectangular Additional Sizes" begin
        test_rectangular_additional_sizes()
    end
    @testset "Dense Rectangular Right Inverse" begin
        test_dense_rectangular_right_inverse()
    end
    @testset "Right Inverse Leading-Block Fallback" begin
        test_right_inverse_singular_leading_block_fallback()
    end
    @testset "Phase0 Regime Matrix Grid" begin
        test_phase0_regime_matrix_grid()
    end
    @testset "Batched Tiny Kernels" begin
        test_batched_tiny_kernels()
    end
    @testset "Batched CUDA Streams" begin
        test_batched_streams()
    end
end
