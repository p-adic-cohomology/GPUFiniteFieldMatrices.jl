#using GPUFiniteFieldMatrices
#using Test
#using CUDA
#using LinearAlgebra
#using BenchmarkTools

"""
Benchmark performance of standard operations vs. in-place operations.
"""
function benchmark_inplace_vs_standard()
    println("Benchmarking in-place operations vs. standard operations...")

    n = 64  # Use multiples of 32 for no padding
    modulus = 11  # Prime modulus

    # Create test matrices
    A_data = rand(0:(modulus-1), n, n)
    B_data = rand(0:(modulus-1), n, n)
    C_data = zeros(Int, n, n)

    A = CuModMatrix(A_data, modulus)
    B = CuModMatrix(B_data, modulus)
    C = CuModMatrix(C_data, modulus)

    println("Matrix dimensions: $n × $n")
    println("Modulus: $modulus")

    # Benchmark standard addition
    println("\nBenchmarking addition operations:")
    t1 = @benchmark $A + $B
    println("Standard addition (A + B):")
    display(t1)
    println()

    # Benchmark in-place addition
    t2 = @benchmark add_inplace!($C, $A, $B)
    println("In-place addition (add_inplace!(C, A, B)):")
    display(t2)
    println()

    # Calculate speedup
    median_standard = median(t1.times)
    median_inplace = median(t2.times)
    speedup = median_standard / median_inplace
    println("Speedup for in-place addition: $(round(speedup, digits=2))x")

    # Benchmark standard matrix multiplication
    println("\nBenchmarking matrix multiplication operations:")
    t3 = @benchmark $A * $B
    println("Standard multiplication (A * B):")
    display(t3)
    println()

    # Benchmark direct matrix multiplication
    t4 = @benchmark mat_mul_gpu_type($A, $B)
    println("Direct multiplication (mat_mul_gpu_type(A, B)):")
    display(t4)
    println()

    # Benchmark in-place matrix multiplication
    t5 = @benchmark mat_mul_gpu_type($A, $B, $C)
    println("In-place multiplication (mat_mul_gpu_type(A, B, C)):")
    display(t5)
    println()

    # Calculate speedups
    median_standard_mul = median(t3.times)
    median_direct_mul = median(t4.times)
    median_inplace_mul = median(t5.times)

    speedup_direct = median_standard_mul / median_direct_mul
    speedup_inplace = median_standard_mul / median_inplace_mul

    println("Speedup for direct multiplication: $(round(speedup_direct, digits=2))x")
    println("Speedup for in-place multiplication: $(round(speedup_inplace, digits=2))x")

    println("All benchmarks completed successfully!")
end

"""
Benchmark performance of different matrix multiplication regimes.
"""
function benchmark_matmul_regimes()
    println("Benchmarking different matrix multiplication regimes...")

    n = 128  # Use multiples of 32
    modulus = 101  # Prime modulus

    A_data = rand(0:(modulus-1), n, n)
    B_data = rand(0:(modulus-1), n, n)

    A = CuModMatrix(A_data, modulus)
    B = CuModMatrix(B_data, modulus)

    println("Matrix dimensions: $n × $n")
    println("Modulus: $modulus")

    # Benchmark standard regime
    println("\nBenchmarking matrix multiplication regimes:")
    t1 = @benchmark mat_mul_gpu_type($A, $B, -1, "⊠")
    println("Standard regime (⊠):")
    display(t1)
    println()

    # Benchmark hybrid regime
    t2 = @benchmark mat_mul_gpu_type($A, $B, -1, "hybrid")
    println("Hybrid regime (hybrid):")
    display(t2)
    println()

    # Calculate speedup
    median_standard = median(t1.times)
    median_hybrid = median(t2.times)

    if median_standard > median_hybrid
        speedup = median_standard / median_hybrid
        println("Hybrid regime is $(round(speedup, digits=2))x faster than standard regime")
    else
        speedup = median_hybrid / median_standard
        println("Standard regime is $(round(speedup, digits=2))x faster than hybrid regime")
    end

    println("All regime benchmarks completed successfully!")
end

"""
Benchmark RREF operations.
"""
function benchmark_rref_operations()
    println("Benchmarking RREF operations...")

    sizes = [32, 64, 128]
    modulus = 17  # Prime modulus

    for n in sizes
        println("\nMatrix dimensions: $n × $n")
        println("Modulus: $modulus")

        # Create test matrix
        A_data = rand(0:(modulus-1), n, n)
        A = CuModMatrix(A_data, modulus)

        # Benchmark standard RREF
        t1 = @benchmark rref_gpu_type($A)
        println("Standard RREF (rref_gpu_type):")
        display(t1)
        println()

        # Benchmark direct RREF
        t2 = @benchmark rref_gpu_type($A)
        println("Direct RREF (rref_gpu_type):")
        display(t2)
        println()

        # Calculate speedup
        median_standard = median(t1.times)
        median_direct = median(t2.times)

        if median_standard > median_direct
            speedup = median_standard / median_direct
            println("Direct RREF is $(round(speedup, digits=2))x faster than standard RREF")
        else
            speedup = median_direct / median_standard
            println("Standard RREF is $(round(speedup, digits=2))x faster than direct RREF")
        end
    end

    println("All RREF benchmarks completed successfully!")
end

function run_benchmarks()
    benchmark_inplace_vs_standard()
    benchmark_matmul_regimes()
    benchmark_rref_operations()

    println("\nAll performance benchmarks completed!")
end

# Run the benchmarks if this file is run directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmarks()
end
