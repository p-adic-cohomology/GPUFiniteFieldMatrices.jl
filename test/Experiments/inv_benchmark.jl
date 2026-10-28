using GPUFiniteFieldMatrices
using CUDA
using Nemo
using Statistics
using Printf
using Random
using LinearAlgebra

struct InverseBenchSpec
    rows::Int
    cols::Int
    p::Int
    elem_type::DataType
end

"""
    make_spec(rows, cols, p, dtype)

Create an `InverseBenchSpec` from tuple fields where `dtype` is one of
`:FP32`, `:FP64`, `Float32`, or `Float64`.
"""
function make_spec(rows::Int, cols::Int, p::Int, dtype)
    et =
        dtype == :FP32 ? Float32 :
        dtype == :FP64 ? Float64 :
        dtype == Float32 ? Float32 :
        dtype == Float64 ? Float64 :
        throw(ArgumentError("dtype must be :FP32, :FP64, Float32, or Float64"))
    return InverseBenchSpec(rows, cols, p, et)
end

"""
    normalize_specs(spec_tuples)

Convert a vector of `(rows, cols, p, dtype)` tuples into benchmark specs.
"""
function normalize_specs(spec_tuples::Vector{<:Tuple})
    return [make_spec(Int(s[1]), Int(s[2]), Int(s[3]), s[4]) for s in spec_tuples]
end

"""
    nemo_inverse_matrix(A_cpu, p)

Compute exact inverse in Nemo over `GF(p)` and return a dense Julia matrix.
"""
function nemo_inverse_matrix(A_cpu::AbstractMatrix, p::Int)
    R = Nemo.GF(p)
    n, m = size(A_cpu)

    A_nemo = Nemo.matrix(R, [R(Int(mod(x, p))) for x in A_cpu])
    Ainv_nemo = inv(A_nemo)
    out = Matrix{Int}(undef, n, n)
    for j = 1:n
        for i = 1:n
            out[i, j] = Int(lift(ZZ, Ainv_nemo[i, j]))
        end
    end
    return out
end

"""
    random_invertible_cumod(spec; max_tries=64, rng=Random.default_rng())

Generate an invertible `CuModMatrix` matching `spec`.
"""
function random_invertible_cumod(
    spec::InverseBenchSpec;
    max_tries::Int = 64,
    rng = Random.default_rng(),
)
    for _ = 1:max_tries
        Ahost = rand(rng, 0:(spec.p-1), spec.rows, spec.cols)
        Atyped = Matrix{spec.elem_type}(Ahost)
        A = CuModMatrix{spec.elem_type}(Atyped; modulus = spec.p)
        if GPUFiniteFieldMatrices.is_invertible(A)
            return A
        end
    end
    throw(
        ErrorException(
            "failed to sample invertible matrix for spec $(spec.rows)x$(spec.cols), p=$(spec.p)",
        ),
    )
end

function random_one_sided_invertible_cumod(
    spec::InverseBenchSpec;
    max_tries::Int = 64,
    rng = Random.default_rng(),
    rectangular_requires_full_rank::Bool = false,
)
    if spec.rows != spec.cols && !rectangular_requires_full_rank
        Ahost = rand(rng, 0:(spec.p-1), spec.rows, spec.cols)
        Atyped = Matrix{spec.elem_type}(Ahost)
        return CuModMatrix{spec.elem_type}(Atyped; modulus = spec.p)
    end
    for _ = 1:max_tries
        Ahost = rand(rng, 0:(spec.p-1), spec.rows, spec.cols)
        Atyped = Matrix{spec.elem_type}(Ahost)
        A = CuModMatrix{spec.elem_type}(Atyped; modulus = spec.p)
        ok = if spec.rows == spec.cols
            GPUFiniteFieldMatrices.is_invertible(A)
        elseif spec.rows < spec.cols
            try
                GPUFiniteFieldMatrices.right_inverse(A)
                true
            catch
                false
            end
        else
            try
                GPUFiniteFieldMatrices.left_inverse(A)
                true
            catch
                false
            end
        end
        ok && return A
    end
    throw(
        ErrorException(
            "failed to sample one-sided invertible matrix for spec $(spec.rows)x$(spec.cols), p=$(spec.p)",
        ),
    )
end

"""
    prime_gpu!(; p=101, elem_type=Float32, n=100)

Warm up the GPU with scalar and batched inverse calls.
"""
function prime_gpu!(; p::Int = 101, elem_type::DataType = Float32, n::Int = 100)
    Ih = Matrix{elem_type}(I, n, n)
    A = CuModMatrix{elem_type}(Ih; modulus = p)
    CUDA.@sync inv(A)
    CUDA.@sync GPUFiniteFieldMatrices.inverse_batched([A])
    return nothing
end

function _time_gpu_call(f)
    t = @elapsed begin
        CUDA.@sync f()
    end
    return t
end

function _time_gpu_batch(f, batch)
    t = @elapsed begin
        f(batch)
        CUDA.synchronize()
    end
    return t / length(batch)
end

function _tiny_inverse_writer(n::Int)
    n == 4 && return GPUFiniteFieldMatrices.inverse_batched_4x4!
    n == 8 && return GPUFiniteFieldMatrices.inverse_batched_8x8!
    n == 16 && return GPUFiniteFieldMatrices.inverse_batched_16x16!
    n == 32 && return GPUFiniteFieldMatrices.inverse_batched_32x32!
    throw(ArgumentError("supported tiny inverse sizes are 4, 8, 16, and 32"))
end

function _warm_gpu_tiny_inverse_batch(f, host_batch, p::Int)
    T = eltype(first(host_batch))
    mats = [CuModMatrix{T}(copy(A); modulus = p) for A in host_batch]
    dests = [
        CuModMatrix(CUDA.zeros(T, size(parent(A))...); modulus = p, new_size = size(A))
        for A in mats
    ]
    f(dests, mats)
    CUDA.synchronize()
    return nothing
end

function _warm_gpu_scalar_inverse_batch(host_batch, p::Int)
    T = eltype(first(host_batch))
    mats = [CuModMatrix{T}(copy(A); modulus = p) for A in host_batch]
    CUDA.synchronize()
    for A in mats
        inv(A)
    end
    CUDA.synchronize()
    return nothing
end

function _time_gpu_tiny_inverse_batch(f, host_batch, p::Int)
    T = eltype(first(host_batch))
    mats = [CuModMatrix{T}(copy(A); modulus = p) for A in host_batch]
    dests = [
        CuModMatrix(CUDA.zeros(T, size(parent(A))...); modulus = p, new_size = size(A))
        for A in mats
    ]
    CUDA.synchronize()
    t = @elapsed begin
        f(dests, mats)
        CUDA.synchronize()
    end
    return t / length(mats)
end

function _time_gpu_scalar_inverse_batch(host_batch, p::Int)
    T = eltype(first(host_batch))
    mats = [CuModMatrix{T}(copy(A); modulus = p) for A in host_batch]
    CUDA.synchronize()
    t = @elapsed begin
        for A in mats
            inv(A)
        end
        CUDA.synchronize()
    end
    return t / length(mats)
end

function _time_cpu_call(f)
    t = @elapsed f()
    return t
end

function _time_cpu_batch(f, batch)
    t = @elapsed begin
        for A in batch
            f(A)
        end
    end
    return t / length(batch)
end

function nemo_is_invertible_with_inverse(A_cpu::AbstractMatrix, p::Int)
    R = Nemo.GF(p)
    n, m = size(A_cpu)
    A_nemo = Nemo.matrix(R, [R(Int(mod(x, p))) for x in A_cpu])
    side = n <= m ? :right : :left
    elapsed = @elapsed begin
        Nemo.is_invertible_with_inverse(A_nemo, side = side)
    end
    invflag, invmat = Nemo.is_invertible_with_inverse(A_nemo, side = side)
    return (elapsed = elapsed, invertible = invflag, inverse = invmat, side = side)
end

"""
    benchmark_inverse_suite(spec_tuples; trials=5, warmup=true, check_against_nemo=false, make_plot=true, plot_path="test/Experiments/inv_benchmark.png", verbose=true)

Benchmark GPU inverse_batched, scalar Base.inv, and Nemo inverse
over a list of tuple specs `(rows, cols, p, dtype)`.
"""
function benchmark_inverse_suite(
    spec_tuples::Vector{<:Tuple};
    trials::Int = 5,
    warmup::Bool = true,
    check_against_nemo::Bool = false,
    make_plot::Bool = true,
    plot_path::String = "test/Experiments/inv_benchmark.png",
    batch_count::Int = 4,
    benchmark_mode::Symbol = :throughput,
    compare_scalar::Bool = false,
    verbose::Bool = true,
)
    if benchmark_mode != :throughput && benchmark_mode != :latency
        throw(ArgumentError("benchmark_mode must be :throughput or :latency"))
    end
    specs = normalize_specs(spec_tuples)
    if warmup
        prime_gpu!()
    end
    results = NamedTuple[]
    for spec in specs
        local_batch =
            benchmark_mode == :latency ? 1 :
            ((max(spec.rows, spec.cols) >= 1024) ? 1 : max(1, batch_count))
        batch = [random_one_sided_invertible_cumod(spec) for _ = 1:local_batch]
        A = batch[1]
        A_cpu_batch = [mod.(round.(Int, Array(B)), spec.p) for B in batch]
        A_cpu = mod.(round.(Int, Array(A)), spec.p)
        nemo_inv =
            check_against_nemo && spec.rows == spec.cols ?
            nemo_inverse_matrix(A_cpu, spec.p) : nothing
        batched_times = Float64[]
        scalar_times = Float64[]
        nemo_times = Float64[]
        for _ = 1:trials
            t_batched = try
                _time_gpu_batch(B -> GPUFiniteFieldMatrices.inverse_batched(B), batch)
            catch
                NaN
            end
            t_scalar = try
                (compare_scalar && spec.rows == spec.cols) ? _time_gpu_call(() -> inv(A)) : NaN
            catch
                NaN
            end
            t_nemo = try
                benchmark_mode == :throughput ?
                _time_cpu_batch(
                    B ->
                        spec.rows == spec.cols ? nemo_inverse_matrix(B, spec.p) :
                        nemo_is_invertible_with_inverse(B, spec.p),
                    A_cpu_batch,
                ) :
                _time_cpu_call(
                    () ->
                        spec.rows == spec.cols ? nemo_inverse_matrix(A_cpu, spec.p) :
                        nemo_is_invertible_with_inverse(A_cpu, spec.p),
                )
            catch
                NaN
            end
            push!(batched_times, t_batched)
            push!(scalar_times, t_scalar)
            push!(nemo_times, t_nemo)
        end
        Ainv_batched = try
            GPUFiniteFieldMatrices.inverse_batched([A])[1]
        catch
            nothing
        end
        Ainv_scalar = try
            (compare_scalar && spec.rows == spec.cols) ? inv(A) : nothing
        catch
            nothing
        end
        batched_ok = true
        scalar_ok = compare_scalar && spec.rows == spec.cols
        if check_against_nemo
            if spec.rows == spec.cols && Ainv_batched !== nothing
                Ainv_batched_cpu = mod.(round.(Int, Array(Ainv_batched)), spec.p)
                batched_ok = Ainv_batched_cpu == nemo_inv
                if compare_scalar && Ainv_scalar !== nothing
                    Ainv_scalar_cpu = mod.(round.(Int, Array(Ainv_scalar)), spec.p)
                    scalar_ok = Ainv_scalar_cpu == nemo_inv
                end
            else
                nres = nemo_is_invertible_with_inverse(A_cpu, spec.p)
                batched_ok = nres.invertible
                scalar_ok = false
            end
        end
        row = (
            rows = spec.rows,
            cols = spec.cols,
            p = spec.p,
            elem_type = spec.elem_type,
            batched_mean = mean(batched_times),
            scalar_mean = (compare_scalar && spec.rows == spec.cols) ?
                          mean(scalar_times) : NaN,
            nemo_mean = mean(nemo_times),
            batched_median = median(batched_times),
            scalar_median = (compare_scalar && spec.rows == spec.cols) ?
                            median(scalar_times) : NaN,
            nemo_median = median(nemo_times),
            batched_ok = batched_ok,
            scalar_ok = scalar_ok,
        )
        push!(results, row)
        if verbose
            @printf(
                "mode=%s spec=(%d,%d,p=%d,%s)  batched=%.6f s  scalar=%.6f s  nemo=%.6f s\n",
                string(benchmark_mode),
                spec.rows,
                spec.cols,
                spec.p,
                string(spec.elem_type),
                row.batched_mean,
                row.scalar_mean,
                row.nemo_mean
            )
        end
    end
    if make_plot
        try
            @eval using Plots
            x = [r.rows for r in results]
            y_new = [r.batched_mean for r in results]
            y_old = [r.scalar_mean for r in results]
            y_nemo = [r.nemo_mean for r in results]
            plt = plot(
                x,
                y_new;
                label = "inverse_batched (GPU)",
                color = :blue,
                marker = :circle,
                linewidth = 2,
                xlabel = "matrix size (rows)",
                ylabel = "time (s)",
                title = "Inverse benchmark: batched vs scalar vs Nemo",
            )
            if compare_scalar
                plot!(
                    plt,
                    x,
                    y_old;
                    label = "Base.inv (GPU square-only)",
                    color = :red,
                    marker = :square,
                    linewidth = 2,
                )
            end
            plot!(
                plt,
                x,
                y_nemo;
                label = "Nemo (CPU)",
                color = :green,
                marker = :diamond,
                linewidth = 2,
            )
            savefig(plt, plot_path)
            if verbose
                println("saved plot to $(plot_path)")
            end
        catch err
            println("plot skipped: $(err)")
        end
    end
    return results
end

"""
    run_default_inverse_benchmark(; check_against_nemo=false, trials=5, warmup=true, make_plot=true)

Run a default benchmark grid and return row-wise results.
"""
function run_default_inverse_benchmark(;
    check_against_nemo::Bool = false,
    trials::Int = 5,
    warmup::Bool = true,
    make_plot::Bool = true,
)
    specs = [
        (8, 8, 101, :FP32),
        (8, 8, 101, :FP64),
        (16, 16, 101, :FP32),
        (16, 16, 101, :FP64),
        (24, 24, 101, :FP32),
        (24, 24, 101, :FP64),
        (32, 32, 101, :FP32),
        (32, 32, 101, :FP64),
        (48, 48, 101, :FP32),
        (48, 48, 101, :FP64),
        (64, 64, 101, :FP32),
        (64, 64, 101, :FP64),
    ]
    return benchmark_inverse_suite(
        specs;
        trials = trials,
        warmup = warmup,
        check_against_nemo = check_against_nemo,
        make_plot = make_plot,
        compare_scalar = false,
    )
end

function run_fastest_vs_nemo_benchmark(;
    check_against_nemo::Bool = false,
    trials::Int = 5,
    warmup::Bool = true,
    make_plot::Bool = true,
    batch_count::Int = 4,
    benchmark_mode::Symbol = :throughput,
)
    specs = [
        (16, 16, 101, :FP32),
        (16, 16, 101, :FP64),
        (64, 64, 101, :FP32),
        (64, 64, 101, :FP64),
        (256, 256, 101, :FP32),
        (256, 256, 101, :FP64),
        (256, 640, 101, :FP32),
        (256, 640, 101, :FP64),
        (1024, 1024, 101, :FP32),
        (1024, 1024, 101, :FP64),
        (2048, 2048, 101, :FP32),
        (2048, 2048, 101, :FP64),
        (4096, 4096, 101, :FP32),
        (4096, 4096, 101, :FP64),
        (8192, 8192, 101, :FP32),
        (8192, 8192, 101, :FP64),
        (16384, 16384, 101, :FP32),
        (16384, 16384, 101, :FP64),
    ]
    return benchmark_inverse_suite(
        specs;
        trials = trials,
        warmup = warmup,
        check_against_nemo = check_against_nemo,
        make_plot = make_plot,
        batch_count = batch_count,
        benchmark_mode = benchmark_mode,
        compare_scalar = false,
    )
end

function run_tiny_batched_kernel_benchmark(;
    trials::Int = 10,
    warmup::Bool = true,
    p::Int = 101,
    elem_type::DataType = Float32,
    batch_size::Int = 64,
)
    warmup && prime_gpu!(p = p, elem_type = elem_type, n = 32)
    ns = (4, 8, 16, 32)
    rng = Random.MersenneTwister(23)
    options = PLUQOptions(inverse_strategy = :augmented)
    rows = NamedTuple[]
    for n in ns
        host_mats = Matrix{elem_type}[]
        for _ = 1:batch_size
            A = Matrix{elem_type}(I, n, n)
            for i = 1:n
                for j = (i+1):n
                    A[i, j] = elem_type(rand(rng, 1:(p-1)))
                end
            end
            push!(host_mats, A)
        end
        inverse! = _tiny_inverse_writer(n)
        measured_tiny_call = (D, B) -> inverse!(D, B; options = options)
        if warmup
            _warm_gpu_tiny_inverse_batch(measured_tiny_call, host_mats, p)
            _warm_gpu_scalar_inverse_batch(host_mats, p)
        end
        t_fast = Float64[]
        t_generic = Float64[]
        for _ = 1:trials
            t1 = _time_gpu_tiny_inverse_batch(measured_tiny_call, host_mats, p)
            t2 = _time_gpu_scalar_inverse_batch(host_mats, p)
            push!(t_fast, t1)
            push!(t_generic, t2)
        end
        push!(
            rows,
            (
                n = n,
                elem_type = elem_type,
                fast_mean = mean(t_fast),
                generic_mean = mean(t_generic),
                speedup = mean(t_generic) / mean(t_fast),
            ),
        )
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_fastest_vs_nemo_benchmark()
end
