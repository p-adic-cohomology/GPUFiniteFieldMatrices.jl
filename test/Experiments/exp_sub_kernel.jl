using CUDA
using BenchmarkTools
using PrettyTables
using Printf
include("pretty_table.jl")

CUDA.allowscalar(false)

const DEFAULT_THREADS = 256

function k_sub_scalar!(A, s, m)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    len = length(A)
    @inbounds for i = idx:stride:len
        A[i] = mod(A[i] - s, m)
    end
    return
end

function k_sub_matrix!(A, B, m)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    len = length(A)
    @inbounds for i = idx:stride:len
        A[i] = mod(A[i] - B[i], m)
    end
    return
end

function launch_sub_scalar!(A, s, m; threads::Int = DEFAULT_THREADS)
    len = length(A)
    blocks = min(cld(len, threads), 65535)
    @cuda threads=threads blocks=blocks k_sub_scalar!(A, s, m)
    return nothing
end

function launch_sub_matrix!(A, B, m; threads::Int = DEFAULT_THREADS)
    len = length(A)
    blocks = min(cld(len, threads), 65535)
    @cuda threads=threads blocks=blocks k_sub_matrix!(A, B, m)
    return nothing
end

sub_scalar_broadcast!(A, s, m) = (A .= mod.(A .- s, m); nothing)
sub_matrix_broadcast!(A, B, m) = (A .= mod.(A .- B, m); nothing)

sub_scalar_gpu!(A, s, m; threads = DEFAULT_THREADS) =
    (launch_sub_scalar!(A, s, m; threads = threads); nothing)
sub_matrix_gpu!(A, B, m; threads = DEFAULT_THREADS) =
    (launch_sub_matrix!(A, B, m; threads = threads); nothing)

function bench_sub(
    specs;
    threads::Int = DEFAULT_THREADS,
    seconds::Float64 = 0.25,
    csv_path::AbstractString = "sub_bench_results.csv",
)
    rows = Any[]
    for (nr, nc, mod_in, T) in specs
        println("\n=== SUB: $(nr)x$(nc), mod=$(mod_in), T=$(T) ===")

        A_cpu = T.(rand(0:(10^6), nr, nc))
        B_cpu = T.(rand(0:(10^6), nr, nc))

        A1 = CuArray(A_cpu)
        A2 = CuArray(A_cpu)
        B = CuArray(B_cpu)

        m = convert(T, mod_in)
        s = convert(T, 7)

        sub_scalar_broadcast!(A1, s, m)
        CUDA.synchronize()
        sub_scalar_gpu!(A2, s, m; threads = threads)
        CUDA.synchronize()

        t_scalar_b = @belapsed begin
            sub_scalar_broadcast!($A1, $s, $m)
            CUDA.synchronize()
        end seconds=seconds

        t_scalar_k = @belapsed begin
            sub_scalar_gpu!($A2, $s, $m; threads = $threads)
            CUDA.synchronize()
        end seconds=seconds

        speedup_scalar = 100 * (t_scalar_b / t_scalar_k - 1)
        push!(rows, (nr, nc, string(T), mod_in, "scalar", "broadcast", 1e3*t_scalar_b, ""))
        push!(
            rows,
            (nr, nc, string(T), mod_in, "scalar", "kernel", 1e3*t_scalar_k, speedup_scalar),
        )

        sub_matrix_broadcast!(A1, B, m)
        CUDA.synchronize()
        sub_matrix_gpu!(A2, B, m; threads = threads)
        CUDA.synchronize()

        t_matrix_b = @belapsed begin
            sub_matrix_broadcast!($A1, $B, $m)
            CUDA.synchronize()
        end seconds=seconds

        t_matrix_k = @belapsed begin
            sub_matrix_gpu!($A2, $B, $m; threads = $threads)
            CUDA.synchronize()
        end seconds=seconds

        speedup_matrix = 100 * (t_matrix_b / t_matrix_k - 1)
        push!(rows, (nr, nc, string(T), mod_in, "matrix", "broadcast", 1e3*t_matrix_b, ""))
        push!(
            rows,
            (nr, nc, string(T), mod_in, "matrix", "kernel", 1e3*t_matrix_k, speedup_matrix),
        )
    end

    headers = [
        "nrows",
        "ncols",
        "dtype",
        "modulus",
        "variant",
        "method",
        "time (ms)",
        "speedup (%)",
    ]
    print_and_save_table(rows, headers; csv_path = csv_path, title = "SUB")
    return nothing
end

specs = [(3200, 3200, 13, Float16), (3200, 3200, 13, Float32), (3200, 3200, 13, Float64)]

bench_sub(specs; threads = 256, seconds = 0.25, csv_path = "sub_bench_results.csv")
nothing
