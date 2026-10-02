using CUDA
using BenchmarkTools
using PrettyTables
using Printf
include("pretty_table.jl")

CUDA.allowscalar(false)

function k_rem!(out, A, m)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    len = length(A)
    @inbounds for i = idx:stride:len
        out[i] = rem(A[i], m)
    end
    return
end

function k_mod!(out, A, m)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    len = length(A)
    @inbounds for i = idx:stride:len
        out[i] = mod(A[i], m)
    end
    return
end

function k_pct!(out, A, m)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    len = length(A)
    @inbounds for i = idx:stride:len
        out[i] = A[i] % m
    end
    return
end

const DEFAULT_THREADS = 256

function launch_rem!(out, A, m; threads::Int = DEFAULT_THREADS)
    len = length(A)
    blocks = min(cld(len, threads), 65535)
    @cuda threads=threads blocks=blocks k_rem!(out, A, m)
    return out
end

function launch_mod!(out, A, m; threads::Int = DEFAULT_THREADS)
    len = length(A)
    blocks = min(cld(len, threads), 65535)
    @cuda threads=threads blocks=blocks k_mod!(out, A, m)
    return out
end

function launch_pct!(out, A, m; threads::Int = DEFAULT_THREADS)
    len = length(A)
    blocks = min(cld(len, threads), 65535)
    @cuda threads=threads blocks=blocks k_pct!(out, A, m)
    return out
end

broadcast_mod!(B, A, m) = (B .= mod.(A, m); nothing)
broadcast_rem!(B, A, m) = (B .= rem.(A, m); nothing)
broadcast_pct!(B, A, m) = (B .= A .% m; nothing)

mod_gpu!(B, A, m; threads = DEFAULT_THREADS) =
    (launch_mod!(B, A, m; threads = threads); nothing)
rem_gpu!(B, A, m; threads = DEFAULT_THREADS) =
    (launch_rem!(B, A, m; threads = threads); nothing)
pct_gpu!(B, A, m; threads = DEFAULT_THREADS) =
    (launch_pct!(B, A, m; threads = threads); nothing)

function bench_mod(
    specs;
    threads::Int = DEFAULT_THREADS,
    seconds::Float64 = 0.25,
    csv_path::AbstractString = "mod_bench_results.csv",
)
    rows = Any[]
    for (nr, nc, mod_in, T) in specs
        println("\n=== MOD: $(nr)x$(nc), mod=$(mod_in), T=$(T) ===")
        A_cpu = T.(rand(0:(10^6), nr, nc))
        A = CuArray(A_cpu)
        B = similar(A)
        m = convert(T, mod_in)

        broadcast_mod!(B, A, m)
        CUDA.synchronize()
        mod_gpu!(B, A, m; threads = threads)
        CUDA.synchronize()
        broadcast_rem!(B, A, m)
        CUDA.synchronize()
        rem_gpu!(B, A, m; threads = threads)
        CUDA.synchronize()
        broadcast_pct!(B, A, m)
        CUDA.synchronize()
        pct_gpu!(B, A, m; threads = threads)
        CUDA.synchronize()

        t_mod_b = @belapsed begin
            broadcast_mod!($B, $A, $m)
            CUDA.synchronize()
        end seconds=seconds

        t_mod_k = @belapsed begin
            mod_gpu!($B, $A, $m; threads = $threads)
            CUDA.synchronize()
        end seconds=seconds

        t_rem_b = @belapsed begin
            broadcast_rem!($B, $A, $m)
            CUDA.synchronize()
        end seconds=seconds

        t_rem_k = @belapsed begin
            rem_gpu!($B, $A, $m; threads = $threads)
            CUDA.synchronize()
        end seconds=seconds

        t_pct_b = @belapsed begin
            broadcast_pct!($B, $A, $m)
            CUDA.synchronize()
        end seconds=seconds

        t_pct_k = @belapsed begin
            pct_gpu!($B, $A, $m; threads = $threads)
            CUDA.synchronize()
        end seconds=seconds

        speedup_mod = 100 * (t_mod_b / t_mod_k - 1)
        speedup_rem = 100 * (t_rem_b / t_rem_k - 1)
        speedup_pct = 100 * (t_pct_b / t_pct_k - 1)

        push!(rows, (nr, nc, string(T), mod_in, "mod", "broadcast", 1e3*t_mod_b, ""))
        push!(rows, (nr, nc, string(T), mod_in, "mod", "kernel", 1e3*t_mod_k, speedup_mod))
        push!(rows, (nr, nc, string(T), mod_in, "rem", "broadcast", 1e3*t_rem_b, ""))
        push!(rows, (nr, nc, string(T), mod_in, "rem", "kernel", 1e3*t_rem_k, speedup_rem))
        push!(rows, (nr, nc, string(T), mod_in, "pct", "broadcast", 1e3*t_pct_b, ""))
        push!(rows, (nr, nc, string(T), mod_in, "pct", "kernel", 1e3*t_pct_k, speedup_pct))
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
    print_and_save_table(rows, headers; csv_path = csv_path, title = "MOD")
    return nothing
end

specs = [(3200, 3200, 13, Float16), (3200, 3200, 13, Float32), (3200, 3200, 13, Float64)]

bench_mod(specs; threads = 256, seconds = 0.25, csv_path = "mod_bench_results.csv")
nothing
