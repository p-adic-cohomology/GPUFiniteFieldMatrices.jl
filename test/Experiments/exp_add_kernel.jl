using CUDA
using BenchmarkTools
using PrettyTables
using Printf
include("pretty_table.jl")

CUDA.allowscalar(false)

const DEFAULT_THREADS = 256

function k_add_scalar!(A, s, m)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    len = length(A)
    @inbounds for i = idx:stride:len
        A[i] = mod(A[i] + s, m)
    end
    return
end

function k_add_matrix!(A, B, m)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    len = length(A)
    @inbounds for i = idx:stride:len
        A[i] = mod(A[i] + B[i], m)
    end
    return
end

function launch_add_scalar!(A, s, m; threads::Int = DEFAULT_THREADS)
    len = length(A)
    blocks = min(cld(len, threads), 65535)
    @cuda threads=threads blocks=blocks k_add_scalar!(A, s, m)
    return nothing
end

function launch_add_matrix!(A, B, m; threads::Int = DEFAULT_THREADS)
    len = length(A)
    blocks = min(cld(len, threads), 65535)
    @cuda threads=threads blocks=blocks k_add_matrix!(A, B, m)
    return nothing
end

add_scalar_broadcast!(A, s, m) = (A .= mod.(A .+ s, m); nothing)
add_matrix_broadcast!(A, B, m) = (A .= mod.(A .+ B, m); nothing)

add_scalar_gpu!(A, s, m; threads = DEFAULT_THREADS) =
    (launch_add_scalar!(A, s, m; threads = threads); nothing)
add_matrix_gpu!(A, B, m; threads = DEFAULT_THREADS) =
    (launch_add_matrix!(A, B, m; threads = threads); nothing)

function bench_add(
    specs;
    threads::Int = DEFAULT_THREADS,
    seconds::Float64 = 0.25,
    csv_path::AbstractString = "bench_add_results.csv",
)
    rows = Any[]
    for (nr, nc, mod_in, T) in specs
        println("\n=== ADD: $(nr)x$(nc), mod=$(mod_in), T=$(T) ===")

        A_cpu = T.(rand(0:(10^6), nr, nc))
        B_cpu = T.(rand(0:(10^6), nr, nc))

        A1 = CuArray(A_cpu)
        A2 = CuArray(A_cpu)
        B = CuArray(B_cpu)

        m = convert(T, mod_in)
        s = convert(T, 7)

        add_scalar_broadcast!(A1, s, m)
        CUDA.synchronize()
        add_scalar_gpu!(A2, s, m; threads = threads)
        CUDA.synchronize()

        t_scalar_b = @belapsed begin
            add_scalar_broadcast!($A1, $s, $m)
            CUDA.synchronize()
        end seconds=seconds

        t_scalar_k = @belapsed begin
            add_scalar_gpu!($A2, $s, $m; threads = $threads)
            CUDA.synchronize()
        end seconds=seconds

        speedup_scalar = 100 * (t_scalar_b / t_scalar_k - 1)
        push!(rows, (nr, nc, string(T), mod_in, "scalar", "broadcast", 1e3*t_scalar_b, ""))
        push!(
            rows,
            (nr, nc, string(T), mod_in, "scalar", "kernel", 1e3*t_scalar_k, speedup_scalar),
        )

        add_matrix_broadcast!(A1, B, m)
        CUDA.synchronize()
        add_matrix_gpu!(A2, B, m; threads = threads)
        CUDA.synchronize()

        t_matrix_b = @belapsed begin
            add_matrix_broadcast!($A1, $B, $m)
            CUDA.synchronize()
        end seconds=seconds

        t_matrix_k = @belapsed begin
            add_matrix_gpu!($A2, $B, $m; threads = $threads)
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
    print_and_save_table(rows, headers; csv_path = csv_path, title = "ADD")

    return nothing
end

specs = [(3200, 3200, 13, Float16), (3200, 3200, 13, Float32), (3200, 3200, 13, Float64)]

bench_add(specs; threads = 256, seconds = 0.25, csv_path = "mod_bench_results.csv")
nothing
