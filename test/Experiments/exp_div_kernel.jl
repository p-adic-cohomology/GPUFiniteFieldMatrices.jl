using CUDA
using BenchmarkTools
using PrettyTables
using Printf
include("pretty_table.jl")

CUDA.allowscalar(false)

const DEFAULT_THREADS = 256

function k_div_scalar!(A, s, m)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    len = length(A)
    @inbounds for i = idx:stride:len
        A[i] = mod(A[i] * s, m)
    end
    return
end

function launch_div_scalar!(A, s, m; threads::Int = DEFAULT_THREADS)
    len = length(A)
    blocks = min(cld(len, threads), 65535)
    @cuda threads=threads blocks=blocks k_div_scalar!(A, s, m)
    return nothing
end

div_scalar_broadcast!(A, s, m) = (A .= mod.(A .* s, m); nothing)
div_scalar_gpu!(A, s, m; threads = DEFAULT_THREADS) =
    (launch_div_scalar!(A, s, m; threads = threads); nothing)

function invmod_int(a::Int, m::Int)
    a = mod(a, m)
    t, newt = 0, 1
    r, newr = m, a
    while newr != 0
        q = r ÷ newr
        t, newt = newt, t - q * newt
        r, newr = newr, r - q * newr
    end
    r == 1 || error("no inverse")
    t < 0 && (t += m)
    return t
end

function bench_div(
    specs;
    threads::Int = DEFAULT_THREADS,
    seconds::Float64 = 0.25,
    csv_path::AbstractString = "div_bench_results.csv",
)
    rows = Any[]
    for (nr, nc, mod_in, T) in specs
        println("\n=== DIV: $(nr)x$(nc), mod=$(mod_in), T=$(T) ===")

        A_cpu = T.(rand(0:(10^6), nr, nc))
        A1 = CuArray(A_cpu)
        A2 = CuArray(A_cpu)

        m = convert(T, mod_in)
        s_inv = convert(T, invmod_int(7, mod_in))

        div_scalar_broadcast!(A1, s_inv, m)
        CUDA.synchronize()
        div_scalar_gpu!(A2, s_inv, m; threads = threads)
        CUDA.synchronize()

        t_scalar_b = @belapsed begin
            div_scalar_broadcast!($A1, $s_inv, $m)
            CUDA.synchronize()
        end seconds=seconds

        t_scalar_k = @belapsed begin
            div_scalar_gpu!($A2, $s_inv, $m; threads = $threads)
            CUDA.synchronize()
        end seconds=seconds

        speedup_scalar = 100 * (t_scalar_b / t_scalar_k - 1)
        push!(rows, (nr, nc, string(T), mod_in, "scalar", "broadcast", 1e3*t_scalar_b, ""))
        push!(
            rows,
            (nr, nc, string(T), mod_in, "scalar", "kernel", 1e3*t_scalar_k, speedup_scalar),
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
    print_and_save_table(rows, headers; csv_path = csv_path, title = "DIV")
    return nothing
end

specs = [(3200, 3200, 13, Float16), (3200, 3200, 13, Float32), (3200, 3200, 13, Float64)]

bench_div(specs; threads = 256, seconds = 0.25, csv_path = "div_bench_results.csv")
nothing
