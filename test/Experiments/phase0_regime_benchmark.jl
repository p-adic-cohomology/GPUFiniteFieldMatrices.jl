using GPUFiniteFieldMatrices
using CUDA
using Random
using Statistics
using Printf
using DelimitedFiles

struct Phase0BenchRegime
    name::Symbol
    square_n::Int
    rect_m::Int
    rect_n::Int
    run_inverse::Bool
    long_only::Bool
end

const PHASE0_BENCH_REGIMES = [
    Phase0BenchRegime(:tiny, 16, 12, 20, true, false),
    Phase0BenchRegime(:small, 96, 72, 128, true, false),
    Phase0BenchRegime(:medium, 384, 256, 640, false, false),
    Phase0BenchRegime(:fivekish, 5008, 640, 5032, false, true),
]

const PHASE0_BENCH_CASES = [
    (:square, Float32, 7),
    (:square, Float32, 11),
    (:square, Float64, 7),
    (:square, Float64, 11),
    (:rectangular, Float32, 7),
    (:rectangular, Float32, 11),
    (:rectangular, Float64, 7),
    (:rectangular, Float64, 11),
]

function _phase0_bench_long_enabled()
    return get(ENV, "GPUFFM_PHASE0_LONG", "0") == "1"
end

function _phase0_bench_id(::Type{T}, n::Int) where {T}
    M = zeros(T, n, n)
    for i = 1:n
        M[i, i] = one(T)
    end
    return M
end

function _phase0_bench_apply_col_perm(A::Matrix{T}, perm::Vector{Int}) where {T}
    out = similar(A)
    for j in eachindex(perm)
        out[:, j] = A[:, perm[j]]
    end
    return out
end

function _phase0_bench_square_host(n::Int, p::Int, ::Type{T}, rng) where {T}
    A = _phase0_bench_id(T, n)
    for j = 2:min(n, 9)
        A[1, j] = T(rand(rng, 0:(p-1)))
    end
    return A
end

function _phase0_bench_rect_host(m::Int, n::Int, p::Int, ::Type{T}, rng) where {T}
    A = Matrix{T}(rand(rng, 0:(p-1), m, n))
    Ipart = _phase0_bench_id(T, m)
    for j = 1:m
        A[:, j] .= Ipart[:, j]
    end
    return _phase0_bench_apply_col_perm(A, randperm(rng, n))
end

function _time_gpu(f)
    t = @elapsed begin
        CUDA.@sync f()
    end
    return t
end

function _time_gpu_batched(f, batch)
    t = @elapsed begin
        for A in batch
            f(A)
        end
        CUDA.synchronize()
    end
    return t / length(batch)
end

function write_phase0_csv(rows, path::String = "test/Experiments/Phase0_benchmark.csv")
    mkpath(dirname(path))
    header = [
        "regime",
        "shape",
        "elem_type",
        "p",
        "rows",
        "cols",
        "pluq_mean",
        "pluq_median",
        "inv_mean",
        "inv_median",
    ]
    table = Matrix{Any}(undef, length(rows) + 1, length(header))
    for j in eachindex(header)
        table[1, j] = header[j]
    end
    for (i, r) in enumerate(rows)
        table[i+1, 1] = String(r.regime)
        table[i+1, 2] = String(r.shape)
        table[i+1, 3] = string(r.elem_type)
        table[i+1, 4] = r.p
        table[i+1, 5] = r.rows
        table[i+1, 6] = r.cols
        table[i+1, 7] = r.pluq_mean
        table[i+1, 8] = r.pluq_median
        table[i+1, 9] = r.inv_mean
        table[i+1, 10] = r.inv_median
    end
    open(path, "w") do io
        for i = 1:size(table, 1)
            println(io, join(table[i, :], ","))
        end
    end
    return path
end

function _read_benchmark_csv(path::String)
    lines = readlines(path)
    length(lines) <= 1 &&
        return Dict{Tuple{String,String,String,Int,Int,Int},Tuple{Float64,Float64}}()
    out = Dict{Tuple{String,String,String,Int,Int,Int},Tuple{Float64,Float64}}()
    for line in lines[2:end]
        cols = split(line, ",")
        key = (
            cols[1],
            cols[2],
            cols[3],
            parse(Int, cols[4]),
            parse(Int, cols[5]),
            parse(Int, cols[6]),
        )
        out[key] = (parse(Float64, cols[7]), parse(Float64, cols[9]))
    end
    return out
end

function write_phase1_comparison_csv(
    rows,
    phase0_csv::String,
    path::String = "test/Experiments/Phase1_speedup_vs_Phase0.csv",
)
    mkpath(dirname(path))
    baseline = _read_benchmark_csv(phase0_csv)
    header = [
        "regime",
        "shape",
        "elem_type",
        "p",
        "rows",
        "cols",
        "phase0_pluq_mean",
        "phase1_pluq_mean",
        "pluq_speedup",
        "phase0_inv_mean",
        "phase1_inv_mean",
        "inv_speedup",
    ]
    open(path, "w") do io
        println(io, join(header, ","))
        for r in rows
            key = (
                String(r.regime),
                String(r.shape),
                string(r.elem_type),
                r.p,
                r.rows,
                r.cols,
            )
            b = get(baseline, key, (NaN, NaN))
            sp_pluq =
                (isfinite(b[1]) && isfinite(r.pluq_mean) && r.pluq_mean != 0.0) ?
                b[1] / r.pluq_mean : NaN
            sp_inv =
                (isfinite(b[2]) && isfinite(r.inv_mean) && r.inv_mean != 0.0) ?
                b[2] / r.inv_mean : NaN
            println(
                io,
                join(
                    (
                        String(r.regime),
                        String(r.shape),
                        string(r.elem_type),
                        string(r.p),
                        string(r.rows),
                        string(r.cols),
                        string(b[1]),
                        string(r.pluq_mean),
                        string(sp_pluq),
                        string(b[2]),
                        string(r.inv_mean),
                        string(sp_inv),
                    ),
                    ",",
                ),
            )
        end
    end
    return path
end

function run_phase0_regime_benchmark(;
    trials::Int = 3,
    warmup::Bool = true,
    seed::Int = 7,
    verbose::Bool = true,
    debug_errors::Bool = false,
    export_csv::Bool = true,
    csv_path::String = "test/Experiments/Phase0_benchmark.csv",
    batch_count::Int = 4,
    options::PLUQOptions = PLUQOptions(),
)
    rng = Random.MersenneTwister(seed)
    if warmup
        A0 = CuModMatrix(Matrix{Float32}(I, 32, 32), 11)
        CUDA.@sync pluq_new(A0)
        CUDA.@sync inverse_new(A0)
    end
    rows = NamedTuple[]
    if verbose
        println(
            "kernel config: pivot_warp_kernel=$(options.pivot_warp_kernel), trsm_mode=$(options.trsm_mode), trsm_warp_threshold=$(options.trsm_warp_threshold), schur_tile=$(options.schur_tile), schur_transpose_u=$(options.schur_transpose_u), mod_backend=$(options.mod_backend), inverse_strategy=$(options.inverse_strategy), autotune=$(options.autotune), batch_streams=$(options.batch_streams)",
        )
    end
    for reg in PHASE0_BENCH_REGIMES
        if reg.long_only && !_phase0_bench_long_enabled()
            continue
        end
        for (shape, T, p) in PHASE0_BENCH_CASES
            local_batch =
                (reg.long_only || max(reg.square_n, reg.rect_n) >= 1024) ? 1 :
                max(1, batch_count)
            batch = CuModMatrix[]
            for _ = 1:local_batch
                A = if shape == :square
                    CuModMatrix(
                        _phase0_bench_square_host(reg.square_n, p, T, rng),
                        p;
                        elem_type = T,
                    )
                else
                    CuModMatrix(
                        _phase0_bench_rect_host(reg.rect_m, reg.rect_n, p, T, rng),
                        p;
                        elem_type = T,
                    )
                end
                push!(batch, A)
            end
            t_pluq = Float64[]
            t_inv = Float64[]
            for _ = 1:trials
                try
                    push!(
                        t_pluq,
                        _time_gpu_batched(A -> pluq_new(A, options = options), batch),
                    )
                catch err
                    if debug_errors
                        println(
                            "pluq error regime=$(reg.name) shape=$(shape) T=$(T) p=$(p): ",
                            sprint(showerror, err),
                        )
                    end
                    push!(t_pluq, NaN)
                end
                if shape == :square
                    if reg.run_inverse
                        try
                            push!(
                                t_inv,
                                _time_gpu_batched(
                                    A -> inverse_new(A, options = options),
                                    batch,
                                ),
                            )
                        catch err
                            if debug_errors
                                println(
                                    "inverse error regime=$(reg.name) shape=$(shape) T=$(T) p=$(p): ",
                                    sprint(showerror, err),
                                )
                            end
                            push!(t_inv, NaN)
                        end
                    else
                        push!(t_inv, NaN)
                    end
                else
                    if reg.run_inverse
                        try
                            push!(
                                t_inv,
                                _time_gpu_batched(
                                    A -> right_inverse_new(A, options = options),
                                    batch,
                                ),
                            )
                        catch err
                            if debug_errors
                                println(
                                    "right_inverse error regime=$(reg.name) shape=$(shape) T=$(T) p=$(p): ",
                                    sprint(showerror, err),
                                )
                            end
                            push!(t_inv, NaN)
                        end
                    else
                        push!(t_inv, NaN)
                    end
                end
            end
            r = (
                regime = reg.name,
                shape = shape,
                elem_type = T,
                p = p,
                rows = shape == :square ? reg.square_n : reg.rect_m,
                cols = shape == :square ? reg.square_n : reg.rect_n,
                pluq_mean = mean(t_pluq),
                pluq_median = median(t_pluq),
                inv_mean = mean(t_inv),
                inv_median = median(t_inv),
            )
            push!(rows, r)
            if verbose
                @printf(
                    "regime=%s shape=%s type=%s p=%d size=%dx%d pluq_mean=%.6f inv_mean=%.6f\n",
                    string(r.regime),
                    string(r.shape),
                    string(r.elem_type),
                    r.p,
                    r.rows,
                    r.cols,
                    r.pluq_mean,
                    r.inv_mean
                )
            end
        end
    end
    if export_csv
        outpath = write_phase0_csv(rows, csv_path)
        if verbose
            println("saved csv to $(outpath)")
        end
    end
    return rows
end

function run_phase1_benchmark(;
    trials::Int = 3,
    warmup::Bool = true,
    seed::Int = 7,
    verbose::Bool = true,
    export_csv::Bool = true,
    csv_path::String = "test/Experiments/Phase1_benchmark.csv",
    batch_count::Int = 4,
    baseline_csv_path::String = "test/Experiments/Phase0_benchmark.csv",
    export_speedup_csv::Bool = true,
    speedup_csv_path::String = "test/Experiments/Phase1_speedup_vs_Phase0.csv",
)
    rows = run_phase0_regime_benchmark(
        trials = trials,
        warmup = warmup,
        seed = seed,
        verbose = verbose,
        export_csv = export_csv,
        csv_path = csv_path,
        batch_count = batch_count,
    )
    if export_speedup_csv && isfile(baseline_csv_path)
        outpath = write_phase1_comparison_csv(rows, baseline_csv_path, speedup_csv_path)
        if verbose
            println("saved speedup csv to $(outpath)")
        end
    end
    return rows
end

function run_phase2_benchmark(;
    trials::Int = 3,
    warmup::Bool = true,
    seed::Int = 7,
    verbose::Bool = true,
    export_csv::Bool = true,
    csv_path::String = "test/Experiments/Phase2_benchmark.csv",
    batch_count::Int = 4,
    baseline_csv_path::String = "test/Experiments/Phase1_benchmark.csv",
    export_speedup_csv::Bool = true,
    speedup_csv_path::String = "test/Experiments/Phase2_speedup_vs_Phase1.csv",
)
    rows = run_phase0_regime_benchmark(
        trials = trials,
        warmup = warmup,
        seed = seed,
        verbose = verbose,
        export_csv = export_csv,
        csv_path = csv_path,
        batch_count = batch_count,
    )
    if export_speedup_csv && isfile(baseline_csv_path)
        outpath = write_phase1_comparison_csv(rows, baseline_csv_path, speedup_csv_path)
        if verbose
            println("saved speedup csv to $(outpath)")
        end
    end
    return rows
end

function run_phase3_benchmark(;
    trials::Int = 3,
    warmup::Bool = true,
    seed::Int = 7,
    verbose::Bool = true,
    export_csv::Bool = true,
    csv_path::String = "test/Experiments/Phase3_benchmark.csv",
    batch_count::Int = 4,
    baseline_csv_path::String = "test/Experiments/Phase2_benchmark.csv",
    export_speedup_csv::Bool = true,
    speedup_csv_path::String = "test/Experiments/Phase3_speedup_vs_Phase2.csv",
)
    rows = run_phase0_regime_benchmark(
        trials = trials,
        warmup = warmup,
        seed = seed,
        verbose = verbose,
        export_csv = export_csv,
        csv_path = csv_path,
        batch_count = batch_count,
    )
    if export_speedup_csv && isfile(baseline_csv_path)
        outpath = write_phase1_comparison_csv(rows, baseline_csv_path, speedup_csv_path)
        if verbose
            println("saved speedup csv to $(outpath)")
        end
    end
    return rows
end

function run_phase4_benchmark(;
    trials::Int = 3,
    warmup::Bool = true,
    seed::Int = 7,
    verbose::Bool = true,
    export_csv::Bool = true,
    csv_path::String = "test/Experiments/Phase4_benchmark.csv",
    batch_count::Int = 4,
    baseline_csv_path::String = "test/Experiments/Phase3_benchmark.csv",
    export_speedup_csv::Bool = true,
    speedup_csv_path::String = "test/Experiments/Phase4_speedup_vs_Phase3.csv",
)
    rows = run_phase0_regime_benchmark(
        trials = trials,
        warmup = warmup,
        seed = seed,
        verbose = verbose,
        export_csv = export_csv,
        csv_path = csv_path,
        batch_count = batch_count,
    )
    if export_speedup_csv && isfile(baseline_csv_path)
        outpath = write_phase1_comparison_csv(rows, baseline_csv_path, speedup_csv_path)
        if verbose
            println("saved speedup csv to $(outpath)")
        end
    end
    return rows
end

function run_phase5_benchmark(;
    trials::Int = 3,
    warmup::Bool = true,
    seed::Int = 7,
    verbose::Bool = true,
    export_csv::Bool = true,
    csv_path::String = "test/Experiments/Phase5_benchmark.csv",
    batch_count::Int = 8,
    baseline_csv_path::String = "test/Experiments/Phase4_benchmark.csv",
    export_speedup_csv::Bool = true,
    speedup_csv_path::String = "test/Experiments/Phase5_speedup_vs_Phase4.csv",
)
    phase5_opts = PLUQOptions(lazy_q = true, nftb = 8)
    rows = run_phase0_regime_benchmark(
        trials = trials,
        warmup = warmup,
        seed = seed,
        verbose = verbose,
        export_csv = export_csv,
        csv_path = csv_path,
        batch_count = batch_count,
        options = phase5_opts,
    )
    if export_speedup_csv && isfile(baseline_csv_path)
        outpath = write_phase1_comparison_csv(rows, baseline_csv_path, speedup_csv_path)
        if verbose
            println("saved speedup csv to $(outpath)")
        end
    end
    return rows
end

function run_phaseB2_benchmark(;
    trials::Int = 3,
    warmup::Bool = true,
    seed::Int = 7,
    verbose::Bool = true,
    export_csv::Bool = true,
    csv_path::String = "test/Experiments/PhaseB2_benchmark.csv",
    batch_count::Int = 8,
    baseline_csv_path::String = "test/Experiments/Phase5_benchmark.csv",
    export_speedup_csv::Bool = true,
    speedup_csv_path::String = "test/Experiments/PhaseB2_speedup_vs_Phase5.csv",
    options::PLUQOptions = PLUQOptions(lazy_q = true, nftb = 8),
)
    rows = run_phase0_regime_benchmark(
        trials = trials,
        warmup = warmup,
        seed = seed,
        verbose = verbose,
        export_csv = export_csv,
        csv_path = csv_path,
        batch_count = batch_count,
        options = options,
    )
    if export_speedup_csv && isfile(baseline_csv_path)
        outpath = write_phase1_comparison_csv(rows, baseline_csv_path, speedup_csv_path)
        if verbose
            println("saved speedup csv to $(outpath)")
        end
    end
    return rows
end

function run_phaseC_benchmark(;
    trials::Int = 3,
    warmup::Bool = true,
    seed::Int = 7,
    verbose::Bool = true,
    debug_errors::Bool = false,
    export_csv::Bool = true,
    csv_path::String = "test/Experiments/PhaseC_benchmark.csv",
    batch_count::Int = 8,
    baseline_csv_path::String = "test/Experiments/PhaseB2_benchmark.csv",
    export_speedup_csv::Bool = true,
    speedup_csv_path::String = "test/Experiments/PhaseC_speedup_vs_PhaseB2.csv",
    options::PLUQOptions = PLUQOptions(
        lazy_q = true,
        nftb = 8,
        trsm_mode = :auto,
        trsm_warp_threshold = 24,
        schur_tile = 16,
        schur_transpose_u = true,
        pivot_warp_kernel = :ballot,
    ),
)
    rows = run_phase0_regime_benchmark(
        trials = trials,
        warmup = warmup,
        seed = seed,
        verbose = verbose,
        debug_errors = debug_errors,
        export_csv = export_csv,
        csv_path = csv_path,
        batch_count = batch_count,
        options = options,
    )
    if export_speedup_csv && isfile(baseline_csv_path)
        outpath = write_phase1_comparison_csv(rows, baseline_csv_path, speedup_csv_path)
        if verbose
            println("saved speedup csv to $(outpath)")
        end
    end
    return rows
end

function write_phaseC_warp_compare_csv(
    rows_ballot,
    rows_shfl,
    path::String = "test/Experiments/PhaseC_warp_compare.csv",
)
    mkpath(dirname(path))
    bmap = Dict{Tuple{String,String,String,Int,Int,Int},Tuple{Float64,Float64}}()
    for r in rows_ballot
        key = (String(r.regime), String(r.shape), string(r.elem_type), r.p, r.rows, r.cols)
        bmap[key] = (r.pluq_mean, r.inv_mean)
    end
    open(path, "w") do io
        println(
            io,
            join(
                (
                    "regime",
                    "shape",
                    "elem_type",
                    "p",
                    "rows",
                    "cols",
                    "ballot_pluq_mean",
                    "shfl_pluq_mean",
                    "pluq_speedup_shfl_vs_ballot",
                    "ballot_inv_mean",
                    "shfl_inv_mean",
                    "inv_speedup_shfl_vs_ballot",
                ),
                ",",
            ),
        )
        for r in rows_shfl
            key = (
                String(r.regime),
                String(r.shape),
                string(r.elem_type),
                r.p,
                r.rows,
                r.cols,
            )
            b = get(bmap, key, (NaN, NaN))
            pluq_sp =
                (isfinite(b[1]) && isfinite(r.pluq_mean) && r.pluq_mean != 0.0) ?
                b[1] / r.pluq_mean : NaN
            inv_sp =
                (isfinite(b[2]) && isfinite(r.inv_mean) && r.inv_mean != 0.0) ?
                b[2] / r.inv_mean : NaN
            println(
                io,
                join(
                    (
                        String(r.regime),
                        String(r.shape),
                        string(r.elem_type),
                        string(r.p),
                        string(r.rows),
                        string(r.cols),
                        string(b[1]),
                        string(r.pluq_mean),
                        string(pluq_sp),
                        string(b[2]),
                        string(r.inv_mean),
                        string(inv_sp),
                    ),
                    ",",
                ),
            )
        end
    end
    return path
end

function run_phaseC_warp_compare_benchmark(;
    trials::Int = 3,
    warmup::Bool = true,
    seed::Int = 7,
    verbose::Bool = true,
    debug_errors::Bool = false,
    batch_count::Int = 8,
    ballot_csv_path::String = "test/Experiments/PhaseC_ballot_benchmark.csv",
    shfl_csv_path::String = "test/Experiments/PhaseC_shfl_benchmark.csv",
    compare_csv_path::String = "test/Experiments/PhaseC_warp_compare.csv",
)
    base_kwargs = (
        trials = trials,
        warmup = warmup,
        seed = seed,
        verbose = verbose,
        debug_errors = debug_errors,
        export_csv = true,
        batch_count = batch_count,
        export_speedup_csv = false,
    )
    ballot_opts = PLUQOptions(
        lazy_q = true,
        nftb = 8,
        trsm_mode = :auto,
        trsm_warp_threshold = 24,
        schur_tile = 16,
        schur_transpose_u = true,
        pivot_warp_kernel = :ballot,
    )
    shfl_opts = PLUQOptions(
        lazy_q = true,
        nftb = 8,
        trsm_mode = :auto,
        trsm_warp_threshold = 24,
        schur_tile = 16,
        schur_transpose_u = true,
        pivot_warp_kernel = :shfl,
    )
    rows_ballot = run_phaseC_benchmark(;
        base_kwargs...,
        csv_path = ballot_csv_path,
        options = ballot_opts,
    )
    rows_shfl = run_phaseC_benchmark(;
        base_kwargs...,
        csv_path = shfl_csv_path,
        options = shfl_opts,
    )
    outpath = write_phaseC_warp_compare_csv(rows_ballot, rows_shfl, compare_csv_path)
    if verbose
        println("saved shfl-vs-ballot csv to $(outpath)")
    end
    return (rows_ballot = rows_ballot, rows_shfl = rows_shfl, compare_csv = outpath)
end

function run_phaseDEF_benchmark(;
    trials::Int = 3,
    warmup::Bool = true,
    seed::Int = 7,
    verbose::Bool = true,
    debug_errors::Bool = false,
    export_csv::Bool = true,
    csv_path::String = "test/Experiments/PhaseDEF_benchmark.csv",
    batch_count::Int = 8,
    baseline_csv_path::String = "test/Experiments/PhaseC_benchmark.csv",
    export_speedup_csv::Bool = true,
    speedup_csv_path::String = "test/Experiments/PhaseDEF_speedup_vs_PhaseC.csv",
    options::PLUQOptions = PLUQOptions(
        lazy_q = true,
        nftb = 8,
        trsm_mode = :auto,
        trsm_warp_threshold = 24,
        schur_tile = 16,
        schur_transpose_u = true,
        pivot_warp_kernel = :ballot,
        mod_backend = :auto,
        inverse_strategy = :pluq,
        autotune = true,
        batch_streams = 2,
    ),
)
    rows = run_phase0_regime_benchmark(
        trials = trials,
        warmup = warmup,
        seed = seed,
        verbose = verbose,
        debug_errors = debug_errors,
        export_csv = export_csv,
        csv_path = csv_path,
        batch_count = batch_count,
        options = options,
    )
    if export_speedup_csv && isfile(baseline_csv_path)
        outpath = write_phase1_comparison_csv(rows, baseline_csv_path, speedup_csv_path)
        if verbose
            println("saved speedup csv to $(outpath)")
        end
    end
    return rows
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_phase0_regime_benchmark()
end
