function _load_capture(value)
    value isa AbstractString && return GFMPerfCore.read_run(value)
    return value
end

function _comparison_text(report)
    io = IOBuffer()
    println(io, "GFM CUDA performance comparison")
    println(io, "Current:  ", report["current_run"])
    println(io, "Baseline: ", report["baseline_run"])
    println(io, "Environment compatible: ", report["compatible"])
    println(io, "Median slowdown flag: >", round(100 * report["threshold_fraction"], digits = 1), "%")
    println(io)
    for row in report["rows"]
        if haskey(row, "ratio")
            println(io, row["status"], "  ", row["case_id"])
            println(io, "  ratio=", round(row["ratio"], digits = 4),
                "  change=", round(row["change_percent"], digits = 2), "%",
                "  noise≈", round(100 * row["noise_estimate_fraction"], digits = 2), "%")
            println(io, "  CPU allocations bytes/count: ", row["current_cpu_alloc_bytes"], "/",
                row["current_cpu_alloc_count"], " vs ", row["baseline_cpu_alloc_bytes"], "/",
                row["baseline_cpu_alloc_count"])
            println(io, "  GPU allocations bytes/count: ", row["current_gpu_alloc_bytes"], "/",
                row["current_gpu_alloc_count"], " vs ", row["baseline_gpu_alloc_bytes"], "/",
                row["baseline_gpu_alloc_count"])
            println(io, "  allocation deltas CPU bytes/count=", row["cpu_alloc_bytes_delta"], "/",
                row["cpu_alloc_count_delta"], " GPU bytes/count=", row["gpu_alloc_bytes_delta"], "/",
                row["gpu_alloc_count_delta"])
        else
            println(io, row["status"], "  ", row["case_id"], "  ", get(row, "detail", ""))
        end
    end
    !isempty(report["flagged_case_ids"]) && println(io, "\nFlagged cases are suspected only. Repeat fresh-process baseline/candidate captures before calling a regression confirmed.")
    return String(take!(io))
end

function _write_comparison_files(report, outdir)
    mkpath(outdir)
    write_json(joinpath(outdir, "comparison.json"), report)
    open(joinpath(outdir, "comparison.txt"), "w") do io
        write(io, _comparison_text(report))
    end
    open(joinpath(outdir, "comparison.csv"), "w") do io
        write(io, csv_line(("case_id", "status", "ratio", "change_percent", "noise_estimate_fraction",
            "current_cpu_alloc_bytes", "baseline_cpu_alloc_bytes", "current_gpu_alloc_bytes",
            "baseline_gpu_alloc_bytes", "cpu_alloc_bytes_delta", "gpu_alloc_bytes_delta", "detail")))
        for row in report["rows"]
            write(io, csv_line((get(row, "case_id", ""), get(row, "status", ""),
                get(row, "ratio", nothing), get(row, "change_percent", nothing),
                get(row, "noise_estimate_fraction", nothing), get(row, "current_cpu_alloc_bytes", nothing),
                get(row, "baseline_cpu_alloc_bytes", nothing), get(row, "current_gpu_alloc_bytes", nothing),
                get(row, "baseline_gpu_alloc_bytes", nothing), get(row, "cpu_alloc_bytes_delta", nothing),
                get(row, "gpu_alloc_bytes_delta", nothing), get(row, "detail", ""))))
        end
    end
    return outdir
end

"""Compare a current run to one reference capture (normally v0.3.2)."""
function compare(current, baseline; threshold = 0.10, min_samples = 5, output_dir = nothing)
    c = _load_capture(current)
    b = _load_capture(baseline)
    report = GFMPerfCore.compare_runs(c, b; threshold = threshold, min_samples = min_samples)
    if isnothing(output_dir)
        current_dir = get(c, "run_dir", pwd())
        output_dir = joinpath(dirname(current_dir), "comparisons", "$(basename(current_dir))-vs-$(basename(get(b, "run_dir", "baseline")))")
    end
    ispath(output_dir) && throw(ArgumentError("comparison output already exists: $output_dir"))
    report["comparison_dir"] = abspath(output_dir)
    _write_comparison_files(report, output_dir)
    print(_comparison_text(report))
    return abspath(output_dir)
end

"""Capture only cases flagged by a prior comparison; invoke in a fresh current-package process."""
function rerun_flagged(comparison_path; label, samples = DEFAULT_SAMPLES,
        seconds = DEFAULT_SECONDS, warmups = DEFAULT_WARMUPS, output_dir = nothing)
    report_file = isdir(comparison_path) ? joinpath(comparison_path, "comparison.json") : comparison_path
    report = JSON3.read(read(report_file, String), Dict{String,Any})
    ids = String.(report["flagged_case_ids"])
    isempty(ids) && throw(ArgumentError("comparison has no flagged cases to rerun"))
    return capture(label; suite = :full, case_ids = ids, role = "current-rerun",
        samples = samples, seconds = seconds, warmups = warmups,
        output_dir = isnothing(output_dir) ? joinpath(pwd(), "perf", "results") : output_dir)
end

"""Review at least three distinct fresh-process comparison reports for confirmation."""
function confirm(comparison_paths; threshold = 0.10, min_independent_runs = 3)
    length(comparison_paths) >= min_independent_runs ||
        throw(ArgumentError("need at least $min_independent_runs independent comparison reports"))
    reports = [JSON3.read(read(isdir(p) ? joinpath(p, "comparison.json") : p, String), Dict{String,Any}) for p in comparison_paths]
    all(get(r, "compatible", false) for r in reports) || throw(ArgumentError("all repeated reports must be environment-compatible"))
    current_runs = [r["current_run"] for r in reports]
    baseline_runs = [r["baseline_run"] for r in reports]
    length(unique(current_runs)) == length(current_runs) || throw(ArgumentError("candidate captures must be distinct"))
    length(unique(baseline_runs)) == length(baseline_runs) || throw(ArgumentError("baseline captures must be distinct"))
    by_report = [Dict(String(row["case_id"]) => row for row in r["rows"]) for r in reports]
    common = intersect((Set(keys(d)) for d in by_report)...)
    results = Dict{String,Any}[]
    for id in sort!(collect(common))
        rows = [d[id] for d in by_report]
        ratios = [get(r, "ratio", nothing) for r in rows]
        if any(isnothing, ratios)
            push!(results, Dict("case_id" => id, "status" => "not-confirmed", "reason" => "a run pair has no valid ratio"))
            continue
        end
        values = Float64.(ratios)
        noise = maximum(Float64(get(r, "noise_estimate_fraction", Inf)) for r in rows)
        med = median(values)
        all_over = all(x -> x > 1 + threshold, values)
        confirmed = all_over && med - 1 > max(threshold, 2noise)
        push!(results, Dict("case_id" => id,
            "status" => confirmed ? "confirmed" : "unconfirmed",
            "independent_pairs" => length(values), "ratios" => values,
            "median_ratio" => med, "noise_estimate_fraction" => noise,
            "threshold_fraction" => threshold,
            "reason" => confirmed ? "all independent pair medians exceed threshold and pooled noise" :
                "repeated results did not clear the threshold and noise rule"))
    end
    report = Dict{String,Any}("schema_version" => 1, "created_at" => string(now(UTC)),
        "current_runs" => current_runs, "baseline_runs" => baseline_runs,
        "minimum_independent_runs" => min_independent_runs,
        "threshold_fraction" => threshold, "rows" => results)
    for row in results
        println(row["status"], "  ", row["case_id"], "  ratios=", get(row, "ratios", Any[]),
            "  ", row["reason"])
    end
    return report
end
