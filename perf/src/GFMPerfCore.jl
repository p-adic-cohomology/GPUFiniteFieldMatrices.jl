module GFMPerfCore

using Dates
using JSON3
using SHA
using Statistics

export CaseSpec,
    quick_cases,
    full_cases,
    case_id,
    summarize_samples,
    compare_runs,
    read_run,
    write_json,
    csv_line,
    validate_output_path

const RUN_SCHEMA = 2
const COMPARISON_SCHEMA = 1

struct CaseSpec
    group::String
    operation::String
    dtype::String
    dims::Vector{Int}
    modulus::Int
    low_modulus::Union{Nothing,Int}
    high_modulus::Union{Nothing,Int}
    layout::String
    seed::Int
    target::String
end

function CaseSpec(group, operation, dtype, dims; modulus = 101, low_modulus = nothing,
        high_modulus = nothing, layout = "package-default", seed = 101, target = "shared")
    CaseSpec(String(group), String(operation), String(dtype), Int.(collect(dims)),
        Int(modulus), isnothing(low_modulus) ? nothing : Int(low_modulus),
        isnothing(high_modulus) ? nothing : Int(high_modulus), String(layout), Int(seed), String(target))
end

function case_id(c::CaseSpec)
    dims = join(c.dims, "x")
    split = isnothing(c.low_modulus) ? "none" : "$(c.low_modulus)/$(c.high_modulus)"
    return join(("group=$(c.group)", "op=$(c.operation)", "dtype=$(c.dtype)",
        "dims=$dims", "mod=$(c.modulus)", "split=$split", "layout=$(c.layout)",
        "seed=$(c.seed)"), "|")
end

function _add!(out, group, operations, dtype, dims; kwargs...)
    for operation in operations
        push!(out, CaseSpec(group, operation, dtype, dims; kwargs...))
    end
    return out
end

function _base_cases()
    cases = CaseSpec[]
    for dtype in ("Float32", "Float64"), dims in ((32, 32, 32), (31, 33, 7))
        _add!(cases, "gemm", ("mul!", "*"), dtype, dims; seed = 101 + sum(dims))
    end
    for dtype in ("Float32", "Float64"), dims in ((32, 32), (33, 65))
        _add!(cases, "gemv", ("mul!", "*"), dtype, dims; seed = 211 + sum(dims))
    end
    for dtype in ("Float32", "Float64"), dims in ((31, 33), (64, 64))
        _add!(cases, "elementwise", ("+", "add!", ".*", "mul_elementwise!", "copy", "construct"), dtype, dims;
            seed = 307 + sum(dims))
    end
    for dims in ((4, 4, 4), (5, 4, 3))
        for operation in ("mul!", "*")
            push!(cases, CaseSpec("karatsuba-gemm", operation, "Float64", dims;
                low_modulus = 101, high_modulus = 1, seed = 401 + sum(dims)))
        end
    end
    for operation in ("mul!", "*")
        push!(cases, CaseSpec("karatsuba-gemv", operation, "Float64", (4, 4);
            low_modulus = 101, high_modulus = 1, seed = 503))
        push!(cases, CaseSpec("karatsuba32-current", operation, "Float32", (4, 4, 4);
            low_modulus = 101, high_modulus = 1, seed = 509, target = "current-only"))
    end
    push!(cases, CaseSpec("inverse-pluq", "inv", "Float64", (8, 8); seed = 601))
    push!(cases, CaseSpec("inverse-pluq", "pluq", "Float64", (8, 8); seed = 607))
    return cases
end

"""Stable small suite; all operands are deterministic per case ID and seed."""
quick_cases() = _base_cases()

"""Opt-in suite with tile boundaries and larger GEMM/GEMV cases."""
function full_cases()
    cases = _base_cases()
    for dtype in ("Float32", "Float64"), n in (31, 32, 33, 63, 64, 65)
        _add!(cases, "gemm-padding", ("mul!", "*"), dtype, (n, n, n);
            seed = 701 + n)
        _add!(cases, "gemv-padding", ("mul!", "*"), dtype, (n, n);
            seed = 809 + n)
    end
    for dtype in ("Float32", "Float64"), n in (127, 128, 129, 256, 1024, 5000)
        _add!(cases, "gemm-scale", ("mul!", "*"), dtype, (n, n, n);
            seed = 907 + n)
    end
    return cases
end

function summarize_samples(samples)
    xs = sort(Float64.(collect(samples)))
    isempty(xs) && return Dict{String,Any}(
        "n" => 0, "median" => nothing, "mad" => nothing,
        "min" => nothing, "p05" => nothing, "p95" => nothing, "max" => nothing)
    med = median(xs)
    q(p) = xs[clamp(round(Int, 1 + p * (length(xs) - 1)), 1, length(xs))]
    return Dict{String,Any}(
        "n" => length(xs), "median" => med,
        "mad" => median(abs.(xs .- med)), "min" => first(xs),
        "p05" => q(0.05), "p95" => q(0.95), "max" => last(xs))
end

function _relative_noise(summary)
    med = summary["median"]
    isnothing(med) || med == 0 ? Inf : 1.4826 * summary["mad"] / med
end

_delta(current, baseline) = (isnothing(current) || isnothing(baseline)) ? nothing : current - baseline

function _by_case(run)
    Dict(String(c["case_id"]) => c for c in run["cases"])
end

function _compatibility(current, baseline)
    keys = ("julia_version", "cuda_version", "benchmarktools_version", "cuda_runtime",
        "cuda_driver", "gpu_identity", "cuda_preferences", "launch_settings")
    diffs = Dict{String,Any}()
    cm = current["metadata"]
    bm = baseline["metadata"]
    for key in keys
        cv = get(cm, key, nothing)
        bv = get(bm, key, nothing)
        cv == bv || (diffs[key] = Dict("current" => cv, "baseline" => bv))
    end
    cm["normalized_dependencies"] == bm["normalized_dependencies"] ||
        (diffs["normalized_dependencies"] = Dict("current" => cm["normalized_dependencies"],
            "baseline" => bm["normalized_dependencies"]))
    cm["harness_hash"] == bm["harness_hash"] ||
        (diffs["harness_hash"] = Dict("current" => cm["harness_hash"], "baseline" => bm["harness_hash"]))
    cp = current["measurement_policy"]
    bp = baseline["measurement_policy"]
    cp == bp || (diffs["measurement_policy"] = Dict("current" => cp, "baseline" => bp))
    get(cm, "capture_valid", true) || (diffs["current_capture_valid"] = false)
    get(bm, "capture_valid", true) || (diffs["baseline_capture_valid"] = false)
    return isempty(diffs), diffs
end

"""Compare two machine-readable captures. Target package source/path/version may differ."""
function compare_runs(current, baseline; threshold = 0.10, min_samples = 5)
    compatible, differences = _compatibility(current, baseline)
    curr_cases = _by_case(current)
    base_cases = _by_case(baseline)
    rows = Dict{String,Any}[]
    all_ids = sort!(collect(union(keys(curr_cases), keys(base_cases))))
    for id in all_ids
        c = get(curr_cases, id, nothing)
        b = get(base_cases, id, nothing)
        row = Dict{String,Any}("case_id" => id)
        if isnothing(c) || isnothing(b)
            row["status"] = "missing"
            row["detail"] = isnothing(c) ? "missing from current capture" : "missing from baseline capture"
        elseif !compatible
            row["status"] = "incompatible"
            row["detail"] = "capture environments or policies differ"
        elseif c["status"] != "ok" || b["status"] != "ok"
            row["status"] = "not-comparable"
            row["current_outcome"] = c["status"]
            row["baseline_outcome"] = b["status"]
            row["current_correctness"] = get(c, "correctness", nothing)
            row["baseline_correctness"] = get(b, "correctness", nothing)
        elseif !get(c, "correctness", false) || !get(b, "correctness", false)
            row["status"] = "not-comparable"
            row["detail"] = "a correctness oracle did not pass"
        else
            cs = summarize_samples(c["times_ns"])
            bs = summarize_samples(b["times_ns"])
            n = min(cs["n"], bs["n"])
            ratio = cs["median"] / bs["median"]
            noise = hypot(_relative_noise(cs), _relative_noise(bs))
            slowdown = ratio - 1
            status = if n < min_samples
                "inconclusive-insufficient-samples"
            elseif slowdown <= threshold
                "within-threshold"
            elseif slowdown <= max(threshold, 2noise)
                "inconclusive-within-noise"
            else
                "flagged-unconfirmed"
            end
            row["status"] = status
            row["current_median_ns"] = cs["median"]
            row["baseline_median_ns"] = bs["median"]
            row["ratio"] = ratio
            row["change_percent"] = 100 * slowdown
            row["noise_estimate_fraction"] = noise
            row["current_spread_ns"] = Dict("mad" => cs["mad"], "p05" => cs["p05"], "p95" => cs["p95"])
            row["baseline_spread_ns"] = Dict("mad" => bs["mad"], "p05" => bs["p05"], "p95" => bs["p95"])
            row["current_cpu_alloc_bytes"] = get(c, "cpu_alloc_bytes", nothing)
            row["baseline_cpu_alloc_bytes"] = get(b, "cpu_alloc_bytes", nothing)
            row["current_cpu_alloc_count"] = get(c, "cpu_alloc_count", nothing)
            row["baseline_cpu_alloc_count"] = get(b, "cpu_alloc_count", nothing)
            row["current_gpu_alloc_bytes"] = get(c, "gpu_alloc_bytes", nothing)
            row["baseline_gpu_alloc_bytes"] = get(b, "gpu_alloc_bytes", nothing)
            row["current_gpu_alloc_count"] = get(c, "gpu_alloc_count", nothing)
            row["baseline_gpu_alloc_count"] = get(b, "gpu_alloc_count", nothing)
            row["cpu_alloc_bytes_delta"] = _delta(row["current_cpu_alloc_bytes"], row["baseline_cpu_alloc_bytes"])
            row["cpu_alloc_count_delta"] = _delta(row["current_cpu_alloc_count"], row["baseline_cpu_alloc_count"])
            row["gpu_alloc_bytes_delta"] = _delta(row["current_gpu_alloc_bytes"], row["baseline_gpu_alloc_bytes"])
            row["gpu_alloc_count_delta"] = _delta(row["current_gpu_alloc_count"], row["baseline_gpu_alloc_count"])
            row["cpu_allocation_growth"] = !isnothing(row["cpu_alloc_bytes_delta"]) && row["cpu_alloc_bytes_delta"] > 0
            row["gpu_allocation_growth"] = !isnothing(row["gpu_alloc_bytes_delta"]) && row["gpu_alloc_bytes_delta"] > 0
        end
        push!(rows, row)
    end
    return Dict{String,Any}(
        "schema_version" => COMPARISON_SCHEMA,
        "created_at" => string(now(UTC)),
        "compatible" => compatible,
        "compatibility_differences" => differences,
        "threshold_fraction" => threshold,
        "minimum_samples" => min_samples,
        "current_run" => current["run_dir"],
        "baseline_run" => baseline["run_dir"],
        "flagged_case_ids" => [r["case_id"] for r in rows if r["status"] == "flagged-unconfirmed"],
        "rows" => rows)
end

function read_run(path)
    file = isdir(path) ? joinpath(path, "run.json") : path
    run = JSON3.read(read(file, String), Dict{String,Any})
    get(run, "schema_version", nothing) == RUN_SCHEMA || error("Unsupported run schema in $file")
    return run
end

function write_json(path, value)
    open(path, "w") do io
        JSON3.pretty(io, value)
        write(io, '\n')
    end
    return path
end

function validate_output_path(path; overwrite = false)
    ispath(path) && !overwrite && throw(ArgumentError("result path already exists: $path"))
    return path
end

function csv_line(values)
    escape(x) = begin
        s = isnothing(x) ? "" : string(x)
        occursin(r"[\",\r\n]", s) ? "\"" * replace(s, "\"" => "\"\"") * "\"" : s
    end
    join(escape.(values), ",") * "\n"
end

end # module
