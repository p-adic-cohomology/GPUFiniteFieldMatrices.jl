function _load_capture(value)
    value isa AbstractString && return GFMPerfCore.read_run(value)
    return value
end

function _load_comparison(value)
    file = isdir(value) ? joinpath(value, "comparison.json") : value
    return JSON3.read(read(file, String), Dict{String,Any})
end

"""Read the stable case IDs flagged by a saved comparison directory or JSON file."""
flagged_case_ids(comparison_path) = String.(_load_comparison(comparison_path)["flagged_case_ids"])

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
    ids = flagged_case_ids(comparison_path)
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

_review_relative_noise(summary) = isnothing(summary["median"]) || summary["median"] == 0 ?
    Inf : 1.4826 * summary["mad"] / summary["median"]

function _review_run_case(run, case_id)
    cases = Dict(String(c["case_id"]) => c for c in run["cases"])
    case = get(cases, case_id, nothing)
    isnothing(case) && return Dict{String,Any}(
        "status" => "missing", "detail" => "case not present in capture",
        "correctness" => nothing, "correctness_detail" => "not run", "sample_count" => 0,
        "median_ns" => nothing, "median_us" => nothing,
        "spread_ns" => Dict("mad" => nothing, "p05" => nothing, "p95" => nothing),
        "spread_us" => Dict("mad" => nothing, "p05" => nothing, "p95" => nothing),
        "cpu_alloc_bytes" => nothing, "cpu_alloc_count" => nothing,
        "gpu_alloc_bytes" => nothing, "gpu_alloc_count" => nothing)
    stats = GFMPerfCore.summarize_samples(get(case, "times_ns", Int[]))
    return Dict{String,Any}(
        "status" => get(case, "status", "unknown"),
        "detail" => get(case, "detail", ""),
        "correctness" => get(case, "correctness", nothing),
        "correctness_detail" => get(case, "correctness_detail", ""),
        "sample_count" => stats["n"],
        "median_ns" => stats["median"],
        "median_us" => isnothing(stats["median"]) ? nothing : stats["median"] / 1_000,
        "spread_ns" => Dict("mad" => stats["mad"], "p05" => stats["p05"], "p95" => stats["p95"]),
        "spread_us" => Dict(k => (isnothing(v) ? nothing : v / 1_000) for (k, v) in
            Dict("mad" => stats["mad"], "p05" => stats["p05"], "p95" => stats["p95"])),
        "cpu_alloc_bytes" => get(case, "cpu_alloc_bytes", nothing),
        "cpu_alloc_count" => get(case, "cpu_alloc_count", nothing),
        "gpu_alloc_bytes" => get(case, "gpu_alloc_bytes", nothing),
        "gpu_alloc_count" => get(case, "gpu_alloc_count", nothing))
end

function _review_pair_row(row)
    isnothing(row) && return Dict{String,Any}("status" => "missing", "detail" => "case missing from pair comparison")
    result = Dict{String,Any}(String(k) => v for (k, v) in row)
    if haskey(row, "ratio")
        result["current_median_us"] = row["current_median_ns"] / 1_000
        result["baseline_median_us"] = row["baseline_median_ns"] / 1_000
        result["current_spread_us"] = Dict(k => v / 1_000 for (k, v) in row["current_spread_ns"])
        result["baseline_spread_us"] = Dict(k => v / 1_000 for (k, v) in row["baseline_spread_ns"])
    end
    return result
end

function _pair_row_map(report)
    Dict(String(row["case_id"]) => row for row in report["rows"])
end

function _baseline_bracket(before, after, case_id)
    before_case = _review_run_case(before, case_id)
    after_case = _review_run_case(after, case_id)
    if before_case["status"] != "ok" || after_case["status"] != "ok" ||
            !get(before_case, "correctness", false) || !get(after_case, "correctness", false)
        return Dict{String,Any}("status" => "not-comparable", "before" => before_case, "after" => after_case,
            "detail" => "a bracket baseline did not have an ok, correctness-passing outcome")
    end
    before_median = before_case["median_ns"]
    after_median = after_case["median_ns"]
    if isnothing(before_median) || isnothing(after_median) || before_median <= 0 || after_median <= 0
        return Dict{String,Any}("status" => "not-comparable", "before" => before_case, "after" => after_case,
            "detail" => "a bracket baseline has no positive median")
    end
    before_raw = get(_case_run_map(before), case_id, Dict{String,Any}())
    after_raw = get(_case_run_map(after), case_id, Dict{String,Any}())
    before_summary = GFMPerfCore.summarize_samples(get(before_raw, "times_ns", Int[]))
    after_summary = GFMPerfCore.summarize_samples(get(after_raw, "times_ns", Int[]))
    noise = hypot(_review_relative_noise(before_summary), _review_relative_noise(after_summary))
    ratio = after_median / before_median
    # Opposite candidate directions, or a baseline shift outside the measured
    # spread, make this bracket inconclusive even if the average looks benign.
    drift = abs(log(ratio)) > 2noise
    return Dict{String,Any}(
        "status" => drift ? "inconclusive-baseline-drift" : "baseline-stable-within-noise",
        "before" => before_case, "after" => after_case,
        "after_over_before_ratio" => ratio,
        "change_percent" => 100 * (ratio - 1),
        "noise_estimate_fraction" => noise,
        "detail" => drift ? "bracketing baselines differ beyond their measured noise" :
            "bracketing baseline shift is within measured noise")
end

function _case_run_map(run)
    Dict(String(c["case_id"]) => c for c in run["cases"])
end

function _cycle_case_record(cycle_index, runs, pair_rows, case_id)
    before, current, after = runs
    before_case = _review_run_case(before, case_id)
    current_case = _review_run_case(current, case_id)
    after_case = _review_run_case(after, case_id)
    before_pair = _review_pair_row(get(pair_rows[1], case_id, nothing))
    after_pair = _review_pair_row(get(pair_rows[2], case_id, nothing))
    bracket = _baseline_bracket(before, after, case_id)
    before_ratio = get(before_pair, "ratio", nothing)
    after_ratio = get(after_pair, "ratio", nothing)
    candidate_disagrees = !isnothing(before_ratio) && !isnothing(after_ratio) &&
        ((before_ratio > 1 && after_ratio < 1) || (before_ratio < 1 && after_ratio > 1))
    if candidate_disagrees
        bracket["status"] = "inconclusive-baseline-drift"
        bracket["detail"] = "candidate-to-baseline comparisons disagree in direction"
    end
    cycle_ratio = isnothing(before_ratio) || isnothing(after_ratio) ? nothing : sqrt(before_ratio * after_ratio)
    return Dict{String,Any}(
        "cycle" => cycle_index,
        "baseline_before_run" => get(before, "run_dir", "unknown"),
        "current_run" => get(current, "run_dir", "unknown"),
        "baseline_after_run" => get(after, "run_dir", "unknown"),
        "runs" => Dict("baseline_before" => before_case, "current" => current_case, "baseline_after" => after_case),
        "current_vs_baseline_before" => before_pair,
        "current_vs_baseline_after" => after_pair,
        "baseline_bracket" => bracket,
        "cycle_ratio" => cycle_ratio,
        "cycle_change_percent" => isnothing(cycle_ratio) ? nothing : 100 * (cycle_ratio - 1))
end

function _review_invalid_run_details(cycles)
    details = String[]
    roles = (("baseline_before", "baseline_before_run"), ("current", "current_run"),
        ("baseline_after", "baseline_after_run"))
    for cycle in cycles, (role, run_key) in roles
        outcome = cycle["runs"][role]
        correctness = get(outcome, "correctness", nothing)
        if get(outcome, "status", "unknown") != "ok" || correctness !== true
            push!(details, "cycle $(cycle["cycle"])/$role ($(cycle[run_key])): " *
                "status=$(get(outcome, "status", "unknown")), correctness=$(correctness), " *
                "detail=$(get(outcome, "detail", "")), " *
                "correctness_detail=$(get(outcome, "correctness_detail", ""))")
        end
    end
    return details
end

function _bracketed_case_status(cycles; compatible, source_identity_ok)
    pair_rows = [vcat(c["current_vs_baseline_before"], c["current_vs_baseline_after"]) for c in cycles]
    rows = reduce(vcat, pair_rows; init = Dict{String,Any}[])
    statuses = String[get(r, "status", "missing") for r in rows]
    if !compatible
        return "incompatible", "one or more of the nine captures differ in environment, harness, or policy", nothing
    elseif !source_identity_ok
        return "incompatible-source-identity", "baseline or candidate source identity changed between cycles", nothing
    elseif any(==("missing"), statuses)
        return "missing-case", "case is missing from at least one of the nine captures", nothing
    elseif any(==("incompatible"), statuses)
        return "incompatible", "one or more captures differ in environment, harness, or measurement policy", nothing
    elseif any(s -> s in ("not-comparable", "missing"), statuses)
        return "not-comparable", "one or more run outcomes failed correctness or timing availability", nothing
    elseif any(==("inconclusive-insufficient-samples"), statuses)
        return "inconclusive-insufficient-samples", "at least one pair has too few samples", nothing
    elseif any(c -> c["baseline_bracket"]["status"] == "inconclusive-baseline-drift", cycles)
        return "inconclusive-baseline-drift", "a current comparison disagrees across its neighboring baselines", nothing
    end
    ratios = Float64[c["cycle_ratio"] for c in cycles]
    any(r -> r > 1, ratios) && any(r -> r < 1, ratios) &&
        return "inconclusive-repeat-disagreement", "independent fresh-process cycles disagree in direction", nothing
    summary = GFMPerfCore.summarize_samples(ratios)
    noise = _review_relative_noise(summary)
    all_pair_ratios = Float64[r["ratio"] for r in rows]
    cycle_noise = [maximum(Float64(get(pair, "noise_estimate_fraction", Inf)) for pair in
        (cycle["current_vs_baseline_before"], cycle["current_vs_baseline_after"])) for cycle in cycles]
    consistency_bounds = 2 .* cycle_noise
    deviations = abs.(log.(ratios ./ summary["median"]))
    between_cycle_consistent = all(deviations[i] <= consistency_bounds[i] for i in 1:3)
    confirmed = all(==("flagged-unconfirmed"), statuses) && between_cycle_consistent
    !between_cycle_consistent && return "inconclusive-between-cycle-variation",
        "fresh-process cycle ratios vary beyond the within-pair noise estimates",
        Dict("cycle_ratio_median" => summary["median"], "cycle_ratio_mad" => summary["mad"],
            "cycle_noise_estimate_fraction" => noise, "between_cycle_deviations" => deviations,
            "between_cycle_consistency_bounds" => consistency_bounds,
            "between_cycle_consistent" => false)
    status = confirmed ? "confirmed-slowdown" :
        any(r -> r > 1, all_pair_ratios) ? "positive-change-observed" : "no-positive-median-change-observed"
    detail = confirmed ? "all six bracket comparisons exceed the screening threshold and the three independent cycle ratios clear measured noise" :
        status == "positive-change-observed" ? "positive changes are shown per pair; this status does not confirm or accept a regression" :
        "no positive median change was observed; this is not a no-regression acceptance"
    return status, detail, Dict("cycle_ratio_median" => summary["median"],
        "cycle_ratio_mad" => summary["mad"], "cycle_noise_estimate_fraction" => noise,
        "between_cycle_deviations" => deviations,
        "between_cycle_consistency_bounds" => consistency_bounds,
        "between_cycle_consistent" => between_cycle_consistent)
end

function _source_identity(run)
    metadata = run["metadata"]
    fields = ("target_commit", "target_source_hash_before", "target_source_hash_after", "target_dirty")
    all(haskey(metadata, field) for field in fields) || return nothing
    identity = Dict{String,Any}(field => metadata[field] for field in fields)
    known_string(value) = value isa AbstractString && !isempty(strip(value)) &&
        !(lowercase(strip(value)) in ("unknown", "unavailable", "missing", "none", "null"))
    all(known_string(identity[field]) for field in fields[1:3]) || return nothing
    identity["target_dirty"] isa Bool || return nothing
    identity["target_source_hash_before"] == identity["target_source_hash_after"] || return nothing
    return identity
end

function _stable_source_group(runs)
    identities = [_source_identity(run) for run in runs]
    all(x -> !isnothing(x), identities) || return false, identities
    return all(==(first(identities)), identities), identities
end

"""Review saved capture paths from three independent baseline → current → baseline cycles.

Reads nine unique run artifacts without modifying them and writes a separate
JSON review directory. Each current capture is compared to both
neighboring baselines; only the three bracketed cycles count as independent
repeats. The operator must use one fresh Julia process for each capture; paths
alone cannot prove process isolation.
"""
function review_bracketed_cycles(cycles; label, threshold = 0.0, min_samples = 5,
        output_dir = joinpath(pwd(), "perf", "results", "reviews"))
    length(cycles) == 3 || throw(ArgumentError("provide exactly three baseline-current-baseline cycles"))
    threshold >= 0 || throw(ArgumentError("threshold must be nonnegative"))
    min_samples > 0 || throw(ArgumentError("min_samples must be positive"))
    occursin(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", String(label)) ||
        throw(ArgumentError("label must use letters, digits, dot, underscore, or dash"))
    paths = String[]
    for (i, cycle) in enumerate(cycles)
        cycle isa Tuple && length(cycle) == 3 ||
            throw(ArgumentError("cycle $i must be a three-tuple of baseline-before, current, baseline-after paths"))
        for path in cycle
            path isa AbstractString || throw(ArgumentError("capture paths must be strings"))
            ispath(path) || throw(ArgumentError("capture path does not exist: $path"))
            push!(paths, realpath(path))
        end
    end
    length(unique(paths)) == 9 || throw(ArgumentError("all nine capture inputs must be distinct"))
    runs = [GFMPerfCore.read_run(path) for path in paths]
    run_ids = String[get(run, "run_dir", path) for (run, path) in zip(runs, paths)]
    length(unique(run_ids)) == 9 || throw(ArgumentError("capture artifacts must identify nine distinct runs"))
    run_by_path = Dict(paths[i] => runs[i] for i in eachindex(paths))
    cycles_runs = [[run_by_path[realpath(path)] for path in cycle] for cycle in cycles]
    compatibility_differences = Dict{String,Any}[]
    reference = first(runs)
    for i in 2:length(runs)
        compatible, differences = GFMPerfCore._compatibility(runs[i], reference)
        compatible || push!(compatibility_differences,
            Dict("run" => get(runs[i], "run_dir", paths[i]), "differences" => differences))
    end
    baseline_runs = [runs_at[i] for runs_at in cycles_runs for i in (1, 3)]
    candidate_runs = [runs_at[2] for runs_at in cycles_runs]
    baseline_source_stable, baseline_identities = _stable_source_group(baseline_runs)
    candidate_source_stable, candidate_identities = _stable_source_group(candidate_runs)
    source_identity_ok = baseline_source_stable && candidate_source_stable
    all_compatible = isempty(compatibility_differences)
    all_case_ids = sort!(collect(union((Set(keys(_case_run_map(run))) for run in runs)...)))
    isempty(all_case_ids) && throw(ArgumentError("captures contain no cases to review"))
    # Cache the two candidate/reference reports for each cycle; each candidate
    # appears twice by design but the enclosing three cycles remain the repeats.
    pair_maps = Vector{Tuple{Dict{String,Any},Dict{String,Any}}}(undef, 3)
    for i in 1:3
        before, current, after = cycles_runs[i]
        pair_before = GFMPerfCore.compare_runs(current, before; threshold, min_samples)
        pair_after = GFMPerfCore.compare_runs(current, after; threshold, min_samples)
        pair_maps[i] = (_pair_row_map(pair_before), _pair_row_map(pair_after))
    end
    records = Dict{String,Any}[]
    for id in all_case_ids
        case_cycles = Dict{String,Any}[]
        for i in 1:3
            push!(case_cycles, _cycle_case_record(i, cycles_runs[i], pair_maps[i], id))
        end
        status, detail, summary = _bracketed_case_status(case_cycles;
            compatible = all_compatible, source_identity_ok)
        invalid_runs = _review_invalid_run_details(case_cycles)
        !isempty(invalid_runs) && (detail *= "; " * join(invalid_runs, " | "))
        summary = isnothing(summary) ? Dict{String,Any}() : summary
        push!(records, Dict{String,Any}("case_id" => id, "status" => status, "detail" => detail,
            "cycle_ratio_median" => get(summary, "cycle_ratio_median", nothing),
            "cycle_ratio_mad" => get(summary, "cycle_ratio_mad", nothing),
            "cycle_noise_estimate_fraction" => get(summary, "cycle_noise_estimate_fraction", nothing),
            "cycles" => case_cycles))
    end
    report = Dict{String,Any}("schema_version" => 1, "created_at" => string(now(UTC)),
        "label" => String(label), "threshold_fraction" => threshold, "minimum_samples" => min_samples,
        "cycle_count" => 3, "independent_repeat_count" => 3,
        "cycle_inputs" => [[String(p) for p in cycle] for cycle in cycles],
        "compatible_across_all_runs" => all_compatible,
        "compatibility_differences" => compatibility_differences,
        "source_identity" => Dict("baseline_stable" => baseline_source_stable,
            "candidate_stable" => candidate_source_stable,
            "baseline_runs" => baseline_identities, "candidate_runs" => candidate_identities),
        "fresh_process_assumption" => "The operator must launch each capture in a separate fresh Julia process; unique artifact paths do not prove process isolation.",
        "rows" => records)
    root = abspath(output_dir)
    mkpath(root)
    target = joinpath(root, String(label))
    ispath(target) && throw(ArgumentError("review output already exists: $target"))
    tmp = joinpath(root, ".$(label).tmp-$(uuid4())")
    mkdir(tmp)
    try
        report["review_dir"] = target
        write_json(joinpath(tmp, "review.json"), report)
        mv(tmp, target)
    catch
        ispath(tmp) && rm(tmp; force = true, recursive = true)
        rethrow()
    end
    for row in records
        println(row["status"], "  ", row["case_id"], "  ", row["detail"])
    end
    if !all_compatible
        for difference in compatibility_differences
            println("incompatible run  ", difference["run"], "  fields=",
                join(sort!(collect(keys(difference["differences"]))), ","))
        end
    end
    if !baseline_source_stable || !candidate_source_stable
        for (name, stable, source_runs) in (("baseline", baseline_source_stable, baseline_runs),
                ("current", candidate_source_stable, candidate_runs))
            stable && continue
            println("source identity unstable  ", name)
            for (run, identity) in zip(source_runs, name == "baseline" ? baseline_identities : candidate_identities)
                println("  run=", get(run, "run_dir", "unknown"), "  identity=", identity)
            end
        end
    end
    return abspath(target)
end
