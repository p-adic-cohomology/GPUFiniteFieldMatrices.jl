const MODE = isempty(ARGS) ? "" : ARGS[1]

if MODE == "capture" || MODE == "review"
    @eval using Revise
    @eval using CUDA
    @eval using GPUFiniteFieldMatrices
    @eval using GFMPerf
    @eval using JSON3
end

function valid_label(value)
    occursin(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", value)
end

function require(condition, message)
    condition || error(message)
end

function command_output(arguments...)
    strip(read(Cmd(String[arguments...]), String))
end

function assert_snapshot(source_root, snapshot_dir, name)
    expected_head = strip(read(joinpath(snapshot_dir, "$name.head"), String))
    actual_head = command_output("git", "-C", source_root, "rev-parse", "HEAD")
    actual_head == expected_head || error("HEAD changed during the capture batch: $source_root")
    expected_diff = read(joinpath(snapshot_dir, "$name.diff"), String)
    actual_diff = read(Cmd(["git", "-C", source_root, "diff", "--no-ext-diff", "--binary",
        expected_head, "--", "src", "perf", "Project.toml"]), String)
    actual_diff == expected_diff || error("tracked benchmark or package source changed during the capture batch: $source_root")
end

function assert_batch_stable(repo_root, snapshot_dir)
    assert_snapshot(repo_root, snapshot_dir, "current")
    assert_snapshot(joinpath(repo_root, ".perf-regressions", "worktrees", "v0.3.1"), snapshot_dir, "baseline")
end

function assert_helper_hash(repo_root, helper_hash)
    helper_path = joinpath(repo_root, "perf", "run_hotpaths.jl")
    actual_hash = command_output("git", "-C", repo_root, "hash-object", "--no-filters", helper_path)
    actual_hash == helper_hash || error("Julia helper changed during the capture batch")
end

function verify_loaded_paths(repo_root, variant)
    expected_env = realpath(joinpath(repo_root, ".perf-regressions", "envs", variant, "Project.toml"))
    actual_env = Base.active_project()
    !isnothing(actual_env) && realpath(actual_env) == expected_env ||
        error("wrong active project: expected $expected_env, found $actual_env")

    expected_harness = realpath(joinpath(repo_root, "perf", "src", "GFMPerf.jl"))
    actual_harness = pathof(GFMPerf)
    !isnothing(actual_harness) && realpath(actual_harness) == expected_harness ||
        error("wrong GFMPerf source: expected $expected_harness, found $actual_harness")

    source_root = variant == "current" ? repo_root :
        joinpath(repo_root, ".perf-regressions", "worktrees", "v0.3.1")
    expected_package = realpath(joinpath(source_root, "src", "GPUFiniteFieldMatrices.jl"))
    actual_package = pathof(GPUFiniteFieldMatrices)
    !isnothing(actual_package) && realpath(actual_package) == expected_package ||
        error("wrong package source: expected $expected_package, found $actual_package")

    if variant == "v0.3.1"
        baseline_pin = "7192d479d2c41d31021a4938beec8223d16d6654"
        actual_head = command_output("git", "-C", source_root, "rev-parse", "HEAD")
        actual_head == baseline_pin || error("v0.3.1 baseline is not at pinned commit $baseline_pin")
    end
    return source_root
end

function assert_cuda_functional()
    functional = try
        CUDA.functional()
    catch exception
        error("CUDA.functional() failed: $(sprint(showerror, exception))")
    end
    functional || error("CUDA.functional() is false; no CPU fallback is measured")
end

function expected_case_ids()
    String.(GFMPerf.case_id.(GFMPerf.hotpath_cases()))
end

function validate_capture(path, label)
    run_file = joinpath(path, "run.json")
    isfile(run_file) || error("capture did not save run.json: $path")
    run = GFMPerf.GFMPerfCore.read_run(path)
    get(run, "run_label", nothing) == label || error("capture label mismatch in $run_file")
    recorded_path = get(run, "run_dir", "")
    !isempty(recorded_path) && realpath(recorded_path) == realpath(path) ||
        error("capture path mismatch in $run_file")

    metadata = get(run, "metadata", Dict{String,Any}())
    get(metadata, "capture_valid", false) === true || error("capture source snapshot is invalid: $path")
    get(metadata, "cuda_functional", false) === true || error("CUDA was unavailable in capture: $path")
    ids = expected_case_ids()
    length(ids) == 15 || error("hotpath catalogue changed from the expected 15 cases")
    String.(get(metadata, "case_selection", String[])) == ids || error("capture selected the wrong cases: $path")

    cases = get(run, "cases", Any[])
    case_ids = String[String(get(case, "case_id", "")) for case in cases]
    case_ids == ids || error("capture saved the wrong case rows: $path")
    bad_cases = filter(c -> get(c, "status", "") != "ok" || get(c, "correctness", nothing) !== true, cases)
    for case in bad_cases
        println(stderr, "invalid case: ", get(case, "case_id", "unknown"),
            " status=", get(case, "status", "missing"),
            " correctness=", get(case, "correctness", "missing"),
            " detail=", get(case, "detail", ""),
            " correctness_detail=", get(case, "correctness_detail", ""))
    end
    isempty(bad_cases) || error("capture failed an existing case oracle; artifact retained at $path")
    return run
end

function run_capture(arguments)
    length(arguments) == 9 || error("usage: run_hotpaths.jl capture REPO VARIANT LABEL SAMPLES SECONDS WARMUPS SNAPSHOT_DIR HELPER_HASH")
    repo_root, variant, label = arguments[2:4]
    samples, seconds, warmups = parse(Int, arguments[5]), parse(Float64, arguments[6]), parse(Int, arguments[7])
    snapshot_dir, helper_hash = arguments[8:9]
    variant in ("v0.3.1", "current") || error("variant must be v0.3.1 or current")
    valid_label(label) || error("invalid capture label")
    (samples, seconds, warmups) == (100, 5.0, 10) || error("capture settings must remain 100 samples, 5 seconds, and 10 warmups")
    VERSION.major == 1 && VERSION.minor == 12 || error("Julia 1.12 is required; found $VERSION")

    assert_helper_hash(repo_root, helper_hash)
    assert_batch_stable(repo_root, snapshot_dir)
    source_root = verify_loaded_paths(repo_root, variant)
    assert_cuda_functional()

    role = variant == "v0.3.1" ? "baseline-v0.3.1" : "current"
    output_dir = joinpath(repo_root, "perf", "results")
    path = GFMPerf.capture(label; suite = :hotpaths, role, samples, seconds, warmups, output_dir)
    assert_batch_stable(repo_root, snapshot_dir)
    assert_helper_hash(repo_root, helper_hash)
    validate_capture(path, label)
    println("Validated capture: $path (source: $source_root)")
end

function run_review(arguments)
    length(arguments) == 14 || error("usage: run_hotpaths.jl review REPO REVIEW_LABEL NINE_CAPTURE_LABELS SNAPSHOT_DIR HELPER_HASH")
    repo_root, label = arguments[2:3]
    capture_labels = arguments[4:12]
    snapshot_dir, helper_hash = arguments[13:14]
    valid_label(label) || error("invalid review label")
    VERSION.major == 1 && VERSION.minor == 12 || error("Julia 1.12 is required; found $VERSION")

    assert_helper_hash(repo_root, helper_hash)
    assert_batch_stable(repo_root, snapshot_dir)
    verify_loaded_paths(repo_root, "current")

    paths = [joinpath(repo_root, "perf", "results", run_label) for run_label in capture_labels]
    length(unique(paths)) == 9 || error("review requires nine distinct capture labels")
    for (path, run_label) in zip(paths, capture_labels)
        isdir(path) || error("capture is missing: $path")
        validate_capture(path, run_label)
    end

    cycles = [(paths[1], paths[2], paths[3]), (paths[4], paths[5], paths[6]),
        (paths[7], paths[8], paths[9])]
    output_dir = joinpath(repo_root, "perf", "results", "reviews")
    review_dir = GFMPerf.review_bracketed_cycles(cycles; label, output_dir)
    assert_batch_stable(repo_root, snapshot_dir)
    assert_helper_hash(repo_root, helper_hash)

    report_path = joinpath(review_dir, "review.json")
    isfile(report_path) || error("review did not save review.json: $review_dir")
    report = JSON3.read(read(report_path, String), Dict{String,Any})
    get(report, "label", nothing) == label || error("review label mismatch: $report_path")
    get(report, "compatible_across_all_runs", false) === true || error("review found incompatible captures: $report_path")
    source_identity = get(report, "source_identity", Dict{String,Any}())
    get(source_identity, "baseline_stable", false) === true || error("review baseline source identity is unstable: $report_path")
    get(source_identity, "candidate_stable", false) === true || error("review current source identity is unstable: $report_path")

    expected_ids = sort(expected_case_ids())
    rows = get(report, "rows", Any[])
    row_ids = sort(String[String(get(row, "case_id", "")) for row in rows])
    row_ids == expected_ids || error("review is missing or has unexpected cases: $report_path")
    for row in rows
        outcome = String(get(row, "status", ""))
        valid_outcome = outcome in ("confirmed-slowdown", "positive-change-observed", "no-positive-median-change-observed") ||
            startswith(outcome, "inconclusive-")
        valid_outcome || error("review contains an invalid or incomparable case $(get(row, "case_id", "unknown")): $outcome")
    end
    get(report, "cycle_count", 0) == 3 || error("review did not contain three cycles: $report_path")
    println("Validated review: $report_path")
end

if MODE == "validate-prefix"
    length(ARGS) == 2 || error("usage: run_hotpaths.jl validate-prefix NAME")
    valid_label(ARGS[2]) || error("prefix must start with a letter or digit and contain only letters, digits, dot, underscore, or dash")
    println("Valid prefix: $(ARGS[2])")
elseif MODE == "capture"
    run_capture(ARGS)
elseif MODE == "review"
    run_review(ARGS)
else
    error("mode must be validate-prefix, capture, or review")
end
