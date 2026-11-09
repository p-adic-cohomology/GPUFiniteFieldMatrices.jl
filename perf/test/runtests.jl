using Test
using GFMPerf
using GFMPerf.GFMPerfCore

function synthetic_run(label; package_version = "0.3.2", package_path = "/variant/package",
        commit = "a", cuda = "5.11.3", gpu = "NVIDIA test fixture", deps = Dict("CUDA" => cuda),
        samples = [100, 101, 99, 100, 100], correctness = true, status = "ok", valid = true)
    metadata = Dict{String,Any}(
        "julia_version" => "1.12.7", "cuda_version" => cuda,
        "benchmarktools_version" => "1.5.0", "cuda_runtime" => "12.6",
        "cuda_driver" => "12060", "gpu_identity" => gpu,
        "cuda_preferences" => Dict("memory_pool" => "default"),
        "launch_settings" => Dict("threads" => "1"),
        "normalized_dependencies" => deps, "harness_hash" => "same-harness",
        "target_package_version" => package_version, "target_package_path" => package_path,
        "target_commit" => commit, "capture_valid" => valid)
    policy = Dict("samples_requested" => 5, "seconds_per_case" => 1.0,
        "warmups_per_case" => 3, "evals_per_sample" => 1)
    case = Dict{String,Any}("case_id" => "gemm-fixture", "status" => status,
        "correctness" => correctness, "times_ns" => collect(samples),
        "cpu_alloc_bytes" => 40, "cpu_alloc_count" => 2,
        "gpu_alloc_bytes" => 0, "gpu_alloc_count" => 0)
    return Dict{String,Any}("schema_version" => 1, "run_label" => label,
        "run_dir" => "/results/$label", "metadata" => metadata,
        "measurement_policy" => policy, "cases" => [case])
end

@testset "stable case catalogue" begin
    quick = quick_cases()
    full = full_cases()
    ids = case_id.(quick)
    @test !isempty(quick)
    @test length(ids) == length(unique(ids))
    @test all(id -> occursin("dtype=", id) && occursin("dims=", id) && occursin("seed=", id), ids)
    @test Set(ids) ⊆ Set(case_id.(full))
    @test any(c -> c.group == "gemm-padding" && c.dims == [31, 31, 31], full)
    @test any(c -> c.group == "gemm-padding" && c.dims == [65, 65, 65], full)
    @test any(c -> c.group == "gemm-scale" && c.dims == [5000, 5000, 5000], full)
    @test any(c -> c.target == "current-only" && c.dtype == "Float32", quick)
end

@testset "sample summaries and comparison policy" begin
    summary = summarize_samples([10, 10, 10, 10, 10])
    @test summary["n"] == 5
    @test summary["median"] == 10
    @test summary["mad"] == 0
    @test summarize_samples(Int[])["median"] === nothing

    baseline = synthetic_run("baseline"; package_version = "0.3.2", package_path = "/old", commit = "pin-032")
    current = synthetic_run("current"; package_version = "0.3.2", package_path = "/dev", commit = "dirty-current",
        samples = [120, 121, 119, 120, 120])
    comparison = compare_runs(current, baseline; threshold = 0.10, min_samples = 5)
    @test comparison["compatible"]
    @test only(comparison["rows"])["status"] == "flagged-unconfirmed"
    @test only(comparison["flagged_case_ids"]) == "gemm-fixture"
    @test only(comparison["rows"])["change_percent"] ≈ 20
    @test only(comparison["rows"])["gpu_alloc_bytes_delta"] == 0

    below = compare_runs(synthetic_run("fast"; samples = [108, 109, 107, 108, 108]), baseline)
    @test only(below["rows"])["status"] == "within-threshold"
    noisy = compare_runs(synthetic_run("noisy"; samples = [100, 110, 120, 120, 130]), baseline)
    @test only(noisy["rows"])["status"] == "inconclusive-within-noise"
    short = compare_runs(synthetic_run("short"; samples = [120, 121, 119]), baseline)
    @test only(short["rows"])["status"] == "inconclusive-insufficient-samples"
end

@testset "compatibility and correctness outcomes" begin
    base = synthetic_run("baseline")
    intended_source_change = synthetic_run("new-target"; package_version = "0.4.0",
        package_path = "/new/package", commit = "new-commit")
    @test compare_runs(intended_source_change, base)["compatible"]

    @test !compare_runs(synthetic_run("new-cuda"; cuda = "6.0.0"), base)["compatible"]
    @test !compare_runs(synthetic_run("new-gpu"; gpu = "other GPU"), base)["compatible"]
    @test !compare_runs(synthetic_run("new-dep"; deps = Dict("CUDA" => "5.11.3", "Foo" => "2.0")), base)["compatible"]
    @test !compare_runs(synthetic_run("mixed-source"; valid = false), base)["compatible"]

    unsupported = compare_runs(synthetic_run("unsupported"; status = "unsupported"), base)
    @test only(unsupported["rows"])["status"] == "not-comparable"
    incorrect = compare_runs(synthetic_run("wrong"; correctness = false, status = "incorrect"), base)
    @test only(incorrect["rows"])["status"] == "not-comparable"
end

@testset "artifact round trip, CSV escaping, and collision guard" begin
    mktempdir() do dir
        run = synthetic_run("roundtrip")
        file = joinpath(dir, "run.json")
        write_json(file, run)
        restored = read_run(file)
        @test restored["run_label"] == "roundtrip"
        @test restored["cases"][1]["times_ns"] == [100, 101, 99, 100, 100]
        @test csv_line(("value,one", "a\"b", "line\nbreak")) == "\"value,one\",\"a\"\"b\",\"line\nbreak\"\n"
        out = joinpath(dir, "already-here")
        mkdir(out)
        @test_throws ArgumentError validate_output_path(out)
        @test validate_output_path(out; overwrite = true) == out
    end
end

@testset "independent confirmation policy" begin
    mktempdir() do dir
        paths = String[]
        for i in 1:3
            baseline = synthetic_run("base-$i"; samples = [100, 100, 100, 100, 100])
            current = synthetic_run("candidate-$i"; samples = [120, 120, 120, 120, 120])
            report = compare_runs(current, baseline; threshold = 0.10, min_samples = 5)
            report["current_run"] = "/results/candidate-$i"
            report["baseline_run"] = "/results/base-$i"
            file = joinpath(dir, "comparison-$i.json")
            write_json(file, report)
            push!(paths, file)
        end
        confirmation = GFMPerf.confirm(paths)
        @test only(confirmation["rows"])["status"] == "confirmed"
        @test only(confirmation["rows"])["independent_pairs"] == 3
        @test_throws ArgumentError GFMPerf.confirm([paths[1], paths[1], paths[2]])
    end
end
