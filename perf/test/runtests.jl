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
    return Dict{String,Any}("schema_version" => GFMPerf.GFMPerfCore.RUN_SCHEMA, "run_label" => label,
        "run_dir" => "/results/$label", "metadata" => metadata,
        "measurement_policy" => policy, "cases" => [case])
end

@testset "real CPU BenchmarkTools Trial allocation extraction" begin
    trialable = GFMPerf.BenchmarkTools.@benchmarkable Vector{Int}(undef, 16) evals=1
    trial = GFMPerf.BenchmarkTools.run(trialable; samples = 7, seconds = 0.25, evals = 1, gctrial = true)
    allocation = GFMPerf._cpu_allocation_summary(trial)
    @test trial.memory isa Int
    @test trial.allocs isa Int
    @test !isempty(trial.times)
    @test allocation.bytes == trial.memory
    @test allocation.count == trial.allocs
    @test allocation.bytes > 0
    @test allocation.count > 0
end

@testset "structured CUDA allocation result mapping" begin
    timed = (time = 0.01, gpu_bytes = 4096, gpu_memtime = 3.0,
        gpu_memstats = (alloc_count = 2,))
    allocation = GFMPerf._cuda_allocation_summary(timed)
    @test allocation.bytes == 4096
    @test allocation.count == 2
    @test allocation.raw["gpu_bytes"] == 4096
    @test allocation.raw["gpu_alloc_count"] == 2
    @test allocation.raw["gpu_memtime"] == 3.0
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

@testset "constructor and storage adapters for the loaded package" begin
    for (T, D) in ((Float32, 1), (Float32, 2), (Float64, 1), (Float64, 2))
        profile = GFMPerf._adapter_profile(T, D)
        cmtype = GFMPerf._cumod_type(T, D)
        @test profile.constructor_style in (:positional_modulus, :keyword_modulus)
        @test profile.storage_field == :data
        @test hasfield(cmtype, :data)
        sample = Array{T,D}(undef, ntuple(_ -> 2, D)...)
        if profile.constructor_style == :positional_modulus
            @test applicable(cmtype, sample, 101)
        else
            @test applicable(cmtype, sample; modulus = 101)
        end
    end
end

@testset "bounded exact large GEMM oracle" begin
    small_a = Float64[1 2 3; 4 5 6]
    small_b = Float64[2 1; 3 4; 5 6]
    exact_small = GFMPerf._oracle_bigint_gemm(small_a, small_b, 101)
    bounded_small = GFMPerf._oracle_float64_gemm(small_a, small_b, 101)
    @test bounded_small == exact_small
    @test GFMPerf._oracle_gemm(small_a, small_b, 101).values == exact_small

    large_a = fill(2.0, 501, 501)
    large_b = fill(3.0, 501, 1)
    large = GFMPerf._oracle_gemm(large_a, large_b, 1000)
    @test occursin("bounded exact Float64 BLAS", large.method)
    @test all(==(6), large.values)

    unsafe_a = fill(Float64(2^27), 501, 501)
    unsafe_b = fill(Float64(2^27), 501, 1)
    @test_throws ArgumentError GFMPerf._oracle_gemm(unsafe_a, unsafe_b, 101)
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
        comparison = joinpath(dir, "comparison.json")
        write_json(comparison, Dict("flagged_case_ids" => ["gemm-fixture"]))
        @test GFMPerf.flagged_case_ids(comparison) == ["gemm-fixture"]
        comparison_dir = joinpath(dir, "saved-comparison")
        mkpath(comparison_dir)
        write_json(joinpath(comparison_dir, "comparison.json"), Dict("flagged_case_ids" => ["gemm-fixture"]))
        @test GFMPerf.flagged_case_ids(comparison_dir) == ["gemm-fixture"]
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
