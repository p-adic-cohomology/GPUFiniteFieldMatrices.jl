using Test
using GFMPerf
using GFMPerf.GFMPerfCore

struct TrackedPluqResult
    factor_reads::Base.RefValue{Int}
    L::Matrix{Int}
    U::Matrix{Int}
    p::Vector{Int}
    q::Vector{Int}
    rank::Int
end

function Base.getproperty(result::TrackedPluqResult, name::Symbol)
    if name === :L || name === :U
        getfield(result, :factor_reads)[] += 1
    end
    return getfield(result, name)
end

function synthetic_run(label; package_version = "0.3.2", package_path = "/variant/package",
        commit = "a", cuda = "5.11.3", gpu = "NVIDIA test fixture", deps = Dict("CUDA" => cuda),
        samples = [100, 101, 99, 100, 100], correctness = true, status = "ok", valid = true,
        source_hash = "source-$commit", source_hash_after = source_hash, harness_hash = "same-harness",
        case_id = "gemm-fixture", run_dir = "/results/$label")
    metadata = Dict{String,Any}(
        "julia_version" => "1.12.7", "cuda_version" => cuda,
        "benchmarktools_version" => "1.5.0", "cuda_runtime" => "12.6",
        "cuda_driver" => "12060", "gpu_identity" => gpu,
        "cuda_preferences" => Dict("memory_pool" => "default"),
        "launch_settings" => Dict("threads" => "1"),
        "normalized_dependencies" => deps, "harness_hash" => harness_hash,
        "target_package_version" => package_version, "target_package_path" => package_path,
        "target_commit" => commit, "target_dirty" => source_hash != "source-$commit",
        "target_source_hash_before" => source_hash, "target_source_hash_after" => source_hash_after,
        "capture_valid" => valid)
    policy = Dict("samples_requested" => 5, "seconds_per_case" => 1.0,
        "warmups_per_case" => 3, "evals_per_sample" => 1)
    case = Dict{String,Any}("case_id" => case_id, "status" => status,
        "correctness" => correctness, "times_ns" => collect(samples),
        "correctness_detail" => correctness ? "fixture oracle passed" : "fixture oracle failed",
        "detail" => status == "ok" ? "fixture ready" : "fixture outcome",
        "cpu_alloc_bytes" => 40, "cpu_alloc_count" => 2,
        "gpu_alloc_bytes" => 0, "gpu_alloc_count" => 0)
    return Dict{String,Any}("schema_version" => GFMPerf.GFMPerfCore.RUN_SCHEMA, "run_label" => label,
        "run_dir" => run_dir, "metadata" => metadata,
        "measurement_policy" => policy, "cases" => [case])
end

function save_fixture_capture(root, label; samples, kwargs...)
    path = joinpath(root, label)
    mkpath(path)
    run = synthetic_run(label; samples, run_dir = path, kwargs...)
    write_json(joinpath(path, "run.json"), run)
    return path
end

function fixture_cycles(root; current_samples = fill(105, 5), baseline_samples = fill(100, 5),
        current_samples_by_cycle = fill(current_samples, 3),
        baseline_before_samples_by_cycle = fill(baseline_samples, 3),
        baseline_after_samples_by_cycle = fill(baseline_samples, 3),
        candidate_hashes = fill("candidate-source", 3), baseline_cuda = fill("5.11.3", 3),
        candidate_cuda = fill("5.11.3", 3))
    cycles = Tuple{String,String,String}[]
    for i in 1:3
        before = save_fixture_capture(root, "before-$i"; samples = baseline_before_samples_by_cycle[i],
            package_path = "/baseline", package_version = "0.3.1", commit = "baseline-commit",
            source_hash = "baseline-source", cuda = baseline_cuda[i])
        current = save_fixture_capture(root, "current-$i"; samples = current_samples_by_cycle[i],
            package_path = "/current", package_version = "0.3.2", commit = "current-commit",
            source_hash = candidate_hashes[i], cuda = candidate_cuda[i])
        after = save_fixture_capture(root, "after-$i"; samples = baseline_after_samples_by_cycle[i],
            package_path = "/baseline", package_version = "0.3.1", commit = "baseline-commit",
            source_hash = "baseline-source", cuda = baseline_cuda[i])
        push!(cycles, (before, current, after))
    end
    return cycles
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
    hotpaths = hotpath_cases()
    addition_diagnostics = addition_diagnostic_cases()
    ids = case_id.(quick)
    @test !isempty(quick)
    @test length(ids) == length(unique(ids))
    @test all(id -> occursin("dtype=", id) && occursin("dims=", id) && occursin("seed=", id), ids)
    @test Set(ids) ⊆ Set(case_id.(full))
    @test any(c -> c.group == "gemm-padding" && c.dims == [31, 31, 31], full)
    @test any(c -> c.group == "gemm-padding" && c.dims == [65, 65, 65], full)
    @test any(c -> c.group == "gemm-scale" && c.dims == [5000, 5000, 5000], full)
    @test any(c -> c.target == "current-only" && c.dtype == "Float32", quick)
    pluq_case = only(filter(c -> c.group == "inverse-pluq" && c.operation == PLUQ_TOTAL_CONTRACT, quick))
    @test occursin("op=pluq-total-materialized-factors-v1", case_id(pluq_case))
    @test GFMPerf._measurement_contract(pluq_case) == PLUQ_TOTAL_CONTRACT
    @test length(hotpaths) == 15
    @test count(c -> c.group == "hotpath-addition" && c.operation == "add!", hotpaths) == 8
    @test count(c -> c.group == "hotpath-square-gemv" && c.operation == "mul!", hotpaths) == 6
    @test count(c -> c.operation == "pluq-total-materialized-factors-v1" && c.dtype == "Float64" &&
        c.dims == [8, 8] && c.seed == 607 && c.modulus == 101, hotpaths) == 1
    @test length(case_id.(hotpaths)) == length(unique(case_id.(hotpaths)))
    for c in filter(c -> c.group == "hotpath-addition", hotpaths)
        @test c.seed == 307 + sum(c.dims)
        diagnostic = only(filter(d -> d.dtype == c.dtype && d.dims == c.dims, addition_diagnostics))
        @test diagnostic.seed == c.seed
    end
    for c in filter(c -> c.group == "hotpath-square-gemv", hotpaths)
        @test c.seed == 211 + sum(c.dims)
    end
    @test length(addition_diagnostics) == 8
    @test all(c -> c.group == "hotpath-addition-diagnostic" && c.operation == "+", addition_diagnostics)
    @test isempty(intersect(Set(case_id.(hotpaths)), Set(case_id.(addition_diagnostics))))
    @test_throws ArgumentError GFMPerf.capture("unknown-hotpath-case"; suite = :hotpaths,
        case_ids = ["unknown-case"], output_dir = mktempdir())
    @test_throws ArgumentError GFMPerf.capture("empty-hotpath-selection"; suite = :hotpaths,
        groups = ["unknown-group"], output_dir = mktempdir())

end

@testset "total PLUQ materialized-factor adapter" begin
    fake_current = Module(:FakeCurrentPluqApi)
    Core.eval(fake_current, :(pluq(A) = A))
    Core.eval(fake_current, :(pluq_gpu_kernel(A) = error("current API should win")))
    api, call = GFMPerf._select_pluq_api(fake_current)
    @test api == :current
    @test call(:value) == :value

    fake_legacy = Module(:FakeLegacyPluqApi)
    Core.eval(fake_legacy, :(pluq_gpu_kernel(A) = A))
    api, call = GFMPerf._select_pluq_api(fake_legacy)
    @test api == :legacy
    @test call(:value) == :value
    @test first(GFMPerf._select_pluq_api(Module(:NoPluqApi))) === nothing

    L = [1 0; 0 1]
    U = [2 0; 0 3]
    factor_reads = Ref(0)
    current = TrackedPluqResult(factor_reads, L, U, [1, 2], [1, 2], 2)
    factors = Ref{Any}(nothing)
    permutations = Ref{Any}(nothing)
    rank = Ref{Any}(nothing)
    calls = Ref(0)
    runner = GFMPerf._materialized_pluq_runner(:current,
        _ -> (calls[] += 1; current), nothing, factors, permutations, rank)
    runner()
    @test calls[] == 1
    @test factor_reads[] == 2 # L and U are extracted before the timed runner returns.
    @test factors[] == (L, U)
    @test permutations[] == ([1, 2], [1, 2])
    @test rank[] == 2

    legacy_factors = Ref{Any}(nothing)
    legacy_permutations = Ref{Any}(nothing)
    legacy_rank = Ref{Any}(nothing)
    legacy_runner = GFMPerf._materialized_pluq_runner(:legacy,
        _ -> (calls[] += 1; (U, L, Int[], Int[])), nothing,
        legacy_factors, legacy_permutations, legacy_rank)
    legacy_runner()
    @test calls[] == 2
    @test legacy_factors[] == (L, U)
    @test legacy_permutations[] == (Int[], Int[])
    @test legacy_rank[] === nothing

    diagonal = [2 0; 0 3]
    @test GFMPerf._cpu_pluq_product_matches(L, U, diagonal, 101)
    @test !GFMPerf._cpu_pluq_product_matches(L, U, [2 1; 0 3], 101)
    @test GFMPerf._cpu_pluq_product_matches(L, U, diagonal, 0) == false
    @test GFMPerf._pluq_permutation_is_identity(:current, ([1, 2], [1, 2]), 2)
    @test !GFMPerf._pluq_permutation_is_identity(:current, ([2, 1], [1, 2]), 2)
    @test GFMPerf._pluq_permutation_is_identity(:legacy, (Int[], Int[]), 2)
    @test !GFMPerf._pluq_permutation_is_identity(:legacy, ([(1, 2)], Int[]), 2)
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

    changed_contract = synthetic_run("changed-contract")
    changed_contract["measurement_policy"] = Dict{String,Any}(changed_contract["measurement_policy"])
    changed_contract["measurement_policy"]["operation_contracts"] =
        Dict("pluq-fixture" => "pluq-total-materialized-factors-v1")
    incompatible_contract = compare_runs(changed_contract, base)
    @test !incompatible_contract["compatible"]
    @test haskey(incompatible_contract["compatibility_differences"], "measurement_policy")

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

@testset "bracketed hotpath review uses pair noise and all nine run identities" begin
    mktempdir() do dir
        cycles = fixture_cycles(dir)
        review_dir = GFMPerf.review_bracketed_cycles(cycles; label = "stable-five-percent",
            output_dir = joinpath(dir, "reviews"))
        review = GFMPerf.JSON3.read(read(joinpath(review_dir, "review.json"), String), Dict{String,Any})
        row = only(review["rows"])
        @test review["threshold_fraction"] == 0.0
        @test review["compatible_across_all_runs"]
        @test review["source_identity"]["baseline_stable"]
        @test review["source_identity"]["candidate_stable"]
        @test review["independent_repeat_count"] == 3
        @test row["status"] == "confirmed-slowdown"
        @test row["cycle_ratio_median"] ≈ 1.05
        @test all(c -> c["current_vs_baseline_before"]["status"] == "flagged-unconfirmed" &&
            c["current_vs_baseline_after"]["status"] == "flagged-unconfirmed", row["cycles"])
        @test first(row["cycles"])["runs"]["current"]["correctness"]
        @test first(row["cycles"])["current_vs_baseline_before"]["current_median_us"] ≈ 0.105
        @test isfile(joinpath(review_dir, "review.json"))
        @test occursin("fresh Julia process", review["fresh_process_assumption"])

        visible = GFMPerf.review_bracketed_cycles(cycles; label = "visible-under-threshold",
            threshold = 0.10, output_dir = joinpath(dir, "reviews"))
        visible_report = GFMPerf.JSON3.read(read(joinpath(visible, "review.json"), String), Dict{String,Any})
        visible_row = only(visible_report["rows"])
        @test visible_row["status"] == "positive-change-observed"
        @test first(visible_row["cycles"])["current_vs_baseline_before"]["change_percent"] ≈ 5
        @test_throws ArgumentError GFMPerf.review_bracketed_cycles(cycles; label = "stable-five-percent",
            output_dir = joinpath(dir, "reviews"))

        noisy = fixture_cycles(joinpath(dir, "noisy");
            baseline_samples = [70, 80, 100, 120, 130],
            current_samples = [74, 84, 105, 126, 137])
        noisy_dir = GFMPerf.review_bracketed_cycles(noisy; label = "noisy-with-same-medians",
            output_dir = joinpath(dir, "reviews"))
        noisy_row = only(GFMPerf.JSON3.read(read(joinpath(noisy_dir, "review.json"), String), Dict{String,Any})["rows"])
        @test noisy_row["status"] == "positive-change-observed"
        @test all(c -> c["current_vs_baseline_before"]["status"] == "inconclusive-within-noise" &&
            c["current_vs_baseline_after"]["status"] == "inconclusive-within-noise", noisy_row["cycles"])

        varying = fixture_cycles(joinpath(dir, "varying");
            current_samples_by_cycle = [fill(105, 5), fill(105, 5), fill(130, 5)])
        varying_dir = GFMPerf.review_bracketed_cycles(varying; label = "between-cycle-variation",
            output_dir = joinpath(dir, "reviews"))
        varying_row = only(GFMPerf.JSON3.read(read(joinpath(varying_dir, "review.json"), String), Dict{String,Any})["rows"])
        @test varying_row["status"] == "inconclusive-between-cycle-variation"

        drifting_baseline = fixture_cycles(joinpath(dir, "drifting-baseline");
            current_samples = fill(130, 5),
            baseline_before_samples_by_cycle = fill(fill(100, 5), 3),
            baseline_after_samples_by_cycle = fill(fill(120, 5), 3))
        drift_dir = GFMPerf.review_bracketed_cycles(drifting_baseline; label = "baseline-bracket-drift",
            output_dir = joinpath(dir, "reviews"))
        drift_row = only(GFMPerf.JSON3.read(read(joinpath(drift_dir, "review.json"), String), Dict{String,Any})["rows"])
        @test drift_row["status"] == "inconclusive-baseline-drift"

        incompatible = fixture_cycles(joinpath(dir, "incompatible");
            baseline_cuda = ["5.11.3", "6.0.0", "5.11.3"],
            candidate_cuda = ["5.11.3", "6.0.0", "5.11.3"])
        incompatible_dir = GFMPerf.review_bracketed_cycles(incompatible; label = "cross-cycle-environment",
            output_dir = joinpath(dir, "reviews"))
        incompatible_report = GFMPerf.JSON3.read(read(joinpath(incompatible_dir, "review.json"), String), Dict{String,Any})
        @test !incompatible_report["compatible_across_all_runs"]
        @test only(incompatible_report["rows"])["status"] == "incompatible"

        changed_source = fixture_cycles(joinpath(dir, "changed-source");
            candidate_hashes = ["candidate-source", "candidate-source", "changed-after-cycle-2"])
        changed_dir = GFMPerf.review_bracketed_cycles(changed_source; label = "candidate-source-drift",
            output_dir = joinpath(dir, "reviews"))
        changed_report = GFMPerf.JSON3.read(read(joinpath(changed_dir, "review.json"), String), Dict{String,Any})
        @test !changed_report["source_identity"]["candidate_stable"]
        @test only(changed_report["rows"])["status"] == "incompatible-source-identity"

        for (label, source_edit) in (("unknown-source-identity", :unknown), ("missing-source-identity", :missing))
            unknown_source = fixture_cycles(joinpath(dir, label))
            path = joinpath(unknown_source[1][1], "run.json")
            run = GFMPerf.JSON3.read(read(path, String), Dict{String,Any})
            if source_edit == :unknown
                run["metadata"]["target_commit"] = "unknown"
            else
                delete!(run["metadata"], "target_commit")
            end
            write_json(path, run)
            unknown_dir = GFMPerf.review_bracketed_cycles(unknown_source; label,
                output_dir = joinpath(dir, "reviews"))
            unknown_report = GFMPerf.JSON3.read(read(joinpath(unknown_dir, "review.json"), String), Dict{String,Any})
            @test !unknown_report["source_identity"]["baseline_stable"]
            @test only(unknown_report["rows"])["status"] == "incompatible-source-identity"
        end

        failed_capture = fixture_cycles(joinpath(dir, "failed-capture"))
        failed_path = joinpath(failed_capture[2][2], "run.json")
        failed_run = GFMPerf.JSON3.read(read(failed_path, String), Dict{String,Any})
        failed_case = only(failed_run["cases"])
        failed_case["status"] = "failed"
        failed_case["correctness"] = false
        failed_case["detail"] = "injected GPU kernel failure"
        failed_case["correctness_detail"] = "host oracle did not complete"
        write_json(failed_path, failed_run)
        failed_dir = GFMPerf.review_bracketed_cycles(failed_capture; label = "failed-capture-review",
            output_dir = joinpath(dir, "reviews"))
        failed_review_row = only(GFMPerf.JSON3.read(read(joinpath(failed_dir, "review.json"), String), Dict{String,Any})["rows"])
        @test failed_review_row["status"] == "not-comparable"
        @test occursin("cycle 2/current", failed_review_row["detail"])
        @test occursin("injected GPU kernel failure", failed_review_row["detail"])
        @test occursin("correctness=false", failed_review_row["detail"])

        missing_case_capture = fixture_cycles(joinpath(dir, "missing-case"))
        missing_path = joinpath(missing_case_capture[1][1], "run.json")
        missing_run = GFMPerf.JSON3.read(read(missing_path, String), Dict{String,Any})
        empty!(missing_run["cases"])
        write_json(missing_path, missing_run)
        missing_dir = GFMPerf.review_bracketed_cycles(missing_case_capture; label = "missing-case-review",
            output_dir = joinpath(dir, "reviews"))
        missing_row = only(GFMPerf.JSON3.read(read(joinpath(missing_dir, "review.json"), String), Dict{String,Any})["rows"])
        @test missing_row["status"] == "missing-case"
        @test occursin("cycle 1/baseline_before", missing_row["detail"])
        @test occursin("case not present in capture", missing_row["detail"])

        duplicate = copy(cycles)
        duplicate[2] = (duplicate[1][1], duplicate[2][2], duplicate[2][3])
        @test_throws ArgumentError GFMPerf.review_bracketed_cycles(duplicate; label = "duplicate-input",
            output_dir = joinpath(dir, "reviews"))
    end
end
