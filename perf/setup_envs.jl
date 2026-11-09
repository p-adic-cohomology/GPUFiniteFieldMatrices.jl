using Pkg
using TOML

length(ARGS) == 5 || error("usage: setup_envs.jl RUN_ROOT REPO V031_WORKTREE V032_WORKTREE CUDA_VERSION")
run_root, repo, v031, v032, cuda_version = ARGS
VERSION.major == 1 && VERSION.minor == 12 || error("Julia 1.12 is required; found $VERSION")

perf_project = joinpath(repo, "perf")
variants = (("v0.3.1", v031), ("v0.3.2", v032), ("current", repo))

function manifest_entry(doc, package)
    entries = get(get(doc, "deps", Dict{String,Any}()), package, nothing)
    isnothing(entries) && return nothing
    return entries isa Vector ? only(entries) : entries
end

function verify_manifest(env, target, perf_project, cuda_version)
    manifest = TOML.parsefile(joinpath(env, "Manifest.toml"))
    harness = manifest_entry(manifest, "GFMPerf")
    package = manifest_entry(manifest, "GPUFiniteFieldMatrices")
    cuda = manifest_entry(manifest, "CUDA")
    !isnothing(harness) && get(harness, "path", nothing) == abspath(perf_project) || error("benchmark environment resolves GFMPerf from an unexpected path: $env")
    !isnothing(package) && get(package, "path", nothing) == abspath(target) || error("benchmark environment resolves GPUFiniteFieldMatrices from an unexpected path: $env")
    !isnothing(cuda) && get(cuda, "version", nothing) == cuda_version || error("benchmark environment CUDA.jl does not match pin $cuda_version")
end

for (name, target) in variants
    env = joinpath(run_root, "envs", name)
    mkpath(env)
    marker = joinpath(env, ".gfm-perf-env.toml")
    expected = Dict("schema_version" => 1, "name" => name, "target" => abspath(target),
        "harness" => abspath(perf_project), "cuda_version" => cuda_version)
    state = "new"
    if isfile(marker)
        actual = TOML.parsefile(marker)
        for (key, value) in expected
            get(actual, key, nothing) == value || error("refusing to reuse benchmark environment with different ownership or pins: $env")
        end
        state = get(actual, "state", "")
        state in ("preparing", "ready") || error("invalid environment setup state at $env")
        state == "ready" && !isfile(joinpath(env, "Manifest.toml")) && error("ready environment lost its manifest: $env")
    else
        entries = readdir(env)
        isempty(entries) || error("refusing to reuse non-empty unmarked environment: $env")
        state = "preparing"
        open(marker, "w") do io
            TOML.print(io, merge(expected, Dict("state" => state)))
        end
    end
    Pkg.activate(env)
    if state == "ready"
        Pkg.instantiate()
    else
        env_doc = isfile(joinpath(env, "Project.toml")) ? TOML.parsefile(joinpath(env, "Project.toml")) : Dict{String,Any}()
        env_deps = get(env_doc, "deps", Dict{String,Any}())
        if haskey(env_deps, "GPUFiniteFieldMatrices") && haskey(env_deps, "GFMPerf")
            manifest = joinpath(env, "Manifest.toml")
            if isfile(manifest)
                doc_manifest = TOML.parsefile(manifest)
                harness = manifest_entry(doc_manifest, "GFMPerf")
                package = manifest_entry(doc_manifest, "GPUFiniteFieldMatrices")
                get(harness, "path", nothing) == abspath(perf_project) || error("interrupted environment points to an unexpected harness path: $env")
                get(package, "path", nothing) == abspath(target) || error("interrupted environment points to an unexpected package path: $env")
            end
        elseif !isempty(env_deps)
            error("interrupted environment contains dependencies that are not this harness: $env")
        end
        specs = [Pkg.PackageSpec(path = perf_project), Pkg.PackageSpec(path = target)]
        Pkg.develop(specs)
        Pkg.add(Pkg.PackageSpec(name = "CUDA", version = VersionNumber(cuda_version)))
        Pkg.resolve()
        Pkg.instantiate()
        verify_manifest(env, target, perf_project, cuda_version)
        marker_value = merge(expected, Dict("state" => "ready"))
        open(marker, "w") do io
            TOML.print(io, marker_value)
        end
    end
    state == "ready" && verify_manifest(env, target, perf_project, cuda_version)
    println("prepared $name environment at $env")
end
