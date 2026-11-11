using Dates
using Libdl
using PackageCompiler
using Pkg
using SHA
using TOML
using UUIDs

const EXPECTED_JULIA_VERSION = v"1.12.4"
const PACKAGE_UUID = UUID("465031f1-b347-4f72-85f6-baf667cdf853")
const PACKAGE_NAME = "GPUFiniteFieldMatrices"

function git_output(repo::String, args...)
    command = Cmd(["git", "-C", repo, String.(args)...])
    return strip(read(command, String))
end

function source_files(repo::String)
    files = String[]
    for name in ("Project.toml", "Manifest.toml", "LocalPreferences.toml")
        path = joinpath(repo, name)
        isfile(path) && push!(files, path)
    end
    src = joinpath(repo, "src")
    for (dir, dirs, names) in walkdir(src)
        sort!(dirs)
        for name in sort(names)
            path = joinpath(dir, name)
            isfile(path) && push!(files, path)
        end
    end
    sort!(files)
    return files
end

function source_fingerprint(repo::String)
    entries = String[]
    for path in source_files(repo)
        relative = relpath(path, repo)
        push!(entries, string(relative, "\0", bytes2hex(sha256(read(path)))))
    end
    return bytes2hex(sha256(join(entries, "\n")))
end

function optional_manifest_package(manifest::Dict, name::String, uuid::UUID)
    entries = get(get(manifest, "deps", Dict()), name, nothing)
    entries === nothing && return nothing
    candidates = entries isa AbstractVector ? entries : [entries]
    for entry in candidates
        get(entry, "uuid", "") == string(uuid) && return entry
    end
    error("Manifest.toml contains $name, but not with the checked-out package UUID $uuid")
end

repo = realpath(ENV["OSCAR_GFM_REPO"])
stage = realpath(ENV["OSCAR_SYSIMAGE_STAGE_DIR"])
final_dir = abspath(get(ENV, "OSCAR_SYSIMAGE_FINAL_DIR", stage))
tool_project = realpath(joinpath(@__DIR__, "Project.toml"))
tool_manifest = joinpath(@__DIR__, "Manifest.toml")
project_file = joinpath(repo, "Project.toml")
manifest_file = joinpath(repo, "Manifest.toml")
image_file = joinpath(stage, "GPUFiniteFieldMatrices.so")

VERSION == EXPECTED_JULIA_VERSION || error("build driver requires Julia $EXPECTED_JULIA_VERSION, found $VERSION")
pkgversion(PackageCompiler) == v"2.4.0" || error("expected PackageCompiler 2.4.0, found $(pkgversion(PackageCompiler))")
isfile(tool_manifest) || error("missing pinned PackageCompiler Manifest.toml in $(@__DIR__)")
isfile(project_file) || error("missing package project: $project_file")
isfile(manifest_file) || error("missing package Manifest.toml: $manifest_file")

project = TOML.parsefile(project_file)
get(project, "name", nothing) == PACKAGE_NAME || error("$project_file is not the $PACKAGE_NAME project")
get(project, "uuid", nothing) == string(PACKAGE_UUID) || error("$project_file has an unexpected package UUID")

manifest = TOML.parsefile(manifest_file)
manifest_julia = get(manifest, "julia_version", "")
startswith(manifest_julia, "1.12.") || error("Manifest resolves for Julia $manifest_julia; prepare an explicit Julia 1.12 environment before building")
package_entry = optional_manifest_package(manifest, PACKAGE_NAME, PACKAGE_UUID)
if package_entry === nothing
    resolved_source = repo
else
    manifest_source = get(package_entry, "path", nothing)
    manifest_source === nothing && error("Manifest resolves $PACKAGE_NAME from a registry/git revision, not the checked-out local package path")
    resolved_source = realpath(isabspath(manifest_source) ? manifest_source : joinpath(repo, manifest_source))
    resolved_source == repo || error("Manifest resolves $PACKAGE_NAME from $resolved_source, expected this checkout $repo")
end

source_status = git_output(repo, "status", "--porcelain=v1", "--untracked-files=all", "--ignored=matching", "--", "src", "Project.toml", "Manifest.toml", "LocalPreferences.toml")
allow_dirty = get(ENV, "OSCAR_SYSIMAGE_ALLOW_DIRTY", "0") == "1"
isempty(source_status) || allow_dirty || error("package source/environment is dirty; inspect it and explicitly set OSCAR_SYSIMAGE_ALLOW_DIRTY=1 to continue")

default_image = realpath(joinpath(Sys.BINDIR, "..", "lib", "julia", "sys." * Libdl.dlext))
loaded_image = realpath(unsafe_string(Base.JLOptions().image_file))
loaded_image == default_image || error("PackageCompiler process is using non-default sysimage $loaded_image (expected $default_image)")

cpu_target = get(ENV, "OSCAR_SYSIMAGE_CPU_TARGET", "generic")
isempty(cpu_target) && error("OSCAR_SYSIMAGE_CPU_TARGET must not be empty")
provenance = joinpath(stage, "provenance")
mkpath(provenance)

cp(project_file, joinpath(provenance, "Project.toml"))
cp(manifest_file, joinpath(provenance, "Manifest.toml"))
preferences_file = joinpath(repo, "LocalPreferences.toml")
if isfile(preferences_file)
    cp(preferences_file, joinpath(provenance, "LocalPreferences.toml"))
end
cp(tool_project, joinpath(provenance, "PackageCompiler-Project.toml"))
cp(tool_manifest, joinpath(provenance, "PackageCompiler-Manifest.toml"))

source_patch = read(Cmd(["git", "-C", repo, "diff", "--binary", "HEAD", "--", "src", "Project.toml", "Manifest.toml", "LocalPreferences.toml"]))
write(joinpath(provenance, "source.patch"), source_patch)
write(joinpath(provenance, "source-status.txt"), source_status * "\n")
manifest_tracked = !isempty(git_output(repo, "ls-files", "--", "Manifest.toml"))
manifest_git_status = git_output(repo, "status", "--porcelain=v1", "--untracked-files=all", "--ignored=matching", "--", "Manifest.toml")
write(joinpath(provenance, "manifest-status.txt"), manifest_git_status * "\n")
write(joinpath(provenance, "module-list.txt"), get(ENV, "OSCAR_SYSIMAGE_MODULE_LIST", "module list unavailable or not captured") * "\n")

Pkg.activate(repo; io=devnull)
active_project = realpath(Base.active_project())
active_project == realpath(project_file) || error("active application project is $active_project, expected $project_file")
resolved = Pkg.dependencies()

open(joinpath(provenance, "dependencies.txt"), "w") do io
    Pkg.status(; io)
end

effective_preferences = Dict{String,Any}()
for (uuid, info) in resolved
    prefs = Base.get_preferences(uuid)
    isempty(prefs) || (effective_preferences[string(uuid)] = prefs)
end
package_preferences = Base.get_preferences(PACKAGE_UUID)
isempty(package_preferences) || (effective_preferences[string(PACKAGE_UUID)] = package_preferences)
open(joinpath(provenance, "effective-preferences.toml"), "w") do io
    TOML.print(io, effective_preferences)
end

archive_paths = ["src", "Project.toml"]
manifest_tracked && push!(archive_paths, "Manifest.toml")
preferences_tracked = !isempty(git_output(repo, "ls-files", "--", "LocalPreferences.toml"))
preferences_tracked && push!(archive_paths, "LocalPreferences.toml")
tracked_archive = read(Cmd(["git", "-C", repo, "archive", "--format=tar", "HEAD", "--", archive_paths...]))
archive_hash = bytes2hex(sha256(tracked_archive))
tracked_patch_hash = bytes2hex(sha256(source_patch))
package_source_hash = source_fingerprint(repo)
manifest_hash = bytes2hex(sha256(read(manifest_file)))
preferences_hash = isfile(preferences_file) ? bytes2hex(sha256(read(preferences_file))) : "absent"
target_version = get(project, "version", "unknown")
build_time = Dates.format(now(UTC), "yyyy-mm-ddTHH:MM:SS")
branch = get(ENV, "OSCAR_SYSIMAGE_BRANCH", "DETACHED")
commit = get(ENV, "OSCAR_SYSIMAGE_COMMIT", git_output(repo, "rev-parse", "HEAD"))
tree = get(ENV, "OSCAR_SYSIMAGE_TREE", git_output(repo, "rev-parse", "HEAD^{tree}"))
julia_bin = get(ENV, "OSCAR_SYSIMAGE_JULIA_BIN", realpath(joinpath(Sys.BINDIR, Base.julia_exename())))

receipt = Dict{String,Any}(
    "target_package" => PACKAGE_NAME,
    "target_package_uuid" => string(PACKAGE_UUID),
    "target_version" => target_version,
    "target_project" => project_file,
    "target_source_path" => repo,
    "target_commit" => commit,
    "target_branch" => branch,
    "target_tree" => tree,
    "tracked_archive_sha256" => archive_hash,
    "tracked_patch_sha256" => tracked_patch_hash,
    "package_source_sha256" => package_source_hash,
    "package_manifest_julia" => manifest_julia,
    "package_manifest_sha256" => manifest_hash,
    "package_manifest_tracked" => manifest_tracked,
    "package_manifest_git_status" => manifest_git_status,
    "local_preferences_sha256" => preferences_hash,
    "source_dirty" => !isempty(source_status),
    "source_dirty_allowed" => allow_dirty,
    "source_status" => source_status,
    "julia_executable" => julia_bin,
    "julia_version" => string(VERSION),
    "active_project" => active_project,
    "loaded_base_sysimage" => loaded_image,
    "default_base_sysimage" => default_image,
    "cpu_target" => cpu_target,
    "operating_system" => string(Sys.KERNEL),
    "architecture" => string(Sys.ARCH),
    "packagecompiler_version" => string(pkgversion(PackageCompiler)),
    "build_time_utc" => build_time * "Z",
    "image_path" => joinpath(final_dir, basename(image_file)),
)
open(joinpath(provenance, "receipt.toml"), "w") do io
    TOML.print(io, receipt)
end

println("PackageCompiler $(pkgversion(PackageCompiler)); Julia $VERSION")
println("Target source: $repo @ $commit (tree $tree)")
println("Manifest resolves $PACKAGE_NAME from the checked-out path: $resolved_source")
println("CPU target: $cpu_target; source fingerprint: $package_source_hash")
println("Effective package preferences recorded at $(joinpath(provenance, "effective-preferences.toml"))")
println("Precompile workload is host-only; no CUDA context or GPU allocation is created.")

PackageCompiler.create_sysimage(
    [PACKAGE_NAME];
    sysimage_path = image_file,
    project = repo,
    precompile_execution_file = joinpath(@__DIR__, "precompile-workload.jl"),
    cpu_target = cpu_target,
    incremental = true,
    include_transitive_dependencies = true,
    import_into_main = true,
)

isfile(image_file) && filesize(image_file) > 0 || error("PackageCompiler returned without creating a nonempty image")
source_fingerprint(repo) == package_source_hash || error("package source/environment changed during image creation; refusing to publish a mixed-source image")
println("Created sysimage: $image_file ($(filesize(image_file)) bytes)")
