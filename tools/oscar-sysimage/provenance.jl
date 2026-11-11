module OscarSysimageProvenance

using SHA
using TOML
using UUIDs

const EXPECTED_JULIA_VERSION = v"1.12.4"
const PACKAGE_UUID = UUID("465031f1-b347-4f72-85f6-baf667cdf853")
const PACKAGE_NAME = "GPUFiniteFieldMatrices"

function selected_manifest(repo::AbstractString)
    project_file = realpath(joinpath(repo, "Project.toml"))
    manifest_file = Base.project_file_manifest_path(project_file)
    manifest_file === nothing && error("Julia $VERSION selected no manifest for $project_file")
    return realpath(manifest_file)
end

function preference_files(repo::AbstractString)
    return [joinpath(repo, name) for name in ("JuliaLocalPreferences.toml", "LocalPreferences.toml") if isfile(joinpath(repo, name))]
end

function source_files(repo::AbstractString)
    files = String[joinpath(repo, "Project.toml")]
    src = joinpath(repo, "src")
    for (dir, dirs, names) in walkdir(src)
        sort!(dirs)
        for name in sort(names)
            path = joinpath(dir, name)
            isfile(path) && push!(files, path)
        end
    end
    return sort!(files)
end

function environment_files(repo::AbstractString)
    return sort!([selected_manifest(repo); preference_files(repo)])
end

function effective_preferences(repo::AbstractString)
    manifest = TOML.parsefile(selected_manifest(repo))
    entries = get(manifest, "deps", manifest)
    preferences = Dict{String,Any}()
    for (name, raw_entries) in entries
        candidates = raw_entries isa AbstractVector ? raw_entries : [raw_entries]
        for entry in candidates
            uuid_text = get(entry, "uuid", nothing)
            uuid_text === nothing && continue
            uuid = UUID(uuid_text)
            values = Base.get_preferences(uuid)
            isempty(values) || (preferences[string(uuid)] = values)
        end
    end
    values = Base.get_preferences(PACKAGE_UUID)
    isempty(values) || (preferences[string(PACKAGE_UUID)] = values)
    return preferences
end

function canonical_value(value)
    if value isa AbstractDict
        keys_sorted = sort!(collect(keys(value)); by=string)
        return "{" * join((repr(key) * ":" * canonical_value(value[key]) for key in keys_sorted), ",") * "}"
    elseif value isa AbstractVector || value isa Tuple
        return "[" * join(canonical_value.(value), ",") * "]"
    else
        return repr(value)
    end
end

function fingerprint(repo::AbstractString, files::AbstractVector{<:AbstractString})
    root = realpath(repo)
    entries = String[]
    for path in files
        isfile(path) || error("fingerprinted file does not exist: $path")
        canonical = realpath(path)
        relative_path = relpath(canonical, root)
        relative = startswith(relative_path, "..") ? canonical : relative_path
        push!(entries, string(relative, "\0", bytes2hex(sha256(read(canonical)))))
    end
    return bytes2hex(sha256(join(entries, "\n")))
end

source_fingerprint(repo::AbstractString) = fingerprint(repo, source_files(repo))
function environment_fingerprint(repo::AbstractString)
    files_hash = fingerprint(repo, environment_files(repo))
    preferences = effective_preferences(repo)
    preference_data = join((string(uuid, "\0", canonical_value(preferences[uuid])) for uuid in sort!(collect(keys(preferences)))), "\n")
    return bytes2hex(sha256(string(files_hash, "\0", preference_data)))
end

function verify_receipt(repo::AbstractString, image::AbstractString; require_loaded::Bool=true)
    VERSION == EXPECTED_JULIA_VERSION || error("sysimage validation requires Julia $EXPECTED_JULIA_VERSION, found $VERSION")
    repo = realpath(repo)
    image = realpath(image)
    active_project = Base.active_project()
    active_project !== nothing && realpath(active_project) == realpath(joinpath(repo, "Project.toml")) ||
        error("active project is $(repr(active_project)), expected $repo/Project.toml")
    realpath(unsafe_string(Base.JLOptions().image_file)) == image ||
        error("Julia did not load the requested sysimage $image")

    receipt_file = joinpath(dirname(image), "provenance", "receipt.toml")
    isfile(receipt_file) || error("missing sysimage receipt: $receipt_file")
    receipt = TOML.parsefile(receipt_file)
    get(receipt, "target_package", "") == PACKAGE_NAME || error("receipt is for a different package")
    get(receipt, "target_package_uuid", "") == string(PACKAGE_UUID) || error("receipt package UUID does not match $PACKAGE_NAME")
    get(receipt, "julia_version", "") == string(VERSION) || error("receipt Julia version does not match this process")
    julia_binary = realpath(joinpath(Sys.BINDIR, Base.julia_exename()))
    get(receipt, "julia_executable", "") == julia_binary || error("receipt Julia executable does not match this process: $julia_binary")
    get(receipt, "target_source_path", "") == repo || error("receipt target path does not match this checkout")

    project = TOML.parsefile(joinpath(repo, "Project.toml"))
    get(project, "name", nothing) == PACKAGE_NAME || error("target project name does not match $PACKAGE_NAME")
    get(project, "uuid", nothing) == string(PACKAGE_UUID) || error("target project UUID does not match $PACKAGE_NAME")
    get(receipt, "target_version", "") == get(project, "version", "unknown") || error("receipt package version does not match the target project")

    manifest = selected_manifest(repo)
    get(receipt, "package_manifest_path", "") == manifest || error("receipt manifest path does not match Julia's selected manifest: $manifest")
    bytes2hex(sha256(read(manifest))) == get(receipt, "package_manifest_sha256", "") ||
        error("selected manifest hash does not match the sysimage receipt")
    source_fingerprint(repo) == get(receipt, "package_source_sha256", "") ||
        error("target source fingerprint does not match the sysimage receipt")
    environment_fingerprint(repo) == get(receipt, "package_environment_sha256", "") ||
        error("target manifest/preferences fingerprint does not match the sysimage receipt")

    if require_loaded
        package_id = Base.PkgId(PACKAGE_UUID, PACKAGE_NAME)
        haskey(Base.loaded_modules, package_id) || error("$PACKAGE_NAME was not loaded from the sysimage at startup")
        package_module = Base.loaded_modules[package_id]
        package_path = realpath(pathof(package_module))
        package_path == realpath(joinpath(repo, "src", "$PACKAGE_NAME.jl")) ||
            error("loaded $PACKAGE_NAME from $package_path, expected checked-out source under $repo")
    end

    return (; receipt, manifest, source_sha256=get(receipt, "package_source_sha256", ""),
            environment_sha256=get(receipt, "package_environment_sha256", ""))
end

end # module OscarSysimageProvenance
