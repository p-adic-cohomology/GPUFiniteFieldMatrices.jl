using SHA
using TOML
using UUIDs

const EXPECTED_JULIA_VERSION = v"1.12.4"
const PACKAGE_UUID = UUID("465031f1-b347-4f72-85f6-baf667cdf853")
const PACKAGE_NAME = "GPUFiniteFieldMatrices"

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

repo = realpath(ENV["OSCAR_GFM_REPO"])
image = realpath(ENV["OSCAR_SYSIMAGE_FILE"])
VERSION == EXPECTED_JULIA_VERSION || error("fresh-process validation requires Julia $EXPECTED_JULIA_VERSION, found $VERSION")
realpath(Base.active_project()) == realpath(joinpath(repo, "Project.toml")) ||
    error("active project is $(Base.active_project()), expected $repo/Project.toml")
realpath(unsafe_string(Base.JLOptions().image_file)) == image ||
    error("Julia did not load the requested sysimage $image")

receipt_file = joinpath(dirname(image), "provenance", "receipt.toml")
isfile(receipt_file) || error("missing sysimage receipt: $receipt_file")
receipt = TOML.parsefile(receipt_file)
get(receipt, "julia_version", "") == string(VERSION) || error("receipt Julia version does not match this process")
get(receipt, "target_source_path", "") == repo || error("receipt target path does not match this checkout")
project = TOML.parsefile(joinpath(repo, "Project.toml"))
get(project, "name", nothing) == PACKAGE_NAME || error("target project name does not match $PACKAGE_NAME")
get(project, "uuid", nothing) == string(PACKAGE_UUID) || error("target project UUID does not match $PACKAGE_NAME")
get(receipt, "target_version", "") == get(project, "version", "unknown") || error("receipt package version does not match the target project")
source_fingerprint(repo) == get(receipt, "package_source_sha256", "") || error("target source/environment fingerprint does not match the sysimage receipt")

package_id = Base.PkgId(PACKAGE_UUID, PACKAGE_NAME)
haskey(Base.loaded_modules, package_id) || error("$PACKAGE_NAME was not loaded from the sysimage at startup")
using GPUFiniteFieldMatrices

package_path = realpath(pathof(GPUFiniteFieldMatrices))
package_path == realpath(joinpath(repo, "src", "GPUFiniteFieldMatrices.jl")) ||
    error("loaded $PACKAGE_NAME from $package_path, expected checked-out source under $repo")

for (name, uuid) in (
    ("CUDA", UUID("052768ef-5323-5732-b1bb-66c8b64840ba")),
    ("DispatchDoctor", UUID("8d63f2c5-f18a-4cf2-ba9d-b3f60fc568c8")),
    ("IterTools", UUID("c8e1da08-722c-5040-9ed9-7db0dc04731e")),
    ("Unroll", UUID("36e64239-2ab9-4580-84fb-56737a503d36")),
)
    dependency_id = Base.PkgId(uuid, name)
    haskey(Base.loaded_modules, dependency_id) || error("expected dependency $name was not loaded")
    println("Loaded dependency: $name $(Base.pkgversion(Base.loaded_modules[dependency_id]))")
end

low = Int[1 2; 3 4]
high = Int[0 1; 1 0]
value = GPUFiniteFieldMatrices.KaratsubaMatrix(
    low,
    high;
    low_modulus = 11,
    high_modulus = 11,
    modulus = 11,
)
expected = low .+ 11 .* high
observed = GPUFiniteFieldMatrices.from_karatsuba(Int, value)
observed == expected || error("fresh-process CPU helper validation failed")

println("Fresh-process sysimage validation passed.")
println("Julia version: $VERSION")
println("Active project: $(realpath(Base.active_project()))")
println("Loaded sysimage: $image")
println("Loaded package path: $package_path")
println("Target source fingerprint: $(receipt["package_source_sha256"])")
println("CPU helper result: $observed")
