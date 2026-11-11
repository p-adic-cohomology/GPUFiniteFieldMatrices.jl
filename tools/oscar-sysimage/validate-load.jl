using UUIDs

include(joinpath(@__DIR__, "provenance.jl"))

repo = realpath(ENV["OSCAR_GFM_REPO"])
image = realpath(ENV["OSCAR_SYSIMAGE_FILE"])
identity = OscarSysimageProvenance.verify_receipt(repo, image)

using GPUFiniteFieldMatrices

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
println("Selected application manifest: $(identity.manifest)")
println("Loaded sysimage: $image")
println("Loaded package path: $(realpath(pathof(GPUFiniteFieldMatrices)))")
println("Target source fingerprint: $(identity.source_sha256)")
println("Target environment fingerprint: $(identity.environment_sha256)")
println("CPU helper result: $observed")
