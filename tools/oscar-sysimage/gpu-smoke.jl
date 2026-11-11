include(joinpath(@__DIR__, "provenance.jl"))

repo = realpath(ENV["OSCAR_GFM_REPO"])
image = realpath(ENV["OSCAR_SYSIMAGE_FILE"])
identity = OscarSysimageProvenance.verify_receipt(repo, image)

using CUDA
using GPUFiniteFieldMatrices

println("Julia version: $VERSION")
println("Active project: $(realpath(Base.active_project()))")
println("Selected application manifest: $(identity.manifest)")
println("Loaded sysimage: $image")
println("Package source: $(realpath(pathof(GPUFiniteFieldMatrices)))")
println("Target source fingerprint: $(identity.source_sha256)")
println("Target environment fingerprint: $(identity.environment_sha256)")
CUDA.versioninfo()
CUDA.functional() || error("CUDA reports no functional NVIDIA GPU; GPU smoke validation cannot pass")
device = CUDA.device()
println("CUDA device: $(CUDA.name(device))")
println("Compute capability: $(CUDA.capability(device))")

host_a = Float32[1 2; 3 4]
host_b = Float32[5 6; 0 1]
gpu_a = GPUFiniteFieldMatrices.CuModMatrix(host_a; modulus = 7)
gpu_b = GPUFiniteFieldMatrices.CuModMatrix(host_b; modulus = 7)
gpu_sum = gpu_a + gpu_b
CUDA.synchronize()
sum_observed = Array(gpu_sum)
sum_expected = mod.(host_a .+ host_b, 7)
sum_observed == sum_expected || error("CuModMatrix addition mismatch: expected $sum_expected, got $sum_observed")
println("CuModMatrix addition passed: $sum_observed")

# Exercise the current GPU-backed Karatsuba matrix product with a tiny exact
# integer case. The host calculation is the independent expected result.
host_left = Float64[21 32; 43 54]
host_right = Float64[15 26; 37 48]
left = GPUFiniteFieldMatrices.CuModMatrix{Float64}(host_left; modulus = 100)
right = GPUFiniteFieldMatrices.CuModMatrix{Float64}(host_right; modulus = 100)
left_k = GPUFiniteFieldMatrices.to_karatsuba(
    left;
    low_modulus = 10,
    high_modulus = 10,
    modulus = 100,
)
right_k = GPUFiniteFieldMatrices.to_karatsuba(
    right;
    low_modulus = 10,
    high_modulus = 10,
    modulus = 100,
)
product = left_k * right_k
CUDA.synchronize()
product_observed = Array(product)
product_expected = mod.(host_left * host_right, 100)
product_observed == product_expected ||
    error("Karatsuba matrix product mismatch: expected $product_expected, got $product_observed")
println("GPU Karatsuba product passed: $product_observed")
println("GPU smoke validation passed on $(CUDA.name(device)).")
