using CUDA
using GPUFiniteFieldMatrices
using UUIDs

const EXPECTED_JULIA_VERSION = v"1.12.4"
const PACKAGE_UUID = UUID("465031f1-b347-4f72-85f6-baf667cdf853")

VERSION == EXPECTED_JULIA_VERSION || error("GPU smoke requires Julia $EXPECTED_JULIA_VERSION, found $VERSION")
repo = realpath(ENV["OSCAR_GFM_REPO"])
image = realpath(ENV["OSCAR_SYSIMAGE_FILE"])
realpath(Base.active_project()) == realpath(joinpath(repo, "Project.toml")) ||
    error("active project is not the requested package checkout: $(Base.active_project())")
realpath(unsafe_string(Base.JLOptions().image_file)) == image ||
    error("Julia did not load the requested sysimage $image")
realpath(pathof(GPUFiniteFieldMatrices)) == realpath(joinpath(repo, "src", "GPUFiniteFieldMatrices.jl")) ||
    error("GPUFiniteFieldMatrices did not load from $repo/src")

println("Julia version: $VERSION")
println("Active project: $(realpath(Base.active_project()))")
println("Loaded sysimage: $image")
println("Package source: $(realpath(pathof(GPUFiniteFieldMatrices)))")
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
