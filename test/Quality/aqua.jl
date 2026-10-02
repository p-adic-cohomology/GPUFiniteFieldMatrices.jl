using Aqua
using GPUFiniteFieldMatrices

target = isempty(ARGS) ? "." : ARGS[1]
println("Running Aqua for module GPUFiniteFieldMatrices (TARGET=$(target))")
Aqua.test_all(GPUFiniteFieldMatrices; stale_deps = false, deps_compat = false)
