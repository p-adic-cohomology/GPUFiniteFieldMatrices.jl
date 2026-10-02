using JET
using GPUFiniteFieldMatrices

target = isempty(ARGS) ? "." : ARGS[1]
println("Running JET report_package for GPUFiniteFieldMatrices (TARGET=$(target))")
report_package(GPUFiniteFieldMatrices; target_modules = (GPUFiniteFieldMatrices,))
