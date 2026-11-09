module GFMPerf

using BenchmarkTools
using CUDA
using Dates
using GPUFiniteFieldMatrices
using JSON3
using LinearAlgebra
using Pkg
using Random
using SHA
using Statistics
using TOML
using UUIDs

include("GFMPerfCore.jl")
using .GFMPerfCore

include("capture.jl")
include("compare.jl")

export capture, compare, confirm, rerun_flagged, quick_cases, full_cases

end # module
