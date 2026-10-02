using ReLint

include("common.jl")

target = parse_target(ARGS)
files = target_julia_files(target)
println("ReLint loaded for TARGET=$(target)")
println("Julia files discovered: $(length(files))")
for file in files
    println(file)
end
