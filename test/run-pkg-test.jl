using Pkg
using TOML

package_view = dirname(Base.active_project())
test_directory = joinpath(package_view, "test")
test_project_path = joinpath(test_directory, "Project.toml")
include(joinpath(test_directory, "backend_test_driver.jl"))
dependencies =
    TOML.parsefile(joinpath(test_directory, "backend_optional_dependencies.toml"))
backend = get(ENV, "GPUFFM_TEST_BACKEND", "")
backend_request = BackendTestDriver.parse_backend_selector(
    isempty(backend) ? String[] : ["--backend=$backend"],
)
selected = BackendTestDriver.backends_to_provision(backend_request)
test_project = TOML.parsefile(test_project_path)

for name in selected
    dependency = dependencies[string(name)]
    test_project["deps"][dependency["name"]] = dependency["uuid"]
    test_project["compat"][dependency["name"]] = "=" * dependency["version"]
end

open(test_project_path, "w") do io
    TOML.print(io, test_project; sorted = true)
end

@info "Fresh package test setup" project=Base.active_project() selected_optional_vendors=selected source=realpath(
    joinpath(package_view, "src", "GPUFiniteFieldMatrices.jl"),
)
Pkg.instantiate()
test_args = backend_request === nothing ? String[] : ["--backend=$backend"]
Pkg.test(; test_args)
