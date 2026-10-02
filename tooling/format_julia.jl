import Pkg

const REQUIRED_FORMATTER_VERSION = v"2.14.0"

function ensure_formatter_version()
    installed_versions = [
        dependency.version for
        dependency in values(Pkg.dependencies()) if dependency.name == "JuliaFormatter"
    ]
    if length(installed_versions) != 1 ||
       only(installed_versions) != REQUIRED_FORMATTER_VERSION
        Pkg.add(
            Pkg.PackageSpec(name = "JuliaFormatter", version = REQUIRED_FORMATTER_VERSION),
        )
    end
end

ensure_formatter_version()

using JuliaFormatter

actual_version = Base.pkgversion(JuliaFormatter)
actual_version == REQUIRED_FORMATTER_VERSION ||
    error("Expected JuliaFormatter $(REQUIRED_FORMATTER_VERSION), found $(actual_version)")

isempty(ARGS) && error("JuliaFormatter hook requires selected tracked Julia filenames")
for path in ARGS
    endswith(path, ".jl") || error("Expected a Julia source file, got $(repr(path))")
    read(`git ls-files --error-unmatch -- $path`, String)
end

already_formatted = JuliaFormatter.format(
    ARGS,
    JuliaFormatter.DefaultStyle();
    throw_on_error = true,
    always_for_in = false,
    always_use_return = false,
    format_docstrings = false,
    import_to_using = false,
    long_to_short_function_def = false,
    pipe_to_function_call = false,
    short_to_long_function_def = false,
)

println("JuliaFormatter $(actual_version), default style")
exit(already_formatted ? 0 : 1)
