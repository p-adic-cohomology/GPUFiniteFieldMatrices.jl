module BackendTestDriver

const OPTIONAL_BACKENDS = (:cuda, :metal)
const VENDOR_PACKAGES = Dict(:cuda => "CUDA", :metal => "Metal")

"""Return the optional vendors a default or explicitly selected test run provisions."""
function backends_to_provision(requested::Union{Nothing,Symbol})
    requested === nothing && return OPTIONAL_BACKENDS
    requested in OPTIONAL_BACKENDS || throw(ArgumentError("unsupported backend $requested"))
    return (requested,)
end

function parse_backend_selector(args::Vector{String})
    selectors = filter(arg -> startswith(arg, "--backend="), args)
    length(selectors) <= 1 || throw(ArgumentError("pass at most one --backend selector"))
    isempty(selectors) && return nothing

    value = split(only(selectors), "="; limit = 2)[2]
    backend = Symbol(value)
    backend in OPTIONAL_BACKENDS || throw(
        ArgumentError(
            "unsupported backend '$value'; choose --backend=cuda or --backend=metal",
        ),
    )
    return backend
end

function optional_backends(
    requested::Union{Nothing,Symbol},
    find_package = Base.find_package,
)
    requested === nothing && return Tuple(
        backend for backend in OPTIONAL_BACKENDS if
        find_package(VENDOR_PACKAGES[backend]) !== nothing
    )
    requested in OPTIONAL_BACKENDS || throw(ArgumentError("unsupported backend $requested"))
    name = VENDOR_PACKAGES[requested]
    find_package(name) === nothing && throw(
        ArgumentError(
            "requested backend $requested is unavailable because $name is not installed in the test environment",
        ),
    )
    return (requested,)
end

end
