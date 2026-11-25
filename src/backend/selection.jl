const _DEFAULT_BACKEND = Ref{Symbol}(:auto)
const _SCOPED_BACKEND = Base.ScopedValues.ScopedValue{Union{Nothing,Symbol}}(nothing)

_backend_available(::Val) = false
_backend_instance(::Val) = nothing

function _check_backend_symbol(backend::Symbol)
    backend in (:cuda, :metal) ||
        throw(ArgumentError("backend must be :cuda or :metal, got $backend"))
    return backend
end

_backend_symbol(::Val{:cuda}) = :cuda
_backend_symbol(::Val{:metal}) = :metal

function _auto_backend(cuda_available::Bool, metal_available::Bool)
    cuda_available && return :cuda
    metal_available && return :metal
    throw(ArgumentError("no functional CUDA or Metal backend is loaded"))
end

"""Set the process default GPU backend to `:cuda`, `:metal`, or automatic selection."""
function set_default_backend!(backend::Symbol)
    if backend === :auto
        _DEFAULT_BACKEND[] = :auto
        return backend
    end
    _check_backend_symbol(backend)
    _DEFAULT_BACKEND[] = backend
    return backend
end

"""Run `f` with a task-scoped GPU backend selection."""
function with_backend(f::F, backend::Symbol) where {F}
    _check_backend_symbol(backend)
    return Base.ScopedValues.with(f, _SCOPED_BACKEND => backend)
end

function default_backend()
    scoped = _SCOPED_BACKEND[]
    scoped === nothing || return scoped

    selected = _DEFAULT_BACKEND[]
    selected === :auto || return selected

    return _auto_backend(_backend_available(Val(:cuda)), _backend_available(Val(:metal)))
end

function _backend_object(backend::Symbol = default_backend())
    _check_backend_symbol(backend)
    _backend_available(Val(backend)) ||
        throw(ArgumentError("the selected $backend backend is not loaded and functional"))
    return _backend_instance(Val(backend))
end

function _require_backend(backend::Symbol)
    _check_backend_symbol(backend)
    _backend_available(Val(backend)) ||
        throw(ArgumentError("the selected $backend backend is not loaded and functional"))
    return backend
end
