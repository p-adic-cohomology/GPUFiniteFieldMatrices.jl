function _package_modules(root::Module)
    modules = Module[root]
    index = 1

    while index <= length(modules)
        parent = modules[index]
        for name in names(parent; all = true, imported = true)
            isdefined(parent, name) || continue
            child = getfield(parent, name)
            child isa Module || continue
            parentmodule(child) === parent || continue
            child in modules && continue
            push!(modules, child)
        end
        index += 1
    end

    return modules
end

function _collect_globalrefs!(globalrefs, value)
    if value isa Core.GlobalRef
        push!(globalrefs, value)
    elseif value isa Core.CodeInfo
        foreach(item -> _collect_globalrefs!(globalrefs, item), value.code)
    elseif value isa Expr
        value.head === :quote && return globalrefs
        foreach(item -> _collect_globalrefs!(globalrefs, item), value.args)
        return globalrefs
    elseif value isa Tuple || value isa AbstractArray
        foreach(item -> _collect_globalrefs!(globalrefs, item), value)
    end

    return globalrefs
end

function _missing_cuda_global_bindings(root::Module, cuda::Module)
    cuda_names = Set(names(cuda; all = true, imported = true))
    scanned_methods = Set{Method}()
    missing = Set{String}()

    for mod in _package_modules(root)
        for name in names(mod; all = true, imported = true)
            isdefined(mod, name) || continue
            value = getfield(mod, name)
            value isa Function || continue

            for method in methods(value)
                method.module === mod || continue
                method in scanned_methods && continue
                push!(scanned_methods, method)

                codeinfo = Base.uncompressed_ast(method)
                codeinfo === nothing && continue
                globalrefs = Core.GlobalRef[]
                _collect_globalrefs!(globalrefs, codeinfo)
                for globalref in globalrefs
                    globalref.mod === mod || continue
                    globalref.name in cuda_names || continue
                    isdefined(mod, globalref.name) && continue

                    push!(
                        missing,
                        "$(mod).$(nameof(value)) at $(method.file):$(method.line) " *
                        "references unbound CUDA name $(globalref.name)",
                    )
                end
            end
        end
    end

    return sort!(collect(missing))
end

@testset "CUDA kernel globals resolve on CPU" begin
    missing = _missing_cuda_global_bindings(GPUFiniteFieldMatrices, CUDA)
    isempty(missing) || @error "Unbound CUDA globals in package methods" missing
    @test isempty(missing)
end
