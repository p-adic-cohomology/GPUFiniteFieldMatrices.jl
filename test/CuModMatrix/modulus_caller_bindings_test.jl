function _modulus_call_targets!(targets, node)
    node isa Expr || return targets
    if node.head === :call
        target = first(node.args)
        qualified =
            target isa Expr &&
            target.head === :. &&
            last(target.args) isa QuoteNode &&
            last(target.args).value === :change_modulus!!
        if target === :change_modulus!! || qualified
            push!(targets, target)
        end
    end
    foreach(argument -> _modulus_call_targets!(targets, argument), node.args)
    return targets
end

@testset "Modulus GPU test callers resolve on CPU" begin
    source = read(joinpath(@__DIR__, "basic_operations_test.jl"), String)
    targets = _modulus_call_targets!(Any[], Meta.parseall(source))
    @test !isempty(targets)
    for target in targets
        @test Core.eval(@__MODULE__, target) === GPUFiniteFieldMatrices.change_modulus!!
    end
end
