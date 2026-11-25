function _karatsuba_metal_wrapper(::Type{T}, ::Val{D}, ::Type{Mode}) where {T,D,Mode}
    return GPUModArray{T,D,FFM.MtlStorage{T,D,Mode,Metal.MtlArray{T,D,Mode}}}
end

function FFM._backend_default_component_type(
    ::Val{:metal},
    ::Type{T},
    ::Val{D},
    storage,
) where {T,D}
    FFM._karatsuba_validate_component_type(T, Val(:metal))
    mode = _metal_mode(storage)
    return _karatsuba_metal_wrapper(
        T,
        Val(D),
        mode === :default ? Metal.DefaultStorageMode : mode,
    )
end

function FFM._karatsuba_resolve_requested_component_type(
    ::Type{MtlModArray{T,D,Mode}},
    ::Type{T},
    ::Val{D},
    storage,
) where {T,D,Mode}
    FFM._karatsuba_validate_component_type(T, Val(:metal))
    mode = _metal_mode(storage)
    mode === :default ||
        mode === Mode ||
        throw(
            ArgumentError(
                "requested Metal storage mode does not match the Karatsuba component type",
            ),
        )
    return _karatsuba_metal_wrapper(T, Val(D), _checked_metal_mode(Mode))
end

FFM._karatsuba_plan_type(::Type{<:MtlModArray{T,D,Mode}}) where {T,D,Mode} =
    _karatsuba_metal_wrapper(T, Val(D), Mode)
FFM._karatsuba_plan_type(::Type{Metal.MtlArray{T,D,Mode}}) where {T,D,Mode} =
    _karatsuba_metal_wrapper(T, Val(D), Mode)

function FFM._karatsuba_zero_component(
    ::Type{S},
    dims::Dims{D},
    component_modulus::Int;
    storage,
    device,
) where {T,D,Mode,S<:MtlModArray{T,D,Mode}}
    FFM._karatsuba_validate_component_type(T, Val(:metal))
    requested = _metal_mode(storage)
    requested === :default ||
        requested === Mode ||
        throw(
            ArgumentError(
                "requested Metal storage mode does not match the Karatsuba component type",
            ),
        )
    S === _karatsuba_metal_wrapper(T, Val(D), Mode) || throw(
        ArgumentError("Karatsuba zero factories require dense Metal component storage"),
    )
    return zeros(
        MtlModArray{T,D},
        dims...;
        modulus = component_modulus,
        storage = Mode,
        device,
    )
end

function FFM._karatsuba_zero_component(
    ::Type{Metal.MtlArray{T,D,Mode}},
    dims::Dims{D},
    ::Int;
    storage,
    device,
) where {T,D,Mode}
    FFM._karatsuba_validate_component_type(T, Val(:metal))
    requested = _metal_mode(storage)
    requested === :default ||
        requested === Mode ||
        throw(
            ArgumentError(
                "requested Metal storage mode does not match the Karatsuba component type",
            ),
        )
    Metal.functional() ||
        throw(ArgumentError("the selected Metal backend is not functional"))
    return _with_metal_device(device) do
        _metal_zeros(T, dims, _checked_metal_mode(Mode))
    end
end
