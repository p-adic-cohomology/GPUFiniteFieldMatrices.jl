_karatsuba_logical_view(A::GPUModArray) = _logical_view(parent(A), size(A))
_karatsuba_logical_view(A::AbstractArray) = A

_karatsuba_native(A::GPUModArray) = parent(A)
_karatsuba_native(A::AbstractArray) = A

# Scope native allocations and launches to the device that owns the data.
function _on_karatsuba_device(f::F, native) where {F}
    root = _elementwise_native_root(native)
    backend = KernelInterface.get_backend(root)
    target = KernelInterface.device(backend, root)
    previous = KernelInterface.device(backend)
    target == previous || KernelInterface.device!(backend, target)
    try
        return f(backend, root)
    finally
        target == previous || KernelInterface.device!(backend, previous)
    end
end

function _karatsuba_device_plan(source, radix::Int)
    native = _karatsuba_native(source)
    _validate_backend_storage(_storage_backend(native), eltype(source), radix)
    return _on_karatsuba_device(native) do _, root
        plan = similar(root, eltype(source), _padded_size(size(source)))
        fill!(plan, zero(eltype(source)))
        GPUModArray(plan; modulus = radix, new_size = size(source))
    end
end

function _karatsuba_check_device_conversion(::Type{T}, backend) where {T}
    (isbitstype(T) && T <: Union{Signed,Unsigned,AbstractFloat} && sizeof(T) <= 8) || throw(
        ArgumentError(
            "Karatsuba GPU conversion requires integer or floating storage up to 64 bits",
        ),
    )
    backend === Val(:metal) &&
        T === Float64 &&
        throw(ArgumentError("Metal Karatsuba conversion does not support Float64 output"))
    return nothing
end

_karatsuba_split_component(::AbstractArray, native, dims, modulus) = native

function _karatsuba_split_component(source::GPUModArray, native, dims, modulus)
    storage = _karatsuba_split_storage(getfield(source, :storage), native)
    return GPUModArray{eltype(native),ndims(native),typeof(storage)}(storage, dims, modulus)
end

_karatsuba_split_storage(::CuStorage{S,N,M}, native::AbstractArray{T,D}) where {S,N,M,T,D} =
    CuStorage{T,D,M,typeof(native)}(native)
_karatsuba_split_storage(
    ::MtlStorage{S,N,M},
    native::AbstractArray{T,D},
) where {S,N,M,T,D} = MtlStorage{T,D,M,typeof(native)}(native)
_karatsuba_split_storage(::AbstractArray, native) = native

function _karatsuba_split_device(::Type{T}, A, N1::Int, N2::Int, M::Int) where {T}
    native = _karatsuba_native(A)
    selected = _storage_backend(native)
    _karatsuba_check_device_conversion(T, selected)
    _karatsuba_check_device_conversion(eltype(A), selected)
    wrapped = A isa GPUModArray
    if wrapped
        _validate_backend_storage(selected, T, N1)
        _validate_backend_storage(selected, T, N2)
    end

    dims = size(A)
    result_dims = wrapped ? _padded_size(dims) : dims
    low_storage, high_storage = _on_karatsuba_device(native) do backend, root
        low = similar(root, T, result_dims)
        high = similar(root, T, result_dims)
        if wrapped
            fill!(low, zero(T))
            fill!(high, zero(T))
        end
        if !isempty(A)
            valid = similar(root, Bool, (length(A),))
            _karatsuba_split_kernel!(backend, _ELEMENTWISE_WORKGROUP_SIZE)(
                _logical_view(low, dims),
                _logical_view(high, dims),
                valid,
                _karatsuba_logical_view(A),
                UInt64(N1);
                ndrange = length(A),
            )
            all(valid) || throw(InexactError(:to_karatsuba, T, "inexact GPU split"))
        end
        return low, high
    end
    low_component = _karatsuba_split_component(A, low_storage, dims, N1)
    high_component = _karatsuba_split_component(A, high_storage, dims, N2)
    return KaratsubaArray(
        low_component,
        high_component;
        low_modulus = N1,
        high_modulus = N2,
        modulus = M,
    )
end

function _karatsuba_check_device(A, B)
    root_A = _elementwise_native_root(_karatsuba_native(A))
    root_B = _elementwise_native_root(_karatsuba_native(B))
    selected = _storage_backend(root_A)
    selected !== nothing && selected === _storage_backend(root_B) ||
        throw(ArgumentError("Karatsuba operands must use the same GPU backend and device"))
    backend = KernelInterface.get_backend(root_A)
    KernelInterface.device(backend, root_A) == KernelInterface.device(backend, root_B) ||
        throw(ArgumentError("Karatsuba operands must use the same GPU backend and device"))
    return nothing
end

function _karatsuba_reconstruct_device(::Type{T}, K::KaratsubaArray) where {T}
    low, high = low_part(K), high_part(K)
    native = _karatsuba_native(low)
    selected = _storage_backend(native)
    _karatsuba_check_device_conversion(T, selected)
    _karatsuba_check_device_conversion(eltype(K), selected)
    _karatsuba_check_device(low, high)
    radix = low_modulus(K)
    if T <: AbstractFloat
        converted = T(radix)
        isfinite(converted) && BigInt(converted) == radix ||
            throw(ArgumentError("low_modulus must be exactly representable in $T"))
    end
    return _on_karatsuba_device(native) do backend, root
        result = similar(root, T, size(K))
        if !isempty(K)
            valid = similar(root, Bool, (length(K),))
            _karatsuba_reconstruct_kernel!(backend, _ELEMENTWISE_WORKGROUP_SIZE)(
                result,
                valid,
                _karatsuba_logical_view(low),
                _karatsuba_logical_view(high),
                radix;
                ndrange = length(K),
            )
            all(valid) ||
                throw(InexactError(:from_karatsuba, T, "inexact GPU reconstruction"))
        end
        return result
    end
end

function _karatsuba_output_component(
    source::AbstractArray{T},
    dims::Dims,
    radix::Int,
) where {T}
    _storage_backend(source) === nothing && return zeros(T, dims)
    native = _karatsuba_native(source)
    wrapped = source isa GPUModArray
    wrapped && _validate_backend_storage(_storage_backend(native), T, radix)
    return _on_karatsuba_device(native) do _, root
        data = similar(root, T, wrapped ? _padded_size(dims) : dims)
        fill!(data, zero(T))
        _karatsuba_split_component(source, data, dims, radix)
    end
end

function _karatsuba_output_like(source::KaratsubaArray, dims::Dims)
    N1, N2, M = low_modulus(source), high_modulus(source), modulus(source)
    low = _karatsuba_output_component(low_part(source), dims, N1)
    high = _karatsuba_output_component(high_part(source), dims, N2)
    return KaratsubaArray(low, high; low_modulus = N1, high_modulus = N2, modulus = M)
end

function _karatsuba_copy_component(A::AbstractArray)
    _storage_backend(A) === nothing && return copy(A)
    return _on_karatsuba_device(_karatsuba_native(A)) do _, _
        copy(A)
    end
end

function Base.copy(A::KaratsubaArray)
    result = KaratsubaArray(
        _karatsuba_copy_component(low_part(A)),
        _karatsuba_copy_component(high_part(A));
        low_modulus = low_modulus(A),
        high_modulus = high_modulus(A),
        modulus = modulus(A),
    )
    A.plan === nothing || initialize_plan!(result)
    return result
end

function Base.copy!(dest::KaratsubaArray, src::KaratsubaArray)
    size(dest) == size(src) || throw(DimensionMismatch("Karatsuba dimensions must match"))
    dest === src && return dest
    to_low, to_high = low_part(dest), high_part(dest)
    from_low, from_high = low_part(src), high_part(src)
    _karatsuba_check_destination_alias(dest, src)

    both_gpu =
        _storage_backend(to_low) !== nothing && _storage_backend(from_low) !== nothing
    if both_gpu
        _karatsuba_check_device(to_low, to_high)
        _karatsuba_check_device(to_low, from_low)
        _karatsuba_check_device(to_low, from_high)
    end
    plan =
        src.plan !== nothing && dest.plan === nothing ? _new_karatsuba_plan(dest) :
        dest.plan
    if both_gpu
        _on_karatsuba_device(_karatsuba_native(to_low)) do _, _
            copy!(to_low, from_low)
            copy!(to_high, from_high)
        end
    else
        copy!(to_low, from_low)
        copy!(to_high, from_high)
    end
    dest.plan = plan
    return dest
end

function _karatsuba_check_matching_split(A, B)
    size(A) == size(B) || throw(DimensionMismatch("Karatsuba dimensions must match"))
    eltype(A) === eltype(B) ||
        throw(ArgumentError("Karatsuba component element types must match"))
    (low_modulus(A), high_modulus(A), modulus(A)) ==
    (low_modulus(B), high_modulus(B), modulus(B)) || throw(
        ModulusMismatchError("Karatsuba component and representation moduli must match"),
    )
    _karatsuba_check_device(low_part(A), low_part(B))
    _karatsuba_check_device(low_part(B), high_part(B))
    return nothing
end

function _karatsuba_check_destination_alias(C, A)
    low, high = _karatsuba_native(low_part(C)), _karatsuba_native(high_part(C))
    a_low, a_high = _karatsuba_native(low_part(A)), _karatsuba_native(high_part(A))
    _check_no_partial_alias(low, high)
    _check_elementwise_no_partial_alias(low, a_low)
    _check_elementwise_no_partial_alias(high, a_high)
    _check_no_partial_alias(low, a_high)
    _check_no_partial_alias(high, a_low)
    return nothing
end

@inline _karatsuba_int64_input(x::AbstractFloat) = _is_exact_integer_input(x)
@inline _karatsuba_int64_input(::Signed) = true
@inline _karatsuba_int64_input(x::Unsigned) = x <= typemax(Int64)

function _karatsuba_check_entries(A)
    for plane in (low_part(A), high_part(A))
        all(_karatsuba_int64_input, _karatsuba_logical_view(plane)) || throw(
            ArgumentError(
                "Karatsuba arithmetic requires finite integer components representable as Int64",
            ),
        )
    end
    return nothing
end

function _karatsuba_check_arithmetic(K::KaratsubaArray)
    T = eltype(K)
    selected = _storage_backend(low_part(K))
    _karatsuba_validate_component_type(T, selected)
    _karatsuba_check_device(low_part(K), high_part(K))
    _karatsuba_check_device_conversion(T, selected)
    _validate_storage_modulus(T, low_modulus(K))
    _validate_storage_modulus(T, high_modulus(K))
    return nothing
end

function _karatsuba_binary!(C::KaratsubaArray, A::KaratsubaArray, B::KaratsubaArray, op)
    _karatsuba_check_arithmetic(C)
    _karatsuba_check_matching_split(C, A)
    _karatsuba_check_matching_split(C, B)
    _karatsuba_check_destination_alias(C, A)
    _karatsuba_check_destination_alias(C, B)
    isempty(C) && return C
    _on_karatsuba_device(_karatsuba_native(low_part(C))) do backend, _
        _karatsuba_check_entries(A)
        A === B || _karatsuba_check_entries(B)
        _karatsuba_binary_kernel!(backend, _ELEMENTWISE_WORKGROUP_SIZE)(
            _karatsuba_logical_view(low_part(C)),
            _karatsuba_logical_view(high_part(C)),
            _karatsuba_logical_view(low_part(A)),
            _karatsuba_logical_view(high_part(A)),
            _karatsuba_logical_view(low_part(B)),
            _karatsuba_logical_view(high_part(B)),
            UInt64(low_modulus(C)),
            UInt64(high_modulus(C)),
            op;
            ndrange = length(C),
        )
    end
    return C
end

function _karatsuba_unary!(C::KaratsubaArray, A::KaratsubaArray, op, scalar)
    _karatsuba_check_arithmetic(C)
    _karatsuba_check_matching_split(C, A)
    _karatsuba_check_destination_alias(C, A)
    isempty(C) && return C
    _on_karatsuba_device(_karatsuba_native(low_part(C))) do backend, _
        _karatsuba_check_entries(A)
        _karatsuba_unary_kernel!(backend, _ELEMENTWISE_WORKGROUP_SIZE)(
            _karatsuba_logical_view(low_part(C)),
            _karatsuba_logical_view(high_part(C)),
            _karatsuba_logical_view(low_part(A)),
            _karatsuba_logical_view(high_part(A)),
            UInt64(low_modulus(C)),
            UInt64(high_modulus(C)),
            op,
            scalar;
            ndrange = length(C),
        )
    end
    return C
end

add!(C::KaratsubaArray, A::KaratsubaArray, B::KaratsubaArray) =
    _karatsuba_binary!(C, A, B, Val(:add))
sub!(C::KaratsubaArray, A::KaratsubaArray, B::KaratsubaArray) =
    _karatsuba_binary!(C, A, B, Val(:sub))
negate!(C::KaratsubaArray, A::KaratsubaArray) = _karatsuba_unary!(C, A, Val(:neg), nothing)

function scalar_multiply!(C::KaratsubaArray, A::KaratsubaArray, scalar::Number)
    scalar isa Real && isfinite(scalar) && isinteger(scalar) ||
        throw(ArgumentError("Karatsuba scalars must be finite real integers"))
    R, H = BigInt(low_modulus(A)), BigInt(high_modulus(A))
    high, low = fldmod(mod(BigInt(scalar), R * H), R)
    return _karatsuba_unary!(C, A, Val(:scale), (UInt64(low), UInt64(high)))
end

function Base.:+(A::KaratsubaArray, B::KaratsubaArray)
    _karatsuba_check_matching_split(A, B)
    return add!(_karatsuba_output_like(A, size(A)), A, B)
end

function Base.:-(A::KaratsubaArray, B::KaratsubaArray)
    _karatsuba_check_matching_split(A, B)
    return sub!(_karatsuba_output_like(A, size(A)), A, B)
end

Base.:*(scalar::Number, A::KaratsubaArray) =
    scalar_multiply!(_karatsuba_output_like(A, size(A)), A, scalar)
Base.:*(A::KaratsubaArray, scalar::Number) = scalar * A
