# Floating GPU storage represents integer residues. Convert those residues to
# Int64 and use overflow-safe UInt64 modular operations so every intermediate
# is exact up to the Float64 storage ceiling (2^53). In particular, Float32
# Metal kernels never promote arithmetic to Float64.
const _ELEMENTWISE_WORKGROUP_SIZE = 256

Base.@noinline function _throw_elementwise_size_mismatch()
    throw(
        DimensionMismatch(
            "Elementwise arrays must have equal logical sizes and backing lengths",
        ),
    )
end

Base.@noinline function _throw_elementwise_modulus_mismatch()
    throw(ModulusMismatchError("Elementwise arrays must have the same modulus N"))
end

function _validate_positive_modulus_override(modulus_override::Nothing)
    return nothing
end

function _validate_positive_modulus_override(modulus_override::Integer)
    modulus_override > 0 ||
        throw(ArgumentError("modulus_override must be positive, got $modulus_override"))
    modulus_override <= typemax(Int) ||
        throw(OverflowError("modulus_override $modulus_override does not fit Int"))
    return Int(modulus_override)
end

@inline function _check_elementwise_metadata(
    size_C::Dims,
    size_A::Dims,
    size_B::Dims,
    modulus_C::Int,
    modulus_A::Int,
    modulus_B::Int,
    length_C::Int,
    length_A::Int,
    length_B::Int,
)
    size_C == size_A == size_B || _throw_elementwise_size_mismatch()
    modulus_C == modulus_A == modulus_B || _throw_elementwise_modulus_mismatch()
    length_C == length_A == length_B || _throw_elementwise_size_mismatch()
    return nothing
end

@inline function _check_elementwise_metadata(
    size_C::Dims,
    size_A::Dims,
    modulus_C::Int,
    modulus_A::Int,
    length_C::Int,
    length_A::Int,
)
    size_C == size_A || _throw_elementwise_size_mismatch()
    modulus_C == modulus_A || _throw_elementwise_modulus_mismatch()
    length_C == length_A || _throw_elementwise_size_mismatch()
    return nothing
end

@inline function _check_elementwise_storage(
    size_C::Dims,
    size_A::Dims,
    length_C::Int,
    length_A::Int,
)
    size_C == size_A || _throw_elementwise_size_mismatch()
    length_C == length_A || _throw_elementwise_size_mismatch()
    return nothing
end

@inline function _check_elementwise_backend(A, B)
    native_A = parent(A)
    native_B = parent(B)
    _storage_backend(native_A) === nothing && throw(
        ArgumentError("Elementwise modular kernels require CUDA or Metal storage"),
    )
    _storage_backend(native_B) === nothing && throw(
        ArgumentError("Elementwise modular kernels require CUDA or Metal storage"),
    )
    interface_A = _elementwise_interface_storage(A, native_A)
    interface_B = _elementwise_interface_storage(B, native_B)
    backend_A = KernelInterface.get_backend(interface_A)
    backend_B = KernelInterface.get_backend(interface_B)
    typeof(backend_A) === typeof(backend_B) ||
        throw(ArgumentError("Elementwise arrays must use the same backend and device"))
    KernelInterface.device(backend_A, interface_A) ==
    KernelInterface.device(backend_B, interface_B) ||
        throw(ArgumentError("Elementwise arrays must use the same backend and device"))
    return nothing
end

@inline _elementwise_interface_storage(A::GPUModArray, native) = getfield(A, :storage)
@inline _elementwise_interface_storage(A, native) = _elementwise_native_root(native)

@inline function _elementwise_native_root(array::AbstractArray)
    native = parent(array)
    native === array && return array
    return _elementwise_native_root(native)
end

@inline function _check_elementwise_no_partial_alias(C::GPUModArray, A::GPUModArray)
    parent(C) === parent(A) && return nothing
    return _check_no_partial_alias(parent(C), parent(A))
end

@inline function _check_elementwise_no_partial_alias(C::AbstractArray, A::AbstractArray)
    C === A && return nothing
    return _check_no_partial_alias(C, A)
end

@inline function _check_elementwise_no_partial_alias(C::GPUModArray, A::AbstractArray)
    parent(C) === A && return nothing
    return _check_no_partial_alias(parent(C), A)
end

@inline function _check_no_partial_alias(C, A)
    Base.mightalias(C, A) && throw(
        ArgumentError(
            "Elementwise destination may alias an input only when both use the same storage object",
        ),
    )
    return nothing
end

@inline function _check_elementwise_compatible(
    C::GPUModArray,
    A::GPUModArray,
    B::GPUModArray,
)
    _check_elementwise_metadata(
        size(C),
        size(A),
        size(B),
        modulus(C),
        modulus(A),
        modulus(B),
        length(parent(C)),
        length(parent(A)),
        length(parent(B)),
    )
    _check_elementwise_backend(C, A)
    _check_elementwise_backend(C, B)
    _check_elementwise_no_partial_alias(C, A)
    _check_elementwise_no_partial_alias(C, B)
    return nothing
end

@inline function _check_elementwise_compatible(C::GPUModArray, A::GPUModArray)
    _check_elementwise_metadata(
        size(C),
        size(A),
        modulus(C),
        modulus(A),
        length(parent(C)),
        length(parent(A)),
    )
    _check_elementwise_backend(C, A)
    _check_elementwise_no_partial_alias(C, A)
    return nothing
end

@inline function _check_elementwise_storage_compatible(C::GPUModArray, A::GPUModArray)
    _check_elementwise_storage(size(C), size(A), length(parent(C)), length(parent(A)))
    _check_elementwise_backend(C, A)
    _check_elementwise_no_partial_alias(C, A)
    return nothing
end

@inline function _check_elementwise_allocating_metadata(
    size_A::Dims,
    size_B::Dims,
    modulus_A::Int,
    modulus_B::Int,
    length_A::Int,
    length_B::Int,
)
    size_A == size_B || _throw_elementwise_size_mismatch()
    modulus_A == modulus_B || _throw_elementwise_modulus_mismatch()
    length_A == length_B || _throw_elementwise_size_mismatch()
    length_A == prod(_padded_size(size_A)) || _throw_elementwise_size_mismatch()
    return nothing
end

@inline function _check_elementwise_allocating_input(size_A::Dims, length_A::Int)
    length_A == prod(_padded_size(size_A)) || _throw_elementwise_size_mismatch()
    return nothing
end

@inline function _check_elementwise_allocating_input(A::GPUModArray)
    _check_elementwise_allocating_input(size(A), length(parent(A)))
    _check_elementwise_backend(A, A)
    return nothing
end

@inline function _check_elementwise_allocating_inputs(A::GPUModArray, B::GPUModArray)
    _check_elementwise_allocating_metadata(
        size(A),
        size(B),
        modulus(A),
        modulus(B),
        length(parent(A)),
        length(parent(B)),
    )
    _check_elementwise_backend(A, B)
    return nothing
end

function _validate_elementwise_modulus(::Type{T}, storage, modulus_value::Integer) where {T}
    _validate_backend_storage(_storage_backend(storage), T, modulus_value)
    return _validate_storage_modulus(T, modulus_value)
end

function _elementwise_working_modulus(A::GPUModArray{T}, modulus_value::Integer) where {T}
    return _validate_elementwise_modulus(T, parent(A), modulus_value)
end

function _elementwise_working_modulus(
    A::GPUModArray,
    modulus_override::Union{Nothing,Integer},
)
    checked_override = _validate_positive_modulus_override(modulus_override)
    selected = checked_override === nothing ? modulus(A) : checked_override
    return _elementwise_working_modulus(A, selected)
end

function _normalize_elementwise_scalar(::Type{T}, scalar::Number, modulus_value::Int) where {T<:AbstractFloat}
    scalar isa Real || throw(ArgumentError("finite-field scalars must be real integers"))
    if scalar isa AbstractFloat
        (isfinite(scalar) && isinteger(scalar)) ||
            throw(ArgumentError("finite-field scalars must be finite integer values"))
    end
    residue = mod(BigInt(scalar), BigInt(modulus_value))
    return UInt64(residue)
end

@inline function _is_exact_integer_input(value::T) where {T<:AbstractFloat}
    return isfinite(value) && trunc(value) == value &&
           value >= T(typemin(Int64)) && value < T(typemax(Int64))
end

function _validate_elementwise_values(A::GPUModArray)
    values = _logical_view(parent(A), size(A))
    all(_is_exact_integer_input, values) || throw(
        ArgumentError(
            "elementwise arithmetic requires finite integer entries representable as Int64",
        ),
    )
    return nothing
end

@inline _residue_u64(value::T, modulus_value::Int64) where {T<:AbstractFloat} =
    UInt64(mod(Int64(value), modulus_value))

@inline function _add_mod_u64(a::UInt64, b::UInt64, modulus_value::UInt64)
    distance_to_wrap = modulus_value - b
    return a >= distance_to_wrap ? a - distance_to_wrap : a + b
end

@inline function _sub_mod_u64(a::UInt64, b::UInt64, modulus_value::UInt64)
    return a >= b ? a - b : modulus_value - (b - a)
end

@inline function _neg_mod_u64(a::UInt64, modulus_value::UInt64)
    return iszero(a) ? zero(UInt64) : modulus_value - a
end

@inline function _mul_mod_u64(a::UInt64, b::UInt64, modulus_value::UInt64)
    result = zero(UInt64)
    addend = a
    multiplier = b
    while !iszero(multiplier)
        isodd(multiplier) && (result = _add_mod_u64(result, addend, modulus_value))
        multiplier >>= 1
        iszero(multiplier) ||
            (addend = _add_mod_u64(addend, addend, modulus_value))
    end
    return result
end

@inline _binary_integer(::Val{:add}, a, b, m) = _add_mod_u64(a, b, m)
@inline _binary_integer(::Val{:sub}, a, b, m) = _sub_mod_u64(a, b, m)
@inline _binary_integer(::Val{:mul}, a, b, m) = _mul_mod_u64(a, b, m)

@inline function _scalar_integer(::Val{:add}, a, b, m)
    return _add_mod_u64(a, b, m)
end
@inline function _scalar_integer(::Val{:sub}, a, b, m)
    return _sub_mod_u64(a, b, m)
end
@inline function _scalar_integer(::Val{:rsub}, a, b, m)
    return _sub_mod_u64(b, a, m)
end
@inline function _scalar_integer(::Val{:mul}, a, b, m)
    return _mul_mod_u64(a, b, m)
end
@inline function _scalar_integer(::Val{:neg}, a, b, m)
    return _neg_mod_u64(a, m)
end
@inline function _scalar_integer(::Val{:mod}, a, b, m)
    return a
end

@kernel function _elementwise_binary_kernel!(
    output::AbstractArray{T},
    left,
    right,
    modulus_i::Int64,
    modulus_u::UInt64,
    operation,
) where {T}
    index = @index(Global, Linear)
    if index <= length(output)
        a = @inbounds left[index]
        b = @inbounds right[index]
        result = _binary_integer(
            operation,
            _residue_u64(a, modulus_i),
            _residue_u64(b, modulus_i),
            modulus_u,
        )
        @inbounds output[index] = convert(T, result)
    end
    return
end

@kernel function _elementwise_scalar_kernel!(
    output::AbstractArray{T},
    input,
    modulus_i::Int64,
    modulus_u::UInt64,
    scalar_u::UInt64,
    operation,
) where {T}
    index = @index(Global, Linear)
    if index <= length(output)
        value = @inbounds input[index]
        result = _scalar_integer(
            operation,
            _residue_u64(value, modulus_i),
            scalar_u,
            modulus_u,
        )
        @inbounds output[index] = convert(T, result)
    end
    return
end

function _launch_binary_arrays!(output, left, right, modulus_value::Int, operation, threads::Int)
    count = length(output)
    count == 0 && return output
    T = eltype(output)
    modulus_i = Int64(modulus_value)
    modulus_u = UInt64(modulus_i)
    backend = KernelInterface.get_backend(output)
    kernel = _elementwise_binary_kernel!(backend, threads)
    kernel(output, left, right, modulus_i, modulus_u, operation; ndrange = count)
    return output
end

function _launch_scalar_arrays!(
    output,
    input,
    modulus_value::Int,
    scalar_u::UInt64,
    operation,
    threads::Int,
)
    count = length(output)
    count == 0 && return output
    T = eltype(output)
    modulus_i = Int64(modulus_value)
    modulus_u = UInt64(modulus_i)
    backend = KernelInterface.get_backend(output)
    kernel = _elementwise_scalar_kernel!(backend, threads)
    kernel(
        output,
        input,
        modulus_i,
        modulus_u,
        scalar_u,
        operation;
        ndrange = count,
    )
    return output
end

@inline function _validate_elementwise_threads(threads::Int)
    threads > 0 || throw(ArgumentError("threads must be a positive integer"))
    return threads
end

function _launch_binary!(C::GPUModArray, A::GPUModArray, B::GPUModArray, m, op, threads)
    output = _logical_view(parent(C), size(C))
    left = _logical_view(parent(A), size(A))
    right = _logical_view(parent(B), size(B))
    _launch_binary_arrays!(output, left, right, m, op, threads)
    return C
end

function _launch_scalar!(C::GPUModArray, A::GPUModArray, m, scalar_u, op, threads)
    output = _logical_view(parent(C), size(C))
    input = _logical_view(parent(A), size(A))
    _launch_scalar_arrays!(output, input, m, scalar_u, op, threads)
    return C
end

function _launch_mod_arrays!(output, input, modulus_value::Int, threads::Int)
    T = eltype(output)
    _launch_scalar_arrays!(
        output,
        input,
        modulus_value,
        zero(UInt64),
        Val(:mod),
        threads,
    )
    return output
end

function _launch_mod!(C::GPUModArray, A::GPUModArray, modulus_value::Int, threads::Int)
    output = _logical_view(parent(C), size(C))
    input = _logical_view(parent(A), size(A))
    _launch_mod_arrays!(output, input, modulus_value, threads)
    return C
end

function _check_raw_array_compatible(C::AbstractArray, A::AbstractArray)
    size(C) == size(A) || throw(DimensionMismatch("Array dimensions must match"))
    _check_elementwise_backend(C, A)
    _check_elementwise_no_partial_alias(C, A)
    return nothing
end

function mod!(C::GPUModArray, A::GPUModArray, modulus_value::Integer; threads::Int = _ELEMENTWISE_WORKGROUP_SIZE)
    _check_elementwise_compatible(C, A)
    _validate_elementwise_values(A)
    checked_modulus = _elementwise_working_modulus(C, modulus_value)
    checked_threads = _validate_elementwise_threads(threads)
    return _launch_mod!(C, A, checked_modulus, checked_threads)
end

function mod!(C::AbstractArray{T,D}, A::AbstractArray{T,D}, modulus_value::Integer; threads::Int = _ELEMENTWISE_WORKGROUP_SIZE) where {T<:AbstractFloat,D}
    _check_raw_array_compatible(C, A)
    _check_elementwise_backend(C, A)
    all(_is_exact_integer_input, A) || throw(
        ArgumentError("elementwise arithmetic requires finite integer entries representable as Int64"),
    )
    checked_modulus = _validate_elementwise_modulus(T, C, modulus_value)
    checked_threads = _validate_elementwise_threads(threads)
    _launch_mod_arrays!(C, A, checked_modulus, checked_threads)
    return C
end

function _elementwise_binary!(
    C::GPUModArray,
    A::GPUModArray,
    B::GPUModArray,
    operation,
    modulus_override,
    threads::Int,
)
    _validate_positive_modulus_override(modulus_override)
    _check_elementwise_compatible(C, A, B)
    _validate_elementwise_values(A)
    _validate_elementwise_values(B)
    checked_modulus = _elementwise_working_modulus(C, modulus_override)
    checked_threads = _validate_elementwise_threads(threads)
    return _launch_binary!(C, A, B, checked_modulus, operation, checked_threads)
end

function add!(
    C::GPUModArray,
    A::GPUModArray,
    B::GPUModArray;
    modulus_override::Union{Nothing,Integer} = nothing,
    threads::Int = _ELEMENTWISE_WORKGROUP_SIZE,
)
    return _elementwise_binary!(C, A, B, Val(:add), modulus_override, threads)
end

function sub!(
    C::GPUModArray,
    A::GPUModArray,
    B::GPUModArray;
    modulus_override::Union{Nothing,Integer} = nothing,
    threads::Int = _ELEMENTWISE_WORKGROUP_SIZE,
)
    return _elementwise_binary!(C, A, B, Val(:sub), modulus_override, threads)
end

function mul_elementwise!(
    C::GPUModArray,
    A::GPUModArray,
    B::GPUModArray;
    modulus_override::Union{Nothing,Integer} = nothing,
    threads::Int = _ELEMENTWISE_WORKGROUP_SIZE,
)
    return _elementwise_binary!(C, A, B, Val(:mul), modulus_override, threads)
end

function elementwise_multiply!(
    C::GPUModArray,
    A::GPUModArray,
    B::GPUModArray;
    modulus_override::Union{Nothing,Integer} = nothing,
    threads::Int = _ELEMENTWISE_WORKGROUP_SIZE,
)
    return mul_elementwise!(C, A, B; modulus_override, threads)
end

function _elementwise_scalar!(
    C::GPUModArray{T},
    A::GPUModArray,
    scalar::Number,
    operation,
    modulus_override,
    threads::Int,
) where {T}
    _validate_positive_modulus_override(modulus_override)
    _check_elementwise_compatible(C, A)
    _validate_elementwise_values(A)
    checked_modulus = _elementwise_working_modulus(C, modulus_override)
    scalar_u = _normalize_elementwise_scalar(T, scalar, checked_modulus)
    checked_threads = _validate_elementwise_threads(threads)
    return _launch_scalar!(C, A, checked_modulus, scalar_u, operation, checked_threads)
end

function scalar_add!(
    C::GPUModArray,
    A::GPUModArray,
    scalar::Number;
    modulus_override::Union{Nothing,Integer} = nothing,
    threads::Int = _ELEMENTWISE_WORKGROUP_SIZE,
)
    return _elementwise_scalar!(C, A, scalar, Val(:add), modulus_override, threads)
end

function scalar_sub!(
    C::GPUModArray,
    A::GPUModArray,
    scalar::Number;
    modulus_override::Union{Nothing,Integer} = nothing,
    threads::Int = _ELEMENTWISE_WORKGROUP_SIZE,
)
    return _elementwise_scalar!(C, A, scalar, Val(:sub), modulus_override, threads)
end

function rscalar_sub!(
    C::GPUModArray,
    A::GPUModArray,
    scalar::Number;
    modulus_override::Union{Nothing,Integer} = nothing,
    threads::Int = _ELEMENTWISE_WORKGROUP_SIZE,
)
    return _elementwise_scalar!(C, A, scalar, Val(:rsub), modulus_override, threads)
end

function rscalar_sub!(C::AbstractArray{T,D}, A::AbstractArray{T,D}, scalar::Number, modulus_value::Integer; threads::Int = _ELEMENTWISE_WORKGROUP_SIZE) where {T<:AbstractFloat,D}
    _check_raw_array_compatible(C, A)
    _check_elementwise_backend(C, A)
    all(_is_exact_integer_input, A) || throw(
        ArgumentError("elementwise arithmetic requires finite integer entries representable as Int64"),
    )
    checked_modulus = _validate_elementwise_modulus(T, C, modulus_value)
    scalar_u = _normalize_elementwise_scalar(T, scalar, checked_modulus)
    checked_threads = _validate_elementwise_threads(threads)
    _launch_scalar_arrays!(C, A, checked_modulus, scalar_u, Val(:rsub), checked_threads)
    return C
end

function neg!(
    C::GPUModArray,
    A::GPUModArray;
    modulus_override::Union{Nothing,Integer} = nothing,
    threads::Int = _ELEMENTWISE_WORKGROUP_SIZE,
)
    return _elementwise_scalar!(C, A, 0, Val(:neg), modulus_override, threads)
end

function div!(
    C::GPUModArray{T},
    A::GPUModArray,
    scalar::Integer;
    modulus_override::Union{Nothing,Integer} = nothing,
    threads::Int = _ELEMENTWISE_WORKGROUP_SIZE,
) where {T}
    _validate_positive_modulus_override(modulus_override)
    _check_elementwise_compatible(C, A)
    _validate_elementwise_values(A)
    checked_modulus = _elementwise_working_modulus(C, modulus_override)
    inverse = mod_inv(BigInt(scalar), BigInt(checked_modulus))
    scalar_u = _normalize_elementwise_scalar(T, inverse, checked_modulus)
    checked_threads = _validate_elementwise_threads(threads)
    return _launch_scalar!(C, A, checked_modulus, scalar_u, Val(:mul), checked_threads)
end

function LinearAlgebra.mul!(
    C::GPUModArray,
    A::GPUModArray,
    scalar::Number;
    modulus_override::Union{Nothing,Integer} = nothing,
    threads::Int = _ELEMENTWISE_WORKGROUP_SIZE,
)
    return _elementwise_scalar!(C, A, scalar, Val(:mul), modulus_override, threads)
end

scalar_multiply!(C::GPUModArray, A::GPUModArray, scalar::Number; kwargs...) =
    LinearAlgebra.mul!(C, A, scalar; kwargs...)

function LinearAlgebra.rmul!(A::GPUModArray, scalar::Number)
    return LinearAlgebra.mul!(A, A, scalar)
end

function LinearAlgebra.lmul!(scalar::Number, A::GPUModArray)
    return LinearAlgebra.mul!(A, A, scalar)
end

function Base.similar(A::GPUModArray)
    return similar_with_modulus(A, modulus(A))
end

_storage_like(::CuStorage{T,D,M}, array::AbstractArray{T,D}) where {T,D,M} =
    CuStorage{T,D,M,typeof(array)}(array)
_storage_like(::MtlStorage{T,D,S}, array::AbstractArray{T,D}) where {T,D,S} =
    MtlStorage{T,D,S,typeof(array)}(array)
_storage_like(::AbstractArray, array::AbstractArray) = array

function _new_modular_wrapper(
    A::GPUModArray{T,D},
    native::AbstractArray{T,D},
    logical_size::Dims{D},
    new_modulus::Integer,
) where {T,D}
    storage = _storage_like(getfield(A, :storage), native)
    return GPUModArray{T,D,typeof(storage)}(storage, logical_size, new_modulus)
end

function similar_with_modulus(A::GPUModArray{T,D}, new_modulus::Integer) where {T,D}
    checked_modulus = _elementwise_working_modulus(A, new_modulus)
    logical_size = size(A)
    native = similar(parent(A), T, _padded_size(logical_size))
    fill!(native, zero(T))
    return _new_modular_wrapper(A, native, logical_size, checked_modulus)
end

function Base.copy(A::GPUModArray{T,D}) where {T,D}
    result = similar_with_modulus(A, modulus(A))
    if !isempty(A)
        source = _logical_view(parent(A), size(A))
        destination = _logical_view(parent(result), size(result))
        copyto!(destination, source)
    end
    return result
end

function Base.copy!(destination::GPUModArray, source::GPUModArray)
    size(destination) == size(source) ||
        throw(DimensionMismatch("Matrix dimensions must match"))
    _check_elementwise_backend(destination, source)
    _check_elementwise_no_partial_alias(destination, source)
    parent(destination) === parent(source) && return destination
    copyto!(
        _logical_view(parent(destination), size(destination)),
        _logical_view(parent(source), size(source)),
    )
    return destination
end

function _copyto_modular_array!(
    destination::GPUModArray{T,D},
    source::AbstractArray{S,D},
) where {T,S,D}
    size(destination) == size(source) ||
        throw(DimensionMismatch("Matrix dimensions must match"))
    _check_elementwise_backend(destination, destination)
    source_backend = _storage_backend(source)
    source_backend === nothing || _check_elementwise_backend(destination, source)
    _check_elementwise_no_partial_alias(destination, source)
    parent(destination) === source && return destination
    copyto!(_logical_view(parent(destination), size(destination)), source)
    return destination
end

Base.copyto!(
    destination::GPUModArray{T,D},
    source::AbstractArray{S,D},
) where {T,S,D} = _copyto_modular_array!(destination, source)

# These intersections keep the broad wrapper copy method unambiguous with the
# sparse and CHOLMOD matrix copy methods. Materialize these host-only formats
# before the ordinary array upload so GPU destinations never use scalar loops.
function _copyto_modular_host_matrix!(destination::GPUModMatrix, source::AbstractMatrix)
    size(destination) == size(source) ||
        throw(DimensionMismatch("Matrix dimensions must match"))
    return _copyto_modular_array!(destination, Matrix(source))
end

Base.copyto!(
    destination::GPUModMatrix{T},
    source::SparseArrays.AbstractSparseMatrixCSC{S},
) where {T,S} = _copyto_modular_host_matrix!(destination, source)

Base.copyto!(
    destination::GPUModMatrix{T},
    source::SparseArrays.CHOLMOD.Dense{S},
) where {T,S<:Union{Float32,Float64,ComplexF64,ComplexF32}} =
    _copyto_modular_host_matrix!(destination, source)

Base.copyto!(
    destination::GPUModMatrix{T},
    source::SparseArrays.CHOLMOD.Dense{T},
) where {T<:Union{Float32,Float64,ComplexF64,ComplexF32}} =
    _copyto_modular_host_matrix!(destination, source)

function Base.copyto!(destination::GPUModArray{T,D}, source::GPUModArray{S,D}) where {T,S,D}
    size(destination) == size(source) ||
        throw(DimensionMismatch("Matrix dimensions must match"))
    _check_elementwise_backend(destination, source)
    _check_elementwise_no_partial_alias(destination, source)
    parent(destination) === parent(source) && return destination
    copyto!(
        _logical_view(parent(destination), size(destination)),
        _logical_view(parent(source), size(source)),
    )
    return destination
end

function Base.fill!(A::GPUModArray{T}, scalar::Number) where {T}
    scalar_u = _normalize_elementwise_scalar(T, scalar, modulus(A))
    fill!(_logical_view(parent(A), size(A)), convert(T, scalar_u))
    return A
end

zero!(A::GPUModArray) = fill!(A, zero(eltype(A)))

function mod_elements!(
    A::GPUModArray;
    modulus_override::Union{Nothing,Integer} = nothing,
)
    _validate_positive_modulus_override(modulus_override)
    _check_elementwise_backend(A, A)
    _validate_elementwise_values(A)
    checked_modulus = _elementwise_working_modulus(A, modulus_override)
    return _launch_mod!(A, A, checked_modulus, _ELEMENTWISE_WORKGROUP_SIZE)
end

function change_modulus(A::GPUModArray, new_modulus::Integer)
    checked_modulus = _elementwise_working_modulus(A, new_modulus)
    _check_elementwise_backend(A, A)
    checked_modulus < modulus(A) && _validate_elementwise_values(A)
    result = similar_with_modulus(A, checked_modulus)
    if !isempty(A)
        source = _logical_view(parent(A), size(A))
        destination = _logical_view(parent(result), size(result))
        if checked_modulus < modulus(A)
            _launch_mod_arrays!(destination, source, checked_modulus, _ELEMENTWISE_WORKGROUP_SIZE)
        else
            copyto!(destination, source)
        end
    end
    return result
end

function change_modulus!!(A::GPUModArray{T,D}, new_modulus::Integer) where {T,D}
    checked_modulus = _elementwise_working_modulus(A, new_modulus)
    _check_elementwise_backend(A, A)
    if checked_modulus < modulus(A)
        _validate_elementwise_values(A)
        _launch_mod!(A, A, checked_modulus, _ELEMENTWISE_WORKGROUP_SIZE)
    end
    return _new_modular_wrapper(A, parent(A), size(A), checked_modulus)
end

function Base.mod(A::GPUModArray, new_modulus::Integer)
    _check_elementwise_allocating_input(A)
    checked_modulus = _elementwise_working_modulus(A, new_modulus)
    _validate_elementwise_values(A)
    result = similar_with_modulus(A, checked_modulus)
    _launch_mod!(result, A, checked_modulus, _ELEMENTWISE_WORKGROUP_SIZE)
    return result
end

function Base.:+(A::GPUModArray, B::GPUModArray)
    _check_elementwise_allocating_inputs(A, B)
    _validate_elementwise_values(A)
    _validate_elementwise_values(B)
    result = similar_with_modulus(A, modulus(A))
    return add!(result, A, B)
end

function Base.:+(A::GPUModArray, scalar::Number)
    _check_elementwise_allocating_input(A)
    checked_modulus = _elementwise_working_modulus(A, modulus(A))
    _normalize_elementwise_scalar(eltype(A), scalar, checked_modulus)
    _validate_elementwise_values(A)
    result = similar_with_modulus(A, checked_modulus)
    return scalar_add!(result, A, scalar)
end

Base.:+(scalar::Number, A::GPUModArray) = A + scalar

function Base.:-(A::GPUModArray, B::GPUModArray)
    _check_elementwise_allocating_inputs(A, B)
    _validate_elementwise_values(A)
    _validate_elementwise_values(B)
    result = similar_with_modulus(A, modulus(A))
    return sub!(result, A, B)
end

function Base.:-(A::GPUModArray, scalar::Number)
    _check_elementwise_allocating_input(A)
    checked_modulus = _elementwise_working_modulus(A, modulus(A))
    _normalize_elementwise_scalar(eltype(A), scalar, checked_modulus)
    _validate_elementwise_values(A)
    result = similar_with_modulus(A, checked_modulus)
    return scalar_sub!(result, A, scalar)
end

function Base.:-(scalar::Number, A::GPUModArray)
    _check_elementwise_allocating_input(A)
    checked_modulus = _elementwise_working_modulus(A, modulus(A))
    _normalize_elementwise_scalar(eltype(A), scalar, checked_modulus)
    _validate_elementwise_values(A)
    result = similar_with_modulus(A, checked_modulus)
    return rscalar_sub!(result, A, scalar)
end

function Base.:*(A::GPUModArray, scalar::Number)
    _check_elementwise_allocating_input(A)
    checked_modulus = _elementwise_working_modulus(A, modulus(A))
    _normalize_elementwise_scalar(eltype(A), scalar, checked_modulus)
    _validate_elementwise_values(A)
    result = similar_with_modulus(A, checked_modulus)
    return LinearAlgebra.mul!(result, A, scalar)
end

Base.:*(scalar::Number, A::GPUModArray) = A * scalar

function Base.:/(A::GPUModArray, scalar::Integer)
    _check_elementwise_allocating_input(A)
    checked_modulus = _elementwise_working_modulus(A, modulus(A))
    inverse = mod_inv(BigInt(scalar), BigInt(checked_modulus))
    _normalize_elementwise_scalar(eltype(A), inverse, checked_modulus)
    _validate_elementwise_values(A)
    result = similar_with_modulus(A, checked_modulus)
    return div!(result, A, scalar)
end
