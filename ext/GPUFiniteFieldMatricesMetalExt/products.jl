import Base.ScopedValues: with
import KernelAbstractions: @index, @kernel

@inline _inside_product(I::CartesianIndex, dims::Dims) = all(map(<=, Tuple(I), dims))

@kernel function _metal_pack_stripe!(packed, source, offset, active_size)
    I = @index(Global, Cartesian)
    @inbounds packed[I] = _inside_product(I, active_size) ? source[I+offset] : 0.0f0
end

@kernel function _metal_write_product!(destination, product, logical_size)
    I = @index(Global, Cartesian)
    @inbounds destination[I] =
        product !== nothing && _inside_product(I, logical_size) ? product[I] : 0.0f0
end

function _metal_product_buffer(array, dims)
    array isa Metal.MtlArray && size(array) == dims && return array
    return similar(_metal_root(array), eltype(array), dims)
end

function _metal_stripe_buffers(output, left, right, output_size, stripe_width)
    packed_output = _metal_product_buffer(output, output_size)
    packed_left = _metal_product_buffer(left, (size(left, 1), stripe_width))
    packed_right = _metal_product_buffer(right, (stripe_width, Base.tail(size(right))...))
    return packed_output, packed_left, packed_right
end

function _metal_check_product_destination(array)
    array isa Metal.MtlArray && return nothing
    array isa LinearAlgebra.AbstractTriangular && throw(
        ArgumentError("a product destination must have writable storage for every entry"),
    )
    return _metal_check_product_destination(parent(array))
end

function _metal_stripe_run!(output, left, right, output_size, q, chunk)
    isempty(output) && return output
    device = Metal.device(_metal_root(output))
    backend = KernelInterface.get_backend(_metal_root(output))
    return _with_metal_device(device) do
        store! = _metal_write_product!(backend, 256)
        inner = size(left, 2)
        if iszero(inner) || any(iszero, output_size)
            store!(output, nothing, output_size; ndrange = size(output))
            return output
        end
        # Reuse one packed pair across stripes. The final short stripe is zero
        # extended on the device so no stale entries contribute to its sum.
        width = min(inner, chunk)
        packed_output, packed_left, packed_right =
            _metal_stripe_buffers(output, left, right, output_size, width)
        pack! = _metal_pack_stripe!(backend, 256)
        right_cols = Base.tail(size(right))
        right_offset_tail = map(_ -> 0, right_cols)

        # Metal uses this scoped selector for both native GEMM and GEMV.
        with(Metal.matmul_alg => :MPS) do
            for stripe in FFM._stripe_ranges(inner, chunk)
                offset = first(stripe) - 1
                active_width = length(stripe)
                if packed_left !== left
                    pack!(
                        packed_left,
                        left,
                        CartesianIndex(0, offset),
                        (size(left, 1), active_width);
                        ndrange = size(packed_left),
                    )
                end
                if packed_right !== right
                    pack!(
                        packed_right,
                        right,
                        CartesianIndex(offset, right_offset_tail...),
                        (active_width, right_cols...);
                        ndrange = size(packed_right),
                    )
                end
                LinearAlgebra.mul!(
                    packed_output,
                    packed_left,
                    packed_right,
                    1.0f0,
                    iszero(offset) ? 0.0f0 : 1.0f0,
                )
                FFM._stripe_reduce!(packed_output, q)
            end
        end
        if packed_output !== output
            store!(output, packed_output, output_size; ndrange = size(output))
        end
        return output
    end
end

function _metal_stripe_product!(C, A, B; modulus_override, input_bound, max_ops)
    eltype(C) === eltype(A) === eltype(B) === Float32 || throw(
        ArgumentError("exact Metal products require Float32 storage for all operands"),
    )
    _metal_check_product_destination(parent(C))
    Cdata, Adata, Bdata, q, chunk =
        FFM._stripe_product_args(C, A, B; modulus_override, input_bound, max_ops)
    output = parent(C)
    # Copyback also writes padding, which can overlap a separately derived input.
    size(output) == size(Cdata) || FFM._stripe_check_alias(output, Adata, Bdata)
    _metal_stripe_run!(output, Adata, Bdata, size(Cdata), q, chunk)
    return C
end

function FFM.stripe_mul!(
    C::MtlModMatrix,
    A::MtlModMatrix,
    B::MtlModMatrix;
    modulus_override = nothing,
    input_bound = nothing,
    max_ops = nothing,
)
    return _metal_stripe_product!(C, A, B; modulus_override, input_bound, max_ops)
end

function FFM.stripe_mul!(
    y::MtlModVector,
    A::MtlModMatrix,
    x::MtlModVector;
    modulus_override = nothing,
    input_bound = nothing,
    max_ops = nothing,
    respect_max_ops::Bool = true,
)
    return _metal_stripe_product!(
        y,
        A,
        x;
        modulus_override,
        input_bound,
        max_ops = respect_max_ops ? max_ops : nothing,
    )
end

# Rebuild Julia's views on the logical native storage before passing them to
# the stripe planner. This also excludes the modular wrapper from GPU code.
const _MtlProductView = Union{
    SubArray{<:Any,2,<:MtlModMatrix},
    LinearAlgebra.Transpose{<:Any,<:MtlModMatrix},
    LinearAlgebra.Adjoint{<:Any,<:MtlModMatrix},
    LinearAlgebra.UpperTriangular{<:Any,<:MtlModMatrix},
    LinearAlgebra.LowerTriangular{<:Any,<:MtlModMatrix},
    LinearAlgebra.UnitUpperTriangular{<:Any,<:MtlModMatrix},
    LinearAlgebra.UnitLowerTriangular{<:Any,<:MtlModMatrix},
}
const _MtlProductMatrix = Union{MtlModMatrix,_MtlProductView}
const _MtlProductVectorView = SubArray{<:Any,1,<:Union{MtlModVector,MtlModMatrix}}
const _MtlProductVector = Union{MtlModVector,_MtlProductVectorView}

_metal_product_view(A::SubArray, data) = view(data, parentindices(A)...)

for Wrapper in (
    LinearAlgebra.Transpose,
    LinearAlgebra.Adjoint,
    LinearAlgebra.UpperTriangular,
    LinearAlgebra.LowerTriangular,
    LinearAlgebra.UnitUpperTriangular,
    LinearAlgebra.UnitLowerTriangular,
)
    @eval _metal_product_view(::$Wrapper, data) = $Wrapper(data)
end

_metal_product_operand(A::MtlModArray) = A
function _metal_product_operand(A::Union{_MtlProductView,_MtlProductVectorView})
    modular = parent(A)
    data = _metal_product_view(A, FFM._logical_view(parent(modular), size(modular)))
    return _wrap_metal_storage(data, nothing, modulus(modular))
end

for (Dest, Left, Right) in (
    (MtlModMatrix, _MtlProductView, _MtlProductMatrix),
    (MtlModMatrix, MtlModMatrix, _MtlProductView),
    (MtlModVector, _MtlProductView, _MtlProductVector),
    (MtlModVector, MtlModMatrix, _MtlProductVectorView),
)
    @eval begin
        function FFM.stripe_mul!(C::$Dest, A::$Left, B::$Right; kwargs...)
            return FFM.stripe_mul!(
                C,
                _metal_product_operand(A),
                _metal_product_operand(B);
                kwargs...,
            )
        end

        function FFM.stripe_mul(A::$Left, B::$Right; kwargs...)
            return FFM.stripe_mul(
                _metal_product_operand(A),
                _metal_product_operand(B);
                kwargs...,
            )
        end

        function LinearAlgebra.mul!(C::$Dest, A::$Left, B::$Right; kwargs...)
            return LinearAlgebra.mul!(
                C,
                _metal_product_operand(A),
                _metal_product_operand(B);
                kwargs...,
            )
        end

        Base.:*(A::$Left, B::$Right) = FFM.stripe_mul(A, B)
    end
end
