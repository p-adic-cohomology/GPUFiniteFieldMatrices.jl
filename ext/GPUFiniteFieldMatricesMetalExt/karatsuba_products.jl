import GPUFiniteFieldMatrices: KaratsubaArray, KaratsubaMatrix, KaratsubaVector
import GPUFiniteFieldMatrices: low_modulus, high_modulus

const _MetalKaratsuba{T,D} =
    KaratsubaArray{T,D,S} where {S<:Union{MtlModArray,Metal.MtlArray}}

const _MetalKaratsubaMatrix = _MetalKaratsuba{Float32,2}
const _MetalKaratsubaVector = _MetalKaratsuba{Float32,1}

function _metal_karatsuba_check_storage(operands)
    all(K -> K isa _MetalKaratsuba{Float32}, operands) ||
        throw(ArgumentError("Metal Karatsuba products require Float32 Metal components"))
    return nothing
end

function _metal_karatsuba_planes(C, A, B)
    _metal_karatsuba_check_storage((C, A, B))
    FFM._karatsuba_product_size(C, A, B)
    c, a = FFM._karatsuba_product_planes(C), FFM._karatsuba_product_planes(A)
    b = A === B ? a : FFM._karatsuba_product_planes(B)
    for plane in (c..., a..., b...)
        FFM._karatsuba_check_device(c[1], plane)
    end
    FFM._karatsuba_check_product_aliases(c, a, b, A === B)
    for dest in (c..., a[3], b[3])
        _metal_check_product_destination(dest)
    end
    return c, a, b
end

function _metal_karatsuba_gemm_args(out, left, right, product)
    args = map(x -> _wrap_metal_storage(x, nothing, product.modulus), (out, left, right))
    FFM._stripe_product_args(args...; product.input_bound, product.max_ops)
    return args
end

function _metal_karatsuba_sums!(a, b, same_input)
    FFM._karatsuba_sum_planes!(a...)
    same_input || FFM._karatsuba_sum_planes!(b...)
    return nothing
end

function _metal_karatsuba_gemm!(C, A, B)
    c, a, b = _metal_karatsuba_planes(C, A, B)
    products = FFM._karatsuba_matmul_product_plan(
        Float32,
        low_modulus(A),
        high_modulus(A),
        low_modulus(B),
        high_modulus(B),
        low_modulus(C),
        high_modulus(C),
    )
    device = Metal.device(_metal_root(c[1]))
    return _with_metal_device(device) do
        low = _metal_karatsuba_gemm_args(c[1], a[1], b[1], products.low)
        middle = _metal_karatsuba_gemm_args(c[2], a[3], b[3], products.middle)
        high = _metal_karatsuba_gemm_args(c[3], a[2], b[2], products.high)
        _metal_karatsuba_sums!(a, b, A === B)
        FFM.stripe_mul!(low...; products.low.input_bound, products.low.max_ops)
        FFM.stripe_mul!(middle...; products.middle.input_bound, products.middle.max_ops)
        FFM.stripe_mul!(high...; products.high.input_bound, products.high.max_ops)

        if !isempty(C)
            backend = KernelInterface.get_backend(c[1])
            FFM._karatsuba_gemm_recombine_kernel!(backend, FFM._ELEMENTWISE_WORKGROUP_SIZE)(
                c...,
                Int64(products.output_radix),
                Int64(products.radix_mod_capacity),
                Int64(products.radix_squared_mod_capacity),
                Int64(products.output_capacity);
                ndrange = length(C),
            )
        end
        return C
    end
end

function _metal_karatsuba_pack(array)
    packed = _metal_product_buffer(array, size(array))
    if packed !== array
        packed .= array
    end
    return packed
end

function _metal_karatsuba_gemv_product!(out, A, x)
    packed = _metal_product_buffer(out, size(out))
    LinearAlgebra.mul!(packed, _metal_karatsuba_pack(A), _metal_karatsuba_pack(x))
    if packed !== out
        out .= packed
    end
    return out
end

function _metal_karatsuba_gemv!(y, A, x)
    c, a, b = _metal_karatsuba_planes(y, A, x)
    dims = (
        low_modulus(A),
        high_modulus(A),
        low_modulus(x),
        high_modulus(x),
        low_modulus(y),
        high_modulus(y),
    )
    FFM._karatsuba_gemv_validate_radices(dims...)
    isempty(y) && return y
    cols = size(A, 2)
    device = Metal.device(_metal_root(c[1]))
    if iszero(cols)
        return _with_metal_device(device) do
            c[1] .= 0.0f0
            c[2] .= 0.0f0
            return y
        end
    end
    FFM._karatsuba_gemv_precision_plan(Float32, cols, dims...)
    return _with_metal_device(device) do
        _metal_karatsuba_sums!(a, b, false)
        # Preserve the full dot totals in the three planes. The whole-dot
        # precision preflight includes the exact Float32 endpoint.
        with(Metal.matmul_alg => :MPS) do
            _metal_karatsuba_gemv_product!(c[1], a[1], b[1])
            _metal_karatsuba_gemv_product!(c[2], a[3], b[3])
            _metal_karatsuba_gemv_product!(c[3], a[2], b[2])
        end
        backend = KernelInterface.get_backend(c[1])
        FFM._karatsuba_gemv_recombine_kernel!(backend, FFM._ELEMENTWISE_WORKGROUP_SIZE)(
            c...,
            Int64(low_modulus(y)),
            Int64(high_modulus(y));
            ndrange = length(y),
        )
        return y
    end
end

# Specialize calls that contain a Metal operand. The intersections give
# mixed-storage calls the same preflight as wholly Metal calls.
for (plain, metal, product) in (
    (KaratsubaMatrix, _MetalKaratsubaMatrix, :_metal_karatsuba_gemm!),
    (KaratsubaVector, _MetalKaratsubaVector, :_metal_karatsuba_gemv!),
)
    for (C, A, B) in Iterators.product(
        (plain, metal),
        (KaratsubaMatrix, _MetalKaratsubaMatrix),
        (plain, metal),
    )
        (C === plain && A === KaratsubaMatrix && B === plain) && continue
        @eval LinearAlgebra.mul!(C::$C, A::$A, B::$B) = $product(C, A, B)
        if product === :_metal_karatsuba_gemv!
            @eval FFM.karatsuba_gemv!(y::$C, A::$A, x::$B) = $product(y, A, x)
        end
    end
    for (A, B) in
        Iterators.product((KaratsubaMatrix, _MetalKaratsubaMatrix), (plain, metal))
        (A === KaratsubaMatrix && B === plain) && continue
        @eval function Base.:*(A::$A, B::$B)
            _metal_karatsuba_check_storage((A, B))
            return FFM._karatsuba_allocating_product(A, B)
        end
    end
end

@doc """
    karatsuba_gemv!(y, A, x)

Store a Float32 Metal Karatsuba matrix-vector product in `y`. Initialize
independent plans and use canonical input digits below the corresponding
split moduli. Input and output low radices must match, and the output high
modulus must divide that radix. The full dot-product budgets are checked
before writing scratch or output storage.
""" FFM.karatsuba_gemv!
