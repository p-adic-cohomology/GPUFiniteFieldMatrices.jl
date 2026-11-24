function _default_cumod_storage_type(::Type{T}, ::Val{D}) where {T,D}
    compiler_return_type = getfield(getfield(Core, :Compiler), :return_type)
    native_type = compiler_return_type(CUDA.zeros, Tuple{Type{T},Vararg{Int,D}})
    isconcretetype(native_type) && native_type <: CUDA.CuArray{T,D} || throw(
        ArgumentError("CUDA.zeros does not infer a concrete CuArray{$T,$D} storage type"),
    )
    memory_type = native_type.parameters[3]
    isconcretetype(memory_type) || throw(
        ArgumentError("CUDA.zeros does not infer a concrete CuArray memory type for $T,$D"),
    )
    storage_type = FFM.CuStorage{T,D,memory_type,native_type}
    return GPUModArray{T,D,storage_type}
end

_backend_default_component_type(::Val{:cuda}, ::Type{T}, ::Val{D}) where {T,D} =
    _default_cumod_storage_type(T, Val(D))
_backend_default_plan_type(::Val{:cuda}, ::Type{T}, ::Val{D}) where {T,D} =
    _default_cumod_storage_type(T, Val(D))

_karatsuba_plan_type(::Type{S}) where {T,D,S<:CuModArray{T,D}} =
    _default_cumod_storage_type(T, Val(D))
_karatsuba_plan_type(::Type{S}) where {T,D,S<:CUDA.CuArray{T,D}} =
    _default_cumod_storage_type(T, Val(D))

@inline function _karatsuba_require_matched_float_type(
    ::Type{TA},
    ::Type{TB},
    ::Type{TC},
) where {TA,TB,TC}
    (TA === TB && TA === TC) || throw(
        ArgumentError(
            "Karatsuba products require matching Float32 or Float64 element types",
        ),
    )
    (TA === Float32 || TA === Float64) ||
        throw(ArgumentError("Karatsuba products support Float32 or Float64 storage"))
    return TA
end

@inline function _karatsuba_initialized_plan(K::KaratsubaArray{T,D,S,P}) where {T,D,S,P}
    plan = K.plan
    plan === nothing && throw(ArgumentError("Must initialize plans first"))
    return plan::P
end

function _karatsuba_validate_gemm(
    A::KaratsubaArray{T,2},
    B::KaratsubaArray{T,2},
    C::KaratsubaArray{T,2},
    a_plan,
    b_plan,
    c_plan,
) where {T}
    for (name, K, plan) in (("A", A, a_plan), ("B", B, b_plan), ("C", C, c_plan))
        low = low_part(K)
        high = high_part(K)
        for (plane_name, plane) in (("low", low), ("high", high), ("plan", plan))
            plane isa CuModArray{T,2} || throw(
                ArgumentError(
                    "Karatsuba matrix multiplication requires CuModMatrix{$T} $name.$plane_name storage",
                ),
            )
            size(parent(plane), 1) >= size(plane, 1) &&
            size(parent(plane), 2) >= size(plane, 2) || throw(
                DimensionMismatch(
                    "$name.$plane_name storage is smaller than its logical size",
                ),
            )
        end
        size(high) == size(low) || throw(
            DimensionMismatch(
                "$name low and high components must have matching logical sizes",
            ),
        )
        size(plan) == size(low) ||
            throw(DimensionMismatch("$name plan must match its component logical size"))
    end

    inputs = (low_part(A), high_part(A), low_part(B), high_part(B))
    outputs = (low_part(C), high_part(C), c_plan)
    scratch = (a_plan, b_plan)
    for output in outputs, input in inputs
        _stripe_mightalias(parent(output), parent(input)) && throw(
            ArgumentError(
                "Karatsuba matrix output storage must not alias input components",
            ),
        )
    end
    for i = 1:length(outputs), j = (i+1):length(outputs)
        _stripe_mightalias(parent(outputs[i]), parent(outputs[j])) && throw(
            ArgumentError("Karatsuba matrix output components must not alias each other"),
        )
    end
    for plan in scratch, input in inputs
        _stripe_mightalias(parent(plan), parent(input)) &&
            throw(ArgumentError("Karatsuba plan scratch must not alias input components"))
    end
    for plan in scratch, output in outputs
        _stripe_mightalias(parent(plan), parent(output)) &&
            throw(ArgumentError("Karatsuba plan scratch must not alias output components"))
    end
    _stripe_mightalias(parent(a_plan), parent(b_plan)) &&
        A !== B &&
        throw(ArgumentError("Karatsuba input plan scratch must not alias each other"))
    return nothing
end

function _karatsuba_require_same_device_gemm(
    A::KaratsubaArray{TA,2},
    a_plan::CuModArray{TA,2},
    B::KaratsubaArray{TB,2},
    b_plan::CuModArray{TB,2},
    C::KaratsubaArray{TC,2},
    c_plan::CuModArray{TC,2},
) where {TA,TB,TC}
    expected_device = CUDA.device(_stripe_storage_root(parent(low_part(C))))
    for storage in (
        parent(high_part(C)),
        parent(c_plan),
        parent(low_part(A)),
        parent(high_part(A)),
        parent(a_plan),
        parent(low_part(B)),
        parent(high_part(B)),
        parent(b_plan),
    )
        CUDA.device(_stripe_storage_root(storage)) == expected_device || throw(
            ArgumentError(
                "Karatsuba components and plans must reside on the same CUDA device",
            ),
        )
    end
    return nothing
end

function _karatsuba_require_same_device_gemv(
    A::KaratsubaArray{TA,2},
    a_plan::CuModArray{TA,2},
    x::KaratsubaArray{TX,1},
    x_plan::CuModArray{TX,1},
    y::KaratsubaArray{TY,1},
    y_plan::CuModArray{TY,1},
) where {TA,TX,TY}
    expected_device = CUDA.device(_stripe_storage_root(parent(low_part(y))))
    for storage in (
        parent(high_part(y)),
        parent(y_plan),
        parent(low_part(A)),
        parent(high_part(A)),
        parent(a_plan),
        parent(low_part(x)),
        parent(high_part(x)),
        parent(x_plan),
    )
        CUDA.device(_stripe_storage_root(storage)) == expected_device || throw(
            ArgumentError(
                "Karatsuba components and plans must reside on the same CUDA device",
            ),
        )
    end
    return nothing
end

function _karatsuba_preflight_stripe_product(Cdata, Adata, Bdata, product)
    return _stripe_preflight(
        Cdata,
        Adata,
        Bdata,
        product.modulus,
        product.input_bound,
        product.max_ops,
    )
end

function _karatsuba_stripe_product!(Cdata, Adata, Bdata, product)
    return _stripe_mul!(
        Cdata,
        Adata,
        Bdata;
        modulus = product.modulus,
        input_bound = product.input_bound,
        max_ops = product.max_ops,
    )
end

function _karatsuba_compute_gemm!(
    C::KaratsubaArray{TC,2},
    A::KaratsubaArray{TA,2},
    B::KaratsubaArray{TB,2},
    a_plan::CuModArray{TA,2},
    b_plan::CuModArray{TB,2},
    c_plan::CuModArray{TC,2},
    data,
    products,
) where {TC,TA,TB}
    T = TC
    rows_a, cols_a = size(low_part(A))
    rows_b, cols_b = size(low_part(B))
    logical_count = max(length(low_part(A)), length(low_part(B)))
    threads = TILE_WIDTH
    blocks = max(1, cld(logical_count, threads))
    @cuda threads=threads blocks=blocks karatsuba_matmul_kernel_1!(
        parent(a_plan),
        parent(low_part(A)),
        parent(high_part(A)),
        rows_a,
        cols_a,
        stride(parent(a_plan), 1),
        stride(parent(a_plan), 2),
        stride(parent(low_part(A)), 1),
        stride(parent(low_part(A)), 2),
        stride(parent(high_part(A)), 1),
        stride(parent(high_part(A)), 2),
        parent(b_plan),
        parent(low_part(B)),
        parent(high_part(B)),
        rows_b,
        cols_b,
        stride(parent(b_plan), 1),
        stride(parent(b_plan), 2),
        stride(parent(low_part(B)), 1),
        stride(parent(low_part(B)), 2),
        stride(parent(high_part(B)), 1),
        stride(parent(high_part(B)), 2),
    )

    _karatsuba_stripe_product!(data.c_low, data.a_low, data.b_low, products.low)
    _karatsuba_stripe_product!(data.c_high, data.a_plan, data.b_plan, products.middle)
    _karatsuba_stripe_product!(data.c_plan, data.a_high, data.b_high, products.high)

    output_rows, output_cols = size(low_part(C))
    output_count = length(low_part(C))
    output_blocks = max(1, cld(output_count, threads))
    output_capacity = T(products.output_capacity)
    radix_mod_capacity = T(products.radix_mod_capacity)
    radix_squared_mod_capacity = T(products.radix_squared_mod_capacity)
    output_radix = T(products.output_radix)
    @cuda threads=threads blocks=output_blocks karatsuba_matmul_kernel_2!(
        parent(low_part(C)),
        parent(high_part(C)),
        parent(c_plan),
        output_rows,
        output_cols,
        stride(parent(low_part(C)), 1),
        stride(parent(low_part(C)), 2),
        stride(parent(high_part(C)), 1),
        stride(parent(high_part(C)), 2),
        stride(parent(c_plan), 1),
        stride(parent(c_plan), 2),
        radix_mod_capacity,
        radix_squared_mod_capacity,
        output_radix,
        output_capacity,
    )

    return C
end

"""
    LinearAlgebra.mul!(C, A, B)

Multiply the Karatsuba matrices A and B, storing the product in C. The
three matrices must share the same representation modulus M, and A's column
count must match B's row count. C must have dimensions (size(A, 1), size(B, 2));
the Karatsuba shape check rejects mismatches before any plan or device writes.
All nine input, output, and plan storage roots must reside on one CUDA device;
this is checked before the staging kernel can write either input plan.
Call initialize_plan! on all
three matrices before multiplying. A and B must share input low radix R; C
may use an independent low radix R_C. The result components represent the
product modulo Q = R_C*high_modulus(C), and checked modular recombination
supports independent R_C without reducing by M. Input component entries must
be canonical nonnegative representatives below their respective low and high
moduli; those ranges are the exclusive stripe input bounds. These ranges are
caller promises and are not scanned at runtime. CuModArray wrappers made with
mod=false retain raw values, and to_karatsuba preserves an exact high quotient
without reducing it by high_modulus, so callers must supply a high_modulus
that bounds those values. The operation supports matching Float32 or Float64
GPU-backed components. Each type is bounded by its cuBLAS significand: 24 exact
integer bits for Float32 and 53 for Float64. Output capacities and stripe
accumulators are checked against that bound before any plan is written.
"""
function LinearAlgebra.mul!(
    C::KaratsubaArray{TC,2},
    A::KaratsubaArray{TA,2},
    B::KaratsubaArray{TB,2},
) where {TC,TA,TB}
    T = _karatsuba_require_matched_float_type(TA, TB, TC)
    if (modulus(A) != modulus(B)) || (modulus(A) != modulus(C))
        throw(ModulusMismatchError("Matrices must have the same modulus m"))
    end
    if size(low_part(A))[2] != size(low_part(B))[1]
        throw(DimensionMismatch("Matrix dimensions don't work for multiplication"))
    end
    a_plan = _karatsuba_initialized_plan(A)
    b_plan = _karatsuba_initialized_plan(B)
    c_plan = _karatsuba_initialized_plan(C)
    size(low_part(C)) == (size(low_part(A), 1), size(low_part(B), 2)) ||
        throw(DimensionMismatch("Output matrix C has incorrect dimensions"))
    _karatsuba_validate_gemm(A, B, C, a_plan, b_plan, c_plan)
    data = (
        a_low = _stripe_logical_data(low_part(A)),
        a_high = _stripe_logical_data(high_part(A)),
        a_plan = _stripe_logical_data(a_plan),
        b_low = _stripe_logical_data(low_part(B)),
        b_high = _stripe_logical_data(high_part(B)),
        b_plan = _stripe_logical_data(b_plan),
        c_low = _stripe_logical_data(low_part(C)),
        c_high = _stripe_logical_data(high_part(C)),
        c_plan = _stripe_logical_data(c_plan),
    )
    products = _karatsuba_matmul_product_plan(
        T,
        low_modulus(A),
        high_modulus(A),
        low_modulus(B),
        high_modulus(B),
        low_modulus(C),
        high_modulus(C),
    )

    # Validate all stripe calls before the sum kernel writes either input plan.
    # The calls below repeat this preflight at their own execution boundary.
    _karatsuba_preflight_stripe_product(data.c_low, data.a_low, data.b_low, products.low)
    _karatsuba_preflight_stripe_product(
        data.c_high,
        data.a_plan,
        data.b_plan,
        products.middle,
    )
    _karatsuba_preflight_stripe_product(
        data.c_plan,
        data.a_high,
        data.b_high,
        products.high,
    )
    # The stripe preflights validate each product group independently. The
    # staging and recombination kernels combine all nine storage roots.
    _karatsuba_require_same_device_gemm(A, a_plan, B, b_plan, C, c_plan)
    _karatsuba_compute_gemm!(C, A, B, a_plan, b_plan, c_plan, data, products)
    return C
end

"""
The number of actual matmuls to multiply one karatsuba matrix.
If a higher Karatsuba multiplication algorithm were implemented,
it could be implemented in a similar way with a bigger constant
"""
const nKMuls = 3

# const _cublas_Apointers_cache_f64 = IdDict{Task,CuArray{Ptr{Float64}}}()
# const _cublas_Bpointers_cache_f64 = IdDict{Task,CuArray{Ptr{Float64}}}()
# const _cublas_Cpointers_cache_f64 = IdDict{Task,CuArray{Ptr{Float64}}}()

# @inline function cublas_Apointers_f64()
#     t = current_task()
#     get!(_cublas_Apointers_cache, t) do
#         CuArray{Ptr{Float64}

# """
# The following is a device vector of device pointers
# """
# mat_ptrs_device = CuArray{CuPtr{Float64}}(undef,nKMuls)
# vec_ptrs_device = CuArray{CuPtr{Float64}}(undef,nKMuls)
# target_ptrs_device = CuArray{CuPtr{Float64}}(undef,nKMuls)

# """
# The following is a host vector of device pointers
# """
# mat_ptrs_host = CUDA.pin(Vector{CuPtr{Float64}}(undef,nKMuls))
# vec_ptrs_host = CUDA.pin(Vector{CuPtr{Float64}}(undef,nKMuls))
# target_ptrs_host = CUDA.pin(Vector{CuPtr{Float64}}(undef,nKMuls))


# the stuff in this function should apply to the general mul!, but with gemm instead of gemv
@inline _karatsuba_gemv_whole_cuarray_parent(A::CUDA.CuArray) = true

@inline function _karatsuba_gemv_whole_cuarray_parent(A::CuModArray)
    storage = parent(A)
    return storage isa CUDA.CuArray && size(storage) == size(A)
end

@inline _karatsuba_gemv_whole_cuarray_parent(::AbstractArray) = false

@inline function _karatsuba_gemv_batch_containers_supported(mats, vecs, targets)
    return length(mats) == 3 &&
           length(vecs) == 3 &&
           length(targets) == 3 &&
           mats isa Vector{<:CUDA.CuArray} &&
           vecs isa Vector{<:CUDA.CuArray} &&
           targets isa Vector{<:CUDA.CuArray} &&
           isconcretetype(eltype(mats)) &&
           isconcretetype(eltype(vecs)) &&
           isconcretetype(eltype(targets)) &&
           ndims(eltype(mats)) == 2 &&
           ndims(eltype(vecs)) == 1 &&
           ndims(eltype(targets)) == 1 &&
           eltype(eltype(mats)) === eltype(eltype(vecs)) &&
           eltype(eltype(mats)) === eltype(eltype(targets))
end

function _karatsuba_validate_gemv(
    A::KaratsubaArray{T,2},
    x::KaratsubaArray{T,1},
    y::KaratsubaArray{T,1},
    a_plan,
    x_plan,
    y_plan,
) where {T}
    for (name, K, plan) in (("A", A, a_plan), ("x", x, x_plan), ("y", y, y_plan))
        low = low_part(K)
        high = high_part(K)
        D = ndims(low)
        for (plane_name, plane) in (("low", low), ("high", high), ("plan", plan))
            plane isa CuModArray{T,D} || throw(
                ArgumentError(
                    "Karatsuba GEMV requires CuModArray{$T,$D} $name.$plane_name storage",
                ),
            )
            all(size(parent(plane), k) >= size(plane, k) for k = 1:D) || throw(
                DimensionMismatch(
                    "$name.$plane_name storage is smaller than its logical size",
                ),
            )
        end
        size(high) == size(low) ||
            throw(DimensionMismatch("$name low and high components must match logically"))
        size(plan) == size(low) ||
            throw(DimensionMismatch("$name plan must match its component logical size"))
    end
    size(low_part(y)) == (size(low_part(A), 1),) ||
        throw(DimensionMismatch("Output vector y has incorrect dimensions"))

    inputs = (low_part(A), high_part(A), low_part(x), high_part(x))
    outputs = (low_part(y), high_part(y), y_plan)
    scratch = (a_plan, x_plan)
    for output in outputs, input in inputs
        _stripe_mightalias(parent(output), parent(input)) && throw(
            ArgumentError("Karatsuba GEMV output storage must not alias input components"),
        )
    end
    for i = 1:length(outputs), j = (i+1):length(outputs)
        _stripe_mightalias(parent(outputs[i]), parent(outputs[j])) && throw(
            ArgumentError("Karatsuba GEMV output components must not alias each other"),
        )
    end
    for plan in scratch, input in inputs
        _stripe_mightalias(parent(plan), parent(input)) && throw(
            ArgumentError("Karatsuba GEMV plan scratch must not alias input components"),
        )
    end
    for plan in scratch, output in outputs
        _stripe_mightalias(parent(plan), parent(output)) && throw(
            ArgumentError("Karatsuba GEMV plan scratch must not alias output components"),
        )
    end
    _stripe_mightalias(parent(a_plan), parent(x_plan)) &&
        throw(ArgumentError("Karatsuba GEMV input plan scratch must not alias each other"))

    return nothing
end

function karatsuba_gemv!(
    C::KaratsubaArray{TC,1},
    A::KaratsubaArray{TA,2},
    B::KaratsubaArray{TB,1},
) where {TC,TA,TB}
    T = _karatsuba_require_matched_float_type(TA, TB, TC)
    if (modulus(A) != modulus(B)) || (modulus(A) != modulus(C))
        throw(ModulusMismatchError("Matrices must have the same modulus m"))
    end
    if size(low_part(A), 2) != length(low_part(B))
        throw(DimensionMismatch("Matrix dimensions don't work for multiplication"))
    end
    a_plan = _karatsuba_initialized_plan(A)
    b_plan = _karatsuba_initialized_plan(B)
    c_plan = _karatsuba_initialized_plan(C)
    tw = TILE_WIDTH
    rows, cols = size(low_part(A))
    _karatsuba_validate_gemv(A, B, C, a_plan, b_plan, c_plan)
    _karatsuba_require_same_device_gemv(A, a_plan, B, b_plan, C, c_plan)

    # The output radix relation is checked for every shape before writes.
    _karatsuba_gemv_validate_radices(
        low_modulus(A),
        high_modulus(A),
        low_modulus(B),
        high_modulus(B),
        low_modulus(C),
        high_modulus(C),
    )
    if rows == 0
        return C
    end
    if cols == 0
        fill!(_logical_view(parent(low_part(C)), size(low_part(C))), zero(T))
        fill!(_logical_view(parent(high_part(C)), size(high_part(C))), zero(T))
        return C
    end

    precision = _karatsuba_gemv_precision_plan(
        T,
        cols,
        low_modulus(A),
        high_modulus(A),
        low_modulus(B),
        high_modulus(B),
        low_modulus(C),
        high_modulus(C),
    )

    logical_count = max(length(a_plan), length(b_plan))
    if logical_count > 0
        threads = tw
        blocks = cld(logical_count, threads)
        @cuda threads=threads blocks=blocks karatsuba_matmul_kernel_1!(
            _logical_view(parent(a_plan), size(a_plan)),
            _logical_view(parent(low_part(A)), size(low_part(A))),
            _logical_view(parent(high_part(A)), size(high_part(A))),
            _logical_view(parent(b_plan), size(b_plan)),
            _logical_view(parent(low_part(B)), size(low_part(B))),
            _logical_view(parent(high_part(B)), size(high_part(B))),
        )
    end
    # A.plan and B.plan now hold the full low-plus-high sums used by P1.

    (zero_ptr, one_ptr) = _stripe_gemv_scalars(T)


    # C.plan is the independent, output-sized destination for the third
    # product. B.plan remains the second product's input sum.
    Cplan_output = _logical_view(parent(c_plan), size(c_plan))
    matrix_parents = (parent(low_part(A)), parent(a_plan), parent(high_part(A)))
    vector_parents = (parent(low_part(B)), parent(b_plan), parent(high_part(B)))
    target_parents = (parent(low_part(C)), parent(high_part(C)), parent(c_plan))
    whole_parent_operands =
        _karatsuba_gemv_whole_cuarray_parent(low_part(A)) &&
        _karatsuba_gemv_whole_cuarray_parent(a_plan) &&
        _karatsuba_gemv_whole_cuarray_parent(high_part(A)) &&
        _karatsuba_gemv_whole_cuarray_parent(low_part(B)) &&
        _karatsuba_gemv_whole_cuarray_parent(b_plan) &&
        _karatsuba_gemv_whole_cuarray_parent(high_part(B)) &&
        _karatsuba_gemv_whole_cuarray_parent(low_part(C)) &&
        _karatsuba_gemv_whole_cuarray_parent(high_part(C)) &&
        _karatsuba_gemv_whole_cuarray_parent(c_plan)
    if whole_parent_operands
        mats = collect(matrix_parents)
        vecs = collect(vector_parents)
        targets = collect(target_parents)
        batch_supported = _karatsuba_gemv_batch_containers_supported(mats, vecs, targets)
        if batch_supported
            lda = stride(mats[1], 2)
            incx = stride(vecs[1], 1)
            incy = stride(targets[1], 1)
            batch_supported =
                all(stride(matrix, 2) == lda for matrix in mats) &&
                all(stride(vector, 1) == incx for vector in vecs) &&
                all(stride(target, 1) == incy for target in targets)
        end
        if batch_supported
            _CUDA_BLAS.gemv_batched!('N', one_ptr, mats, vecs, zero_ptr, targets)
        else
            for i in eachindex(mats)
                _CUDA_BLAS.gemv!('N', one_ptr, mats[i], vecs[i], zero_ptr, targets[i])
            end
        end
    else
        mats = [
            _logical_view(parent(low_part(A)), size(low_part(A))),
            _logical_view(parent(a_plan), size(a_plan)),
            _logical_view(parent(high_part(A)), size(high_part(A))),
        ]
        vecs = [
            _logical_view(parent(low_part(B)), size(low_part(B))),
            _logical_view(parent(b_plan), size(b_plan)),
            _logical_view(parent(high_part(B)), size(high_part(B))),
        ]
        targets = [
            _logical_view(parent(low_part(C)), size(low_part(C))),
            _logical_view(parent(high_part(C)), size(high_part(C))),
            Cplan_output,
        ]
        for i in eachindex(mats)
            _CUDA_BLAS.gemv!('N', one_ptr, mats[i], vecs[i], zero_ptr, targets[i])
        end
    end

    # CUDA.CUBLAS.gemv!('N',one_ptr,parent(low_part(A)),parent(low_part(B)),zero_ptr,parent(low_part(C)))
    # CUDA.CUBLAS.gemv!('N',one_ptr,parent(A.plan),parent(B.plan),zero_ptr,parent(high_part(C)))
    # CUDA.CUBLAS.gemv!('N',one_ptr,parent(high_part(A)),parent(high_part(B)),zero_ptr,parent(B.plan))

    totalsize = rows
    if totalsize > 0
        threads = tw
        blocks = cld(totalsize, threads)
        @cuda threads=threads blocks=blocks karatsuba_matmul_kernel_2!(
            _logical_view(parent(low_part(C)), size(low_part(C))),
            _logical_view(parent(high_part(C)), size(high_part(C))),
            Cplan_output,
            T(low_modulus(C)),
            T(high_modulus(C)),
        )
    end


    C
end

function karatsuba_gemv!(C::KaratsubaArray, A::KaratsubaArray, B::KaratsubaArray)
    throw(
        ArgumentError(
            "karatsuba_gemv! requires KaratsubaMatrix, KaratsubaVector, KaratsubaVector",
        ),
    )
end

function LinearAlgebra.mul!(
    y::KaratsubaArray{TY,1},
    A::KaratsubaArray{TA,2},
    x::KaratsubaArray{TX,1},
) where {TY,TA,TX}
    return karatsuba_gemv!(y, A, x)
end

function LinearAlgebra.mul!(C::KaratsubaArray, A::KaratsubaArray, B::KaratsubaArray)
    throw(ArgumentError("Karatsuba mul! supports matrix-matrix and matrix-vector products"))
end

_karatsuba_gpu_storage(A::CuModArray) = parent(A)
_karatsuba_gpu_storage(A::CUDA.CuArray) = A
_karatsuba_gpu_strides(A::CuModArray) = strides(parent(A))
_karatsuba_gpu_strides(A::CUDA.CuArray) = strides(A)
@stable default_mode = "disable" default_union_limit = 2 begin
    _karatsuba_logical_view(A::CuModArray) = _logical_view(parent(A), size(A))
end
_karatsuba_logical_view(A::CUDA.CuArray) = A

function _karatsuba_gpu_float_precision(::Type{Float16})
    11
end
function _karatsuba_gpu_float_precision(::Type{Float32})
    24
end
function _karatsuba_gpu_float_precision(::Type{Float64})
    53
end

function _karatsuba_gpu_component(::Type{T}, x::S) where {T<:Integer,S<:Integer}
    fits = if T <: Signed
        if S <: Signed
            sizeof(S) <= sizeof(T) || (x >= typemin(T) && x <= typemax(T))
        else
            sizeof(S) < sizeof(T) || x <= typemax(T)
        end
    elseif S <: Signed
        x >= 0 && (sizeof(S) < sizeof(T) || x <= typemax(T))
    else
        sizeof(S) <= sizeof(T) || x <= typemax(T)
    end
    return fits ? (convert(T, x), true) : (zero(T), false)
end

function _karatsuba_gpu_component(::Type{T}, x::S) where {T<:AbstractFloat,S<:Integer}
    converted = T(x)
    isfinite(converted) || return zero(T), false
    lower = S <: Signed ? T(typemin(S)) : zero(T)
    upper = T(typemax(S))
    upper_ok =
        _karatsuba_gpu_float_precision(T) >= 8 * sizeof(S) ? converted <= upper :
        converted < upper
    (converted >= lower && upper_ok) || return zero(T), false
    return converted, S(converted) == x
end

function _karatsuba_gpu_component(::Type{T}, x::S) where {T<:Integer,S<:AbstractFloat}
    (isfinite(x) && isinteger(x)) || return zero(T), false
    lower = T <: Signed ? S(typemin(T)) : zero(S)
    upper = S(typemax(T))
    upper_ok = _karatsuba_gpu_float_precision(S) >= 8 * sizeof(T) ? x <= upper : x < upper
    (x >= lower && upper_ok) || return zero(T), false
    return convert(T, x), true
end

function _karatsuba_gpu_component(::Type{T}, x::S) where {T<:AbstractFloat,S<:AbstractFloat}
    (isfinite(x) && isinteger(x)) || return zero(T), false
    converted = T(x)
    (isfinite(converted) && S(converted) == x) || return zero(T), false
    return converted, true
end

function _karatsuba_gpu_reconstruct(
    ::Type{T},
    low::T,
    high::T,
    radix::Int,
) where {T<:Signed}
    radix > 0 || return zero(T), false
    if high > 0 && (high > typemax(T) ÷ radix)
        return zero(T), false
    elseif high < 0 && (high < typemin(T) ÷ radix)
        return zero(T), false
    end
    product = high * radix
    if product > 0 && low > typemax(T) - product
        return zero(T), false
    elseif product < 0 && low < typemin(T) - product
        return zero(T), false
    end
    return convert(T, low + product), true
end

function _karatsuba_gpu_reconstruct(
    ::Type{T},
    low::T,
    high::T,
    radix::Int,
) where {T<:Unsigned}
    radix > 0 || return zero(T), false
    if high != 0
        radix <= typemax(T) || return zero(T), false
        high <= typemax(T) ÷ radix || return zero(T), false
    end
    product = high == 0 ? zero(T) : high * T(radix)
    low <= typemax(T) - product || return zero(T), false
    return low + product, true
end

function _karatsuba_gpu_reconstruct(
    ::Type{T},
    low::T,
    high::T,
    radix::Int,
) where {T<:AbstractFloat}
    radix > 0 || return zero(T), false
    radix_value = T(radix)
    isfinite(radix_value) && isinteger(radix_value) || return zero(T), false
    product = radix_value * high
    isfinite(product) || return zero(T), false
    product_error = fma(radix_value, high, -product)
    result = product + low
    isfinite(result) && isinteger(result) || return zero(T), false
    sum_virtual = result - product
    sum_error = (product - (result - sum_virtual)) + (low - sum_virtual)
    return result, product_error == -sum_error
end

function _karatsuba_gpu_linear_index(
    k::Int,
    dims::Dims{D},
    source_strides::Dims{D},
) where {D}
    remaining = k - 1
    offset = 1
    @inbounds for d = 1:D
        coordinate = rem(remaining, dims[d])
        remaining = div(remaining, dims[d])
        offset += coordinate * source_strides[d]
    end
    return offset
end

function _karatsuba_reconstruct_gpu_kernel!(
    result,
    valid,
    low,
    high,
    dims,
    low_strides,
    high_strides,
    radix,
)
    k = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if k <= length(result)
        low_index = _karatsuba_gpu_linear_index(k, dims, low_strides)
        high_index = _karatsuba_gpu_linear_index(k, dims, high_strides)
        low_value, low_ok = _karatsuba_gpu_component(eltype(result), low[low_index])
        high_value, high_ok = _karatsuba_gpu_component(eltype(result), high[high_index])
        value, reconstruction_ok =
            _karatsuba_gpu_reconstruct(eltype(result), low_value, high_value, radix)
        valid[k] = low_ok && high_ok && reconstruction_ok
        result[k] = valid[k] ? value : zero(eltype(result))
    end
    return
end

function _karatsuba_reconstruct_gpu(
    ::Type{T},
    K::KaratsubaArray{S,D,Storage,Plan},
) where {T,S,D,Storage<:AbstractArray{S,D},Plan<:AbstractArray{S,D}}
    supported(T) = isbitstype(T) && T <: Union{Integer,AbstractFloat} && sizeof(T) <= 8
    supported(S) || throw(
        ArgumentError(
            "from_karatsuba GPU components support isbits integer or floating types up to 64 bits",
        ),
    )
    supported(T) || throw(
        ArgumentError(
            "from_karatsuba GPU output supports isbits integer or floating types up to 64 bits",
        ),
    )
    low_modulus(K) > 0 ||
        throw(ArgumentError("from_karatsuba requires a positive low_modulus on GPU"))
    radix = try
        Int(low_modulus(K))
    catch
        throw(
            ArgumentError(
                "from_karatsuba GPU reconstruction requires low_modulus to fit in Int",
            ),
        )
    end
    if T <: AbstractFloat
        exact_radix = try
            BigInt(T(radix)) == BigInt(radix)
        catch
            false
        end
        exact_radix || throw(
            ArgumentError(
                "from_karatsuba GPU reconstruction requires low_modulus to be exactly representable in the requested floating type",
            ),
        )
    end

    result = CUDA.CuArray{T,D}(undef, size(K))
    isempty(result) && return result
    valid = CUDA.CuArray{Bool}(undef, length(result))
    low = _karatsuba_gpu_storage(low_part(K))
    high = _karatsuba_gpu_storage(high_part(K))
    threads = 256
    blocks = cld(length(result), threads)
    @cuda threads=threads blocks=blocks _karatsuba_reconstruct_gpu_kernel!(
        result,
        valid,
        low,
        high,
        size(K),
        _karatsuba_gpu_strides(low_part(K)),
        _karatsuba_gpu_strides(high_part(K)),
        radix,
    )
    all(valid) || throw(
        InexactError(
            :from_karatsuba,
            T,
            "GPU components or reconstructed values are not exactly representable",
        ),
    )
    return result
end

function from_karatsuba(
    ::Type{T},
    K::KaratsubaArray{S,D,Storage,Plan},
) where {T,S,D,Storage<:CuModArray{S,D},Plan<:AbstractArray{S,D}}
    return _karatsuba_reconstruct_gpu(T, K)::CUDA.CuArray{T,D}
end

function from_karatsuba(
    ::Type{T},
    K::KaratsubaArray{S,D,Storage,Plan},
) where {T,S,D,Storage<:CUDA.CuArray{S,D},Plan<:AbstractArray{S,D}}
    return _karatsuba_reconstruct_gpu(T, K)::CUDA.CuArray{T,D}
end

function _karatsuba_host_array(A::CuModArray)
    Array(A)
end

function _karatsuba_component(
    ::Type{T},
    values::AbstractArray,
    source::CuModArray,
    component_modulus,
) where {T}
    return CuModArray{T,ndims(values)}(
        values;
        modulus = Int(component_modulus),
        mod = false,
    )
end

function _karatsuba_component(
    ::Type{T},
    values::AbstractArray,
    source::CUDA.CuArray,
    component_modulus,
) where {T}
    return CUDA.CuArray(values)
end

function _karatsuba_zero_component(
    ::Type{S},
    dims::Dims{D},
    component_modulus::Int,
) where {T,D,S<:CuModArray{T,D}}
    return zeros(CuModArray{T,D}, dims...; modulus = component_modulus)
end

function _karatsuba_zero_component(
    ::Type{S},
    dims::Dims{D},
    ::Int,
) where {T,D,S<:CuArray{T,D}}
    isconcretetype(S) || throw(
        ArgumentError("CuArray zero factories require a concrete component storage type"),
    )
    component = CUDA.zeros(T, dims...)
    typeof(component) === S || throw(
        ArgumentError(
            "CUDA.zeros returned $(typeof(component)) for requested component type $S",
        ),
    )
    return component
end

function _karatsuba_zero_component(
    ::Type{S},
    ::Dims{D},
    ::Int,
) where {T,D,S<:AbstractArray{T,D}}
    throw(
        ArgumentError("Karatsuba zero factories do not support component storage type $S"),
    )
end
function Base.zeros(
    ::Type{KaratsubaArray{T,2}},
    rows::Integer,
    cols::Integer;
    kwargs...,
) where {T}
    return _karatsuba_zeros(T, CuModArray{T,2}, (Int(rows), Int(cols)); kwargs...)
end

function Base.zeros(::Type{KaratsubaArray{T,1}}, length::Integer; kwargs...) where {T}
    return _karatsuba_zeros(T, CuModArray{T,1}, (Int(length),); kwargs...)
end
function initialize_plan!(K::KaratsubaArray{T,D,S,P}) where {T,D,S,P}
    expected_plan_type = _karatsuba_plan_type(S)
    P === expected_plan_type || throw(
        ArgumentError(
            "initialize_plan! requires plan storage type $expected_plan_type, got $P",
        ),
    )
    K.plan = zeros(CuModArray{T,D}, size(low_part(K))...; modulus = Int(low_modulus(K)))
    return K
end

import Base: +, -, *

function _karatsuba_zero_component(source, dims::Dims, component_modulus)
    T = eltype(source)
    D = length(dims)
    if source isa CuModArray
        return zeros(CuModArray{T,D}, dims...; modulus = Int(component_modulus))
    elseif source isa CuArray
        return CUDA.zeros(T, dims...)
    end
    return Base.zeros(T, dims)
end

function _karatsuba_output_like(
    source::KaratsubaArray{T,D,S,P},
    dims::Dims{DOut},
) where {T,D,S,P,DOut}
    data1 = _karatsuba_zero_component(low_part(source), dims, low_modulus(source))
    data2 = _karatsuba_zero_component(high_part(source), dims, high_modulus(source))
    return KaratsubaArray(
        data1,
        data2;
        low_modulus = low_modulus(source),
        high_modulus = high_modulus(source),
        modulus = modulus(source),
    )
end

function +(A::KaratsubaArray, B::KaratsubaArray)
    K = _karatsuba_output_like(A, size(low_part(A)))
    add!(K, A, B)
    K
end

function -(A::KaratsubaArray, B::KaratsubaArray)
    K = _karatsuba_output_like(A, size(low_part(A)))
    sub!(K, A, B)
end

function *(a::Number, A::KaratsubaArray)
    K = _karatsuba_output_like(A, size(low_part(A)))
    scalar_multiply!(K, A, a)
    K
end

function *(A::KaratsubaArray, a::Number)
    a * A
end

function _karatsuba_product_workspace_like(K::KaratsubaArray{T,D,S,P}) where {T,D,S,P}
    return KaratsubaArray{T,D,S,P}(
        low_part(K),
        high_part(K);
        low_modulus = low_modulus(K),
        high_modulus = high_modulus(K),
        modulus = modulus(K),
    )
end

function _karatsuba_validate_allocating_product_storage(K::KaratsubaArray{T,D}) where {T,D}
    (T === Float32 || T === Float64) || throw(
        ArgumentError("allocating Karatsuba products support Float32 or Float64 storage"),
    )
    (low_part(K) isa CuModArray{T,D} && high_part(K) isa CuModArray{T,D}) || throw(
        ArgumentError(
            "allocating Karatsuba products require GPU-backed CuModArray{$T,$D} components",
        ),
    )
    return nothing
end

"""
    A * B

Allocate the product of GPU-backed `Float32` or `Float64` Karatsuba matrices. `A` and `B`
must satisfy the same modulus, dimension, common input low-radix, and
canonical-component requirements as `LinearAlgebra.mul!(C, A, B)`. The result uses `A`'s component split and
representation metadata; its output capacity is `low_modulus(A) *
high_modulus(A)`, which is distinct from the metadata modulus `modulus(A)`.

The operation delegates to the same three-argument `mul!` implementation using
private plan scratch. The input component storage is shared read-only, so the
caller plans may be uninitialized and any existing caller plan is left
unchanged.
"""
function *(A::KaratsubaArray{TA,2}, B::KaratsubaArray{TB,2}) where {TA,TB}
    _karatsuba_require_matched_float_type(TA, TB, TA)
    _karatsuba_validate_allocating_product_storage(A)
    _karatsuba_validate_allocating_product_storage(B)

    A_work = _karatsuba_product_workspace_like(A)
    B_work = _karatsuba_product_workspace_like(B)
    C = _karatsuba_output_like(A, (size(A, 1), size(B, 2)))
    foreach(initialize_plan!, (A_work, B_work, C))
    return LinearAlgebra.mul!(C, A_work, B_work)
end

function *(A::KaratsubaArray{TA,2}, x::KaratsubaArray{TB,1}) where {TA,TB}
    _karatsuba_require_matched_float_type(TA, TB, TA)
    _karatsuba_validate_allocating_product_storage(A)
    _karatsuba_validate_allocating_product_storage(x)

    A_work = _karatsuba_product_workspace_like(A)
    x_work = _karatsuba_product_workspace_like(x)
    y = _karatsuba_output_like(A, (size(A, 1),))
    foreach(initialize_plan!, (A_work, x_work, y))
    return LinearAlgebra.mul!(y, A_work, x_work)
end

function *(A::KaratsubaArray, B::KaratsubaArray)
    throw(
        ArgumentError(
            "allocating Karatsuba multiplication supports matrix-matrix and matrix-vector products with matching Float32 or Float64 GPU components",
        ),
    )
end

# """

# Works even for multidimensional arrays because CuArrays support linear indexing

# """
# function karatsuba_add_kernel!(Kplan,Kdata1,Kdata2,Adata1,Adata2,Bdata1,Bdata2,N1,N2)
#     i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

#     Kplan[i] = (Adata1[i] + Bdata1[i]) % (2*N1)
#     Kplan[i] = div(Kplan[i],N1)
#     Kdata2[i] = (Kplan[i] + Adata2[i]) % (2*N2)
#     Kdata2[i] = (Kdata2[i] + Bdata2[i]) % (2*N2)

#     Kplan[i] = (Kplan[i] * N1) % (N1^2)
#     Kplan[i] = (Bdata1[i] - Kplan[i]) % (N1^2)
#     Kdata1[i] = (Adata1[i] + Kplan[i]) % (N1^2)

#     Kdata1[i] %= N1
#     Kdata2[i] %= N2

#     return
# end

function add!(K::KaratsubaArray, A::KaratsubaArray, B::KaratsubaArray)
    if (modulus(A) != modulus(B)) || (modulus(A) != modulus(K))
        throw(ModulusMismatchError("Matrices must have the same modulus m"))
    end
    if (size(low_part(A)) != size(low_part(B))) ||
       (size(low_part(A)) != (size(low_part(K))))
        throw(DimensionMismatch("Matrix dimensions must match"))
    end

    tw = TILE_WIDTH

    totalsize = length(_karatsuba_logical_view(low_part(A)))

    threads = tw
    if totalsize > 0
        blocks = cld(totalsize, tw)
        @cuda threads=threads blocks=blocks karatsuba_add_kernel!(
            _karatsuba_logical_view(low_part(K)),
            _karatsuba_logical_view(high_part(K)),
            _karatsuba_logical_view(low_part(A)),
            _karatsuba_logical_view(high_part(A)),
            _karatsuba_logical_view(low_part(B)),
            _karatsuba_logical_view(high_part(B)),
            low_modulus(A),
            high_modulus(A),
        )
    end


    K
end

#"""
#This is a fallback implementation, and shouldn't be used
#unless the dimension is greater than or equal to 3.

#This is not safe if K == B, but it is safe if K == A.
#"""
#function add!(K::KaratsubaArray, A::KaratsubaArray, B::KaratsubaArray)
#    if (modulus(A) != modulus(B)) || (modulus(A) != modulus(K))
#        throw(ModulusMismatchError("Matrices must have the same modulus m"))
#    end
#    if (size(low_part(A)) != size(low_part(B))) || (size(low_part(A)) != (size(low_part(K))))
#        throw(DimensionMismatch("Matrix dimensions must match"))
#    end
#    if (A.plan == nothing) || (K.plan == nothing)
#        throw(ArgumentError("Must initialize plans first before using fallback implementation"))
#    end
#    #=
#    high_part(K) .= mod.(trunc.((low_part(A)+low_part(B))./modulus(A)),high_modulus(A))
#    low_part(K) .= mod.(low_part(A) + low_part(B) - modulus(A)*high_part(K),low_modulus(A))
#    high_part(K) .= mod.(high_part(K) + high_part(A) + high_part(B),high_modulus(A))
#    =#

#    add!(K.plan,low_part(A),low_part(B),2*low_modulus(A))
#    divide_elements!(K.plan,K.plan,low_modulus(A))
#    add!(high_part(K),K.plan,high_part(A),2*high_modulus(A))
#    add!(high_part(K),high_part(K),high_part(B),2*high_modulus(A))
#    LinearAlgebra.mul!(K.plan,K.plan,low_modulus(A),low_modulus(A)^2)
#    sub!(K.plan,low_part(B),K.plan,low_modulus(B)^2)
#    add!(low_part(K),low_part(A),K.plan,low_modulus(A)^2)
#    #=
#    LinearAlgebra.mul!(high_part(K),high_part(K),1/modulus(A))
#    trunc_elements!(high_part(K))
#    =#
#    mod_elements!(low_part(K); modulus_override=low_modulus(A))
#    mod_elements!(high_part(K); modulus_override=high_modulus(A))
#    #=
#    add!(low_part(K),low_part(A),low_part(B),2*modulus(A))
#    add!(high_part(K),high_part(A),high_part(B))
#    add!(high_part(K),high_part(K),divides(low_part(K),modulus(K)),high_modulus(K))
#    mod_elements!(low_part(K); modulus_override=low_modulus(K))
#    =#

#    K
#end

function sub!(K::KaratsubaArray, A::KaratsubaArray, B::KaratsubaArray)
    if (modulus(A) != modulus(B)) || (modulus(A) != modulus(K))
        throw(ModulusMismatchError("Matrices must have the same modulus m"))
    end
    if (size(low_part(A)) != size(low_part(B))) ||
       (size(low_part(A)) != (size(low_part(K))))
        throw(DimensionMismatch("Matrix dimensions must match"))
    end

    #=
    high_part(C) .= mod.(trunc.((low_part(A)-low_part(B))./modulus(A)),high_modulus(A))
    low_part(C) .= mod.(low_part(A) - low_part(B) - modulus(A)*high_part(C),low_modulus(A))
    high_part(C) .= mod.(high_part(C) + high_part(A) - high_part(B) - (low_part(A).<low_part(B)),high_modulus(A))

    sub!(high_part(K),low_part(A),low_part(B))
    LinearAlgebra.mul!(high_part(K),high_part(K),1/modulus(A))
    trunc_elements!(high_part(K))
    mod_elements!(high_part(K); modulus_override=high_modulus(A))
    LinearAlgebra.mul!(low_part(K),low_part(K),modulus(A),modulus(A)^2)
    sub!(low_part(K),low_part(B),low_part(K))
    sub!(low_part(K),low_part(A),low_part(K))
    mod_elements!(low_part(K); modulus_override=low_modulus(A))
    add!(high_part(K),high_part(K),high_part(A))
    sub!(high_part(K),high_part(K),high_part(B))
    sub!(high_part(K),high_part(K),low_part(A).<low_part(B))
    mod_elements!(high_part(K); modulus_override=high_modulus(A))
    =#
    tw = TILE_WIDTH

    totalsize = length(_karatsuba_logical_view(low_part(A)))

    threads = tw
    if totalsize > 0
        blocks = cld(totalsize, tw)
        @cuda threads=threads blocks=blocks karatsuba_sub_kernel!(
            _karatsuba_logical_view(low_part(K)),
            _karatsuba_logical_view(high_part(K)),
            _karatsuba_logical_view(low_part(A)),
            _karatsuba_logical_view(high_part(A)),
            _karatsuba_logical_view(low_part(B)),
            _karatsuba_logical_view(high_part(B)),
            low_modulus(A),
            high_modulus(A),
            modulus(A),
        )
    end

    # negate!(K,B)
    # add!(K,K,A)
    K
end

function scalar_multiply!(B::KaratsubaArray, A::KaratsubaArray, s::Number)
    if modulus(A) != modulus(B)
        throw(ModulusMismatchError("Matrices must have the same modulus m"))
    end
    if size(low_part(A)) != size(low_part(B))
        throw(DimensionMismatch("Matrix dimensions must match"))
    end

    # LinearAlgebra.mul!(high_part(B),low_part(A),s,low_modulus(A)^2)
    # divide_elements!(high_part(B),high_part(B),low_modulus(A))
    # LinearAlgebra.mul!(low_part(B),high_part(A),s,high_modulus(A))
    # add!(high_part(B),high_part(B),low_part(B))
    # LinearAlgebra.mul!(low_part(B),low_part(A),s,low_modulus(A))

    tw = TILE_WIDTH

    totalsize = length(_karatsuba_logical_view(low_part(A)))

    threads = tw
    if totalsize > 0
        blocks = cld(totalsize, tw)
        @cuda threads=threads blocks=blocks karatsuba_scalar_multiply_kernel!(
            _karatsuba_logical_view(low_part(B)),
            _karatsuba_logical_view(high_part(B)),
            _karatsuba_logical_view(low_part(A)),
            _karatsuba_logical_view(high_part(A)),
            s,
            low_modulus(A),
            high_modulus(A),
        )
    end

    #=
    high_part(B) .= mod.(trunc.((s*low_part(A))/modulus(A)),high_modulus(A))
    low_part(B) .= mod.(s*low_part(A) - modulus(B)*high_part(B),low_modulus(A))
    high_part(B) .= mod.(high_part(B) + s*high_part(A),high_modulus(A))
    =#
    B
end



# function negate!(K::KaratsubaArray, A::KaratsubaArray)

#     tw = TILE_WIDTH

#     totalsize = length(parent(low_part(A)))

#     threads = tw
#     blocks = totalsize ÷ tw

# @cuda threads=threads blocks=blocks karatsuba_negate_kernel!(parent(low_part(K)),
#                                                              parent(high_part(K)),
#                                                              parent(low_part(A)),
#                                                              parent(high_part(A)),
#                                                              low_modulus(A),
#                                                              high_modulus(A),
#                                                             modulus(A))


#     K
# end

function negate!(K::KaratsubaArray, A::KaratsubaArray)
    tw = TILE_WIDTH

    totalsize = length(_karatsuba_logical_view(low_part(A)))

    threads = tw
    if totalsize > 0
        blocks = cld(totalsize, tw)
        @cuda threads=threads blocks=blocks karatsuba_negate_kernel!(
            _karatsuba_logical_view(low_part(K)),
            _karatsuba_logical_view(high_part(K)),
            _karatsuba_logical_view(low_part(A)),
            _karatsuba_logical_view(high_part(A)),
            low_modulus(A),
            high_modulus(A),
            modulus(A),
        )
    end



    #=
    high_part(K) .= trunc.((modulus(A) .- low_part(A))/modulus(A))
    low_part(K) .= mod.(modulus(A) - low_part(A) - modulus(A)*high_part(K),low_modulus(A))
    high_part(K) .= mod.(mod.(high_part(K) + modulus(A) - high_part(A) - 1,modulus(A)),high_modulus(A))
    =#
    # mul!(high_part(K),high_part(K),0)
    # scalar_add!(high_part(K),high_part(K),low_modulus(A),2*low_modulus(A))
    # sub!(high_part(K),high_part(K),low_part(A),2*low_modulus(A))
    #=
    LinearAlgebra.mul!(high_part(K),high_part(K),1/low_modulus(A))
    trunc_elements!(high_part(K))
     =#
    # divide_elements!(high_part(K),high_part(K),low_modulus(A))
    # LinearAlgebra.mul!(low_part(K),high_part(K),low_modulus(A),modulus(A)^2)
    # add!(low_part(K),low_part(K),low_part(A),modulus(A)^2)
    # negate!(low_part(K),low_part(K),low_modulus(A))
    # mod_elements!(low_part(K); modulus_override=low_modulus(A))
    # scalar_add!(high_part(K),high_part(K),high_modulus(A),modulus(A)^2)
    # scalar_sub!(high_part(K),high_part(K),1,modulus(A)^2)
    # sub!(high_part(K),high_part(K),high_part(A),modulus(A))
    # mod_elements!(high_part(K); modulus_override=high_modulus(A))
    K
end

function divide_elements!(B::CuModArray, A::CuModArray, N::Integer)
    parent(B) .= div.(parent(A), N)
    return A
end

function Base.copy(A::KaratsubaArray{T,D,S,P}) where {T,D,S,P}
    compute_copy =
        () -> begin
            B = KaratsubaArray{T,D,S,P}(
                copy(low_part(A)),
                copy(high_part(A));
                low_modulus = copy(low_modulus(A)),
                high_modulus = copy(high_modulus(A)),
                modulus = modulus(A),
            )
            if !(A.plan == nothing)
                initialize_plan!(B)
            end
            B
        end

    component = low_part(A)
    component_storage = component isa CuModArray ? parent(component) : component
    if component_storage isa CUDA.CuArray
        return CUDA.device!(compute_copy, CUDA.device(component_storage))
    end

    return compute_copy()
end
