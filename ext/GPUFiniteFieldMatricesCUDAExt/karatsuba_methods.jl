function _karatsuba_cuda_wrapper(::Type{T}, ::Val{D}, ::Type{M}) where {T,D,M}
    return GPUModArray{T,D,FFM.CuStorage{T,D,M,CuArray{T,D,M}}}
end

function _default_cumod_storage_type(::Type{T}, ::Val{D}) where {T,D}
    return _karatsuba_cuda_wrapper(T, Val(D), CUDA.CUDACore.default_memory)
end

function _backend_default_component_type(
    ::Val{:cuda},
    ::Type{T},
    ::Val{D},
    storage,
) where {T,D}
    _cuda_storage_mode(storage)
    return _default_cumod_storage_type(T, Val(D))
end

_karatsuba_plan_type(::Type{<:CuModArray{T,D,M}}) where {T,D,M} =
    _karatsuba_cuda_wrapper(T, Val(D), M)
_karatsuba_plan_type(::Type{CuArray{T,D,M}}) where {T,D,M} =
    _karatsuba_cuda_wrapper(T, Val(D), M)

function _karatsuba_zero_component(
    ::Type{S},
    dims::Dims{D},
    component_modulus::Int;
    storage,
    device,
) where {T,D,M,S<:CuModArray{T,D,M}}
    _cuda_storage_mode(storage)
    _validate_backend_storage(Val(:cuda), T, component_modulus)
    S === _karatsuba_cuda_wrapper(T, Val(D), M) || throw(
        ArgumentError("Karatsuba zero factories require dense CUDA component storage"),
    )
    CUDA.functional() || throw(ArgumentError("the selected CUDA backend is not functional"))
    native = _with_cuda_device(device) do
        result = CuArray{T,D,M}(undef, _padded_size(dims))
        fill!(result, zero(T))
    end
    return _construct_cuda_storage(T, native, dims, component_modulus)
end

function _karatsuba_zero_component(
    ::Type{CuArray{T,D,M}},
    dims::Dims{D},
    ::Int;
    storage,
    device,
) where {T,D,M}
    _cuda_storage_mode(storage)
    CUDA.functional() || throw(ArgumentError("the selected CUDA backend is not functional"))
    return _with_cuda_device(device) do
        fill!(CuArray{T,D,M}(undef, dims), zero(T))
    end
end

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

import Base: *

_karatsuba_output_like(source::KaratsubaArray, dims::Dims) =
    FFM._karatsuba_output_like(source, dims)

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

function divide_elements!(B::CuModArray, A::CuModArray, N::Integer)
    parent(B) .= div.(parent(A), N)
    return A
end
