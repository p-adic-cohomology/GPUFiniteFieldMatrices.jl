function Base.:^(A::CuModMatrix, n::Integer)
    if rows(A) != cols(A)
        throw(DimensionMismatch("Matrix must be square for power operation"))
    end

    if n == 0
        I_mat = Matrix{eltype(parent(A))}(I, rows(A), cols(A))
        return CuModMatrix(I_mat; modulus = modulus(A))
    elseif n < 0
        return inv(A)^(-n)
    else
        if n == 1
            return A
        elseif n % 2 == 0
            half_pow = A^(n ÷ 2)
            return half_pow * half_pow
        else
            return A * (A^(n-1))
        end
    end
end

function Base.transpose(A::CuModMatrix)
    return CuModMatrix(transpose(parent(A)); modulus = modulus(A), new_size = size(A))
end

rows(A::CuModMatrix) = size(A, 1)
cols(A::CuModMatrix) = size(A, 2)
Base.getindex(A::CuModArray, i::Int, j::Int) = parent(A)[i, j]
Base.setindex!(A::CuModArray, value, i::Int, j::Int) =
    (parent(A)[i, j] = mod(value, modulus(A)))
Base.setindex!(A::CuModArray, value, i::Int) = (parent(A)[i] = mod(value, modulus(A)))

function Base.Array(A::CuModArray)
    data = Array(parent(A))
    return data[CartesianIndices(map(d -> 1:d, size(A)))]
end

unsafe_Array(A::CuModArray) = Array(parent(A))

function _show_cumod_header(io::IO, name::AbstractString, A::CuModArray)
    print(io, join(size(A), "×"), " ", name, " modulo ", modulus(A))
end

Base.show(io::IO, A::CuModArray) = _show_cumod_header(io, "CuModArray", A)
Base.show(io::IO, ::MIME"text/plain", A::CuModArray) = begin
    _show_cumod_header(io, "CuModArray", A)
    print(io, '\n')
    show(io, Array(A))
end

function _setup_PLUQ(A::CuModMatrix)
    return pluq(A)
end

function _is_invertible(F)
    m = rows(F.LU)
    n = cols(F.LU)
    return m == n && F.rank == m
end

# The documented singular and inverse results are two intentional concrete
# tuple alternatives; keep their union allowance local to this API method.
@stable default_mode = "disable" default_union_limit = 2 begin
    """
        is_invertible_with_inverse(A::CuModMatrix)

    Checks if a matrix is invertible. If not, returns
    false and nothing. If it is, returns true and the
    inverse matrix.
    """
    function is_invertible_with_inverse(A::CuModMatrix; debug::Bool = false)
        F = _setup_PLUQ(A)
        if !_is_invertible(F)
            return false, nothing
        end
        A_inv = inv(A)
        if debug
            println("A")
            display(A)
            println("A_inv")
            display(A_inv)
            println("A * A_inv")
            display(A * A_inv)
        end
        return true, A_inv
    end
end

function inverse_permutation(P::CuDeviceVector{Int,1})
    # TODO: Pad the perm array
    P_inv = CuDeviceVector{Int,1}(undef, length(P))

    @cuda threads=TILE_WIDTH blocks=div(length(P), TILE_WIDTH) _inverse_permutation_kernel!(
        P_inv,
        P,
    )

    return P_inv
end

function _inverse_permutation_kernel!(
    P_inv::CuDeviceVector{Int,1},
    P::CuDeviceVector{Int,1},
)
    # TODO: Pad the perm array
    tid = threadIdx().x
    bid = blockIdx().x
    row = (bid - 1) * TILE_WIDTH + tid
    n = length(P)

    if row <= n
        P_inv[P[row]] = row
    end

    return
end

function Base.zeros(::Type{CuModArray{T,1}}, length::Integer; modulus::Int) where {T}
    padded_entries = _padded_extent(length)
    data = CUDA.zeros(T, padded_entries)
    return CuModArray{T,1}(data; modulus = modulus, mod = false, new_size = (length,))
end

function Base.zeros(
    ::Type{CuModArray{T,2}},
    rows::Integer,
    cols::Integer;
    modulus::Int,
    new_size::Tuple{Int,Int} = (rows, cols),
) where {T}
    padded_rows = _padded_extent(rows)
    padded_cols = _padded_extent(cols)
    data = CUDA.zeros(T, padded_rows, padded_cols)
    return CuModArray{T,2}(data; modulus = modulus, mod = false, new_size = new_size)
end

"""
    rand(CuModVector{T}, length; modulus)
    rand(CuModMatrix{T}, rows, cols; modulus)

Creates a random vector or matrix in the finite field modulo `modulus`.
Only logical entries are sampled; the zero-initialized padding is untouched.
"""
function Base.rand(::Type{CuModArray{T,1}}, length::Integer; modulus::Int) where {T}
    dims = (Int(length),)
    data = CUDA.zeros(T, _padded_size(dims)...)
    if !isempty(data)
        logical_data = CUDA.rand(T, dims...)
        if T <: AbstractFloat
            logical_data .= floor.(logical_data .* modulus)
        end
        mod!(logical_data, logical_data, modulus)
        copyto!(_logical_view(data, dims), logical_data)
    end
    return CuModArray{T,1}(data; modulus = modulus, mod = false, new_size = dims)
end

function Base.rand(
    ::Type{CuModArray{T,2}},
    rows::Integer,
    cols::Integer;
    modulus::Int,
) where {T}
    dims = (Int(rows), Int(cols))
    data = CUDA.zeros(T, _padded_size(dims)...)
    if !isempty(data)
        logical_data = CUDA.rand(T, dims...)
        if T <: AbstractFloat
            logical_data .= floor.(logical_data .* modulus)
        end
        mod!(logical_data, logical_data, modulus)
        copyto!(_logical_view(data, dims), logical_data)
    end
    return CuModArray{T,2}(data; modulus = modulus, mod = false, new_size = dims)
end

"""
    elementwise_multiply!(C, A, B; modulus_override = nothing)

In-place element-wise multiplication: C = A .* B mod N. No allocation is performed.
Only a positive `modulus_override` replaces `modulus(C)` for the reduction modulus.
A zero or negative value throws `ArgumentError`. All stored moduli must still
match, including when an override is supplied. Logical sizes and backing lengths
must also match. These compatibility checks may be elided by an `@inbounds`
caller, who is responsible for satisfying them.
"""
Base.@propagate_inbounds @inline function elementwise_multiply!(
    C::CuModArray,
    A::CuModArray,
    B::CuModArray,
    ;
    modulus_override::Union{Nothing,Integer} = nothing,
)
    _validate_positive_modulus_override(modulus_override)
    N = modulus_override === nothing ? modulus(C) : modulus_override

    mul_elementwise!(C, A, B; modulus_override = N)
    return C
end

"""
    neg!(B, A; modulus_override = nothing)

In-place negation: B = -A mod N. No allocation is performed.
Only a positive `modulus_override` replaces `modulus(B)` for the reduction modulus.
A zero or negative value throws `ArgumentError`. Stored moduli, logical sizes
and backing lengths must match. Compatibility checks may be elided by an
`@inbounds` caller, who is responsible for satisfying them.
"""
Base.@propagate_inbounds @inline function neg!(
    B::CuModArray,
    A::CuModArray;
    modulus_override::Union{Nothing,Integer} = nothing,
)
    _validate_positive_modulus_override(modulus_override)
    @boundscheck _check_elementwise_compatible(B, A)
    N = modulus_override === nothing ? modulus(B) : modulus_override

    rscalar_sub!(B, A, 0; modulus_override = N)
    return B
end

"""
    rmul!(A,s)

Scales the array A by s, overwriting A.
As ZZ/n is commutative, this has the same behavior as lmul!

Extends LinearAlgebra.rmul! and LinearAlgebra.lmul!
"""
Base.@propagate_inbounds @inline function LinearAlgebra.rmul!(A::CuModArray, s::Number)
    return mul!(A, A, s)
end

"""
    lmul!(s,A)

Scales the array A by s, overwriting A.
As ZZ/n is commutative, this has the same behavior as lmul!
"""
Base.@propagate_inbounds @inline function LinearAlgebra.lmul!(s::Number, A::CuModArray)
    return mul!(A, A, s)
end

function dumb_copy_kernel!(B, A)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    if i <= length(A) && i <= length(B)
        B[i] = A[i]
    end

    nothing
end

"""
    copy!(B, A)

In-place copy: B = A. No allocation is performed.
B is updated with the contents of A.

This copies all logical entries across valid backing layouts. Padding is outside
the copy and is left unchanged in the destination. The operation does not test
the modulus or normalize entries.
"""
function Base.copy!(B::CuModArray, A::CuModArray)
    if size(A) != size(B)
        throw(DimensionMismatch("Matrix dimensions must match"))
    end

    source = _logical_view(parent(A), size(A))
    destination = _logical_view(parent(B), size(B))
    threads = TILE_WIDTH
    count = length(source)
    if count > 0
        blocks = cld(count, threads)
        @cuda threads=threads blocks=blocks dumb_copy_kernel!(destination, source)
    end

    return B
end

"""
    copyto!(dest::CuModArray, src::DenseArray)

Copy the data from dest to src. This is useful for sending
data to the gpu.

Does not mod.
"""
function Base.copyto!(dest::CuModArray{T,D}, src::DenseArray{T,D}) where {T,D}
    if size(dest) != size(src)
        throw(DimensionMismatch("Matrix dimensions must match"))
    end

    #rangesize = map(x -> 1:x,size(dest))
    #data_inds = CartesianIndices(rangesize)
    inds = CartesianIndices(src)

    #dataview = @view dest[data_inds]
    copyto!(parent(dest), inds, src, inds)
    return dest
end

function Base.copyto!(
    dest::CuModMatrix{T},
    src::SparseArrays.CHOLMOD.Dense{T},
) where {T<:Union{Float32,Float64,ComplexF32,ComplexF64}}
    return copyto!(dest, Matrix(src))
end


"""
    fill!(A::CuModArray{T,D}, s::T) where {T,D}

fills (in place) A with the value s (mod N)
"""
function Base.fill!(A::CuModArray, s::Number)
    smod = mod(s, modulus(A))

    fill!(parent(A), smod)
end

"""
    zero!(A::CuModArray)

Sets all of the entries of A to zero, in place.

"""
zero!(A::CuModArray) = fill!(parent(A), zero(eltype(parent(A))))

"""
    mod_elements!(A; modulus_override = nothing)

Apply modulus to all elements of A in-place.
Only a positive `modulus_override` replaces `modulus(A)` for the reduction modulus.
A zero or negative value throws `ArgumentError`. The stored modulus on `A`
is unchanged.
"""
function mod_elements!(A::CuModArray; modulus_override::Union{Nothing,Integer} = nothing)
    _validate_positive_modulus_override(modulus_override)
    N = modulus_override === nothing ? modulus(A) : modulus_override
    mod!(parent(A), parent(A), N)
    return A
end

function trunc_elements!(A::CuModArray)
    parent(A) .= trunc.(parent(A))
    return A
end

# Utility functions to change modulus
"""
    change_modulus(A, new_N)

Create an independent CuModArray with modulus `new_N`. Logical elements are
reduced only when `new_N` is smaller than `modulus(A)`; otherwise they are copied.
The new storage rounds each logical
dimension up to a multiple of `TILE_WIDTH` and has zero padding.
"""
function change_modulus(A::CuModArray, new_N::Integer)
    T = eltype(parent(A))
    D = ndims(A)
    dims = size(A)
    new_modulus = Int(new_N)
    data = CUDA.zeros(T, _padded_size(dims)...)
    result = CuModArray{T,D}(data; modulus = new_modulus, mod = false, new_size = dims)

    if !isempty(A)
        logical_source = _logical_view(parent(A), dims)
        logical_result = _logical_view(parent(result), dims)
        if new_N < modulus(A)
            logical_result .= mod.(logical_source, new_N)
        else
            copyto!(logical_result, logical_source)
        end
    end

    return result
end

"""
    change_modulus!!(A, new_N)

Consumes `A` and returns a new immutable `CuModArray` wrapper over the same GPU
storage with modulus `new_N`. The wrapper shares its parent storage with `A`.
Use the returned wrapper, for example `A = change_modulus!!(A, p)`, and do not
use the consumed wrapper afterward. Aliases to the parent share storage and
observe the in-place reduction when the modulus shrinks.

`new_N` must be positive. A smaller modulus performs one in-place reduction.
Equal or larger moduli leave values unchanged and reinterpret them.
The wrapper may allocate small host metadata; no GPU storage is allocated.
"""
function change_modulus!!(A::CuModArray{T,D}, new_N::Integer) where {T,D}
    new_N > 0 || throw(ArgumentError("new_N must be positive"))
    new_modulus = Int(new_N)

    if new_modulus < modulus(A)
        mod!(parent(A), parent(A), new_modulus)
    end

    return CuModArray{T,D}(parent(A); modulus = new_modulus, new_size = size(A))
end

"""
    mul!(C, A, B; max_ops=nothing, input_bound=nothing)

In-place matrix multiplication: `C = A * B mod modulus(C)`. `input_bound` is the
exclusive upper bound on nonnegative multiplicands. `max_ops` counts product
contributions accumulated per output entry between modular reductions. When
omitted, it is derived from the effective input bound and reduction modulus; a
safe explicit value is honored, while an unsafe value throws `ArgumentError`.
An explicit `input_bound` may require multiple stripes when the inner dimension
exceeds the safe budget.
"""
function LinearAlgebra.mul!(
    C::CuModMatrix,
    A::CuModMatrix,
    B::CuModMatrix;
    max_ops = nothing,
    input_bound = nothing,
)

    if (modulus(A) != modulus(B) || modulus(A) != modulus(C))
        throw(ModulusMismatchError("All matrices must have the same modulus N"))
    end
    if cols(A) != rows(B)
        throw(DimensionMismatch("Matrix dimensions do not match for multiplication"))
    end
    if rows(C) != rows(A) || cols(C) != cols(B)
        throw(DimensionMismatch("Output matrix C has incorrect dimensions"))
    end

    stripe_mul!(C, A, B; max_ops = max_ops, input_bound = input_bound)
    return C
end

"""
    mul!(z, A, x; input_bound=nothing, modulus_override=nothing)

In-place matrix-vector multiplication: `z = A * x`. `input_bound` is the
exclusive upper bound on nonnegative multiplicands and controls the default
`max_ops`. A supplied `modulus_override` must be positive and sets the per-call
reduction modulus; when omitted, `modulus(z)` is used and its stored modulus is
unchanged.
"""
function LinearAlgebra.mul!(
    z::CuModVector,
    A::CuModMatrix,
    x::CuModVector;
    input_bound = nothing,
    modulus_override = nothing,
)

    _validate_positive_modulus_override(modulus_override)
    if (
        modulus(A) != modulus(x) ||
        (modulus_override === nothing && modulus(A) != modulus(z))
    )
        throw(ModulusMismatchError("All matrices must have the same modulus N"))
    end
    if cols(A) != length(x)
        throw(DimensionMismatch("Matrix dimensions do not match for multiplication"))
    end
    if length(z) != rows(A)
        throw(DimensionMismatch("Output matrix C has incorrect dimensions"))
    end

    stripe_mul!(z, A, x; input_bound = input_bound, modulus_override = modulus_override)
    return z
end

"""
    copy(A::CuModArray)

Copy the logical data and metadata into newly allocated storage. The new
storage rounds each logical dimension up to a multiple of `TILE_WIDTH` and has
zero padding, regardless of the source padding contents.
"""
function Base.copy(A::CuModArray{T,D}) where {T,D}
    return CUDA.device!(CUDA.device(parent(A))) do
        dims = size(A)
        data = CUDA.zeros(T, _padded_size(dims)...)
        if !isempty(A)
            copyto!(_logical_view(data, dims), _logical_view(parent(A), dims))
        end
        CuModArray{T,D}(data; modulus = modulus(A), mod = false, new_size = dims)
    end
end

#TODO: addmul! (add and scalar multiply), gemv!, gemm!
#
#Note: gemv! and gemm! could be more efficient by incorporating the add into the
#CUBLAS gemm/gemv that happens in stripe_mul!
