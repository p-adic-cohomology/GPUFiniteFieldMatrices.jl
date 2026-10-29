
const TILE_WIDTH = 32
const DEFAULT_TYPE = Float32

@inline _padded_extent(d::Integer) = cld(d, TILE_WIDTH) * TILE_WIDTH
@inline _padded_size(dims::Dims{D}) where {D} = ntuple(i -> _padded_extent(dims[i]), D)

@inline function _logical_view(data::AbstractArray{T,D}, dims::Dims{D}) where {T,D}
    return view(data, ntuple(i -> 1:dims[i], D)...)
end

struct CuModArraySizeMismatchException <: Exception
    message::String
end

struct CuModArrayModulusMismatchException <: Exception
    message::String
end

struct CuModMatrixTooLargeException <: Exception
    message::String
end

struct CuModMatrixNotSquareException <: Exception
    message::String
end

struct CuModMatrixModulusNotPrimeException <: Exception
    message::String
end

struct InverseOverflowError <: Exception
    message::String
end

struct InverseNotDefinedException <: Exception
    message::String
end

"""
    CuModArray{T}

A matrix over a finite field mod N, implemented using CuArray. Newly allocated
GPU storage rounds each dimension up to a multiple of `TILE_WIDTH` (32) and
starts with zero padding. Later operations may change padding values; logical
operations ignore padding contents. Wrapping an existing `CuArray` preserves
its storage and does not pad or clear it.

Use `modulus(A)` for the modulus, `size(A)` for the logical dimensions, and
`parent(A)` for the padded GPU storage. The `data`, `size`, and `N` fields are
internal implementation details.

Note matrices over the permitted size for efficient matmul may be created, but matmul 
will throw an exception if these matrices are multiplied.
"""
struct CuModArray{T,D} <: AbstractArray{T,D}
    data::CuArray{T,D}  # Padded CuArray data
    size::Dims{D}        # True dimensions
    N::Int               # The modulus N

    """
        CuModArray{T,D}(A::AbstractArray{T}; modulus)

    General contructor from abstract CPU array.
    Pads data on the GPU to fit a multiple of 32.
    """
    function CuModArray{T,D}(
        A::AbstractArray{S,D},
        ;
        modulus::Int,
        mod = true,
        new_size = nothing,
    ) where {T,D,S}

        N = modulus

        if 2^52 < N
            throw(
                CuModArrayModulusMismatchException(
                    "Modulus $N is bigger than 2^52 = $(2^52), the largest supported modulus",
                ),
            )
        end

        padded_size = _padded_size(size(A))

        # Initialize the padded areas to zero
        data = CUDA.fill(zero(T), padded_size...)
        # data = CUDA.CuArray{T}(I, padded_size...)

        A_inds = CartesianIndices(A)
        #rangesize = map(x -> 1:x,size(A))
        #data_inds = CartesianIndices(rangesize)

        # The desired behavior is for this to error if the conversion is
        # impossible
        if !(S <: T)
            converted = convert.(T, A)
        else
            converted = A
        end

        #TODO: We would like to use an in-place version:
        #`copyto!(data, data_inds, converted, A_inds)`
        #but this doesn't seem to be implemented in CUDA.jl.
        #For now, we can deal with a little extra (cpu) allocation
        #in creation of matrices.
        #dataview = @view data[data_inds]
        copyto!(data, A_inds, converted, A_inds)

        if mod && !isempty(data)
            mod!(data, data, N)
        end

        if new_size != nothing
            new{T,D}(data, new_size, N)
        else
            new{T,D}(data, size(A), N)
        end
    end

    """
        CuModArray{T,D}(data::CuArray; modulus)

    Wrapper constructor for results already on the GPU.
    Does not pad.

    Thus, for some functionality to work properly, you must manage
    the padding yourself.

    This constructor is useful for creating a new CuModArray while 
    keeping the same data as a previous CuModarray.
    """
    function CuModArray{T,D}(
        A::CuArray{T,D},
        ;
        modulus::Int,
        mod = false,
        new_size = nothing,
    ) where {T,D}
        data = A
        N = modulus

        if new_size != nothing
            new{T,D}(data, new_size, N)
        else
            new{T,D}(data, size(data), N)
        end
    end
end

"""Return the modulus associated with `A`."""
modulus(A::CuModArray) = getfield(A, :N)

"""
    parent(A::CuModArray)

Return the GPU storage backing `A`. Newly allocated storage rounds each logical
dimension up to a multiple of `TILE_WIDTH`; a wrapped `CuArray` keeps its
original dimensions. `size(A)` reports logical dimensions in either case.
"""
Base.parent(A::CuModArray) = getfield(A, :data)

"""
    CuModArray(A::AbstractArray; modulus)
    CuModArray{T,D}(A::AbstractArray; modulus)

Construct from a CPU array. Newly allocated storage rounds each logical
dimension up to a multiple of 32 and starts with zero padding.
The unparameterized constructor defaults to Float32.
"""
function CuModArray(
    A::AbstractArray{T,D},
    ;
    modulus::Int,
    mod = true,
    new_size = nothing,
) where {T,D}
    CuModArray{DEFAULT_TYPE,D}(A; modulus = modulus, mod = mod, new_size = new_size)
end

const CuModMatrix{T} = CuModArray{T,2}
const CuModVector{T} = CuModArray{T,1}

"""
    CuModMatrix(A::AbstractMatrix; modulus)
    CuModMatrix{T}(A::AbstractMatrix; modulus)

Construct from a CPU matrix. Newly allocated storage rounds each logical
dimension up to a multiple of 32 and starts with zero padding.
The unparameterized constructor defaults to Float32.
"""
function CuModMatrix(
    A::AbstractMatrix{T};
    modulus::Int,
    mod = true,
    new_size = nothing,
) where {T}
    CuModArray{DEFAULT_TYPE,2}(A; modulus = modulus, mod = mod, new_size = new_size)
end

"""
    CuModMatrix(data::CuArray; modulus)

Wrapper constructor for results already on the GPU.
Does not pad.

Thus, for some functionality to work properly, you must manage
the padding yourself.

This constructor is useful for creating a new CuModMatrix while 
keeping the same data as a previous CuModMatrix.
"""
function CuModMatrix(
    A::CuArray{T,2};
    modulus::Int,
    mod = false,
    new_size = nothing,
) where {T}
    CuModArray{T,2}(A; modulus = modulus, mod = mod, new_size = new_size)
end

"""
    CuModVector(A::AbstractVector; modulus)
    CuModVector{T}(A::AbstractVector; modulus)

Construct from a CPU vector. Newly allocated storage rounds its logical length
up to a multiple of 32 and starts with zero padding.
The unparameterized constructor defaults to Float32.
"""
function CuModVector(
    A::AbstractVector{T};
    modulus::Int,
    mod = true,
    new_size = nothing,
) where {T}
    CuModArray{DEFAULT_TYPE,1}(A; modulus = modulus, mod = mod, new_size = new_size)
end

"""
    CuModVector(data::CuArray; modulus)

Wrapper constructor for results already on the GPU.
Does not pad.
"""
function CuModVector(
    A::CuArray{T,1};
    modulus::Int,
    mod = false,
    new_size = nothing,
) where {T}
    CuModArray{T,1}(A; modulus = modulus, mod = mod, new_size = new_size)
end

function (::Type{CuModArray{T,2}})(
    ::LinearAlgebra.UniformScaling,
    rows::Integer,
    cols::Integer;
    modulus::Int,
) where {T}
    return CuModArray{T,2}(Matrix{T}(I, rows, cols); modulus = modulus)
end

"""
    identity_matrix(CuModMatrix{T}, n; modulus)

Creates an `n × n` identity matrix in the finite field modulo `modulus`.
Its storage rounds each dimension up to a multiple of `TILE_WIDTH` and starts
with zero padding.
"""
function identity_matrix(::Type{CuModArray{T,2}}, n::Integer; modulus::Int) where {T}
    return CuModArray{T,2}(I, n, n; modulus = modulus)
end

"""
    get_bits(type)

Return the integer precision budget for a supported element type. Floating
formats use their integer significand width; signed integers exclude the sign
bit, and unsigned integers use their full fixed width.
"""
function get_bits(type::Type)
    if type <: AbstractFloat
        type === Float16 && return 11
        type === Float32 && return 24
        type === Float64 && return 53
    elseif type <: Unsigned
        type in (UInt8, UInt16, UInt32, UInt64, UInt128) || error(
            "unsupported type $type; expected Float16, Float32, Float64, or a fixed-width signed or unsigned integer type",
        )
        return 8 * sizeof(type)
    elseif type <: Signed
        type in (Int8, Int16, Int32, Int64, Int128) || error(
            "unsupported type $type; expected Float16, Float32, Float64, or a fixed-width signed or unsigned integer type",
        )
        return 8 * sizeof(type) - 1
    end

    error(
        "unsupported type $type; expected Float16, Float32, Float64, or a fixed-width signed or unsigned integer type",
    )
end
get_bits(type) = error(
    "unsupported type $type; expected Float16, Float32, Float64, or a fixed-width signed or unsigned integer type",
)

"""
    find_max_ops(type, N)

Return the maximum operation count permitted by the precision budget for a
supported element type and dimension `N`.
"""
function find_max_ops(type, N)
    bits = get_bits(type)
    maxterm = BigInt(max(N - 1, 1))^2
    value = (BigInt(2)^bits - 1) ÷ maxterm - 1
    return value > typemax(Int) ? typemax(Int) : max(Int(value), 0)
end

Base.size(A::CuModArray) = A.size
Base.size(A::CuModArray, i::Int) = A.size[i]

rows(A::CuModArray{T,2}) where {T} = size(A)[1]
cols(A::CuModArray{T,2}) where {T} = size(A)[2]

Base.length(A::CuModArray{T,1}) where {T} = size(A)[1]
Base.length(A::CuModArray{T,2}) where {T} = rows(A)*cols(A)
Base.length(A::CuModArray) = prod(size(A))

Base.eltype(A::CuModArray) = eltype(parent(A))

# User needs to do CUDA.@allowscalar to use these
#
# Also, 3D indexing is currently not supported because we have no
# use cases.
Base.getindex(A::CuModArray, i::Int) = parent(A)[i]
Base.getindex(A::CuModArray, i::Int, j::Int) = parent(A)[i, j]

Base.setindex!(A::CuModArray, v, i::Int, j::Int) = parent(A)[i, j] = mod(v, modulus(A))
Base.setindex!(A::CuModArray, v, i::Int) = parent(A)[i] = mod(v, modulus(A))

Base.getindex(A::CuModArray, I::AbstractArray{Int}, J::AbstractArray{Int}) =
    CuModArray(Array(parent(A)[I, J]); modulus = modulus(A))
Base.getindex(A::CuModArray, I::AbstractArray{Int}, j::Int) =
    CuModArray(reshape(Array(parent(A)[I, j]), length(I), 1); modulus = modulus(A))
Base.getindex(A::CuModArray, i::Int, J::AbstractArray{Int}) =
    CuModArray(reshape(Array(parent(A)[i, J]), 1, length(J)); modulus = modulus(A))

# Convert back to CPU array, with padding
# Unsafe because there is no guarantee that the padding is zero
function unsafe_Array(A::CuModArray)
    Array(parent(A))
end

# Convert back to CPU array, without padding
function Base.Array(A::CuModArray)

    rangesize = map(x -> 1:x, size(A))
    inds = CartesianIndices(rangesize)
    Array(parent(A))[inds]
end

# Display methods write only to the caller's IO. Keep the compact representation
# suitable for repr/string interpolation, and use Base's text/plain array layout
# for the logical contents without its redundant Array summary line.
function _show_cumod_header(io::IO, type_name::AbstractString, A::CuModArray)
    print(io, join(size(A), "×"), " ", type_name, " modulo ", modulus(A))
end

function _show_cumod_text(io::IO, type_name::AbstractString, A::CuModArray)
    _show_cumod_header(io, type_name, A)
    print(io, ':')

    array_text = repr(MIME"text/plain"(), Array(A); context = io)
    newline = findfirst('\n', array_text)
    if newline !== nothing
        print(io, '\n', SubString(array_text, nextind(array_text, newline)))
    end
end

Base.show(io::IO, A::CuModArray) = _show_cumod_header(io, "CuModArray", A)
Base.show(io::IO, ::MIME"text/plain", A::CuModArray) = _show_cumod_text(io, "CuModArray", A)

Base.show(io::IO, A::CuModVector) = _show_cumod_header(io, "CuModVector", A)
Base.show(io::IO, ::MIME"text/plain", A::CuModVector) =
    _show_cumod_text(io, "CuModVector", A)

Base.show(io::IO, A::CuModMatrix) = _show_cumod_header(io, "CuModMatrix", A)
Base.show(io::IO, ::MIME"text/plain", A::CuModMatrix) =
    _show_cumod_text(io, "CuModMatrix", A)
# Recursive binary exponentiation algorithm
function Base.:^(A::CuModMatrix, n::Integer)
    if rows(A) != cols(A)
        throw(CuModMatrixNotSquareException("Matrix must be square for power operation"))
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

function _setup_PLUQ(A::CuModMatrix)
    return pluq(A)
end

function _is_invertible(F)
    m = rows(F.LU)
    n = cols(F.LU)
    return m == n && F.rank == m
end

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

function inverse_permutation(P::Vector{Int})
    P_inv = Array{Int}(undef, length(P))
    for i = 1:length(P)
        P_inv[P[i]] = i
    end
    return P_inv
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

function mod_inv(a::Integer, m::Integer)
    aa = mod(a, m)
    t = 0
    newt = 1
    r = m
    newr = aa
    while newr != 0
        q = div(r, newr)
        t, newt = newt, t - q * newt
        r, newr = newr, r - q * newr
    end
    if r != 1
        throw(DomainError((a, m), "inverse does not exist"))
    end
    return mod(t, m)
end

function mod_inv(a::Number, m::Integer)
    return mod_inv(Int(round(a)), m)
end

# Additional utility functions

"""
    zeros(CuModMatrix{T}, rows, cols; modulus)
    zeros(CuModVector{T}, length; modulus)

Creates a zero matrix or vector in the finite field modulo `modulus`.
Each storage dimension is rounded up to a multiple of `TILE_WIDTH`.
"""
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
A zero or negative value throws `ArgumentError`. The stored modulus on `C` is unchanged.
"""
function elementwise_multiply!(
    C::CuModArray,
    A::CuModArray,
    B::CuModArray,
    ;
    modulus_override::Union{Nothing,Integer} = nothing,
)
    _validate_positive_modulus_override(modulus_override)
    N = modulus_override === nothing ? modulus(C) : modulus_override

    if modulus_override === nothing &&
       (modulus(A) != modulus(B) || modulus(A) != modulus(C))
        throw(
            CuModArrayModulusMismatchException(
                "All matrices must have the same modulus N or provide a modulus_override",
            ),
        )
    end
    if size(A) != size(B) || size(A) != size(C)
        throw(CuModArraySizeMismatchException("All matrix dimensions must match"))
    end

    mul_elementwise!(C, A, B; modulus_override = N)
    return C
end

"""
    neg!(B, A; modulus_override = nothing)

In-place negation: B = -A mod N. No allocation is performed.
Only a positive `modulus_override` replaces `modulus(B)` for the reduction modulus.
Zero or a negative value uses `modulus(B)`. The stored modulus on `B` is unchanged.
"""
function neg!(
    B::CuModArray,
    A::CuModArray;
    modulus_override::Union{Nothing,Integer} = nothing,
)
    N =
        modulus_override === nothing || modulus_override <= 0 ? modulus(B) :
        modulus_override

    if (modulus_override === nothing || modulus_override <= 0) && modulus(A) != modulus(B)
        throw(
            CuModArrayModulusMismatchException(
                "Both matrices must have the same modulus N or provide a modulus_override",
            ),
        )
    end

    if size(A) != size(B)
        throw(CuModArraySizeMismatchException("Matrix dimensions must match"))
    end

    rscalar_sub!(B, A, 0; modulus_override = N)
    return B
end

"""
    rmul!(A,s)

Scales the array A by s, overwriting A.
As ZZ/n is commutative, this has the same behavior as lmul!

Extends LinearAlgebra.rmul! and LinearAlgebra.lmul!
"""
LinearAlgebra.rmul!(A::CuModArray, s::Number) = mul!(A, A, s)

"""
    lmul!(s,A)

Scales the array A by s, overwriting A.
As ZZ/n is commutative, this has the same behavior as lmul!
"""
LinearAlgebra.lmul!(s::Number, A::CuModArray) = mul!(A, A, s)

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
        throw(CuModArraySizeMismatchException("Matrix dimensions must match"))
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
        throw(CuModArraySizeMismatchException("Matrix dimensions must match"))
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
Zero or a negative value uses `modulus(A)`. The stored modulus on `A` is unchanged.
"""
function mod_elements!(A::CuModArray; modulus_override::Union{Nothing,Integer} = nothing)
    N =
        modulus_override === nothing || modulus_override <= 0 ? modulus(A) :
        modulus_override
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

Creates a new CuModArray with the same values as A but with a different modulus.
Logical elements are reduced modulo new_N. The new storage rounds each logical
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
    change_modulus!(A, new_N)

Changes the modulus of A in-place to new_N.
All elements are reduced modulo new_N.
"""
#TODO: we are not allowed to actually change an immutable struct.
#This returns a new struct, but modifies the underlying data
#
#This will allocate a litte bit the way the constructor is written right now.
function change_modulus_no_alloc!(A::CuModArray, new_N::Integer)
    if new_N < modulus(A)
        parent(A) .= mod.(parent(A), new_N)
        mod!(parent(A), parent(A), new_N)
    end

    return CuModArray(parent(A); modulus = Int(new_N), new_size = size(A))
end

"""
    mul!(C, A, B; max_ops=nothing, input_bound=nothing, modulus_override=nothing)

In-place matrix multiplication: `C = A * B mod N`. By default `N` is
`modulus(C)`; a positive `modulus_override` selects another working reduction
modulus without changing the stored modulus of `C`. The input matrices must
still have matching stored moduli. `input_bound` is the exclusive upper bound
on nonnegative multiplicands. `max_ops` counts product
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
    modulus_override = nothing,
)

    _validate_positive_modulus_override(modulus_override)
    if modulus(A) != modulus(B) ||
       (modulus_override === nothing && modulus(A) != modulus(C))
        throw(
            CuModArrayModulusMismatchException("All matrices must have the same modulus N"),
        )
    end
    if cols(A) != rows(B)
        throw(
            CuModArraySizeMismatchException(
                "Matrix dimensions do not match for multiplication",
            ),
        )
    end
    if rows(C) != rows(A) || cols(C) != cols(B)
        throw(CuModArraySizeMismatchException("Output matrix C has incorrect dimensions"))
    end

    stripe_mul!(
        C,
        A,
        B;
        max_ops = max_ops,
        input_bound = input_bound,
        modulus_override = modulus_override,
    )
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
        throw(
            CuModArrayModulusMismatchException("All matrices must have the same modulus N"),
        )
    end
    if cols(A) != length(x)
        throw(
            CuModArraySizeMismatchException(
                "Matrix dimensions do not match for multiplication",
            ),
        )
    end
    if length(z) != rows(A)
        throw(CuModArraySizeMismatchException("Output matrix C has incorrect dimensions"))
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
    dims = size(A)
    data = CUDA.zeros(T, _padded_size(dims)...)
    if !isempty(A)
        copyto!(_logical_view(data, dims), _logical_view(parent(A), dims))
    end
    return CuModArray{T,D}(data; modulus = modulus(A), mod = false, new_size = dims)
end

#TODO: addmul! (add and scalar multiply), gemv!, gemm!
#
#Note: gemv! and gemm! could be more efficient by incorporating the add into the
#CUBLAS gemm/gemv that happens in stripe_mul!
