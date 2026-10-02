mutable struct KaratsubaArray{T,D} <: AbstractArray{T,D}
    data1::AbstractArray{T,D}
    data2::AbstractArray{T,D}
    plan::Union{Nothing,AbstractArray{T,D}}
    N1::Integer
    N2::Integer
    M::Integer

    function KaratsubaArray{T,D}(A::AbstractArray{T,D}, B::AbstractArray{T,D}) where {T,D}
        if size(A) != size(B)
            error("Dimensions of matrices must match")
        end

        M = find_max_ops(eltype(A), max(size(A)..., size(B)...))
        return new{T,D}(A, B, nothing, M, M, M)
    end

    function KaratsubaArray{T,D}(
        A::AbstractArray{T,D},
        B::AbstractArray{T,D},
        N1::Integer,
        N2::Integer,
        M::Integer,
    ) where {T,D}
        type = eltype(A)

        if size(A) != size(B)
            error("Dimensions of matrices must match")
        end
        #=
        if !(all(x->x<N,A) && all(x->x<N,B))
            error("Cannot have entries larger than modulus in matrices")
        end
        =#
        if occursin("Float", string(type))
            bits_dict = Dict("64" => 51, "32" => 22, "16" => 9)
            bits_match = match(r"\d+", string(type))
            bits = get(bits_dict, bits_match.match, -1)
        elseif occursin("UInt", string(type))
            bits_match = match(r"\d+", string(type))
            bits = parse(Int, bits_match.match) - 1
        elseif occursin("Int", string(type))
            bits_match = match(r"\d+", string(type))
            bits = parse(Int, bits_match.match)
        else
            error("The input type is neither Int, UInt, nor Float.")
        end

        if bits == -1
            error("Input type is not recognized.")
        end
        if M >= BigInt(2)^bits
            error("Modulus too large")
        end

        return new{T,D}(A, B, nothing, N1, N2, M)
    end
end

const KaratsubaMatrix{T} = KaratsubaArray{T,2}
const KaratsubaVector{T} = KaratsubaArray{T,1}

function KaratsubaArray(
    A::AbstractArray{T,D},
    B::AbstractArray{T,D},
    N1::Integer,
    N2::Integer,
    M::Integer,
) where {T,D}
    KaratsubaArray{T,D}(A, B, N1, N2, M)
end

function KaratsubaMatrix(A::AbstractMatrix{T}, B::AbstractMatrix{T}) where {T}
    KaratsubaArray{T,2}(A, B)
end

function KaratsubaMatrix(A::AbstractMatrix{T}, B::AbstractMatrix{T}, M::Integer) where {T}
    KaratsubaArray{T,2}(A, B, M, M, M)
end

function KaratsubaMatrix(
    A::AbstractMatrix{T},
    B::AbstractMatrix{T},
    N1::Integer,
    N2::Integer,
    M::Integer,
) where {T}
    KaratsubaArray{T,2}(A, B, N1, N2, M)
end

function KaratsubaVector(A::AbstractVector{T}, B::AbstractVector{T}) where {T}
    KaratsubaArray{T,1}(A, B)
end

function KaratsubaVector(
    A::AbstractVector{T},
    B::AbstractVector{T},
    N1::Integer,
    N2::Integer,
    M::Integer,
) where {T}
    KaratsubaArray{T,1}(A, B, N1, N2, M)
end

#=
struct KMatMulPlan{T,D}
    temp1::AbstractArray{T,D}
    temp2::AbstractArray{T,D}
    temp1mod::Integer
    temp2mod::Integer

    function KMatMulPlan{T,D}(A::AbstractArray{T,D},B::AbstractArray{T,D},N1::Integer,N2::Integer) where {T,D}
        new{T,D}(A,B,N1,N2)
    end
end
=#

#=
function zerosplan(T::Type,rows::Integer,cols::Integer,N1::Integer,N2::Integer,use_gpu=false)
    if use_gpu == true
        return KMatMulPlan{T,2}(zeros(T,rows,cols,N1),zeros(T,rows,cols,N2),N1,N2)
    else
        return KMatMulPlan{T,2}(zeros(T,rows,cols),zeros(T,rows,cols),N1,N2)
    end
end
=#

function find_max_ops_karatsuba(type, N)

    if occursin("Float", string(type))
        bits_dict = Dict("64" => 51, "32" => 22, "16" => 9)
        bits_match = match(r"\d+", string(type))
        bits = get(bits_dict, bits_match.match, -1)
    elseif occursin("UInt", string(type))
        bits_match = match(r"\d+", string(type))
        bits = parse(Int, bits_match.match) - 1
    elseif occursin("Int", string(type))
        bits_match = match(r"\d+", string(type))
        bits = parse(Int, bits_match.match)
    else
        error("The input type is neither Int, UInt, nor Float.")
    end

    if bits == -1
        error("Input type is not recognized.")
    end

    if 64 ≤ bits
        floor(BigInt, (BigInt(2)^bits - 1) / N^2) - 1
    else
        floor(Int, (2^bits - 1) / N^2) - 1
    end


end

function KMatMul!(C::KaratsubaArray, A::KaratsubaArray, B::KaratsubaArray)
    if (A.M != B.M) || (A.M != C.M)
        error("Matrices must have the same modulus m")
    end
    if size(A.data1)[2] != size(B.data1)[1]
        error("Matrix dimensions don't work for multiplication")
    end
    if (A.plan == nothing) || (B.plan == nothing) || (C.plan == nothing)
        error("Must initialize plans first")
    end
    if eltype(A.data1.data) != Float64
        error("Not implemented for non-Float64 entries")
    end
    # TODO: enable this
    # maxN = max(A.N1,A.N2)
    # if find_max_stripe_ops(eltype(A.data1.data), maxN) < size(A,1)
    #     error("Matrix is too big for a single matmul, this is currently not implemented")
    # end
    #=
    tw = TILE_WIDTH

    totalsize = length(C.data1.data) # the vector length

    threads = tw
    blocks = totalsize ÷ tw
    =#

    tw = TILE_WIDTH

    totalsize = length(A.data1.data)

    threads = tw
    blocks = totalsize ÷ tw

    #A_cols = size(A.data1,2)
    @cuda threads=threads blocks=blocks karatsuba_matmul_kernel_1!(
        A.plan.data,
        A.data1.data,
        A.data2.data,
        #A_cols,
        B.plan.data,
        B.data1.data,
        B.data2.data,
        A.N1,
    )
    # add!(A.plan,A.data1,A.data2; mod_N=2*A.N1)
    # add!(B.plan,B.data1,B.data2; mod_N=2*B.N1)

    # use gemm

    LinearAlgebra.mul!(
        C.data1,
        A.data1,
        B.data1,
        R = A.N1,
        P = A.N1^2,
        maxopsOverride = false,
    )

    LinearAlgebra.mul!(
        C.data2,
        A.plan,
        B.plan,
        R = (2*A.N1),
        P = ((4*A.N1)^2),
        maxopsOverride = false,
    )
    LinearAlgebra.mul!(
        B.plan,
        A.data2,
        B.data2,
        R = (A.N1),
        P = A.N1^2,
        maxopsOverride = false,
    )


    tw = TILE_WIDTH

    totalsize = length(C.data1.data)

    threads = tw
    blocks = totalsize ÷ tw


    @cuda threads=threads blocks=blocks karatsuba_matmul_kernel_2!(
        C.data1.data,
        C.data2.data,
        B.plan.data,
        C.N1,
        C.N2,
    )


    C
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


# the stuff in this function should apply to the general KMatMul!, but with gemm instead of gemv
function KMatMul_gemv!(C::KaratsubaArray, A::KaratsubaArray, B::KaratsubaArray)
    if (A.M != B.M) || (A.M != C.M)
        error("Matrices must have the same modulus m")
    end
    if size(A.data1)[2] != size(B.data1)[1]
        error("Matrix dimensions don't work for multiplication")
    end
    if (A.plan == nothing) || (B.plan == nothing) || (C.plan == nothing)
        error("Must initialize plans first")
    end
    if eltype(A.data1.data) != Float64
        error("Not implemented for non-Float64 entries")
    end
    # TOD0: enable this
    # maxN = max(A.N1,A.N2)
    # if find_max_stripe_ops(eltype(A.data1.data), maxN) < size(A,1)
    #     error("Matrix is too big for a single matmul, this is currently not implemented")
    # end

    tw = TILE_WIDTH

    totalsize = length(A.data1.data)

    threads = tw
    blocks = totalsize ÷ tw

    @cuda threads=threads blocks=blocks karatsuba_matmul_kernel_1!(
        A.plan.data,
        A.data1.data,
        A.data2.data,
        B.plan.data,
        B.data1.data,
        B.data2.data,
        A.N1,
    )
    # add!(A.plan,A.data1,A.data2; mod_N=2*A.N1)
    # add!(B.plan,B.data1,B.data2; mod_N=2*B.N1)

    (zero_ptr, one_ptr) = cublas_scalars_f64()


    targets = [C.data1.data, C.data2.data, B.plan.data]
    mats = [A.data1.data, A.plan.data, A.data2.data]
    vecs = [B.data1.data, B.plan.data, B.data2.data]
    CUDA.CUBLAS.gemv_batched!('N', one_ptr, mats, vecs, zero_ptr, targets)

    # CUDA.CUBLAS.gemv!('N',one_ptr,A.data1.data,B.data1.data,zero_ptr,C.data1.data)
    # CUDA.CUBLAS.gemv!('N',one_ptr,A.plan.data,B.plan.data,zero_ptr,C.data2.data)
    # CUDA.CUBLAS.gemv!('N',one_ptr,A.data2.data,B.data2.data,zero_ptr,B.plan.data)

    tw = TILE_WIDTH

    totalsize = length(C.data1.data)

    threads = tw
    blocks = totalsize ÷ tw
    @cuda threads=threads blocks=blocks karatsuba_matmul_kernel_2!(
        C.data1.data,
        C.data2.data,
        B.plan.data,
        C.N1,
        C.N2,
    )


    C
end

Base.size(A::KaratsubaArray) = size(A.data1)

Base.getindex(A::KaratsubaArray, i::Int, j::Int) = A.data1[i, j] + A.N1*A.data2[i, j]
Base.getindex(A::KaratsubaArray, i::Int) = A.data1[i] + A.N1*A.data2[i]

function Base.setindex!(A::KaratsubaArray, v, i::Int, j::Int)
    A.data1[i, j] = rem(v, A.N1)
    A.data2[i, j] = div(v, A.N1)
end

function Base.setindex!(A::KaratsubaArray, v, i::Int)
    A.data1[i] = rem(v, A.N1)
    A.data2[i] = div(v, A.N1)
end

#Converts KMat to Mat
function Base.Array(K::KaratsubaArray)
    if K.N1*K.N2 < BigInt(2)^(64)
        type = Int64
    else
        type = Int128
    end
    A = Base.zeros(Int128, size(K.data1)...)
    cpudata1 = Base.zeros(eltype(K.data1), size(K.data1)...)
    cpudata2 = Base.zeros(eltype(K.data1), size(K.data1)...)
    inds = CartesianIndices(K.data1)
    copyto!(cpudata1, inds, K.data1.data, inds)
    copyto!(cpudata2, inds, K.data2.data, inds)
    intdata1 = convert.(type, cpudata1)
    intdata2 = convert.(type, cpudata2)
    A .= intdata1 .+ K.N1 .* intdata2
    #scalar_multiply!(A,K.data2,K.N1)
    #add!(A,A,K.data1)
    A
end

function Base.copy!(B::KaratsubaArray, A::KaratsubaArray)
    copy!(B.data1, A.data1)
    copy!(B.data2, A.data2)
    if A.plan != nothing && B.plan == nothing
        initialize_plan!(B)
    end
    return B
end

function zero!(A::KaratsubaArray)
    zero!(A.data1)
    zero!(A.data2)
end

function KMatToMat(T::Type, K::KaratsubaArray)
    A = Base.zeros(T, size(K.data1)...)
    A = K.data1 + K.N1*K.data2
    A
end

function MatToKMat(A::AbstractArray, M::Integer)
    MatToKMat(eltype(A), A, M)
end

function MatToKMat(A::AbstractArray)
    M = find_max_ops(eltype(A), max(size(A)...))
    MatToKMat(eltype(A), A, M)
end

function MatToKMat(T::Type, A::AbstractArray, M::Integer)
    KaratsubaMatrix(T, A, M, M, M)
end

# right now, this assumes that A is already a CuModMatrix
function KaratsubaMatrix(T::Type, A::AbstractArray, N1::Integer, N2::Integer, M::Integer)
    #if occursin("CuMod", string(typeof(A)))
    K = KaratsubaArray(
        zeros(eltype(A), size(A)..., N1),
        zeros(eltype(A), size(A)..., N1),
        N1,
        N2,
        M,
    )
    #=
    elseif occursin("Cu", string(typeof(A)))
        K = KaratsubaArray(CUDA.zeros(T,size(A)...),CUDA.zeros(T,size(A)...),N1,N2,M)
        M = Int(M)
    else
        K = KaratsubaArray(zeros(T,size(A)...),zeros(T,size(A)...),N1,N2,M)
    end
    =#
    #=
    K.data2 .= mod.(trunc.(A./M),N2)
    K.data1 .= mod.(A - Int(K.M)*K.data2,N1)
    =#
    #=
    LinearAlgebra.mul!(K.data2,A,1/M,N2)
    trunc_elements!(K.data2)
    =#
    divide_elements!(K.data2, A, N1)
    mod_elements!(K.data2, N2)
    LinearAlgebra.mul!(K.data1, K.data2, Int(N1))
    sub!(K.data1, A, K.data1; mod_N = N1)
    mod_elements!(K.data1, N1)
    K
end

function MatToKMat(T::Type, A::AbstractArray)
    M = find_max_ops(T, max(size(A)...))
    MatToKMat(T, A, M)
end

function KaratsubaZeros(T, rows, cols, N1, N2, M, use_gpu)
    if use_gpu == true
        K = KaratsubaMatrix(zeros(T, rows, cols, N1), zeros(T, rows, cols, N1), N1, N2, M)
    else
        K = KaratsubaMatrix(zeros(T, rows, cols), zeros(T, rows, cols), N1, N2, M)
    end
    K
end

function KaratsubaZeros(T, length, N1, N2, M, use_gpu)
    if use_gpu == true
        K = KaratsubaVector(zeros(T, length, N1), zeros(T, length, N1), N1, N2, M)
    else
        K = KaratsubaVector(zeros(T, length), zeros(T, length), N1, N2, M)
    end
    K
end

function initialize_plan!(K::KaratsubaArray)
    K.plan = zeros(eltype(K.data1), size(K.data1)..., K.N1)
end

import Base: +, -, *

function +(A::KaratsubaArray, B::KaratsubaArray)
    if occursin("CuMod", string(typeof(A.data1)))
        K = KaratsubaMatrix(
            zeros(eltype(A.data1), size(A.data1)..., A.data1.N),
            zeros(eltype(A.data1), size(A.data1)..., A.data1.N),
            A.N1,
            A.N2,
            A.M,
        )
    elseif occursin("Cu", string(typeof(A.data1)))
        K = KaratsubaMatrix(
            CUDA.zeros(eltype(A.data1), size(A.data1)...),
            CUDA.zeros(eltype(A.data1), size(A.data1)...),
            A.N1,
            A.N2,
            A.M,
        )
    else
        K = KaratsubaMatrix(
            zeros(eltype(A.data1), size(A.data1)...),
            zeros(eltype(A.data1), size(A.data1)...),
            A.N1,
            A.N2,
            A.M,
        )
    end
    add!(K, A, B)
    K
end

function -(A::KaratsubaArray, B::KaratsubaArray)
    if occursin("CuMod", string(typeof(A.data1)))
        K = KaratsubaMatrix(
            zeros(eltype(A.data1), size(A.data1)..., A.data1.N),
            zeros(eltype(A.data1), size(A.data1)..., A.data1.N),
            A.N1,
            A.N2,
            A.M,
        )
    elseif occursin("Cu", string(typeof(A.data1)))
        K = KaratsubaMatrix(
            CUDA.zeros(eltype(A.data1), size(A.data1)...),
            CUDA.zeros(eltype(A.data1), size(A.data1)...),
            A.N1,
            A.N2,
            A.M,
        )
    else
        K = KaratsubaMatrix(
            zeros(eltype(A.data1), size(A.data1)...),
            zeros(eltype(A.data1), size(A.data1)...),
            A.N1,
            A.N2,
            A.M,
        )
    end
    sub!(K, A, B)
end

function *(a::Number, A::KaratsubaArray)
    if occursin("CuMod", string(typeof(A.data1)))
        K = KaratsubaMatrix(
            zeros(eltype(A.data1), size(A.data1)..., A.data1.N),
            zeros(eltype(A.data1), size(A.data1)..., A.data1.N),
            A.N1,
            A.N2,
            A.M,
        )
    elseif occursin("Cu", string(typeof(A.data1)))
        K = KaratsubaMatrix(
            CUDA.zeros(eltype(A.data1), size(A.data1)...),
            CUDA.zeros(eltype(A.data1), size(A.data1)...),
            A.N1,
            A.N2,
            A.M,
        )
    else
        K = KaratsubaMatrix(
            zeros(eltype(A.data1), size(A.data1)...),
            zeros(eltype(A.data1), size(A.data1)...),
            A.N1,
            A.N2,
            A.M,
        )
    end
    scalar_multiply!(K, A, a)
    K
end

function *(A::KaratsubaArray, a::Number)
    a * A
end

function *(A::KaratsubaArray, B::KaratsubaArray)
    if occursin("CuMod", string(typeof(A.data1)))
        K = KaratsubaMatrix(
            zeros(eltype(A.data1), size(A.data1)[1], size(B.data1)[2], A.M),
            zeros(eltype(A.data1), size(A.data1)[1], size(B.data1)[2], A.M),
            A.N1,
            A.N2,
            A.M,
        )
        plan = KaratsubaMatrix(
            zeros(eltype(A.data1), size(A.data1)[1], size(B.data1)[2], A.M),
            zeros(eltype(A.data1), size(A.data1)[1], size(B.data1)[2], A.M),
            A.N1,
            A.N2,
            A.M,
        )
        #plan = KMatMulPlan(zeros(eltype(A.data1),size(A.data1)...,A.M),zeros(eltype(B.data1),size(B.data1)...,A.M),A.N1,A.N2)
    elseif occursin("Cu", string(typeof(A.data1)))
        K = KaratsubaMatrix(
            CUDA.zeros(eltype(A.data1), size(A.data1)...),
            CUDA.zeros(eltype(A.data1), size(A.data1)...),
            A.N1,
            A.N2,
            A.M,
        )
    else
        K = KaratsubaMatrix(
            zeros(eltype(A.data1), size(A.data1)[1], size(B.data1)[2]),
            zeros(eltype(A.data1), size(A.data1)[1], size(B.data1)[2]),
            A.N1,
            A.N2,
            A.M,
        )
        plan = KMatMulPlan{Matrix{eltype(A.data1)}}(
            zeros(eltype(A.data1), size(A.data1)...),
            zeros(eltype(B.data1), size(B.data1)...),
            A.N1,
            A.N2,
        )
    end
    KMatMul!(K, A, B, plan)
    K
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
    if (A.M != B.M) || (A.M != K.M)
        error("Matrices must have the same modulus m")
    end
    if (size(A.data1) != size(B.data1)) || (size(A.data1) != (size(K.data1)))
        error("Matrix dimensions must match")
    end

    #TODO: remove plans
    if (K.plan == nothing)
        error("Must initialize plan first before using fallback implementation")
    end

    tw = TILE_WIDTH

    totalsize = length(A.data1.data)

    threads = tw
    blocks = totalsize ÷ tw

    @cuda threads=threads blocks=blocks karatsuba_add_kernel!(
        K.data1.data,
        K.data2.data,
        A.data1.data,
        A.data2.data,
        B.data1.data,
        B.data2.data,
        A.N1,
        A.N2,
    )


    K
end

#"""
#This is a fallback implementation, and shouldn't be used
#unless the dimension is greater than or equal to 3.

#This is not safe if K == B, but it is safe if K == A.
#"""
#function add!(K::KaratsubaArray, A::KaratsubaArray, B::KaratsubaArray)
#    if (A.M != B.M) || (A.M != K.M)
#        error("Matrices must have the same modulus m")
#    end
#    if (size(A.data1) != size(B.data1)) || (size(A.data1) != (size(K.data1)))
#        error("Matrix dimensions must match")
#    end
#    if (A.plan == nothing) || (K.plan == nothing)
#        error("Must initialize plans first before using fallback implementation")
#    end
#    #=
#    K.data2 .= mod.(trunc.((A.data1+B.data1)./A.M),A.N2)
#    K.data1 .= mod.(A.data1 + B.data1 - A.M*K.data2,A.N1)
#    K.data2 .= mod.(K.data2 + A.data2 + B.data2,A.N2)
#    =#

#    add!(K.plan,A.data1,B.data1,2*A.N1)
#    divide_elements!(K.plan,K.plan,A.N1)
#    add!(K.data2,K.plan,A.data2,2*A.N2)
#    add!(K.data2,K.data2,B.data2,2*A.N2)
#    LinearAlgebra.mul!(K.plan,K.plan,A.N1,A.N1^2)
#    sub!(K.plan,B.data1,K.plan,B.N1^2)
#    add!(K.data1,A.data1,K.plan,A.N1^2)
#    #=
#    LinearAlgebra.mul!(K.data2,K.data2,1/A.M)
#    trunc_elements!(K.data2)
#    =#
#    mod_elements!(K.data1,A.N1)
#    mod_elements!(K.data2,A.N2)
#    #=
#    add!(K.data1,A.data1,B.data1,2*A.M)
#    add!(K.data2,A.data2,B.data2)
#    add!(K.data2,K.data2,divides(K.data1,K.M),K.N2)
#    mod_elements!(K.data1,K.N1)
#    =#

#    K
#end

function sub!(K::KaratsubaArray, A::KaratsubaArray, B::KaratsubaArray)
    if (A.M != B.M) || (A.M != K.M)
        error("Matrices must have the same modulus m")
    end
    if (size(A.data1) != size(B.data1)) || (size(A.data1) != (size(K.data1)))
        error("Matrix dimensions must match")
    end

    #=
    C.data2 .= mod.(trunc.((A.data1-B.data1)./A.M),A.N2)
    C.data1 .= mod.(A.data1 - B.data1 - A.M*C.data2,A.N1)
    C.data2 .= mod.(C.data2 + A.data2 - B.data2 - (A.data1.<B.data1),A.N2)

    sub!(K.data2,A.data1,B.data1)
    LinearAlgebra.mul!(K.data2,K.data2,1/A.M)
    trunc_elements!(K.data2)
    mod_elements!(K.data2,A.N2)
    LinearAlgebra.mul!(K.data1,K.data1,A.M,A.M^2)
    sub!(K.data1,B.data1,K.data1)
    sub!(K.data1,A.data1,K.data1)
    mod_elements!(K.data1,A.N1)
    add!(K.data2,K.data2,A.data2)
    sub!(K.data2,K.data2,B.data2)
    sub!(K.data2,K.data2,A.data1.<B.data1)
    mod_elements!(K.data2,A.N2)
    =#
    tw = TILE_WIDTH

    totalsize = length(A.data1.data)

    threads = tw
    blocks = totalsize ÷ tw

    @cuda threads=threads blocks=blocks karatsuba_sub_kernel!(
        K.data1.data,
        K.data2.data,
        A.data1.data,
        A.data2.data,
        B.data1.data,
        B.data2.data,
        A.N1,
        A.N2,
        A.M,
    )

    # negate!(K,B)
    # add!(K,K,A)
    K
end

function scalar_multiply!(B::KaratsubaArray, A::KaratsubaArray, s::Number)
    if A.M != B.M
        error("Matrices must have the same modulus m")
    end
    if size(A.data1) != size(B.data1)
        error("Matrix dimensions must match")
    end

    # LinearAlgebra.mul!(B.data2,A.data1,s,A.N1^2)
    # divide_elements!(B.data2,B.data2,A.N1)
    # LinearAlgebra.mul!(B.data1,A.data2,s,A.N2)
    # add!(B.data2,B.data2,B.data1)
    # LinearAlgebra.mul!(B.data1,A.data1,s,A.N1)

    tw = TILE_WIDTH

    totalsize = length(A.data1.data)

    threads = tw
    blocks = totalsize ÷ tw

    @cuda threads=threads blocks=blocks karatsuba_scalar_multiply_kernel!(
        B.data1.data,
        B.data2.data,
        A.data1.data,
        A.data2.data,
        s,
        A.N1,
        A.N2,
    )

    #=
    B.data2 .= mod.(trunc.((s*A.data1)/A.M),A.N2)
    B.data1 .= mod.(s*A.data1 - B.M*B.data2,A.N1)
    B.data2 .= mod.(B.data2 + s*A.data2,A.N2)
    =#
    B
end



# function negate!(K::KaratsubaArray, A::KaratsubaArray)

#     tw = TILE_WIDTH

#     totalsize = length(A.data1.data)

#     threads = tw
#     blocks = totalsize ÷ tw

# @cuda threads=threads blocks=blocks karatsuba_negate_kernel!(K.data1.data,
#                                                              K.data2.data,
#                                                              A.data1.data,
#                                                              A.data2.data,
#                                                              A.N1,
#                                                              A.N2,
#                                                             A.M)


#     K
# end

function negate!(K::KaratsubaArray, A::KaratsubaArray)
    tw = TILE_WIDTH

    totalsize = length(A.data1.data)

    threads = tw
    blocks = totalsize ÷ tw

    @cuda threads=threads blocks=blocks karatsuba_negate_kernel!(
        K.data1.data,
        K.data2.data,
        A.data1.data,
        A.data2.data,
        A.N1,
        A.N2,
        A.M,
    )



    #=
    K.data2 .= trunc.((A.M .- A.data1)/A.M)
    K.data1 .= mod.(A.M - A.data1 - A.M*K.data2,A.N1)
    K.data2 .= mod.(mod.(K.data2 + A.M - A.data2 - 1,A.M),A.N2)
    =#
    # mul!(K.data2,K.data2,0)
    # scalar_add!(K.data2,K.data2,A.N1,2*A.N1)
    # sub!(K.data2,K.data2,A.data1,2*A.N1)
    #=
    LinearAlgebra.mul!(K.data2,K.data2,1/A.N1)
    trunc_elements!(K.data2)
     =#
    # divide_elements!(K.data2,K.data2,A.N1)
    # LinearAlgebra.mul!(K.data1,K.data2,A.N1,A.M^2)
    # add!(K.data1,K.data1,A.data1,A.M^2)
    # negate!(K.data1,K.data1,A.N1)
    # mod_elements!(K.data1,A.N1)
    # scalar_add!(K.data2,K.data2,A.N2,A.M^2)
    # scalar_sub!(K.data2,K.data2,1,A.M^2)
    # sub!(K.data2,K.data2,A.data2,A.M)
    # mod_elements!(K.data2,A.N2)
    K
end

function divide_elements!(B::CuModArray, A::CuModArray, N::Integer)
    @. B.data = div(A.data, N)
    return A
end

function Karatsubacopy(A::KaratsubaArray{T,D}) where {T,D}
    B = KaratsubaArray{T,D}(
        CuModcopy(A.data1),
        CuModcopy(A.data2),
        copy(A.N1),
        copy(A.N2),
        A.N1*A.N2,
    )
    if !(A.plan==nothing)
        initialize_plan!(B)
    end
    B
end
