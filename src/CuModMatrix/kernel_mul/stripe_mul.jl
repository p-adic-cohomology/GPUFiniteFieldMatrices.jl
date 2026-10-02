
#function gemv!(transpose::Bool,alpha::Integer,A::CuModMatrix,x::CuModVector,beta::Integer,y::CuModVector)
#
#end

"""
    find_max_stripe_ops(type,N)

Given a type `type` that is supported by CUBLAS, 
find out how many columns we can include in a stripe
(in the stripe multiplication algorithm)
"""
function find_max_stripe_ops(type, N)
    if occursin("Float", string(type))
        bits_dict = Dict("64" => 53, "32" => 24, "16" => 11)
        bits_match = match(r"\d+", string(type))
        bits = get(bits_dict, bits_match.match, -1)
    else
        throw(ArgumentError("The input type is not supported for CUBLAS gemm"))
    end

    if bits == -1
        throw(ArgumentError("Input type is not recognized."))
    end

    floor(Int, (2^bits - 1) / (N-1)^2) - 1
end

"""
     unsafe_gemm!(transposeA::Bool,transposeB::Bool,alpha::Integer,A::CuModMatrix,B::CuModMatrix,beta::Integer,C::CuModMatrix)

Light wrapper over the provided interface in CUDA.jl. 

Does not keep track of overflow, so errors could happen if the matrices are too big.

Also does not check to see if the modulus of the varios matrices agree

Since we are working mod N, `alpha` and `beta` must be integers
"""
function unsafe_gemm!(
    transposeA::Bool,
    transposeB::Bool,
    alpha::Integer,
    A::CuModMatrix,
    B::CuModMatrix,
    beta::Integer,
    C::CuModMatrix,
)

    tAchar = transposeA ? 'Y' : 'N'
    tBchar = transposeB ? 'Y' : 'N'

    #TODO: convert alpha and beta to floats?

    CUDA.CUBLAS.gemm!(tAchar, tBchar, alpha, A.data, B.data, beta, C.data)

    mod!(C.data, C.data, C.N)
end

const _cublas_scalar_cache_f64 = IdDict{Task,Tuple}()
const _cublas_scalar_cache_f32 = IdDict{Task,Tuple}()

"""
returns (0, 1) as pointers that can be used with low-level CUBLAS APIs
"""
@inline function cublas_scalars_f64()
    t = current_task()
    get!(_cublas_scalar_cache_f64, t) do
        (CUDA.CUBLAS.CuRef(Float64(0.0)), CUDA.CUBLAS.CuRef(Float64(1.0)))
    end
end

"""
returns (0, 1) as pointers that can be used with low-level CUBLAS APIs
"""
@inline function cublas_scalars_f32()
    t = current_task()
    get!(_cublas_scalar_cache_f32, t) do
        (CUDA.CUBLAS.CuRef(Float32(0.0)), CUDA.CUBLAS.CuRef(Float32(1.0)))
    end
end

"""
    stripe_mul!(z::CuModVector,A::CuModMatrix,x::CuModVector)

Matrix-vector multiplication based on stripes
"""
#TODO: replayce CuVector{Float64} with padded custom type.
#the ".data" stuff won't work until then
function stripe_mul!(
    z::CuModVector,
    A::CuModMatrix,
    x::CuModVector;
    M = nothing,
    R = nothing,
    N = nothing,
    maxopsOverride = true,
)
    #TODO: add new signature that takes in custom M and N (and can copy the old one into new one)
    # so we can update the modulus but use the old bound for coeffs.

    if A.N != z.N || z.N != z.N
        throw(ArgumentError("Mismatched modulus in matmul"))
    elseif cols(A) != length(z)
        throw(DimensionMismatch(""))
    elseif eltype(A.data) ∉ [Float64, Float32, Float16, ComplexF32, ComplexF64]
        throw(ArgumentError("Element type $(eltype(A.data)) unsupported by CUBLAS"))
    elseif eltype(A.data) != eltype(z.data) || eltype(z.data) != eltype(x.data)
        throw(ArgumentError("Mismatched element types in matmul"))
    end # possibly also enforce that the eltypes are the same

    if R==nothing
        R = z.N
    end

    if maxopsOverride == true
        if M == nothing
            if N == nothing
                M = find_max_stripe_ops(eltype(A.data), R)
            else
                M = find_max_stripe_ops(eltype(A.data), R)
            end
        end
    else
        M = find_max_stripe_ops(eltype(A.data), R)
    end

    if N==nothing
        N = z.N
    end

    if M < 1
        throw(
            ArgumentError(
                "cannot perform a single multiplication for modulus $(R) with datatype $(eltype(A.data))",
            ),
        )
    end

    if eltype(A.data) == Float64
        (zero_ptr, one_ptr) = cublas_scalars_f64()
    elseif eltype(A.data) == Float32
        (zero_ptr, one_ptr) = cublas_scalars_f32()
    else
        zero_ptr = CUDA.CUBLAS.CuRef(eltype(A.data)(0.0))
        one_ptr = CUDA.CUBLAS.CuRef(eltype(A.data)(1.0))
    end

    summed_size = cols(A)#size(A,2)

    num_stripes = div(summed_size, M) + 1


    if num_stripes == 1
        CUDA.CUBLAS.gemv!('N', one_ptr, A.data, x.data, zero_ptr, z.data)
        mod!(z.data, z.data, N)
        return
    end


    i = 1

    range = 1:M
    A_temp = @view A.data[:, range]
    x_temp = @view x.data[range]
    CUDA.CUBLAS.gemv!('N', one_ptr, A_temp, x_temp, zero_ptr, z.data)
    mod!(z.data, z.data, N)

    i += 1

    while i < num_stripes
        range = (M*(i-1)+1):(M*i)
        A_temp = @view A.data[:, range]
        x_temp = @view x.data[range]
        CUDA.CUBLAS.gemv!('N', one_ptr, A_temp, x_temp, one_ptr, z.data)
        mod!(z.data, z.data, N)

        i += 1
    end
    # i == num_stripes

    range = (M*(i-1)+1):cols(A)
    A_temp = @view A.data[:, range]
    x_temp = @view x.data[range]
    CUDA.CUBLAS.gemv!('N', one_ptr, A_temp, x_temp, one_ptr, z.data)
    mod!(z.data, z.data, N)

end

"""
    stripe_mul!(C::CuModMatrix,A::CuModMatrix,B::CuModMatrix)

Matrix multiplication mod N based on stripes.
"""
function stripe_mul!(
    C::CuModMatrix,
    A::CuModMatrix,
    B::CuModMatrix;
    M = nothing,
    N = nothing,
)

    if A.N != B.N || B.N != C.N
        throw(ArgumentError("Mismatched modulus in matmul"))
    elseif cols(A) != rows(B)
        throw(DimensionMismatch(""))
    elseif eltype(A.data) ∉ [Float64, Float32, Float16, ComplexF32, ComplexF64]
        throw(ArgumentError("Element type $(eltype(A.data)) unsupported by CUBLAS"))
    elseif eltype(A.data) != eltype(B.data) || eltype(B.data) != eltype(C.data)
        throw(ArgumentError("Mismatched element types in matmul"))
    end # possibly also enforce that the eltypes are the same

    #TODO
    # basicallty adapt a new P which is the actual maxium, not the modulus
    # Be careful not to mod by P though
    # Also update all the versions of stripe_mul!

    if M == nothing
        if N == nothing
            M = find_max_stripe_ops(eltype(A.data), A.N)
        else
            M = find_max_stripe_ops(eltype(A.data), N)
        end
    end

    if M < 1
        throw(
            ArgumentError(
                "cannot perform a single multiplication for modulus $(A.N) with datatype $(eltype(A.data))",
            ),
        )
    end

    summed_size = cols(A)#size(A,2)

    if N == nothing
        num_stripes = div(summed_size, M) + 1
    else
        num_stripes = 1
    end

    if num_stripes == 1
        mul!(C.data, A.data, B.data)
        mod!(C.data, C.data, C.N)
        return
    end

    i = 1

    range = 1:M
    A_temp = @view A.data[:, range]
    B_temp = @view B.data[range, :]
    CUDA.CUBLAS.gemm!('N', 'N', 1, A_temp, B_temp, 0, C.data)
    mod!(C.data, C.data, C.N)

    i += 1

    while i < num_stripes
        range = (M*(i-1)+1):(M*i)
        A_temp = @view A.data[:, range]
        B_temp = @view B.data[range, :]
        CUDA.CUBLAS.gemm!('N', 'N', 1, A_temp, B_temp, 1, C.data)
        mod!(C.data, C.data, C.N)

        i += 1
    end
    # i == num_stripes

    range = (M*(i-1)+1):cols(A)
    A_temp = @view A.data[:, range]
    B_temp = @view B.data[range, :]
    CUDA.CUBLAS.gemm!('N', 'N', 1, A_temp, B_temp, 1, C.data)
    mod!(C.data, C.data, C.N)
end
