
#function gemv!(transpose::Bool,alpha::Integer,A::CuModMatrix,x::CuModVector,beta::Integer,y::CuModVector)
#
#end

"""
    find_max_stripe_ops(type, input_bound)

Given a type `type` that is supported by cuBLAS,
find the maximum number of product contributions that can be accumulated per output
entry before a modular reduction, for multiplicands in `0:(input_bound - 1)`.
"""
function find_max_stripe_ops(type, input_bound)
    if occursin("Float", string(type))
        bits_dict = Dict("64" => 53, "32" => 24, "16" => 11)
        bits_match = match(r"\d+", string(type))
        bits = get(bits_dict, bits_match.match, -1)
    else
        throw(ArgumentError("The input type is not supported for cuBLAS gemm"))
    end

    if bits == -1
        throw(ArgumentError("Input type is not recognized."))
    end

    floor(Int, (2^bits - 1) / (input_bound - 1)^2) - 1
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

    _CUDA_BLAS.gemm!(tAchar, tBchar, alpha, A.data, B.data, beta, C.data)

    mod!(C.data, C.data, C.N)
end

const _cublas_scalar_cache_f64 = IdDict{Task,Tuple}()
const _cublas_scalar_cache_f32 = IdDict{Task,Tuple}()

"""
returns (0, 1) as pointers that can be used with low-level cuBLAS APIs
"""
@inline function cublas_scalars_f64()
    t = current_task()
    get!(_cublas_scalar_cache_f64, t) do
        (CuRef(Float64(0.0)), CuRef(Float64(1.0)))
    end
end

"""
returns (0, 1) as pointers that can be used with low-level cuBLAS APIs
"""
@inline function cublas_scalars_f32()
    t = current_task()
    get!(_cublas_scalar_cache_f32, t) do
        (CuRef(Float32(0.0)), CuRef(Float32(1.0)))
    end
end

"""
    stripe_mul!(z::CuModVector, A::CuModMatrix, x::CuModVector;
        max_ops=nothing, input_bound=nothing, modulus_override=nothing)

Matrix-vector multiplication based on stripes. `input_bound` is the exclusive
bound on nonnegative input values and controls the default `max_ops`. `max_ops`
is the number of product contributions accumulated per output entry between
modular reductions. `modulus_override` changes the per-call reduction modulus;
when omitted, `z.N` is used and its stored modulus is unchanged. The low-level
`respect_max_ops=false` option recomputes the budget even when `max_ops` is given.
"""
#TODO: replayce CuVector{Float64} with padded custom type.
#the ".data" stuff won't work until then
function stripe_mul!(
    z::CuModVector,
    A::CuModMatrix,
    x::CuModVector;
    max_ops = nothing,
    input_bound = nothing,
    modulus_override = nothing,
    respect_max_ops::Bool = true,
)
    if A.N != z.N || z.N != z.N
        throw(ArgumentError("Mismatched modulus in matmul"))
    elseif cols(A) != length(z)
        throw(DimensionMismatch(""))
    elseif eltype(A.data) ∉ [Float64, Float32, Float16, ComplexF32, ComplexF64]
        throw(ArgumentError("Element type $(eltype(A.data)) unsupported by cuBLAS"))
    elseif eltype(A.data) != eltype(z.data) || eltype(z.data) != eltype(x.data)
        throw(ArgumentError("Mismatched element types in matmul"))
    end # possibly also enforce that the eltypes are the same

    bound = input_bound === nothing ? z.N : input_bound
    if !respect_max_ops || max_ops === nothing
        max_ops = find_max_stripe_ops(eltype(A.data), bound)
    end
    reduction_modulus = modulus_override === nothing ? z.N : modulus_override

    if max_ops < 1
        throw(
            ArgumentError(
                "cannot perform a single multiplication for input bound $(bound) with datatype $(eltype(A.data))",
            ),
        )
    end

    if eltype(A.data) == Float64
        (zero_ptr, one_ptr) = cublas_scalars_f64()
    elseif eltype(A.data) == Float32
        (zero_ptr, one_ptr) = cublas_scalars_f32()
    else
        zero_ptr = CuRef(eltype(A.data)(0.0))
        one_ptr = CuRef(eltype(A.data)(1.0))
    end

    summed_size = cols(A)#size(A,2)

    num_stripes = div(summed_size, max_ops) + 1


    if num_stripes == 1
        _CUDA_BLAS.gemv!('N', one_ptr, A.data, x.data, zero_ptr, z.data)
        mod!(z.data, z.data, reduction_modulus)
        return
    end


    i = 1

    range = 1:max_ops
    A_temp = @view A.data[:, range]
    x_temp = @view x.data[range]
    _CUDA_BLAS.gemv!('N', one_ptr, A_temp, x_temp, zero_ptr, z.data)
    mod!(z.data, z.data, reduction_modulus)

    i += 1

    while i < num_stripes
        range = (max_ops*(i-1)+1):(max_ops*i)
        A_temp = @view A.data[:, range]
        x_temp = @view x.data[range]
        _CUDA_BLAS.gemv!('N', one_ptr, A_temp, x_temp, one_ptr, z.data)
        mod!(z.data, z.data, reduction_modulus)

        i += 1
    end
    # i == num_stripes

    range = (max_ops*(i-1)+1):cols(A)
    A_temp = @view A.data[:, range]
    x_temp = @view x.data[range]
    _CUDA_BLAS.gemv!('N', one_ptr, A_temp, x_temp, one_ptr, z.data)
    mod!(z.data, z.data, reduction_modulus)

end

"""
    stripe_mul!(C::CuModMatrix, A::CuModMatrix, B::CuModMatrix;
        max_ops=nothing, input_bound=nothing)

Matrix multiplication mod N based on stripes. `input_bound` is the exclusive
bound on nonnegative input values and controls the default `max_ops`. Supplying
`input_bound` retains the existing single-stripe route. `max_ops` is the number
of product contributions accumulated per output entry between reductions on
the multi-stripe route. The single-stripe route may accumulate more than
`max_ops` contributions before its reduction.
"""
function stripe_mul!(
    C::CuModMatrix,
    A::CuModMatrix,
    B::CuModMatrix;
    max_ops = nothing,
    input_bound = nothing,
)

    if A.N != B.N || B.N != C.N
        throw(ArgumentError("Mismatched modulus in matmul"))
    elseif cols(A) != rows(B)
        throw(DimensionMismatch(""))
    elseif eltype(A.data) ∉ [Float64, Float32, Float16, ComplexF32, ComplexF64]
        throw(ArgumentError("Element type $(eltype(A.data)) unsupported by cuBLAS"))
    elseif eltype(A.data) != eltype(B.data) || eltype(B.data) != eltype(C.data)
        throw(ArgumentError("Mismatched element types in matmul"))
    end # possibly also enforce that the eltypes are the same

    if max_ops === nothing
        bound = input_bound === nothing ? A.N : input_bound
        max_ops = find_max_stripe_ops(eltype(A.data), bound)
    end

    if max_ops < 1
        throw(
            ArgumentError(
                "cannot perform a single multiplication for modulus $(A.N) with datatype $(eltype(A.data))",
            ),
        )
    end

    summed_size = cols(A)#size(A,2)

    if input_bound === nothing
        num_stripes = div(summed_size, max_ops) + 1
    else
        num_stripes = 1
    end

    if num_stripes == 1
        mul!(C.data, A.data, B.data)
        mod!(C.data, C.data, C.N)
        return
    end

    i = 1

    range = 1:max_ops
    A_temp = @view A.data[:, range]
    B_temp = @view B.data[range, :]
    _CUDA_BLAS.gemm!('N', 'N', 1, A_temp, B_temp, 0, C.data)
    mod!(C.data, C.data, C.N)

    i += 1

    while i < num_stripes
        range = (max_ops*(i-1)+1):(max_ops*i)
        A_temp = @view A.data[:, range]
        B_temp = @view B.data[range, :]
        _CUDA_BLAS.gemm!('N', 'N', 1, A_temp, B_temp, 1, C.data)
        mod!(C.data, C.data, C.N)

        i += 1
    end
    # i == num_stripes

    range = (max_ops*(i-1)+1):cols(A)
    A_temp = @view A.data[:, range]
    B_temp = @view B.data[range, :]
    _CUDA_BLAS.gemm!('N', 'N', 1, A_temp, B_temp, 1, C.data)
    mod!(C.data, C.data, C.N)
end
