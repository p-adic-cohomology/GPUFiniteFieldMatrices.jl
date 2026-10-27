for i = 1000:1000:20000
    p = 11
    R = GF(p)
    MS = matrix_space(R, i, 1000)
    A_nemo = MS([R(x) for x in rand(1:p, i, 1000)])
    println("Testing matrix of size $(size(A_nemo))")
    @btime Nemo.is_invertible_with_inverse($A_nemo, side = :left)
end

for i = 1000:1000:10000
    p = 11
    A = rand(1:p, i, i)
    d_A = CuModMatrix(A, p)
    println("Testing matrix of size $(size(d_A))")
    @btime GPUFiniteFieldMatrices.is_invertible_with_inverse($d_A)
end

function gpu_inverse(
    A::CuArray{T,2},
) where {T<:Union{Float32,Float64,ComplexF32,ComplexF64}}
    n = size(A, 1)
    @assert size(A, 1) == size(A, 2) "Matrix must be square"

    A_fact = copy(A)
    ipiv = CuArray{Int32}(undef, n)

    # Step 1: LU factorization
    CUDA.CUSOLVER.getrf!(A_fact, ipiv)

    # Step 2: Solve AX = I
    I_gpu = CuArray(Matrix{T}(I, n, n))
    CUDA.CUSOLVER.getrs!('N', A_fact, ipiv, I_gpu)

    return I_gpu
end

for i = 1000:1000:10000
    p = 11
    A = Float64.(rand(1:p, i, i))
    println("Testing matrix of size $(size(A))")
    @btime cuinv($A)
    # isapprox(cuinv(A), inv(A))
end

function cuinv(m::Matrix{T}) where {T}
    if size(m, 1) != size(m, 2)
        throw(ArgumentError("Matrix not square."))
    end
    A = CuArray(m)
    B = CuArray(Matrix{T}(I(size(A, 1))))
    A, ipiv = CUDA.CUSOLVER.getrf!(A)
    Matrix{T}(CUDA.CUSOLVER.getrs!('N', A, ipiv, B))
end

for i = 1000:1000:20000
    p = 11
    R = GF(p)
    MS = matrix_space(R, i, 1000)
    A_nemo = MS([R(x) for x in rand(1:p, i, 1000)])
    println("Testing LU decomposition of size $(size(A_nemo))")
    @btime Nemo.lu($A_nemo)
end
