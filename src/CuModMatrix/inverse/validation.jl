"""
    pluq_nonzero_mod_kernel!(flag, D, n, N)

Set `flag[1] = 1` if any entry in the top-left `n x n` block of `D`
is nonzero modulo `N`.
"""
function pluq_nonzero_mod_kernel!(flag, D, n::Int32, N::Int32)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total = n * n
    stride = blockDim().x * gridDim().x
    while idx <= total
        j = (idx - 1) ÷ n + 1
        i = (idx - 1) % n + 1
        if _pluq_mod_t(D[i, j], N) != zero(eltype(D))
            CUDA.@atomic flag[1] = max(flag[1], Int32(1))
        end
        idx += stride
    end
    return
end

"""
    pluq_check_identity(F, Aorig)

Check the PLUQ identity on GPU:
`P*A*Q == L*U (mod N)`.
Returns `true` if the identity holds.
"""
function pluq_check_identity(F::PLUQFactorization, Aorig::CuModMatrix)
    n = rows(Aorig)
    N = Aorig.N
    pdev = CuArray(Int32.(F.p))
    qdev = CuArray(Int32.(F.q))
    PAQ = zeros(eltype(Aorig.data), n, n, N)
    tx = 16
    ty = 16
    @cuda threads=(tx, ty) blocks=(max(1, cld(n, tx)), max(1, cld(n, ty))) pluq_apply_paq_kernel!(
        PAQ.data,
        Aorig.data,
        pdev,
        qdev,
        Int32(n),
    )
    L = pluq_extract_L(F)
    U = pluq_extract_U(F)
    LU = zeros(eltype(Aorig.data), n, n, N)
    mul!(LU, L, U)
    D = zeros(eltype(Aorig.data), n, n, N)
    sub!(D, PAQ, LU)
    flag = CUDA.zeros(Int32, 1)
    threads = 256
    @cuda threads=threads blocks=max(1, cld(n * n, threads)) pluq_nonzero_mod_kernel!(
        flag,
        D.data,
        Int32(n),
        Int32(N),
    )
    return Array(flag)[1] == 0
end
