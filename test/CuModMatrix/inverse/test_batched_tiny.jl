using Random

function _tiny_invertible_host(n::Int, p::Int, ::Type{T}, rng) where {T}
    A = Matrix{T}(I, n, n)
    for i = 1:n
        for j = 1:n
            if i != j
                A[i, j] = T(rand(rng, 0:(p-1)))
            end
        end
    end
    return A
end

function _check_identity_batch(A::CuModMatrix, Ainv::CuModMatrix, p::Int)
    got = mod.(round.(Int, Array(A * Ainv)), p)
    n = size(A, 1)
    want = Matrix{Int}(LinearAlgebra.I, n, n)
    return got == want
end

function test_batched_tiny_kernels()
    rng = Random.MersenneTwister(17)
    p = 101
    for n in (4, 8, 16, 32)
        mats = CuModMatrix[]
        origs = CuModMatrix[]
        for _ = 1:4
            H = _tiny_invertible_host(n, p, Float32, rng)
            push!(mats, CuModMatrix(H, p; elem_type = Float32))
            push!(origs, CuModMatrix(copy(H), p; elem_type = Float32))
        end
        Fs = if n == 4
            GPUFiniteFieldMatrices.pluq_batched_4x4!(mats)
        elseif n == 8
            GPUFiniteFieldMatrices.pluq_batched_8x8!(mats)
        elseif n == 16
            GPUFiniteFieldMatrices.pluq_batched_16x16!(mats)
        else
            GPUFiniteFieldMatrices.pluq_batched_32x32!(mats)
        end
        @test length(Fs) == length(mats)
        for i in eachindex(mats)
            @test Fs[i].rank == n
            @test GPUFiniteFieldMatrices.pluq_check_identity(Fs[i], origs[i])
        end

        mats2 = CuModMatrix[]
        for _ = 1:4
            push!(
                mats2,
                CuModMatrix(
                    _tiny_invertible_host(n, p, Float32, rng),
                    p;
                    elem_type = Float32,
                ),
            )
        end
        invs = if n == 4
            GPUFiniteFieldMatrices.inverse_batched_4x4!(mats2)
        elseif n == 8
            GPUFiniteFieldMatrices.inverse_batched_8x8!(mats2)
        elseif n == 16
            GPUFiniteFieldMatrices.inverse_batched_16x16!(mats2)
        else
            GPUFiniteFieldMatrices.inverse_batched_32x32!(mats2)
        end
        @test length(invs) == length(mats2)
        for i in eachindex(mats2)
            @test _check_identity_batch(mats2[i], invs[i], p)
        end

        S = Matrix{Float32}(I, n, n)
        S[2, :] .= S[1, :]
        singular = [CuModMatrix(S, p; elem_type = Float32)]
        if n == 4
            @test_throws GPUFiniteFieldMatrices.InverseNotDefinedException GPUFiniteFieldMatrices.inverse_batched_4x4!(
                singular,
            )
        elseif n == 8
            @test_throws GPUFiniteFieldMatrices.InverseNotDefinedException GPUFiniteFieldMatrices.inverse_batched_8x8!(
                singular,
            )
        elseif n == 16
            @test_throws GPUFiniteFieldMatrices.InverseNotDefinedException GPUFiniteFieldMatrices.inverse_batched_16x16!(
                singular,
            )
        else
            @test_throws GPUFiniteFieldMatrices.InverseNotDefinedException GPUFiniteFieldMatrices.inverse_batched_32x32!(
                singular,
            )
        end
    end
end
