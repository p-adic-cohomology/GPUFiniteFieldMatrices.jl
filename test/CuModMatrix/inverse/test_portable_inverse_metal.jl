using LinearAlgebra: I, SingularException, det, rank

function _portable_test_matrix(a, storage; N = 101)
    G = GPUFiniteFieldMatrices
    m, n = size(a)
    padded = fill(-17.0f0, m + 4, n + 6)
    padded[1:m, 1:n] .= a
    native = Metal.MtlArray{Float32,2,storage}(padded)
    return G.MtlModMatrix{Float32}(native; modulus = N, new_size = (m, n), mod = false)
end

_portable_test_product(a, b, N) = mod.(BigInt.(a) * BigInt.(b), BigInt(N))
_portable_test_eye(n) = Matrix{BigInt}(I, n, n)

function _portable_test_padding(A, before)
    m, n = size(A)
    after = Array(parent(A))
    return after[(m+1):end, :] == before[(m+1):end, :] &&
           after[:, (n+1):end] == before[:, (n+1):end]
end

function _portable_test_factor(F, original, expected_rank, N)
    m, n = size(original)
    @test rank(F) == expected_rank
    @test sort(F.p) == collect(1:m)
    @test sort(F.q) == collect(1:n)
    @test size(F.L) == (m, m)
    @test size(F.U) == (m, n)
    @test _portable_test_product(Array(F.L), Array(F.U), N) ==
          mod.(BigInt.(original[F.p, F.q]), N)
end

function _portable_test_square_data(n)
    a = zeros(Int, n, n)
    for j = 1:n, i = 1:j
        a[i, j] = i == j ? 1 + mod(i, 5) : mod(3i + 5j, 13)
    end
    a[1, :], a[n, :] = a[n, :], a[1, :]
    return a
end

@testset "Portable inverse API on Metal" begin
    G = GPUFiniteFieldMatrices
    N = 101
    aug = G.PLUQOptions(inverse_strategy = :augmented, check_prime = true)
    Metal.allowscalar(false)
    try
        for storage in (Metal.SharedStorage, Metal.PrivateStorage)
            @testset "$(nameof(storage))" begin
                host = [0 2 0; 3 1 4; 0 0 5]
                A = _portable_test_matrix(host, storage; N)
                before = Array(parent(A))
                F = G.pluq(A)
                @test F.LU !== A
                _portable_test_factor(F, host, 3, N)
                @test G.pluq_check_identity(F, A)
                raw_host = copy(host)
                raw_host[1, 2] -= N
                raw = _portable_test_matrix(raw_host, storage; N)
                raw_before = Array(parent(raw))
                @test G.pluq_check_identity(G.pluq(raw), raw)
                @test Array(parent(raw)) == raw_before
                repeated = G.PLUQ(F.LU, [1, 1, 3], F.q, F.rank)
                @test_throws ArgumentError G.pluq_check_identity(repeated, A)
                @test rank(A) == 3
                @test det(F) == det(A) == 71
                @test G.is_invertible(A)
                inplace = _portable_test_matrix(host, storage; N)
                @test G.pluq!(inplace).LU === inplace
                @test _portable_test_padding(inplace, before)
                for X in (inv(A), inv(A; options = aug), G.inverse_pluq(A))
                    @test Metal.storagemode(parent(X)) === storage
                    @test _portable_test_product(host, Array(X), N) == _portable_test_eye(3)
                    @test _portable_test_product(Array(X), host, N) == _portable_test_eye(3)
                end
                yes, X = G.is_invertible_with_inverse(A)
                @test yes
                @test _portable_test_product(Array(X), host, N) == _portable_test_eye(3)
                @test Array(parent(A)) == before

                wide0 = [0 2 0 1 0; 0 0 0 0 4; 0 3 5 1 0]
                tall0 = Matrix(transpose(wide0))
                wide = _portable_test_matrix(wide0, storage; N)
                tall = _portable_test_matrix(tall0, storage; N)
                wide_before, tall_before = Array(parent(wide)), Array(parent(tall))
                _portable_test_factor(G.pluq(wide), wide0, 3, N)
                _portable_test_factor(G.pluq(tall), tall0, 3, N)
                @test G.pluq_check_identity(G.pluq(wide), wide)
                @test G.pluq_check_identity(G.pluq(tall), tall)
                @test !G.is_invertible(wide)
                @test G.is_invertible_with_inverse(wide) == (false, nothing)
                for options in (G.PLUQOptions(), aug)
                    xr = G.right_inverse(wide; options)
                    xl = G.left_inverse(tall; options)
                    @test size(xr) == (5, 3)
                    @test size(xl) == (3, 5)
                    @test _portable_test_product(wide0, Array(xr), N) ==
                          _portable_test_eye(3)
                    @test _portable_test_product(Array(xl), tall0, N) ==
                          _portable_test_eye(3)
                end
                @test Array(parent(wide)) == wide_before
                @test Array(parent(tall)) == tall_before
                @test_throws DimensionMismatch inv(wide)
                @test_throws DimensionMismatch det(tall)
                @test_throws DimensionMismatch G.right_inverse(tall)
                @test_throws DimensionMismatch G.left_inverse(wide)

                singular = _portable_test_matrix([2 3; 4 6], storage; N)
                singular_before = Array(parent(singular))
                _portable_test_factor(G.pluq(singular), [2 3; 4 6], 1, N)
                @test det(singular) == 0
                @test G.is_invertible_with_inverse(singular) == (false, nothing)
                @test_throws SingularException inv(singular)
                @test_throws SingularException inv(singular; options = aug)
                @test Array(parent(singular)) == singular_before
                composite = _portable_test_matrix([1 0; 0 1], storage; N = 15)
                composite_before = Array(parent(composite))
                @test_throws DomainError G.pluq!(composite; options = aug)
                @test Array(parent(composite)) == composite_before

                zero = _portable_test_matrix(zeros(Int, 0, 0), storage; N)
                @test rank(zero) == 0
                @test det(zero) == 1
                @test size(inv(zero; options = aug)) == (0, 0)
                @test size(inv(zero)) == (0, 0)
                @test G.is_invertible(zero)
                empty_wide = _portable_test_matrix(zeros(Int, 0, 5), storage; N)
                empty_tall = _portable_test_matrix(zeros(Int, 5, 0), storage; N)
                @test size(G.right_inverse(empty_wide)) == (5, 0)
                @test size(G.left_inverse(empty_tall)) == (0, 5)

                edgeN = 16_777_213
                edge = [edgeN-2 edgeN-1; 0 edgeN-3]
                edge_a = _portable_test_matrix(edge, storage; N = edgeN)
                edge_x = inv(edge_a; options = aug)
                @test _portable_test_product(edge, Array(edge_x), edgeN) ==
                      _portable_test_eye(2)
                @test_throws OverflowError G.inverse_pluq(edge_a)

                host35 = _portable_test_square_data(35)
                A35 = _portable_test_matrix(host35, storage; N)
                options = G.PLUQOptions(blocksize = 8, basecase = 8, schur_tile = 8)
                _portable_test_factor(G.pluq(A35; options), host35, 35, N)
                @test _portable_test_product(host35, Array(inv(A35; options)), N) ==
                      _portable_test_eye(35)
            end
        end
    finally
        Metal.allowscalar(true)
    end
end

@testset "Portable tiny batches on Metal" begin
    G = GPUFiniteFieldMatrices
    N = 101
    aug = G.PLUQOptions(inverse_strategy = :augmented)
    multi = G.PLUQOptions(inverse_strategy = :augmented, batch_streams = 2)
    Metal.allowscalar(false)
    try
        for storage in (Metal.SharedStorage, Metal.PrivateStorage)
            @testset "$(nameof(storage))" begin
                for (n, factor!, inverse!) in (
                    (4, G.pluq_batched_4x4!, G.inverse_batched_4x4!),
                    (8, G.pluq_batched_8x8!, G.inverse_batched_8x8!),
                    (16, G.pluq_batched_16x16!, G.inverse_batched_16x16!),
                    (32, G.pluq_batched_32x32!, G.inverse_batched_32x32!),
                )
                    host = _portable_test_square_data(n)
                    inputs = [
                        _portable_test_matrix(h, storage; N) for
                        h in (host, Matrix(transpose(host)))
                    ]
                    snapshot = Array.(parent.(inputs))
                    dests = [
                        _portable_test_matrix(zeros(Int, n, n), storage; N) for _ in inputs
                    ]
                    padding = Array.(parent.(dests))
                    @test inverse!(dests, inputs; options = aug) === dests
                    for i in eachindex(inputs)
                        @test _portable_test_padding(dests[i], padding[i])
                        @test _portable_test_product(
                            Array(inputs[i]),
                            Array(dests[i]),
                            N,
                        ) == _portable_test_eye(n)
                    end
                    @test Array.(parent.(inputs)) == snapshot
                    allocated = G.inverse_batched(inputs; options = aug)
                    @test Array.(dests) == Array.(allocated)

                    work = [_portable_test_matrix(Array(A), storage; N) for A in inputs]
                    @test inverse!(work; options = aug) === work
                    @test Array.(work) == Array.(dests)

                    work = [_portable_test_matrix(Array(A), storage; N) for A in inputs]
                    factors = factor!(work)
                    for i in eachindex(work)
                        @test factors[i].LU === work[i]
                        _portable_test_factor(factors[i], Array(inputs[i]), n, N)
                        @test _portable_test_padding(work[i], snapshot[i])
                    end
                    @test factors[1].q !== factors[2].q

                    fill!(dests[1], 9)
                    bad = _portable_test_matrix(zeros(Int, n, n), storage; N)
                    dest_before = Array.(parent.(dests))
                    mixed = [inputs[1], bad]
                    mixed_before = Array.(parent.(mixed))
                    @test_throws SingularException inverse!(dests, mixed; options = aug)
                    @test_throws SingularException inverse!(mixed; options = aug)
                    @test Array.(parent.(dests)) == dest_before
                    @test Array.(parent.(mixed)) == mixed_before

                    if n == 4
                        @test_throws SingularException inverse!(dests, mixed)
                        @test Array.(parent.(dests)) == dest_before
                        @test_throws ArgumentError inverse!(dests, inputs; options = multi)
                        @test_throws ArgumentError G.pluq_batched!(
                            dests,
                            inputs;
                            options = multi,
                        )
                        @test_throws ArgumentError inverse!(inputs, inputs; options = aug)
                        @test_throws ArgumentError G.pluq_batched!([inputs[1], inputs[1]])
                        backing = Metal.MtlArray{Float32,2,storage}(ones(Float32, 5, 6))
                        aview = G.GPUModArray(view(backing, 1:4, 1:4); modulus = N)
                        bview = G.GPUModArray(view(backing, 2:5, 1:4); modulus = N)
                        @test_throws ArgumentError G.pluq_batched!([aview, bview])
                        @test Array(backing) == ones(Float32, 5, 6)
                        @test_throws ArgumentError G.pluq_batched!(
                            [dests[1], dests[1]],
                            inputs,
                        )
                        @test_throws DimensionMismatch inverse!(
                            dests[1:1],
                            inputs;
                            options = aug,
                        )
                        wrong = _portable_test_matrix(zeros(Int, 4, 4), storage; N = 103)
                        @test_throws G.ModulusMismatchError inverse!(
                            [dests[1], wrong],
                            inputs;
                            options = aug,
                        )
                        @test Array.(parent.(dests)) == dest_before
                        @test Array.(parent.(inputs)) == snapshot

                        factors = G.pluq_batched!(dests, inputs)
                        @test factors[1].LU === dests[1]
                        @test factors[2].LU === dests[2]
                        @test Array.(parent.(inputs)) == snapshot
                        @test all(
                            i -> _portable_test_padding(dests[i], padding[i]),
                            eachindex(dests),
                        )
                        empty = eltype(inputs)[]
                        @test isempty(G.pluq_batched!(empty; options = multi))
                        @test isempty(G.inverse_batched(empty; options = multi))
                        @test inverse!(empty; options = multi) === empty
                    end
                end

                rect0 = [0 1 2; 0 0 3]
                t0 = Matrix(transpose(rect0))
                mixed = [_portable_test_matrix(h, storage; N) for h in (rect0, t0)]
                before = Array.(parent.(mixed))
                @test_throws DimensionMismatch G.pluq_batched_NxN!(mixed)
                f = G.pluq_batched(mixed)
                @test all(F -> F.rank == 2, f)
                x = G.inverse_batched(mixed; options = aug)
                @test _portable_test_product(rect0, Array(x[1]), N) == _portable_test_eye(2)
                @test _portable_test_product(Array(x[2]), t0, N) == _portable_test_eye(2)
                @test Array.(parent.(mixed)) == before
            end
        end
    finally
        Metal.allowscalar(true)
    end
end
