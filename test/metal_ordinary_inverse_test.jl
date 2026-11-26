function _ordinary_test_native(values, storage, ::Type{T} = Float32) where {T}
    return Metal.MtlArray{T,2,storage}(T.(values))
end

function _ordinary_test_strided(values, storage, ::Type{T} = Float32) where {T}
    m, n = size(values)
    rows, cols = 2:2:(2m), (2n+1):-2:3
    host = fill(T(-37), 2m + 2, 2n + 3)
    host[rows, cols] .= values
    backing = _ordinary_test_native(host, storage, T)
    return (; data = view(backing, rows, cols), backing, host, rows, cols)
end

function _ordinary_expected_backing(fixture, values)
    result = copy(fixture.host)
    result[fixture.rows, fixture.cols] .= values
    return result
end

_ordinary_test_values(m, n, q) =
    [q - 1 - mod(813i * j + 193i + 179j, q) for i = 1:m, j = 1:n]

function _ordinary_test_triangular(n, q, lower)
    return [
        i == j ? mod(2i, q - 1) + 1 :
        (lower ? i > j : i < j) ? q - 1 - mod(813i * j + 193i + 179j, q) : 0 for
        i = 1:n, j = 1:n
    ]
end

function _ordinary_reference_inverse(A, N, lower)
    q = BigInt(N)
    n = size(A, 1)
    inverses = [invmod(BigInt(A[i, i]), q) for i = 1:n]
    result = zeros(BigInt, n, n)
    for col = 1:n, row in (lower ? (1:n) : (n:-1:1))
        terms = lower ? (1:(row-1)) : ((row+1):n)
        dot = BigInt(0)
        for k in terms
            dot += BigInt(A[row, k]) * result[k, col]
        end
        result[row, col] = mod(inverses[row] * (Int(row == col) - dot), q)
    end
    return result
end

function _ordinary_reference_schur(A, q, first, last)
    expected = BigInt.(A)
    rows = (last+1):size(A, 1)
    cols = (last+1):size(A, 2)
    panel = first:last
    expected[rows, cols] .= mod.(
        expected[rows, cols] - expected[rows, panel] * expected[panel, cols],
        BigInt(q),
    )
    return expected
end

@testset "Ordinary exact arithmetic" begin
    G = GPUFiniteFieldMatrices
    for q in (UInt64(1) << 32, (UInt64(1) << 32) + 15, (UInt64(1) << 53) - 111)
        a, b = q - 7, q - 19
        @test G._ordinary_residue_product(a, b, q) == mod(BigInt(a) * BigInt(b), BigInt(q))
        @test G._triangular_diagonal_inverse(q - 1, q) == q - 1
    end
end

@testset verbose = true "Metal ordinary Schur and triangular kernels" begin
    G = GPUFiniteFieldMatrices
    Metal.allowscalar(false)
    try
        for storage in (Metal.SharedStorage, Metal.PrivateStorage)
            @testset "$(nameof(storage))" begin
                @testset "Ordinary tiled products" begin
                    for (m, k, n, tile, transposed, q) in (
                        (0, 5, 3, 8, false, 101),
                        (5, 3, 0, 8, true, 101),
                        (4, 0, 7, 8, false, 101),
                        (1, 1, 1, 16, true, 65521),
                        (17, 19, 23, 8, false, 2^24),
                        (33, 35, 21, 16, true, 2^24),
                    )
                        a = _ordinary_test_values(m, k, q)
                        b = _ordinary_test_values(k, n, q)
                        A = _ordinary_test_strided(a, storage)
                        B = _ordinary_test_strided(b, storage)
                        C = _ordinary_test_strided(fill(-59, m, n), storage)
                        @test G._mat_mul_ops!(
                            C.data,
                            A.data,
                            B.data,
                            q;
                            tile,
                            transpose_b = transposed,
                        ) === C.data
                        expected = mod.(BigInt.(a) * BigInt.(b), BigInt(q))
                        @test Array(C.backing) == _ordinary_expected_backing(C, expected)
                        @test Array(A.backing) == A.host
                        @test Array(B.backing) == B.host
                    end
                end

                @testset "Schur updates" begin
                    for (m, n, first, last, tile, transposed, q) in (
                        (0, 4, 1, 0, 8, false, 101),
                        (5, 8, 4, 3, 16, false, 101),
                        (4, 11, 1, 4, 16, true, 101),
                        (1, 1, 1, 1, 8, true, 65521),
                        (35, 23, 3, 5, 8, false, 101),
                        (21, 25, 2, 11, 8, true, 65521),
                        (35, 35, 9, 22, 32, false, 2^24),
                        (33, 31, 1, 7, 16, true, 2^24),
                    )
                        packed = fill(-71, m + 2, n + 3)
                        a = _ordinary_test_values(m, n, q)
                        if last == first - 1
                            # A zero-width panel must leave even noncanonical entries alone.
                            a .= -69
                        end
                        packed[1:m, 1:n] .= a
                        A = _ordinary_test_strided(packed, storage)
                        options =
                            PLUQOptions(schur_tile = tile, schur_transpose_u = transposed)
                        if m == n
                            @test G.pluq_schur_update_gpu!(
                                A.data,
                                q,
                                first,
                                last,
                                n;
                                options,
                            ) === A.data
                        else
                            @test G.pluq_schur_update_rect_gpu!(
                                A.data,
                                q,
                                first,
                                last,
                                m,
                                n;
                                options,
                            ) === A.data
                        end
                        if first <= last && last < min(m, n)
                            packed[1:m, 1:n] .= _ordinary_reference_schur(a, q, first, last)
                        end
                        @test Array(A.backing) == _ordinary_expected_backing(A, packed)
                    end
                end

                @testset "Destination-first triangular tiles" begin
                    for (n, N) in ((0, 101), (1, 101), (5, 101), (31, 2^24), (32, 65536)),
                        lower in (false, true)

                        a = _ordinary_test_triangular(n, N, lower)
                        expected = _ordinary_reference_inverse(a, N, lower)
                        for j = 1:n, i = 1:n
                            if lower ? i < j : i > j
                                a[i, j] = -83
                            end
                        end
                        source = fill(-89, n + 3, n + 5)
                        source[2:(n+1), 4:(n+3)] .= a
                        A = _ordinary_test_strided(source, storage)
                        dest = fill(-97, n + 2, n + 4)
                        D = _ordinary_test_strided(dest, storage)
                        solve! =
                            lower ? G._lower_triangular_inv_32! :
                            G._upper_triangular_inv_32!
                        @test solve!(D.data, A.data, N, 1, 3, n) === D.data
                        dest[1:n, 1:n] .= expected
                        @test Array(D.backing) == _ordinary_expected_backing(D, dest)
                        @test Array(A.backing) == A.host
                    end
                end

                @testset "Square and wide triangular inverse" begin
                    for (m, n, q) in (
                            (0, 5, 101),
                            (1, 1, 101),
                            (31, 39, 101),
                            (32, 32, 101),
                            (33, 49, 101),
                            (65, 79, 101),
                            (35, 42, 2^24),
                        ),
                        lower in (false, true)

                        a = _ordinary_test_values(m, n, q)
                        a[:, 1:m] .= _ordinary_test_triangular(m, q, lower)
                        A = _ordinary_test_strided(a, storage)
                        wrapper = G.MtlModMatrix(A.data; modulus = q)
                        invert =
                            lower ? G.lower_triangular_inverse_no_copy :
                            G.upper_triangular_inverse_no_copy
                        inverse = invert(wrapper)
                        expected = zeros(BigInt, n, m)
                        expected[1:m, :] .= _ordinary_reference_inverse(a[:, 1:m], q, lower)
                        @test size(inverse) == (n, m)
                        @test G.modulus(inverse) == q
                        @test Metal.storagemode(parent(inverse)) === storage
                        physical = zeros(Float32, size(parent(inverse)))
                        physical[1:n, 1:m] .= expected
                        @test Array(parent(inverse)) == physical
                        @test mod.(BigInt.(a) * BigInt.(Array(inverse)), q) ==
                              Matrix{Int}(I, m, m)
                        @test Array(A.backing) == A.host
                    end
                    tall = _ordinary_test_triangular(5, 101, true)
                    a = vcat(tall, _ordinary_test_values(3, 5, 101))
                    A = G.MtlModMatrix(a; modulus = 101, storage)
                    X = G.lower_triangular_inverse_no_copy(A)
                    @test mod.(BigInt.(Array(X)) * BigInt.(a), 101) == Matrix{Int}(I, 5, 5)
                end

                @testset "Structured factors and Float16 storage" begin
                    n, q = 35, 101
                    host = _ordinary_test_values(n, n, q)
                    raw = _ordinary_test_strided(host, storage)
                    for (triangle, invert, lower) in (
                        (UnitUpperTriangular, G.upper_triangular_inverse_no_copy, false),
                        (UnitLowerTriangular, G.lower_triangular_inverse_no_copy, true),
                    )
                        A = G.MtlModArray{Float32,2}(triangle(raw.data); modulus = q)
                        @test Array(invert(A)) ==
                              _ordinary_reference_inverse(triangle(host), q, lower)
                    end
                    @test Array(raw.backing) == raw.host

                    q = 2048
                    a = _ordinary_test_triangular(n, q, false)
                    M = G.MtlModArray{Float16,2}(a; modulus = q, storage)
                    X = G.upper_triangular_inverse_no_copy(M)
                    @test eltype(X) === Float16
                    @test Array(X) == _ordinary_reference_inverse(a, q, false)
                end

                @testset "Validation before destination writes" begin
                    q = 91
                    n = 7
                    source = _ordinary_test_triangular(n, q, true)
                    for i = 1:n
                        source[i, i] = 1
                    end
                    source[n, n] = 7
                    A = _ordinary_test_strided(source, storage)
                    D = _ordinary_test_strided(fill(-61, n, n), storage)
                    @test_throws LinearAlgebra.SingularException G._lower_triangular_inv_32!(
                        D.data,
                        A.data,
                        q,
                        0,
                        0,
                        n,
                    )
                    @test_throws LinearAlgebra.SingularException G.lower_triangular_inverse_no_copy(
                        G.MtlModMatrix(A.data; modulus = q),
                    )
                    @test_throws DimensionMismatch G._upper_triangular_inv_32!(
                        D.data,
                        A.data,
                        q,
                        0,
                        1,
                        n,
                    )
                    @test_throws DimensionMismatch G._lower_triangular_inv_32!(
                        D.data,
                        A.data,
                        q,
                        0,
                        0,
                        33,
                    )
                    @test_throws ArgumentError G._upper_triangular_inv_32!(
                        A.data,
                        A.data,
                        q,
                        0,
                        0,
                        n,
                    )
                    @test_throws ArgumentError G._mat_mul_ops!(A.data, A.data, A.data, q)
                    @test_throws ArgumentError G._mat_mul_ops!(D.data, A.data, A.data, 0)
                    @test_throws ArgumentError G._mat_mul_ops!(
                        D.data,
                        A.data,
                        A.data,
                        q;
                        tile = 32,
                    )
                    @test_throws DimensionMismatch G.pluq_schur_update_rect_gpu!(
                        A.data,
                        q,
                        1,
                        2,
                        n + 1,
                        n,
                    )
                    @test_throws DimensionMismatch G.pluq_schur_update_gpu!(
                        A.data,
                        q,
                        4,
                        2,
                        n,
                    )
                    @test_throws DomainError G.pluq_schur_update_gpu!(
                        A.data,
                        2^24 + 1,
                        1,
                        2,
                        n,
                    )
                    @test Array(A.backing) == A.host
                    @test Array(D.backing) == D.host
                end
            end
        end
    finally
        Metal.allowscalar(true)
    end
end
