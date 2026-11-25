function _metal_product_test_native(values, storage)
    return Metal.MtlArray{Float32,ndims(values),storage}(Float32.(values))
end

function _metal_product_test_padded(values, storage, q; padding = 0, sentinel = -911.0f0)
    dims = size(values)
    backing = fill(sentinel, map(d -> d + padding, dims))
    backing[map(d->1:d, dims)...] .= values
    data = _metal_product_test_native(backing, storage)
    return GPUFiniteFieldMatrices.MtlModArray(data; modulus = q, new_size = dims)
end

_metal_product_test_reference(A, B, q) = mod.(BigInt.(A) * BigInt.(B), BigInt(q))

function _metal_product_test_backing(expected, destination)
    backing = zeros(Float32, size(parent(destination)))
    backing[map(d->1:d, size(expected))...] .= expected
    return backing
end

@testset verbose = true "Metal exact Float32 products" begin
    G = GPUFiniteFieldMatrices
    ext = Base.get_extension(G, :GPUFiniteFieldMatricesMetalExt)
    Metal.allowscalar(false)
    try
        for storage in (Metal.SharedStorage, Metal.PrivateStorage)
            @testset "$(nameof(storage))" begin
                @testset "GEMM and GEMV dimensions" begin
                    q = 101
                    for (m, k, n) in (
                        (0, 5, 3),
                        (9, 0, 11),
                        (5, 3, 0),
                        (1, 1, 1),
                        (31, 33, 7),
                        (32, 32, 32),
                        (33, 31, 11),
                        (257, 255, 13),
                    )
                        a = [mod(13i + 19j, q) for i = 1:m, j = 1:k]
                        b = [mod(17i + 23j, q) for i = 1:k, j = 1:n]
                        x = [mod(29i + 5, q) for i = 1:k]
                        A = _metal_product_test_padded(a, storage, q; padding = 3)
                        B = _metal_product_test_padded(b, storage, q; padding = 5)
                        X = _metal_product_test_padded(x, storage, q; padding = 4)
                        C = _metal_product_test_padded(zeros(m, n), storage, q; padding = 2)
                        Y = _metal_product_test_padded(zeros(m), storage, q; padding = 7)
                        expected_C = _metal_product_test_reference(a, b, q)
                        expected_Y = _metal_product_test_reference(a, x, q)
                        original_C, original_Y = parent(C), parent(Y)
                        before_A, before_B, before_X =
                            Array(parent(A)), Array(parent(B)), Array(parent(X))
                        @test LinearAlgebra.mul!(C, A, B) === C
                        @test LinearAlgebra.mul!(Y, A, X) === Y
                        @test parent(C) === original_C
                        @test parent(Y) === original_Y
                        @test Array(original_C) ==
                              _metal_product_test_backing(expected_C, C)
                        @test Array(original_Y) ==
                              _metal_product_test_backing(expected_Y, Y)
                        @test Array(parent(A)) == before_A
                        @test Array(parent(B)) == before_B
                        @test Array(parent(X)) == before_X
                    end
                end

                @testset "Stripe exactness and public products" begin
                    bound = 2049
                    q = 2^23
                    a = repeat([2048 2048 2047 0 2048 2048 1], 33, 1)
                    x = [2047, 2048, 1, 0, 2048, 2048, 2]
                    b = hcat(x, reverse(x), x .÷ 2)
                    @test G.find_max_stripe_ops(Float32, bound; modulus = q) == 2
                    @test sum(BigInt.(a[1, 1:6]) .* x[1:6]) == 2^24 - 1
                    A = G.MtlModMatrix(a; modulus = bound, storage)
                    B = G.MtlModMatrix(b; modulus = bound, storage)
                    X = G.MtlModVector(x; modulus = bound, storage)
                    C = G.MtlModMatrix(fill(29, 33, 3); modulus = q, storage)
                    Y = G.MtlModVector(fill(31, 33); modulus = bound, storage)
                    expected_C = _metal_product_test_reference(a, b, q)
                    expected_Y = _metal_product_test_reference(a, x, q)
                    for max_ops in (1, 2, nothing)
                        @test G.stripe_mul!(C, A, B; input_bound = bound, max_ops) === C
                        @test G.stripe_mul!(
                            Y,
                            A,
                            X;
                            modulus_override = q,
                            input_bound = bound,
                            max_ops,
                        ) === Y
                        @test Array(C) == expected_C
                        @test Array(Y) == expected_Y
                        @test G.modulus(Y) == bound
                    end
                    @test G.stripe_mul!(
                        Y,
                        A,
                        X;
                        modulus_override = q,
                        max_ops = 3,
                        respect_max_ops = false,
                    ) === Y
                    @test Array(Y) == expected_Y
                    for (right, expected) in ((B, expected_C), (X, expected_Y))
                        allocated = G.stripe_mul(A, right; modulus_override = q)
                        @test allocated isa G.MtlModArray{Float32}
                        @test G.modulus(allocated) == q
                        @test Array(allocated) == expected
                    end
                    @test Array(A * B) == _metal_product_test_reference(a, b, bound)
                    @test Array(A * X) == _metal_product_test_reference(a, x, bound)
                end

                @testset "Native and Julia views" begin
                    q = 97
                    a = [mod(3i + 5j, q) for i = 1:33, j = 1:31]
                    b = [mod(7i + 11j, q) for i = 1:31, j = 1:6]
                    x = [mod(31i + 11, q) for i = 1:31]
                    arows, acols = 2:2:66, 63:-2:3
                    a_backing = fill(-57.0f0, 68, 66)
                    a_backing[arows, acols] .= a
                    a_data = _metal_product_test_native(a_backing, storage)
                    A = G.MtlModMatrix(view(a_data, arows, acols); modulus = q)
                    b_data = _metal_product_test_native(transpose(b), storage)
                    B = G.MtlModArray{Float32,2}(adjoint(b_data); modulus = q)
                    xinds = 3:2:63
                    x_backing = fill(-59.0f0, 65)
                    x_backing[xinds] .= x
                    x_data = _metal_product_test_native(x_backing, storage)
                    X = G.MtlModVector(view(x_data, xinds); modulus = q)

                    crows, ccols = 2:2:70, 3:2:17
                    c_backing = fill(-61.0f0, 19, 72)
                    c_data = _metal_product_test_native(c_backing, storage)
                    c_view = transpose(view(c_data, ccols, crows))
                    C = G.MtlModArray{Float32,2}(c_view; modulus = q, new_size = (33, 6))
                    expected_C = _metal_product_test_reference(a, b, q)
                    expected_backing_C = copy(c_backing)
                    expected_backing_C[ccols, crows] .=
                        transpose(_metal_product_test_backing(expected_C, C))

                    yinds = 4:2:74
                    y_backing = fill(-67.0f0, 76)
                    y_data = _metal_product_test_native(y_backing, storage)
                    y_view = view(y_data, yinds)
                    Y = G.MtlModVector(y_view; modulus = q, new_size = (33,))
                    expected_Y = _metal_product_test_reference(a, x, q)
                    expected_backing_Y = copy(y_backing)
                    expected_backing_Y[yinds] .= _metal_product_test_backing(expected_Y, Y)

                    @test G.stripe_mul!(C, A, B; max_ops = 7) === C
                    @test G.stripe_mul!(Y, A, X; max_ops = 7) === Y
                    @test parent(C) === c_view
                    @test parent(Y) === y_view
                    @test Array(c_data) == expected_backing_C
                    @test Array(y_data) == expected_backing_Y
                    @test Array(a_data) == a_backing
                    @test Array(x_data) == x_backing
                    @test Array(b_data) == transpose(b)

                    AP = _metal_product_test_padded(a_backing, storage, q; padding = 2)
                    BP = _metal_product_test_padded(transpose(b), storage, q; padding = 3)
                    XP = _metal_product_test_padded(x_backing, storage, q; padding = 2)
                    AV = view(AP, arows, acols)
                    XV = view(XP, xinds)
                    @test LinearAlgebra.mul!(C, AV, transpose(BP); max_ops = 7) === C
                    @test Array(c_data) == expected_backing_C
                    @test Array(AV * adjoint(BP)) == expected_C
                    @test LinearAlgebra.mul!(Y, AV, XV) === Y
                    @test Array(y_data) == expected_backing_Y
                    @test Array(A * XV) == expected_Y
                    Vt = G.MtlModVector(ones(Int, 33); modulus = q, storage)
                    Zt = G.MtlModVector(zeros(Int, 31); modulus = q, storage)
                    @test G.stripe_mul!(Zt, transpose(A), Vt; max_ops = 7) === Zt
                    @test Array(Zt) ==
                          _metal_product_test_reference(transpose(a), ones(Int, 33), q)

                    square = [mod(17i + 19j, q) for i = 1:31, j = 1:31]
                    S = _metal_product_test_padded(square, storage, q; padding = 3)
                    Ct = G.MtlModMatrix(zeros(Int, 33, 31); modulus = q, storage)
                    for triangular in (
                        LinearAlgebra.UpperTriangular,
                        LinearAlgebra.LowerTriangular,
                        LinearAlgebra.UnitUpperTriangular,
                        LinearAlgebra.UnitLowerTriangular,
                    )
                        T = triangular(S)
                        ref = triangular(square)
                        @test Array(T * X) == _metal_product_test_reference(ref, x, q)
                        expected_T = _metal_product_test_reference(a, ref, q)
                        @test Array(A * T) == expected_T
                        @test LinearAlgebra.mul!(Ct, A, T; max_ops = 7) === Ct
                        @test Array(Ct) == expected_T
                        @test Array(LinearAlgebra.UpperTriangular(S) * T) ==
                              _metal_product_test_reference(
                            LinearAlgebra.UpperTriangular(square),
                            ref,
                            q,
                        )
                    end
                end

                @testset "Validation before writes" begin
                    q = 8192
                    A = G.MtlModMatrix(fill(3, 5, 7); modulus = q, storage)
                    B = G.MtlModMatrix(fill(5, 7, 3); modulus = q, storage)
                    X = G.MtlModVector(fill(11, 7); modulus = q, storage)
                    C = G.MtlModMatrix(fill(13, 5, 3); modulus = q, storage)
                    Y = G.MtlModVector(fill(17, 5); modulus = q, storage)
                    fill!(parent(C), 13.0f0)
                    fill!(parent(Y), 17.0f0)
                    @test_throws ArgumentError G.stripe_mul!(C, A, B)
                    @test_throws ArgumentError G.stripe_mul!(Y, A, X)
                    @test_throws ArgumentError G.stripe_mul!(C, A, B; input_bound = 1)
                    @test_throws ArgumentError G.stripe_mul!(C, A, B; modulus_override = 0)
                    @test_throws ArgumentError G.stripe_mul!(
                        C,
                        A,
                        B;
                        modulus_override = 2^24,
                        input_bound = 12,
                    )
                    @test_throws ArgumentError G.stripe_mul!(Y, A, X; max_ops = 0)
                    @test_throws ArgumentError G.stripe_mul!(
                        C,
                        A,
                        B;
                        input_bound = 2049,
                        max_ops = 3,
                    )
                    @test_throws ArgumentError G.stripe_mul!(
                        Y,
                        A,
                        X;
                        input_bound = 2049,
                        max_ops = big(typemax(Int)) + 1,
                    )
                    @test_throws DimensionMismatch G.stripe_mul!(C, A, A; input_bound = 12)
                    @test_throws DimensionMismatch G.stripe_mul!(Y, A, Y; input_bound = 18)
                    B_wrong_modulus = G.MtlModMatrix(fill(3, 7, 3); modulus = 97, storage)
                    @test_throws G.ModulusMismatchError G.stripe_mul!(C, A, B_wrong_modulus)
                    @test all(==(13.0f0), Array(parent(C)))
                    @test all(==(17.0f0), Array(parent(Y)))
                    H = G.MtlModArray{Float16,2}(fill(5, 3, 3); modulus = 11, storage)
                    D = G.MtlModArray{Float16,2}(fill(7, 3, 3); modulus = 11, storage)
                    @test_throws ArgumentError G.stripe_mul!(D, H, H)
                    @test Array(D) == fill(7, 3, 3)

                    shared = _metal_product_test_native(fill(19, 9, 9), storage)
                    L = G.MtlModMatrix(view(shared, 2:6, 1:7); modulus = q)
                    C_alias = G.MtlModMatrix(view(shared, 3:7, 3:5); modulus = q)
                    @test_throws ArgumentError G.stripe_mul!(
                        C_alias,
                        L,
                        B;
                        input_bound = 20,
                    )
                    @test all(==(19.0f0), Array(shared))
                    v = _metal_product_test_native(fill(23, 20), storage)
                    Z_alias = G.MtlModVector(view(v, 1:10); modulus = q, new_size = (5,))
                    X_alias = G.MtlModVector(view(v, 7:13); modulus = q)
                    @test_throws ArgumentError G.stripe_mul!(
                        Z_alias,
                        A,
                        X_alias;
                        input_bound = 24,
                    )
                    @test all(==(23.0f0), Array(v))
                    raw_triangle = _metal_product_test_native(fill(29, 3, 3), storage)
                    triangular_C = G.MtlModArray{Float32,2}(
                        LinearAlgebra.UpperTriangular(raw_triangle);
                        modulus = q,
                    )
                    dense = G.MtlModMatrix(fill(7, 3, 3); modulus = q, storage)
                    @test_throws ArgumentError G.stripe_mul!(
                        triangular_C,
                        dense,
                        dense;
                        input_bound = 8,
                    )
                    @test all(==(29.0f0), Array(raw_triangle))

                    Ae = G.MtlModMatrix(zeros(Int, 5, 0); modulus = q, storage)
                    Be = G.MtlModMatrix(zeros(Int, 0, 3); modulus = q, storage)
                    Xe = G.MtlModVector(Int[]; modulus = q, storage)
                    @test_throws ArgumentError G.stripe_mul!(C, Ae, Be; max_ops = 1)
                    @test_throws ArgumentError G.stripe_mul!(Y, Ae, Xe; max_ops = 1)
                    @test all(==(13.0f0), Array(parent(C)))
                    @test all(==(17.0f0), Array(parent(Y)))
                    @test G.stripe_mul!(C, Ae, Be) === C
                    @test G.stripe_mul!(Y, Ae, Xe) === Y
                    @test all(iszero, Array(parent(C)))
                    @test all(iszero, Array(parent(Y)))
                end

                @testset "Scratch capacity and dense reuse" begin
                    a = _metal_product_test_native(fill(2, 33, 31), storage)
                    b = _metal_product_test_native(fill(3, 31, 9), storage)
                    x = _metal_product_test_native(fill(5, 31), storage)
                    c = _metal_product_test_native(fill(NaN32, 33, 9), storage)
                    y = _metal_product_test_native(fill(NaN32, 33), storage)
                    @test ext._metal_stripe_buffers(c, a, b, size(c), 31) === (c, a, b)
                    @test ext._metal_stripe_buffers(y, a, x, size(y), 31) === (y, a, x)
                    for (dest, right, right_size) in ((c, b, (7, 9)), (y, x, (7,)))
                        packed = ext._metal_stripe_buffers(dest, a, right, size(dest), 7)
                        @test packed[1] === dest
                        @test size(packed[2]) == (33, 7)
                        @test size(packed[3]) == right_size
                        @test all(buffer -> Metal.storagemode(buffer) === storage, packed)
                    end
                    A = G.MtlModMatrix(a; modulus = 101)
                    B = G.MtlModMatrix(b; modulus = 101)
                    X = G.MtlModVector(x; modulus = 101)
                    C = G.MtlModMatrix(c; modulus = 101)
                    Y = G.MtlModVector(y; modulus = 101)
                    @test G.stripe_mul!(C, A, B; max_ops = 31) === C
                    @test G.stripe_mul!(Y, A, X; max_ops = 31) === Y
                    @test Array(c) == fill(mod(big(2) * 3 * 31, 101), 33, 9)
                    @test Array(y) == fill(mod(big(2) * 5 * 31, 101), 33)
                end
            end
        end
    finally
        Metal.allowscalar(true)
    end
end
