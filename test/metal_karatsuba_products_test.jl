module MetalKaratsubaProductsTests

using GPUFiniteFieldMatrices, LinearAlgebra, Metal, Random, Test
using Base.ScopedValues: with

const FFM = GPUFiniteFieldMatrices
const SCRATCH_SENTINEL = -137.0f0

upload(input, ::Type{S}) where {S} = MtlArray{eltype(input),ndims(input),S}(input)

function fixture_region(dims, layout)
    if layout === :offset && length(dims) == 1
        return (2:2:(2*only(dims)),)
    end
    offset = layout === :offset ? 2 : 1
    return map(n -> offset:(offset+n-1), dims)
end

function plane(values, storage, layout, metadata)
    dims = size(values)
    layout === :native && return upload(Float32.(values), storage)
    region = fixture_region(dims, layout)
    physical = fill(-991.0f0, map(range -> last(range) + 5, region))
    physical[region...] .= values
    native = upload(physical, storage)
    if layout === :offset
        return MtlModArray{Float32,ndims(values)}(
            view(native, region...);
            modulus = metadata,
        )
    end
    return MtlModArray{Float32,ndims(values)}(native; modulus = metadata, new_size = dims)
end

function product_fixture(values, R, H, M, storage; layout = :padded)
    lo = plane(mod.(values, R), storage, layout, 107)
    hi = plane(div.(values, R), storage, layout, 233)
    return KaratsubaArray(lo, hi; low_modulus = R, high_modulus = H, modulus = M)
end

parts(K) = (low_part(K), high_part(K), K.plan)
backing(A) = FFM._elementwise_native_root(FFM._karatsuba_native(A))
snapshot(K) = map(A -> A === nothing ? nothing : Array(backing(A)), parts(K))
logical(A) = Array(A)
reconstruct(K) =
    BigInt.(logical(low_part(K))) + BigInt(low_modulus(K)) * BigInt.(logical(high_part(K)))

function plan_with_sentinel!(K)
    FFM.initialize_plan!(K)
    fill!(backing(K.plan), SCRATCH_SENTINEL)
    return K
end

function check_padding(K, before, layout)
    for (index, A) in pairs(parts(K))
        after = Array(backing(A))
        region = fixture_region(size(K), index <= 2 ? layout : :padded)
        @test all(CartesianIndices(after)) do I
            inside = all(i -> I[i] in region[i], 1:ndims(K))
            inside || after[I] == before[index][I]
        end
    end
end

function check_allocating(A, B, expected, storage)
    before_a, before_b = snapshot(A), snapshot(B)
    plans = (A.plan, B.plan)
    result = A * B
    @test reconstruct(result) == expected
    @test Array(result) == expected
    @test (low_modulus(result), high_modulus(result), modulus(result)) ==
          (low_modulus(A), high_modulus(A), modulus(A))
    @test (A.plan, B.plan) === plans
    @test snapshot(A) == before_a
    @test snapshot(B) == before_b
    @test result.plan !== nothing
    @test result.plan !== A.plan && result.plan !== B.plan
    @test backing(low_part(result)) isa Metal.MtlArray{Float32,ndims(result),storage}
    @test backing(result.plan) isa Metal.MtlArray{Float32,ndims(result),storage}
    return result
end

function test_gemm(storage)
    rng = MersenneTwister(205)
    R, HA, HB, RC, HC, M = 611, 614, 603, 53, 17, 317
    precision = FFM._karatsuba_matmul_product_plan(Float32, R, HA, R, HB, RC, HC)
    k = 2 * precision.middle.max_ops + 1
    @test length(collect(FFM._stripe_ranges(k, precision.middle.max_ops))) == 3
    a = rand(rng, 0:(R*HA-1), 5, k)
    b = rand(rng, 0:(R*HB-1), k, 3)
    A = product_fixture(a, R, HA, M, storage)
    B = product_fixture(b, R, HB, M, storage; layout = :offset)
    C = product_fixture(fill(5, 5, 3), RC, HC, M, storage; layout = :offset)

    independent = mod.(BigInt.(a) * BigInt.(b), RC * HC)
    @test RC * HC != M
    @test (modulus(low_part(A)), modulus(high_part(A))) == (107, 233)
    @test reconstruct(A) == a && reconstruct(B) == b

    # The outer scoped selector would reject native products if inherited.
    with(Metal.matmul_alg => :invalid_test_selector) do
        check_allocating(A, B, mod.(BigInt.(a) * BigInt.(b), R * HA), storage)
        @test Metal.matmul_alg[] === :invalid_test_selector
        foreach(plan_with_sentinel!, (A, B, C))
        before_a, before_b, before_c = snapshot(A), snapshot(B), snapshot(C)
        @test LinearAlgebra.mul!(C, A, B) === C
        @test reconstruct(C) == independent
        @test BigInt.(logical(A.plan)) == mod.(a, R) + div.(a, R)
        @test BigInt.(logical(B.plan)) == mod.(b, R) + div.(b, R)
        @test BigInt.(logical(C.plan)) ==
              mod.(BigInt.(div.(a, R)) * BigInt.(div.(b, R)), RC * HC)
        @test snapshot(A)[1:2] == before_a[1:2]
        @test snapshot(B)[1:2] == before_b[1:2]
        check_padding(A, before_a, :padded)
        check_padding(B, before_b, :offset)
        check_padding(C, before_c, :offset)
        check_allocating(A, B, mod.(BigInt.(a) * BigInt.(b), R * HA), storage)
    end

    vals = [5 15; 16 10]
    square = product_fixture(vals, 6, 3, M, storage; layout = :native)
    square_result = check_allocating(square, square, mod.(BigInt.(vals)^2, 18), storage)
    @test low_part(square_result) isa Metal.MtlMatrix{Float32,storage}
    plan_with_sentinel!(square)
    @test LinearAlgebra.mul!(square_result, square, square) === square_result
    @test reconstruct(square_result) == mod.(BigInt.(vals)^2, 18)
end

function test_gemv(storage)
    a = [4 23 18 6 11; 11 7 18 6 3; 20 13 7 12 17]
    x = [15, 6, 9, 10, 1]
    A = product_fixture(a, 6, 6, 41, storage; layout = :offset)
    X = product_fixture(x, 6, 3, 41, storage)
    y = product_fixture([7, 1, 4], 6, 2, 41, storage; layout = :offset)
    full = BigInt.(a) * BigInt.(x)
    check_allocating(A, X, mod.(full, 36), storage)
    foreach(plan_with_sentinel!, (A, X, y))
    before_a, before_x, before_y = snapshot(A), snapshot(X), snapshot(y)
    with(Metal.matmul_alg => :invalid_test_selector) do
        @test FFM.karatsuba_gemv!(y, A, X) === y
        @test Metal.matmul_alg[] === :invalid_test_selector
    end
    @test reconstruct(y) == mod.(full, 12)
    @test BigInt.(logical(A.plan)) == mod.(a, 6) + div.(a, 6)
    @test BigInt.(logical(X.plan)) == mod.(x, 6) + div.(x, 6)
    @test BigInt.(logical(y.plan)) == BigInt.(div.(a, 6)) * BigInt.(div.(x, 6))
    @test snapshot(A)[1:2] == before_a[1:2]
    @test snapshot(X)[1:2] == before_x[1:2]
    check_padding(A, before_a, :offset)
    check_padding(X, before_x, :padded)
    check_padding(y, before_y, :offset)
    @test LinearAlgebra.mul!(y, A, X) === y
    @test reconstruct(y) == mod.(full, 12)

    # Four products of 2048² reach the exact Float32 endpoint. A fifth term
    # violates the established whole-dot precision budget.
    endpoint = FFM._karatsuba_gemv_precision_plan(Float32, 4, 2049, 1, 2049, 1, 2049, 1)
    @test endpoint.middle_dot_bound == BigInt(1) << 24
    for terms in (4, 5)
        left = product_fixture(fill(2048, 1, terms), 2049, 1, 53, storage)
        right = product_fixture(fill(2048, terms), 2049, 1, 53, storage)
        out = product_fixture([3], 2049, 1, 53, storage)
        foreach(plan_with_sentinel!, (left, right, out))
        if terms == 4
            @test LinearAlgebra.mul!(out, left, right) === out
            @test reconstruct(out) == [mod(BigInt(terms) * BigInt(2048)^2, 2049)]
        else
            before = map(snapshot, (left, right, out))
            @test_throws ArgumentError LinearAlgebra.mul!(out, left, right)
            @test map(snapshot, (left, right, out)) == before
        end
    end

    # This asymmetric dot budget is safe even though squaring the larger
    # exclusive bound would exceed the Float32 stripe budget.
    wide = reshape([299_997, 299_995], 2, 1)
    left = product_fixture(wide, 4, 100_000, 43, storage; layout = :native)
    right = product_fixture([3], 4, 1, 43, storage; layout = :native)
    out = product_fixture([2, 1], 4, 2, 43, storage; layout = :native)
    foreach(plan_with_sentinel!, (left, right, out))
    @test LinearAlgebra.mul!(out, left, right) === out
    @test reconstruct(out) == mod.(BigInt.(wide) * BigInt[3], 8)
end

function test_failures(storage)
    m = 73
    A = product_fixture([1 7; 8 4], 6, 3, m, storage)
    B = product_fixture([2 3; 4 5], 6, 3, m, storage)
    C = product_fixture([6 5; 8 4], 6, 3, m, storage)
    foreach(plan_with_sentinel!, (B, C))
    before = map(snapshot, (A, B, C))
    @test_throws ArgumentError LinearAlgebra.mul!(C, A, B)
    @test map(snapshot, (A, B, C)) == before
    plan_with_sentinel!(A)

    for (failure, bad_C, bad_A, bad_B) in (
        (ArgumentError, A, A, B),
        (ArgumentError, product_fixture(zeros(Int, 2, 2), 4096, 4096, m, storage), A, B),
        (DimensionMismatch, product_fixture(fill(7, 2, 3), 6, 3, m, storage), A, B),
        (FFM.ModulusMismatchError, product_fixture(fill(3, 2, 2), 6, 3, 79, storage), A, B),
        (FFM.ModulusMismatchError, C, A, product_fixture(fill(4, 2, 2), 5, 3, m, storage)),
    )
        for K in (bad_C, bad_A, bad_B)
            K.plan === nothing && plan_with_sentinel!(K)
        end
        old = map(snapshot, (bad_C, bad_A, bad_B))
        @test_throws failure LinearAlgebra.mul!(bad_C, bad_A, bad_B)
        @test map(snapshot, (bad_C, bad_A, bad_B)) == old
    end

    # A plan may have the right storage type and shape while sharing an
    # input backing buffer. Reject before staging either input scratch.
    A.plan = low_part(A)
    @test typeof(A.plan) === typeof(low_part(A))
    old = map(snapshot, (A, B, C))
    @test_throws ArgumentError LinearAlgebra.mul!(C, A, B)
    @test map(snapshot, (A, B, C)) == old

    x = product_fixture([7, 13], 6, 3, m, storage)
    y = product_fixture([3, 7], 6, 4, m, storage)
    plan_with_sentinel!(x)
    plan_with_sentinel!(y)
    A = plan_with_sentinel!(product_fixture([7 12; 15 8], 6, 3, m, storage))
    old = map(snapshot, (A, x, y))
    @test_throws ArgumentError LinearAlgebra.mul!(y, A, x)
    @test map(snapshot, (A, x, y)) == old
    wrong_radix = plan_with_sentinel!(product_fixture([6, 7], 5, 1, m, storage))
    old = map(snapshot, (A, x, wrong_radix))
    @test_throws FFM.ModulusMismatchError FFM.karatsuba_gemv!(wrong_radix, A, x)
    @test map(snapshot, (A, x, wrong_radix)) == old
    host_y = KaratsubaArray(
        zeros(Float32, 2),
        zeros(Float32, 2);
        low_modulus = 6,
        high_modulus = 2,
        modulus = m,
    )
    @test_throws ArgumentError FFM.karatsuba_gemv!(host_y, A, x)
    @test Array(host_y) == zeros(Int, 2)
    # The allocating result inherits the left operand's split.
    indivisible = product_fixture([7 12; 15 8], 6, 4, m, storage)
    @test_throws ArgumentError indivisible * x
    @test indivisible.plan === nothing
    host = KaratsubaArray(
        ones(Float32, 2, 2),
        zeros(Float32, 2, 2);
        low_modulus = 6,
        high_modulus = 3,
        modulus = m,
    )
    @test_throws ArgumentError A * host
    @test_throws ArgumentError host * A
    @test_throws ArgumentError LinearAlgebra.mul!(C, A, host)
    @test_throws ArgumentError LinearAlgebra.mul!(host, A, B)
end

function test_empty(storage)
    left = product_fixture(zeros(Int, 2, 0), 6, 3, 23, storage)
    right = product_fixture(zeros(Int, 0, 4), 6, 3, 23, storage)
    dest = product_fixture(fill(7, 2, 4), 6, 3, 23, storage)
    x = product_fixture(Int[], 6, 3, 23, storage)
    y = product_fixture([7, 10], 6, 3, 23, storage)
    foreach(plan_with_sentinel!, (left, right, dest, x, y))
    before_dest, before_y = snapshot(dest), snapshot(y)
    @test LinearAlgebra.mul!(dest, left, right) === dest
    @test reconstruct(dest) == zeros(Int, 2, 4)
    @test logical(dest.plan) == zeros(Int, 2, 4)
    check_padding(dest, before_dest, :padded)
    @test LinearAlgebra.mul!(y, left, x) === y
    @test reconstruct(y) == zeros(Int, 2)
    @test snapshot(y)[3] == before_y[3]
    check_padding(y, before_y, :padded)
    a = product_fixture(zeros(Int, 0, 2), 6, 3, 23, storage)
    b = product_fixture([4, 5], 6, 3, 23, storage)
    z = product_fixture(Int[], 6, 3, 23, storage)
    foreach(plan_with_sentinel!, (a, b, z))
    old = map(snapshot, (a, b, z))
    @test LinearAlgebra.mul!(z, a, b) === z
    @test size(z) == (0,)
    @test map(snapshot, (a, b, z)) == old
end

function run_tests()
    Metal.allowscalar(false)
    @testset "Metal Karatsuba products" begin
        @test Metal.functional()
        for storage in (Metal.SharedStorage, Metal.PrivateStorage)
            @testset "$storage" begin
                @testset "GEMM planes and stripes" test_gemm(storage)
                @testset "GEMV precision and divisor" test_gemv(storage)
                @testset "Preflight and ownership" test_failures(storage)
                @testset "Empty products" test_empty(storage)
            end
        end
    end
end

end
