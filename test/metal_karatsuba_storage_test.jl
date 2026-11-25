function _test_metal_karatsuba_factories(::Type{Mode}) where {Mode}
    G = GPUFiniteFieldMatrices
    device = Metal.device()
    opts = (low_modulus = 17, high_modulus = 19, modulus = 23)
    Z = @inferred zeros(
        G.KaratsubaMatrix{Float32,G.MtlModArray{Float32,2,Mode}},
        3,
        35;
        opts...,
    )
    @test size(Z) == (3, 35)
    @test size(parent(low_part(Z))) == (32, 64)
    @test Metal.storagemode(parent(low_part(Z))) === Mode
    @test Metal.device(parent(low_part(Z))) == device
    @test Array(Z) == zeros(Int, 3, 35)
    @test Z.plan === nothing
    @test (@inferred G.initialize_plan!(Z)) === Z
    @test typeof(Z.plan) === typeof(low_part(Z))
    @test parent(Z.plan) !== parent(low_part(Z))
    @test G.modulus(Z.plan) == opts.low_modulus
    @test all(iszero, Array(parent(Z.plan)))

    raw = @inferred zeros(
        G.KaratsubaVector{Float32,Metal.MtlArray{Float32,1,Mode}},
        35;
        opts...,
    )
    @test low_part(raw) isa Metal.MtlArray{Float32,1,Mode}
    @test size(low_part(raw)) == (35,)
    @test G.initialize_plan!(raw).plan isa G.MtlModArray{Float32,1,Mode}

    selected = G.with_backend(:metal) do
        zeros(G.KaratsubaMatrix{Float32}, 3, 0; storage = Mode, opts...)
    end
    @test low_part(selected) isa G.MtlModArray{Float32,2,Mode}
    @test size(G.initialize_plan!(selected).plan) == (3, 0)
    @test size(from_karatsuba(Int64, selected)) == (3, 0)
    @test G.negate!(selected, selected) === selected
    @test G.add!(selected, selected, selected) === selected

    @test_throws ArgumentError zeros(
        G.KaratsubaVector{Float64},
        35;
        backend = :metal,
        opts...,
    )
    @test_throws ArgumentError zeros(
        G.KaratsubaVector{Float64,G.MtlModArray{Float64,1,Mode}},
        35;
        opts...,
    )
    @test_throws DomainError zeros(
        G.KaratsubaVector{Float32,G.MtlModArray{Float32,1,Mode}},
        35;
        low_modulus = 17,
        high_modulus = 2^24 + 1,
        modulus = 23,
    )
end

function _test_metal_karatsuba_conversion(::Type{Mode}) where {Mode}
    G = GPUFiniteFieldMatrices
    host = Float32[-11 40 122; -9 15 2^24]
    native = Metal.MtlArray{Float32,2,Mode}(host)
    A = G.MtlModArray{Float32,2}(native; modulus = 27, mod = false)
    opts = (low_modulus = 17, high_modulus = 19, modulus = 23)
    K = @inferred to_karatsuba(Float32, A; opts...)
    expected = BigInt.(host)
    @test Array(K) == expected
    @test (low_modulus(K), high_modulus(K), G.modulus(K)) == (17, 19, 23)
    @test Array(low_part(K)) == mod.(expected, 17)
    @test Array(high_part(K)) == fld.(expected, 17)
    @test maximum(Array(high_part(K))) > high_modulus(K)
    @test parent(low_part(K)) !== native
    @test Metal.storagemode(parent(low_part(K))) === Mode
    @test Array(native) == host
    result = @inferred from_karatsuba(Int64, K)
    @test result isa Metal.MtlArray{Int64,2,Mode}
    @test Metal.device(result) == Metal.device(native)
    @test Array(result) == expected
    @test Array(from_karatsuba(Float32, K)) == host
    @test_throws ArgumentError to_karatsuba(Float64, A; opts...)
    @test_throws ArgumentError from_karatsuba(Float64, K)

    bad = Metal.MtlArray{Float32,1,Mode}(Float32[1.5, NaN])
    @test_throws InexactError to_karatsuba(bad; opts...)
    big = Float32[2.0f0^80, -2.0f0^80, 2.0f0^64, -2.0f0^64]
    wide = Metal.MtlArray{Float32,1,Mode}(big)
    W = to_karatsuba(wide; low_modulus = 2, high_modulus = 7, modulus = 23)
    @test Array(W) == BigInt.(big)
    @test Array(from_karatsuba(Float32, W)) == big

    inexact = KaratsubaVector(
        Metal.MtlArray{Float32,1,Mode}(Float32[1]),
        Metal.MtlArray{Float32,1,Mode}(Float32[2^24]);
        low_modulus = 2,
        high_modulus = 7,
        modulus = 23,
    )
    @test_throws InexactError from_karatsuba(Float32, inexact)
    @test Array(from_karatsuba(Int64, inexact)) == [2^25 + 1]
end

function _test_metal_karatsuba_elementwise(::Type{Mode}) where {Mode}
    G = GPUFiniteFieldMatrices
    dims = (3, 35)
    R, H, M = 2^24, 2^24 - 3, 23
    values = reshape(Float32.(mod.(1:prod(dims), 37)), dims)
    low = values .+ Float32(R - 20)
    high = values .+ Float32(H - 20)
    # Use a sentinel beyond the logical rectangle to detect backing writes.
    backing(v, N) = begin
        padded = fill(91.0f0, (5, 38))
        padded[1:dims[1], 1:dims[2]] .= v
        native = Metal.MtlArray{Float32,2,Mode}(padded)
        G.MtlModArray{Float32,2}(native; modulus = N, new_size = dims, mod = false)
    end
    wrap(a, b; modulus = M) =
        KaratsubaMatrix(a, b; low_modulus = R, high_modulus = H, modulus)
    A = wrap(backing(low, R), backing(high, H))
    B = wrap(backing(reverse(low; dims = 2), R), backing(reverse(high; dims = 2), H))
    @test low_part(A) isa G.MtlModArray{Float32,2,Mode}
    C = wrap(backing(zero(low), R), backing(zero(high), H))
    c_low, c_high = low_part(C), high_part(C)
    @test G.initialize_plan!(C) === C
    plan = C.plan
    fill!(parent(plan), 31.0f0)
    a, b = Array(A), Array(B)
    capacity = BigInt(R) * H
    scalar = -(BigInt(1) << 140) + 34
    pivot = (1, 1)
    @test G.add!(C, A, B) === C
    @test Array(C) == mod.(a .+ b, capacity)
    @test G.sub!(C, A, B) === C
    @test Array(C) == mod.(a .- b, capacity)
    @test G.scalar_multiply!(C, B, scalar) === C
    @test Array(C) == mod.(b .* scalar, capacity)
    @test G.negate!(C, A) === C
    @test Array(C) == mod.(-a, capacity)
    @test low_part(C) === c_low && high_part(C) === c_high
    @test C.plan === plan
    @test all(==(31.0f0), Array(parent(plan)))
    for component in (c_low, c_high)
        data = Array(parent(component))
        @test all(==(91.0f0), data[(dims[1]+1):end, :])
        @test all(==(91.0f0), data[:, (dims[2]+1):end])
    end

    @test Array(A + B) == mod.(a .+ b, capacity)
    @test Array(A - B) == mod.(a .- b, capacity)
    @test Array(A * scalar) == mod.(a .* scalar, capacity)
    @test (A + B).plan === nothing
    negated = -C
    @test typeof(negated) === typeof(C)
    @test negated.plan !== plan
    @test all(iszero, Array(parent(negated.plan)))
    @test Array(negated) == mod.(a, capacity)
    @test Metal.storagemode(parent(low_part(negated))) === Mode

    original = Array(C)
    wrong_modulus = wrap(low_part(B), high_part(B); modulus = M + 1)
    @test_throws G.ModulusMismatchError G.add!(C, A, wrong_modulus)
    @test_throws ArgumentError G.scalar_multiply!(C, A, 1.25)
    bad = copy(high)
    bad[pivot...] = 1.5f0
    @test_throws ArgumentError G.negate!(C, wrap(low_part(A), backing(bad, H)))
    @test Array(C) == original
    @test G.add!(C, C, A) === C
    @test Array(C) == mod.(original .+ a, capacity)
    @test copy!(A, C) === A
    @test Array(A) == Array(C)
    @test A.plan !== C.plan
    @test all(iszero, Array(parent(A.plan)))
    @test G.zero!(C) === C
    @test all(iszero, Array(C))

    # An overlapping destination view must fail before either plane is written.
    native = parent(low_part(A))
    overlapping = G.MtlModArray{Float32,2}(
        view(native, 2:4, 1:dims[2]);
        modulus = R,
        new_size = dims,
        mod = false,
    )
    before = Array(native)
    high_native = parent(backing(zero(high), H))
    high_view = G.MtlModArray{Float32,2}(
        view(high_native, 2:4, 1:dims[2]);
        modulus = H,
        new_size = dims,
        mod = false,
    )
    overlap = wrap(overlapping, high_view)
    @test G.initialize_plan!(overlap).plan isa G.MtlModArray{Float32,2,Mode}
    @test !Base.mightalias(parent(overlap.plan), native)
    @test_throws ArgumentError G.add!(overlap, A, B)
    @test Array(native) == before

    raw_low = Metal.MtlArray{Float32,1,Mode}(Float32[-7, R+2, 0, 9])
    raw_high = Metal.MtlArray{Float32,1,Mode}(Float32[0, H+1, -3, 7])
    raw = KaratsubaVector(raw_low, raw_high; low_modulus = R, high_modulus = H, modulus = M)
    @test Array(-raw) == mod.(-Array(raw), capacity)
    @test low_part(-raw) isa Metal.MtlArray{Float32,1,Mode}
end

@testset "Metal Karatsuba storage and elementwise" begin
    @test Metal.functional()
    Metal.allowscalar(false)
    try
        for mode in (Metal.SharedStorage, Metal.PrivateStorage)
            @testset "$(nameof(mode))" begin
                _test_metal_karatsuba_factories(mode)
                _test_metal_karatsuba_conversion(mode)
                _test_metal_karatsuba_elementwise(mode)
            end
        end
    finally
        Metal.allowscalar(true)
    end
end
