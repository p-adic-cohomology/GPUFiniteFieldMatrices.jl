@testset "Exact stripe planning" begin
    G = GPUFiniteFieldMatrices

    for (T, bits) in ((Float16, 11), (Float32, 24), (Float64, 53))
        exact_max = big(2)^bits - 1
        input_bound = 17
        term = big(input_bound - 1)^2
        for reduction_modulus in (17, exact_max - 2term + 1, exact_max - term + 1)
            safe = min(exact_max ÷ term - 1, (exact_max - reduction_modulus + 1) ÷ term)
            @test G.find_max_stripe_ops(T, input_bound; modulus = reduction_modulus) == safe
            @test G._stripe_safe_chunk(T, reduction_modulus, input_bound, nothing) == safe
            @test G._stripe_safe_chunk(T, reduction_modulus, input_bound, big(safe)) == safe
            @test_throws ArgumentError G._stripe_safe_chunk(
                T,
                reduction_modulus,
                input_bound,
                safe + 1,
            )
        end
    end

    @test G.find_max_stripe_ops(Float32, 8192) == 0
    @test G._stripe_safe_chunk(Float32, 8192, nothing, nothing; empty_inner = true) == 1
    @test_throws ArgumentError G._stripe_safe_chunk(Float32, 8192, nothing, nothing)
    @test_throws ArgumentError G._stripe_safe_chunk(
        Float32,
        8192,
        nothing,
        1;
        empty_inner = true,
    )
    for bound in (1, 0, -1, 2.5)
        @test_throws ArgumentError G._stripe_safe_chunk(
            Float32,
            8192,
            bound,
            nothing;
            empty_inner = true,
        )
    end
    for max_ops in (0, -1, 2.5, big(typemax(Int)) + 1)
        @test_throws ArgumentError G._stripe_safe_chunk(Float32, 127, nothing, max_ops)
    end
    @test_throws ArgumentError G._stripe_safe_chunk(
        Float32,
        0,
        8192,
        nothing;
        empty_inner = true,
    )

    for (k, expected) in
        ((0, []), (1, [1:1]), (31, [1:31]), (32, [1:32]), (33, [1:32, 33:33]))
        @test collect(G._stripe_ranges(k, 32)) == expected
    end
    @test collect(G._stripe_ranges(typemax(Int), typemax(Int) - 2)) ==
          [1:(typemax(Int)-2), (typemax(Int)-1):typemax(Int)]
end

@testset "Stripe product preflight and empty results" begin
    G = GPUFiniteFieldMatrices
    @test G._stripe_output_size(zeros(2, 0), zeros(0, 3)) == (2, 3)
    @test G._stripe_output_size(zeros(2, 0), Float64[]) == (2,)
    @test_throws DimensionMismatch G._stripe_output_size(zeros(2, 3), zeros(4, 1))
    @test_throws DimensionMismatch G._stripe_output_size(zeros(2, 3), zeros(4))

    C = fill(7, 2, 3)
    z = fill(7, 2)
    @test G._stripe_run!(C, zeros(Int, 2, 0), zeros(Int, 0, 3), 8192, 1) === C
    @test G._stripe_run!(z, zeros(Int, 2, 0), Int[], 8192, 1) === z
    @test C == zeros(Int, 2, 3)
    @test z == zeros(Int, 2)

    @test G._stripe_require_same_device(0, 0, 0) === nothing
    @test G._stripe_require_same_device(1, 1, 1) === nothing
    for devices in ((0, 1, 0), (0, 0, 1), (0, 1, 1))
        @test_throws ArgumentError G._stripe_require_same_device(devices...)
    end
    @test G._stripe_mightalias(view(C, 1:1, :), view(C, 2:2, :))
    @test !G._stripe_mightalias(C, copy(C))

    host = fill(7.0f0, 2, 2)
    cpu = G.GPUModArray{Float32,2,Matrix{Float32}}(host, (2, 2), 127)
    @test_throws ArgumentError G.stripe_mul!(cpu, cpu, cpu)
    @test_throws ArgumentError G.stripe_mul(cpu, cpu)
    @test host == fill(7.0f0, 2, 2)

    values = Float32[7, 11]
    resized = G.GPUModArray{Float32,1,Vector{Float32}}(values, (2,), 127)
    resize!(values, 1)
    @test_throws DimensionMismatch G._stripe_logical_data(resized)
end
