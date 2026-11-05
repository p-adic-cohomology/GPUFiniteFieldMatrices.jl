function test_karatsuba_raw_unary()
    G = GPUFiniteFieldMatrices
    @testset "Raw CuArray Karatsuba unary minus" begin
        for T in (Float32, Float64), dims in ((4,), (2, 2)), planned in (false, true)
            low_values = reshape(T[0, 1, 9, 3], dims)
            high_values = reshape(T[1, 0, 2, 8], dims)
            A = G.KaratsubaArray(
                CUDA.CuArray(low_values),
                CUDA.CuArray(high_values);
                low_modulus = 10,
                high_modulus = 11,
                modulus = 17,
            )
            planned && G.initialize_plan!(A)
            original_low, original_high, original_plan =
                G.low_part(A), G.high_part(A), A.plan
            if planned
                fill!(parent(original_plan), T(7))
            end
            plan_before = planned ? Array(parent(original_plan)) : nothing
            source_device = CUDA.device(original_low)
            devices = collect(CUDA.devices())
            caller_device =
                length(devices) > 1 ? only(filter(!=(source_device), devices)[1:1]) :
                source_device
            negative = CUDA.device!(caller_device) do
                result = -A
                @test CUDA.device() == caller_device
                result
            end
            expected = mod.(-(BigInt.(low_values) .+ 10 .* BigInt.(high_values)), 110)
            @test Array(negative) == expected
            @test negative isa typeof(A)
            @test size(negative) == dims
            @test (
                G.low_modulus(negative),
                G.high_modulus(negative),
                G.modulus(negative),
            ) == (10, 11, 17)
            @test G.low_part(negative) isa CUDA.CuArray
            @test G.high_part(negative) isa CUDA.CuArray
            @test G.low_part(negative) !== original_low
            @test G.high_part(negative) !== original_high
            @test CUDA.device(G.low_part(negative)) == source_device
            @test CUDA.device(G.high_part(negative)) == source_device
            @test (negative.plan !== nothing) == planned
            @test G.low_part(A) === original_low && G.high_part(A) === original_high
            @test A.plan === original_plan
            @test Array(original_low) == low_values
            @test Array(original_high) == high_values
            if planned
                @test negative.plan !== original_plan
                @test CUDA.device(parent(negative.plan)) == source_device
                @test Array(parent(original_plan)) == plan_before
            end
        end
    end
end
