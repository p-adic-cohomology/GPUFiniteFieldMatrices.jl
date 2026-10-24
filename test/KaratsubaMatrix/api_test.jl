function test_karatsuba_public_api_values()
    @testset "Karatsuba public API values" begin
        mod = GPUFiniteFieldMatrices
        modulus = 100
        a_values = Float64[20 30; 40 50]
        b_values = Float64[20 30; 40 50]
        a = mod.CuModMatrix{Float64}(a_values, modulus)
        b = mod.CuModMatrix{Float64}(b_values, modulus)

        explicit = mod.to_karatsuba(a, modulus)
        automatic = mod.to_karatsuba(a)
        @test Array(mod.from_karatsuba(Float64, explicit)) == a_values
        @test Array(mod.from_karatsuba(Float64, automatic)) == a_values

        ak = mod.to_karatsuba(a, modulus)
        bk = mod.to_karatsuba(b, modulus)
        ck = mod.to_karatsuba(
            mod.CuModMatrix{Float64}(zeros(Float64, 2, 2), modulus),
            modulus,
        )
        foreach(mod.initialize_plan!, (ak, bk, ck))

        LinearAlgebra.mul!(ck, ak, bk)
        @test Array(ck) == Base.mod.(a_values * b_values, modulus^2)
    end
end
