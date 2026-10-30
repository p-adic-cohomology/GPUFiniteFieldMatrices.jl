using Test
using LinearAlgebra
#using Oscar
using CUDA
#using GPUFiniteFieldMatrices

function test_basic_operations()
    #=
    A = [1 2; 3 4]
    B = [5 6; 7 8]
    a = 9
    AK = to_karatsuba(A)
    BK = to_karatsuba(B)
    CK = KaratsubaMatrix([1 0 3; 8 11 2; 2 4 6], [10 3 2; 7 13 7; 1 0 9])
    DK = KaratsubaMatrix([9 6 5; 2 17 9; 4 4 23], [5 5 11; 6 4 3; 7 3 19])
    C = Array(CK)
    D = Array(DK)

    @testset "Cpu Matrices" begin
        @test A + B == Array(AK + BK)
        @test A - B == Array(AK - BK)
        @test a*A == Array(a*AK)
        #@test A*B == Array(AK*BK)
        @test C + D == Array(CK + DK)
        @test C - D == Array(CK - DK)
        @test a*C == Array(a*CK)
        #@test C*D == Array(CK*DK)
    end

    A = CuArray([1 2; 3 4])
    B = CuArray([5 6; 7 8])
    a = 9
    AK = to_karatsuba(A; modulus = 100)
    BK = to_karatsuba(B; modulus = 100)
    CK = KaratsubaMatrix([1 0 3; 8 11 2; 2 4 6], [10 3 2; 7 13 7; 1 0 9]; modulus = 100)
    DK = KaratsubaMatrix([9 6 5; 2 17 9; 4 4 23], [5 5 11; 6 4 3; 7 3 19]; modulus = 100)
    C = CuArray(Array(CK))
    D = CuArray(Array(DK))

    @testset "Gpu Matrices" begin
        @test A + B == CuArray(Array(AK + BK))
        @test A - B == CuArray(Array(AK - BK))
        @test a*A == CuArray(Array(a*AK))
        #@test A*B == CuArray(Array(AK*BK))
        @test C + D == CuArray(Array(CK + DK))
        @test C - D == CuArray(Array(CK - DK))
        @test a*C == CuArray(Array(a*CK))
        #@test C*D == CuArray(Array(CK*DK))
    end
    =#
    A = GPUFiniteFieldMatrices.CuModMatrix([1 2; 3 4]; modulus = 100)
    B = GPUFiniteFieldMatrices.CuModMatrix([5 6; 7 8]; modulus = 100)
    a = 9
    AK = to_karatsuba(A; modulus = 100)
    BK = to_karatsuba(B; modulus = 100)
    CK = KaratsubaMatrix([1 0 3; 8 11 2; 2 4 6], [10 3 2; 7 13 7; 1 0 9]; modulus = 100)
    DK = KaratsubaMatrix([9 6 5; 2 17 9; 4 4 23], [5 5 11; 6 4 3; 7 3 19]; modulus = 100)
    C = GPUFiniteFieldMatrices.CuModMatrix(Array(CK); modulus = 100^2)
    D = GPUFiniteFieldMatrices.CuModMatrix(Array(DK); modulus = 100^2)

    @testset "Gpu Mod Matrices" begin
        @test A + B == GPUFiniteFieldMatrices.CuModMatrix(Array(AK + BK); modulus = 100^2)
        @test A - B == GPUFiniteFieldMatrices.CuModMatrix(Array(AK - BK); modulus = 100^2)
        @test a*A == GPUFiniteFieldMatrices.CuModMatrix(Array(a*AK); modulus = 100^2)
        @test A*B == GPUFiniteFieldMatrices.CuModMatrix(Array(AK*BK); modulus = 100^2)
        #=
        @test C + D == GPUFiniteFieldMatrices.CuModMatrix(Array(CK + DK); modulus = 100^2)
        @test C - D == GPUFiniteFieldMatrices.CuModMatrix(Array(CK - DK); modulus = 100^2)
        @test a*C == GPUFiniteFieldMatrices.CuModMatrix(Array(a*CK); modulus = 100^2)
        @test C*D == GPUFiniteFieldMatrices.CuModMatrix(Array(CK*DK); modulus = 100^2)
        =#
    end
end

function test_karatsuba_operations()
    n = 500
    N1 = 13^4
    N2 = 13^3

    @testset "Karatsuba Operations" begin

        for i = 1:100
            A1_data = Base.rand(0:(N1-1), n, n)
            A2_data = Base.rand(0:(N2-1), n, n)
            B1_data = Base.rand(0:(N1-1), n, n)
            B2_data = Base.rand(0:(N2-1), n, n)
            C1_data = Base.rand(0:(N1-1), n, n)
            C2_data = Base.rand(0:(N2-1), n, n)
            x1_data = Base.rand(0:(N1-1), n)
            x2_data = Base.rand(0:(N2-1), n)
            y1_data = Base.rand(0:(N1-1), n)
            y2_data = Base.rand(0:(N2-1), n)
            a = Base.rand(0:100)

            A1 = GPUFiniteFieldMatrices.CuModMatrix{Float64}(A1_data; modulus = N1)
            A2 = GPUFiniteFieldMatrices.CuModMatrix{Float64}(A2_data; modulus = N1)
            B1 = GPUFiniteFieldMatrices.CuModMatrix{Float64}(B1_data; modulus = N1)
            B2 = GPUFiniteFieldMatrices.CuModMatrix{Float64}(B2_data; modulus = N1)
            C1 = GPUFiniteFieldMatrices.CuModMatrix{Float64}(C1_data; modulus = N1)
            C2 = GPUFiniteFieldMatrices.CuModMatrix{Float64}(C2_data; modulus = N1)
            x1 = GPUFiniteFieldMatrices.CuModVector{Float64}(x1_data; modulus = N1)
            x2 = GPUFiniteFieldMatrices.CuModVector{Float64}(x2_data; modulus = N1)
            y1 = GPUFiniteFieldMatrices.CuModVector{Float64}(y1_data; modulus = N1)
            y2 = GPUFiniteFieldMatrices.CuModVector{Float64}(y2_data; modulus = N1)

            AK = GPUFiniteFieldMatrices.KaratsubaMatrix(
                A1,
                A2;
                low_modulus = N1,
                high_modulus = N2,
                modulus = N1 * N2,
            )
            BK = GPUFiniteFieldMatrices.KaratsubaMatrix(
                B1,
                B2;
                low_modulus = N1,
                high_modulus = N2,
                modulus = N1 * N2,
            )
            CK = GPUFiniteFieldMatrices.KaratsubaMatrix(
                C1,
                C2;
                low_modulus = N1,
                high_modulus = N2,
                modulus = N1 * N2,
            )
            xK = GPUFiniteFieldMatrices.KaratsubaVector(
                x1,
                x2;
                low_modulus = N1,
                high_modulus = N2,
                modulus = N1 * N2,
            )
            yK = GPUFiniteFieldMatrices.KaratsubaVector(
                y1,
                y2;
                low_modulus = N1,
                high_modulus = N2,
                modulus = N1 * N2,
            )

            GPUFiniteFieldMatrices.initialize_plan!(AK)
            GPUFiniteFieldMatrices.initialize_plan!(BK)
            GPUFiniteFieldMatrices.initialize_plan!(CK)
            GPUFiniteFieldMatrices.initialize_plan!(xK)
            GPUFiniteFieldMatrices.initialize_plan!(yK)

            ACpu = Array(AK)
            BCpu = Array(BK)
            CCpu = Array(CK)
            xCpu = Array(xK)
            yCpu = Array(yK)

            #@testset "Karatsuba Operations" begin
            #@test Array(GPUFiniteFieldMatrices.add!(CK,AK,BK)) == mod.((ACpu + BCpu),(N1*N2))
            #@test Array(GPUFiniteFieldMatrices.sub!(CK,AK,BK)) == mod.((ACpu - BCpu),(N1*N2))
            @test Array(LinearAlgebra.mul!(yK, AK, xK)) == mod.((ACpu*xCpu), (N1*N2))
            #@test Array(GPUFiniteFieldMatrices.karatsuba_gemv!(yK,AK,xK)) == mod.((ACpu*xCpu),(N1*N2))
            #@test Array(GPUFiniteFieldMatrices.scalar_multiply!(CK,AK,a)) == mod.((ACpu*a),(N1*N2))

            #end
        end
    end
end
