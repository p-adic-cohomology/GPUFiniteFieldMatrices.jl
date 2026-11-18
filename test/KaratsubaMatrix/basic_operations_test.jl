using Test
using LinearAlgebra
#using Oscar
using CUDA
#using GPUFiniteFieldMatrices

function test_karatsuba_basic_operations()
    A = GPUFiniteFieldMatrices.CuModMatrix{Float64}([1 2; 3 4]; modulus = 100)
    B = GPUFiniteFieldMatrices.CuModMatrix{Float64}([5 6; 7 8]; modulus = 100)
    a = 9
    AK = to_karatsuba(A; low_modulus = 10, high_modulus = 10, modulus = 100)
    BK = to_karatsuba(B; low_modulus = 10, high_modulus = 10, modulus = 100)
    CK = KaratsubaMatrix([1 0 3; 8 11 2; 2 4 6], [10 3 2; 7 13 7; 1 0 9]; modulus = 100)
    DK = KaratsubaMatrix([9 6 5; 2 17 9; 4 4 23], [5 5 11; 6 4 3; 7 3 19]; modulus = 100)
    C = GPUFiniteFieldMatrices.CuModMatrix(Array(CK); modulus = 100)
    D = GPUFiniteFieldMatrices.CuModMatrix(Array(DK); modulus = 100)

    @testset "Gpu Mod Matrices" begin
        @test A + B == GPUFiniteFieldMatrices.CuModMatrix(Array(AK + BK); modulus = 100)
        @test A - B == GPUFiniteFieldMatrices.CuModMatrix(Array(AK - BK); modulus = 100)
        @test a*A == GPUFiniteFieldMatrices.CuModMatrix(Array(a*AK); modulus = 100)
        @test A*B == GPUFiniteFieldMatrices.CuModMatrix(Array(AK*BK); modulus = 100)
        #=
        @test C + D == GPUFiniteFieldMatrices.CuModMatrix(Array(CK + DK); modulus = 100)
        @test C - D == GPUFiniteFieldMatrices.CuModMatrix(Array(CK - DK); modulus = 100)
        @test a*C == GPUFiniteFieldMatrices.CuModMatrix(Array(a*CK); modulus = 100)
        @test C*D == GPUFiniteFieldMatrices.CuModMatrix(Array(CK*DK); modulus = 100)
        =#
    end
end

function _karatsuba_operations_diagnostic(iteration, phase)
    peak_host_rss_mib = Sys.maxrss() / 2^20
    println(
        stderr,
        "Karatsuba iteration=$iteration phase=$phase pid=$(getpid()) peak_host_rss_mib=$peak_host_rss_mib",
    )
    flush(stderr)
end

function test_karatsuba_operations()
    n = 500
    N1 = 13^4
    N2 = 13^3

    @testset "Karatsuba Operations" begin

        for i = 1:100
            _karatsuba_operations_diagnostic(i, "allocating")

            A1_data = Base.rand(0:(N1-1), n, n)
            A2_data = Base.rand(0:(N2-1), n, n)
            x1_data = Base.rand(0:(N1-1), n)
            x2_data = Base.rand(0:(N2-1), n)
            y1_data = Base.rand(0:(N1-1), n)
            y2_data = Base.rand(0:(N2-1), n)

            A1 = GPUFiniteFieldMatrices.CuModMatrix{Float64}(A1_data; modulus = N1)
            A2 = GPUFiniteFieldMatrices.CuModMatrix{Float64}(A2_data; modulus = N1)
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
            GPUFiniteFieldMatrices.initialize_plan!(xK)
            GPUFiniteFieldMatrices.initialize_plan!(yK)

            _karatsuba_operations_diagnostic(i, "CPU reconstruction")
            ACpu = Array(AK)
            xCpu = Array(xK)

            _karatsuba_operations_diagnostic(i, "GEMV")
            CUDA.@sync GPUFiniteFieldMatrices.karatsuba_gemv!(yK, AK, xK)

            _karatsuba_operations_diagnostic(i, "download")
            actual = Array(yK)

            _karatsuba_operations_diagnostic(i, "CPU oracle")
            expected = mod.((BigInt.(ACpu)*BigInt.(xCpu)), (N1*N2))

            _karatsuba_operations_diagnostic(i, "comparison")
            @test actual == expected
            _karatsuba_operations_diagnostic(i, "finished")
        end
    end
end
