@testset "Consistent exceptions on CPU" begin
    G = GPUFiniteFieldMatrices
    old_types = (
        :CuModArraySizeMismatchException,
        :CuModArrayModulusMismatchException,
        :CuModMatrixTooLargeException,
        :CuModMatrixNotSquareException,
        :CuModMatrixModulusNotPrimeException,
        :InverseOverflowError,
        :InverseNotDefinedException,
    )
    @test all(name -> !isdefined(G, name), old_types)
    @test sprint(showerror, G.ModulusMismatchError("different moduli")) ==
          "different moduli"
    @test_throws LinearAlgebra.SingularException G.mod_inv(2, 4)
    @test_throws LinearAlgebra.SingularException G.pluq_mod_inv(0, 101)
    @test_throws LinearAlgebra.SingularException G.pluq_mod_inv(2, 4)
    @test G.mod_inv(3, 7) == G.pluq_mod_inv(3, 7) == 5
    @test_throws DimensionMismatch G.KaratsubaMatrix(zeros(2, 2), zeros(2, 3); modulus = 3)
    @test_throws ArgumentError G.KaratsubaMatrix(zeros(2, 2), zeros(2, 2); modulus = 2^52)
    @test_throws OverflowError G._karatsuba_checked_sum_bound(typemax(Int), 1, "sum")
    @test_throws DimensionMismatch G._stripe_mul(zeros(2, 3), zeros(2, 2); modulus = 101)
    @test_throws OverflowError G._stripe_mul(zeros(2, 2), zeros(2, 2); modulus = 2^52)
end
