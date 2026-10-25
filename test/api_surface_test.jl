@testset "CPU API surface and dispatch" begin
    mod = GPUFiniteFieldMatrices
    matrix_type = mod.CuModMatrix{Float32}
    karatsuba_type = mod.KaratsubaArray{Int,2}

    exported = (
        :CuModArray,
        :CuModMatrix,
        :CuModVector,
        :KaratsubaArray,
        :KaratsubaMatrix,
        :KaratsubaVector,
        :to_karatsuba,
        :from_karatsuba,
        :PLUQOptions,
        :PLUQ,
        :pluq,
        :pluq!,
        :pluq_batched,
        :inverse_batched,
        :is_invertible,
        :is_invertible_with_inverse,
        :right_inverse,
        :left_inverse,
        :change_modulus,
        :eye,
        :elementwise_multiply!,
        :scalar_add!,
        :scalar_sub!,
        :mod_elements!,
    )
    @test all(name -> Base.isexported(mod, name), exported)
    module_names = filter(!=(nameof(mod)), names(mod; all = true, imported = false))
    @test Set(filter(name -> Base.isexported(mod, name), module_names)) == Set(exported)

    public_names = (
        :add!,
        :sub!,
        :neg!,
        :zero!,
        :zeros,
        :rand,
        :similar_with_modulus,
        :change_modulus_no_alloc!,
        :mod_inv,
        :karatsuba_gemv!,
        :inverse_pluq,
        :pluq_batched_4x4!,
        :pluq_batched_8x8!,
        :pluq_batched_16x16!,
        :pluq_batched_32x32!,
        :inverse_batched_4x4!,
        :inverse_batched_8x8!,
        :inverse_batched_16x16!,
        :inverse_batched_32x32!,
        :upper_triangular_inverse_no_copy,
        :lower_triangular_inverse_no_copy,
        :forward_sub_gpu_type_32,
        :backward_sub_gpu_type_32,
        :mat_mul_gpu_type,
        :mat_mul_type_inplace!,
        :stripe_mul!,
        :TILE_WIDTH,
    )
    @test all(name -> isdefined(mod, name), public_names)
    @test all(name -> Base.ispublic(mod, name), public_names)
    @test all(name -> !Base.isexported(mod, name), public_names)
    expected_public = union(Set(exported), Set(public_names))
    @test Set(filter(name -> Base.ispublic(mod, name), module_names)) == expected_public
    @test !Base.ispublic(mod, :pluq_check_identity)
    @test !Base.isexported(mod, :pluq_check_identity)

    removed = (
        :CuModcopy,
        :mulN!,
        :inverse,
        :inverse_new,
        :is_invertible_new,
        :inverse_pluq_new,
        :right_inverse_new,
        :left_inverse_new,
        :pluq_new,
        :pluq_new!,
        :pluq_new_batch,
        :inverse_new_batch,
        :PLUQFactorization,
        :pluq_extract_L,
        :pluq_extract_U,
        :KMatMul!,
        :KMatMul_gemv!,
        :Karatsubacopy,
        :MatToKMat,
        :KMatToMat,
    )
    @test all(name -> !isdefined(mod, name), removed)

    @test which(mod.CuModMatrix, Tuple{Matrix{Int},Int}).module === mod
    @test which(mod.CuModMatrix{Float64}, Tuple{Matrix{Int},Int}).module === mod
    @test which(Base.copy, Tuple{matrix_type}).module === mod
    @test which(Base.inv, Tuple{matrix_type}).module === mod
    @test which(mod.pluq, Tuple{matrix_type}).module === mod
    @test which(mod.is_invertible, Tuple{matrix_type}).module === mod
    @test which(getproperty, Tuple{mod.PLUQ{Float32},Symbol}).module === mod
    @test which(propertynames, Tuple{mod.PLUQ{Float32}}).module === mod
    @test which(mod.similar_with_modulus, Tuple{matrix_type,Int}).module === mod
    @test which(Base.similar, Tuple{matrix_type,Int}).module !== mod
    @test which(LinearAlgebra.mul!, Tuple{matrix_type,matrix_type,matrix_type}).module ===
          mod
    # The package's mul! overload is matrix-only; vectors use karatsuba_gemv!.
    @test which(LinearAlgebra.mul!, Tuple{karatsuba_type,karatsuba_type,karatsuba_type}).module ===
          mod
    @test which(Base.copy, Tuple{karatsuba_type}).module === mod
    @test which(
        mod.karatsuba_gemv!,
        Tuple{karatsuba_type,karatsuba_type,karatsuba_type},
    ).module === mod
    @test which(mod.to_karatsuba, Tuple{Matrix{Int},Int}).module === mod
    @test which(mod.to_karatsuba, Tuple{Type{Int},Matrix{Int},Int}).module === mod
    @test which(mod.from_karatsuba, Tuple{Type{Int},karatsuba_type}).module === mod
    @test which(LinearAlgebra.rmul!, Tuple{matrix_type,Float64}).module === mod
    @test which(LinearAlgebra.lmul!, Tuple{Float64,matrix_type}).module === mod

    # Unsupported elem_type keywords fail in keyword dispatch before the
    # constructor body can allocate a GPU array.
    @test_throws MethodError mod.CuModMatrix(Float32[1 2; 3 4], 101; elem_type = Float32)

    K = karatsuba_type([1 2; 3 4], [5 6; 7 8], 3, 5, 15)
    Kcopy = copy(K)
    @test Kcopy !== K
    @test Kcopy.data1 == K.data1
    @test Kcopy.data2 == K.data2
    @test Kcopy.data1 !== K.data1

    # Characterize the documented legacy changed-M behavior; this does not
    # assert metadata preservation or repair Base.copy.
    legacy_data1 = [1 2; 3 4]
    legacy_data2 = [5 6; 7 8]
    expected_legacy_data1 = copy(legacy_data1)
    expected_legacy_data2 = copy(legacy_data2)
    legacy_source = karatsuba_type(legacy_data1, legacy_data2, 3, 3, 3)
    legacy_copy = copy(legacy_source)
    @test legacy_copy.M == 9
    @test legacy_copy.M != legacy_source.M
    @test legacy_source.M == 3
    @test legacy_source.data1 == expected_legacy_data1
    @test legacy_source.data2 == expected_legacy_data2
    @test legacy_copy.data1 == expected_legacy_data1
    @test legacy_copy.data2 == expected_legacy_data2
    @test legacy_copy.data1 !== legacy_source.data1
    @test legacy_copy.data2 !== legacy_source.data2
    legacy_source.data1[1, 1] = -1
    legacy_source.data2[1, 1] = -1
    @test legacy_copy.data1 == expected_legacy_data1
    @test legacy_copy.data2 == expected_legacy_data2

    # The explicit mul! checks run before CUDA work and can be checked on CPU.
    assert_guard_error_message = function (operation, expected_message)
        err = try
            operation()
            nothing
        catch caught
            caught
        end
        @test err isa ErrorException
        if err isa ErrorException
            @test sprint(showerror, err) == expected_message
        end
    end
    mul_left =
        mod.KaratsubaArray{Float64,2}(zeros(Float64, 2, 2), zeros(Float64, 2, 2), 3, 5, 15)
    mul_right_wrong_modulus =
        mod.KaratsubaArray{Float64,2}(zeros(Float64, 2, 2), zeros(Float64, 2, 2), 3, 5, 16)
    mul_dest =
        mod.KaratsubaArray{Float64,2}(zeros(Float64, 2, 2), zeros(Float64, 2, 2), 3, 5, 15)
    assert_guard_error_message(
        () -> LinearAlgebra.mul!(mul_dest, mul_left, mul_right_wrong_modulus),
        "Matrices must have the same modulus m",
    )

    mul_left_wrong_shape =
        mod.KaratsubaArray{Float64,2}(zeros(Float64, 2, 3), zeros(Float64, 2, 3), 3, 5, 15)
    mul_right_wrong_shape =
        mod.KaratsubaArray{Float64,2}(zeros(Float64, 2, 2), zeros(Float64, 2, 2), 3, 5, 15)
    assert_guard_error_message(
        () -> LinearAlgebra.mul!(mul_dest, mul_left_wrong_shape, mul_right_wrong_shape),
        "Matrix dimensions don't work for multiplication",
    )

    mul_unplanned_right =
        mod.KaratsubaArray{Float64,2}(zeros(Float64, 2, 2), zeros(Float64, 2, 2), 3, 5, 15)
    assert_guard_error_message(
        () -> LinearAlgebra.mul!(mul_dest, mul_left, mul_unplanned_right),
        "Must initialize plans first",
    )
end
