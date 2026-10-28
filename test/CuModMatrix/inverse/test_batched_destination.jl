function _batch_destination_host(m::Int, n::Int, variant::Int = 0)
    A = zeros(Float32, m, n)
    for i = 1:min(m, n)
        A[i, i] = 1
    end
    if m >= 2 && n >= 2
        A[1, 1] = 0
        A[1, 2] = 1
        A[2, 1] = 1
        A[2, 2] = 0
    end
    if variant > 0 && m >= 3 && n >= 3
        A[3, 3] = Float32(variant + 1)
    end
    return A
end

function _batch_destination_matrix(A::CuModMatrix)
    return CuModMatrix(copy(parent(A)); modulus = modulus(A), new_size = size(A))
end

function _batch_destination_blank(A::CuModMatrix; value::Float32 = 37)
    data = CUDA.fill(value, size(parent(A))...)
    return CuModMatrix(data; modulus = modulus(A), new_size = size(A))
end

function _check_batch_factor_storage(Fs, mats, originals)
    @test length(Fs) == length(mats) == length(originals)
    for i in eachindex(mats)
        @test parent(Fs[i].LU) === parent(mats[i])
        @test Fs[i].rank == min(size(originals[i])...)
        @test sort(Fs[i].p) == collect(1:length(Fs[i].p))
        @test sort(Fs[i].q) == collect(1:length(Fs[i].q))
        @test GPUFiniteFieldMatrices.pluq_check_identity(Fs[i], originals[i])
    end
    if length(Fs) > 1
        @test Fs[1].p !== Fs[2].p
        @test Fs[1].q !== Fs[2].q
    end
end

function _check_pluq_batch_forms(inputs; options = PLUQOptions())
    originals = [
        CuModMatrix(copy(parent(A)); modulus = modulus(A), new_size = size(A)) for
        A in inputs
    ]
    input_storage = [Array(parent(A)) for A in inputs]

    inplace = [_batch_destination_matrix(A) for A in inputs]
    Fs_inplace = GPUFiniteFieldMatrices.pluq_batched!(inplace; options = options)
    _check_batch_factor_storage(Fs_inplace, inplace, originals)
    @test all(i -> Array(parent(inputs[i])) == input_storage[i], eachindex(inputs))

    dests = [_batch_destination_blank(A) for A in inputs]
    dest_storage = [Array(parent(A)) for A in dests]
    Fs_dest = GPUFiniteFieldMatrices.pluq_batched!(dests, inputs; options = options)
    _check_batch_factor_storage(Fs_dest, dests, originals)
    @test all(i -> Array(parent(inputs[i])) == input_storage[i], eachindex(inputs))
    @test all(i -> Array(parent(dests[i])) != dest_storage[i], eachindex(dests))

    Fs_allocating = GPUFiniteFieldMatrices.pluq_batched(inputs; options = options)
    _check_batch_factor_storage(Fs_allocating, [F.LU for F in Fs_allocating], originals)
    @test all(i -> Array(parent(inputs[i])) == input_storage[i], eachindex(inputs))
    @test all(i -> parent(Fs_allocating[i].LU) !== parent(inputs[i]), eachindex(inputs))
    return nothing
end

function test_batched_destination_api()
    p = 101
    for n in (4, 8, 16, 32)
        inputs = [
            CuModMatrix{Float32}(_batch_destination_host(n, n, k); modulus = p) for k = 0:1
        ]
        _check_pluq_batch_forms(inputs; options = PLUQOptions(check_prime = true))
        Fs_square = GPUFiniteFieldMatrices.pluq_batched_NxN!(
            [_batch_destination_matrix(A) for A in inputs];
            options = PLUQOptions(check_prime = true),
        )
        @test all(F -> F.rank == n, Fs_square)
    end

    # A non-default PLUQ setting must route tiny square batches through the
    # configurable scalar fallback, which still overwrites each supplied input.
    option_inputs =
        [CuModMatrix{Float32}(_batch_destination_host(4, 4, k); modulus = p) for k = 0:1]
    _check_pluq_batch_forms(option_inputs; options = PLUQOptions(blocksize = 16))

    # The general fallback supports both square sizes outside the tiny set and
    # rectangular matrices.
    fallback_inputs =
        [CuModMatrix{Float32}(_batch_destination_host(33, 33, k); modulus = p) for k = 0:1]
    _check_pluq_batch_forms(fallback_inputs; options = PLUQOptions(batch_streams = 2))
    rectangular_inputs = [
        CuModMatrix{Float32}(_batch_destination_host(4, 2, 0); modulus = p),
        CuModMatrix{Float32}(_batch_destination_host(2, 4, 1); modulus = p),
    ]
    _check_pluq_batch_forms(rectangular_inputs)

    # Destination errors are found before any destination storage is changed.
    inputs =
        [CuModMatrix{Float32}(_batch_destination_host(4, 4, k); modulus = p) for k = 0:1]
    input_storage = [Array(parent(A)) for A in inputs]
    good = _batch_destination_blank(inputs[1])
    bad_modulus = CuModMatrix{Float32}(_batch_destination_host(4, 4), modulus = 103)
    dests = [good, bad_modulus]
    dest_storage = [Array(parent(A)) for A in dests]
    @test_throws GPUFiniteFieldMatrices.CuModArrayModulusMismatchException GPUFiniteFieldMatrices.pluq_batched!(
        dests,
        inputs,
    )
    @test all(i -> Array(parent(dests[i])) == dest_storage[i], eachindex(dests))
    @test all(i -> Array(parent(inputs[i])) == input_storage[i], eachindex(inputs))

    wrong_shape = CuModMatrix{Float32}(zeros(Float32, 5, 5); modulus = p)
    @test_throws GPUFiniteFieldMatrices.CuModArraySizeMismatchException GPUFiniteFieldMatrices.pluq_batched!(
        [good, wrong_shape],
        inputs,
    )
    @test Array(parent(good)) == dest_storage[1]

    wrong_type = CuModMatrix{Float64}(_batch_destination_host(4, 4); modulus = p)
    @test_throws ArgumentError GPUFiniteFieldMatrices.pluq_batched!(
        [good, wrong_type],
        inputs,
    )
    @test Array(parent(good)) == dest_storage[1]

    @test_throws ArgumentError GPUFiniteFieldMatrices.pluq_batched!([good], inputs)
    @test Array(parent(good)) == dest_storage[1]

    # Both direct and cross-batch input/destination aliases are rejected before writes.
    @test_throws ArgumentError GPUFiniteFieldMatrices.pluq_batched!(
        [inputs[1]],
        [inputs[1]],
    )
    cross_alias_dests = [inputs[2], _batch_destination_blank(inputs[1])]
    @test_throws ArgumentError GPUFiniteFieldMatrices.pluq_batched!(
        cross_alias_dests,
        inputs,
    )
    @test all(i -> Array(parent(inputs[i])) == input_storage[i], eachindex(inputs))

    aliased_dests =
        [_batch_destination_blank(inputs[1]), _batch_destination_blank(inputs[2])]
    aliased_dests[2] = aliased_dests[1]
    aliased_dest_storage = Array(parent(aliased_dests[1]))
    @test_throws ArgumentError GPUFiniteFieldMatrices.pluq_batched!(aliased_dests, inputs)
    @test Array(parent(aliased_dests[1])) == aliased_dest_storage

    composite = [CuModMatrix{Float32}(_batch_destination_host(4, 4); modulus = 15)]
    composite_storage = Array(parent(composite[1]))
    @test_throws GPUFiniteFieldMatrices.CuModMatrixModulusNotPrimeException GPUFiniteFieldMatrices.pluq_batched!(
        composite;
        options = PLUQOptions(check_prime = true),
    )
    @test Array(parent(composite[1])) == composite_storage

    # The tiny inverse backend implements the augmented strategy. PLUQ strategy
    # requests use the scalar fallback, and both paths preserve their inputs.
    inverse_inputs =
        [CuModMatrix{Float32}(_batch_destination_host(4, 4, k); modulus = p) for k = 0:1]
    inverse_storage = [Array(parent(A)) for A in inverse_inputs]
    augmented = GPUFiniteFieldMatrices.inverse_batched(
        inverse_inputs;
        options = PLUQOptions(inverse_strategy = :augmented, check_prime = true),
    )
    pluq_strategy = GPUFiniteFieldMatrices.inverse_batched(
        inverse_inputs;
        options = PLUQOptions(inverse_strategy = :pluq),
    )
    @test all(
        i -> Array(parent(inverse_inputs[i])) == inverse_storage[i],
        eachindex(inverse_inputs),
    )
    @test all(
        i -> _check_identity_batch(inverse_inputs[i], augmented[i], p),
        eachindex(inverse_inputs),
    )
    @test all(
        i -> _check_identity_batch(inverse_inputs[i], pluq_strategy[i], p),
        eachindex(inverse_inputs),
    )
    return nothing
end
