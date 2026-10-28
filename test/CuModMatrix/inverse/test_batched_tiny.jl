using Random

function _tiny_invertible_host(n::Int, p::Int, ::Type{T}, rng) where {T}
    L = Matrix{BigInt}(I, n, n)
    U = Matrix{BigInt}(I, n, n)
    for i = 1:n
        for j = 1:(i-1)
            L[i, j] = BigInt(rand(rng, 0:(p-1)))
        end
        for j = (i+1):n
            U[i, j] = BigInt(rand(rng, 0:(p-1)))
        end
    end
    n > 1 && (L[2, 1] = 0)
    A = _tiny_host_modular_product(L, U, p)
    A[[1, 2], :] = copy(A[[2, 1], :])
    return T.(A)
end

function _tiny_host_modular_product(A, B, p::Int)
    m, k = size(A)
    k == size(B, 1) || throw(DimensionMismatch("modular product dimensions do not match"))
    n = size(B, 2)
    N = BigInt(p)
    C = Matrix{BigInt}(undef, m, n)
    for i = 1:m, j = 1:n
        value = BigInt(0)
        for t = 1:k
            a = BigInt(round(Int, A[i, t]))
            b = BigInt(round(Int, B[t, j]))
            value = mod(value + a * b, N)
        end
        C[i, j] = value
    end
    return C
end

function _tiny_host_inverse_identity(A, B, p::Int)
    n = size(A, 1)
    size(A) == (n, n) && size(B) == (n, n) || return false
    identity = Matrix{BigInt}(I, n, n)
    return _tiny_host_modular_product(A, B, p) == identity &&
           _tiny_host_modular_product(B, A, p) == identity
end

@testset "Tiny inverse host modular oracle" begin
    A = BigInt[1 2; 3 5]
    Ainv = BigInt[96 2; 3 100]
    @test _tiny_host_inverse_identity(A, Ainv, 101)
    corrupted = copy(Ainv)
    corrupted[1, 1] += 1
    @test !_tiny_host_inverse_identity(A, corrupted, 101)
end

function _tiny_inverse_bang(n::Int)
    n == 4 && return GPUFiniteFieldMatrices.inverse_batched_4x4!
    n == 8 && return GPUFiniteFieldMatrices.inverse_batched_8x8!
    n == 16 && return GPUFiniteFieldMatrices.inverse_batched_16x16!
    n == 32 && return GPUFiniteFieldMatrices.inverse_batched_32x32!
    throw(ArgumentError("supported tiny inverse sizes are 4, 8, 16, and 32"))
end

function _tiny_gpu_batch(hosts, p::Int)
    return [CuModMatrix{Float32}(copy(A); modulus = p) for A in hosts]
end

function _tiny_blank_destination(A::CuModMatrix; value::Float32 = 73.0f0)
    data = CUDA.fill(value, size(parent(A))...)
    return CuModMatrix(data; modulus = modulus(A), new_size = size(A))
end

_tiny_logical_host(A::CuModMatrix, n::Int) = Array(parent(A))[1:n, 1:n]

function _test_tiny_inverse_destination_forms(n::Int, hosts, p::Int)
    inverse! = _tiny_inverse_bang(n)
    sources = _tiny_gpu_batch(hosts, p)
    source_storage = [Array(parent(A)) for A in sources]

    dests = [_tiny_blank_destination(A) for A in sources]
    dest_storage = [Array(parent(A)) for A in dests]
    result = inverse!(dests, sources; options = PLUQOptions(check_prime = true))
    @test result === dests
    @test all(i -> result[i] === dests[i], eachindex(dests))
    @test all(i -> parent(result[i]) === parent(dests[i]), eachindex(dests))
    @test all(i -> parent(dests[i]) !== parent(sources[i]), eachindex(dests))
    @test all(i -> Array(parent(sources[i])) == source_storage[i], eachindex(sources))
    @test all(i -> Array(parent(dests[i])) != dest_storage[i], eachindex(dests))
    @test all(
        i -> _tiny_host_inverse_identity(hosts[i], _tiny_logical_host(dests[i], n), p),
        eachindex(dests),
    )

    allocated = GPUFiniteFieldMatrices.inverse_batched(
        sources;
        options = PLUQOptions(inverse_strategy = :augmented, check_prime = true),
    )
    @test all(i -> Array(parent(sources[i])) == source_storage[i], eachindex(sources))
    @test all(i -> parent(allocated[i]) !== parent(sources[i]), eachindex(sources))
    @test all(
        i -> _tiny_host_inverse_identity(hosts[i], _tiny_logical_host(allocated[i], n), p),
        eachindex(allocated),
    )

    inplace = _tiny_gpu_batch(hosts, p)
    inplace_storage = [Array(parent(A)) for A in inplace]
    inplace_result = inverse!(inplace; options = PLUQOptions(check_prime = true))
    @test inplace_result === inplace
    @test all(i -> inplace_result[i] === inplace[i], eachindex(inplace))
    @test all(i -> parent(inplace_result[i]) === parent(inplace[i]), eachindex(inplace))
    @test all(i -> Array(parent(inplace[i])) != inplace_storage[i], eachindex(inplace))
    @test all(
        i -> _tiny_host_inverse_identity(hosts[i], _tiny_logical_host(inplace[i], n), p),
        eachindex(inplace),
    )
    return nothing
end

function _tiny_assert_destination_rejected_without_write(
    inverse!,
    dests,
    sources,
    expected_exception;
    options::PLUQOptions = PLUQOptions(),
)
    input_storage = [Array(parent(A)) for A in sources]
    destination_storage = [Array(parent(A)) for A in dests]
    @test_throws expected_exception inverse!(dests, sources; options = options)
    @test all(i -> Array(parent(sources[i])) == input_storage[i], eachindex(sources))
    @test all(i -> Array(parent(dests[i])) == destination_storage[i], eachindex(dests))
    return nothing
end

function _tiny_assert_inplace_rejected_without_write(
    inverse!,
    mats,
    expected_exception;
    options::PLUQOptions = PLUQOptions(),
)
    input_storage = [Array(parent(A)) for A in mats]
    @test_throws expected_exception inverse!(mats; options = options)
    @test all(i -> Array(parent(mats[i])) == input_storage[i], eachindex(mats))
    return nothing
end

function test_batched_tiny_destination_preflight()
    p = 101
    inverse! = _tiny_inverse_bang(4)
    hosts = [
        _tiny_invertible_host(4, p, Float32, Random.MersenneTwister(101)),
        _tiny_invertible_host(4, p, Float32, Random.MersenneTwister(102)),
    ]
    sources = _tiny_gpu_batch(hosts, p)
    good = [_tiny_blank_destination(A) for A in sources]

    _tiny_assert_destination_rejected_without_write(
        inverse!,
        good[1:1],
        sources,
        ArgumentError,
    )

    wrong_shape = CuModMatrix{Float32}(zeros(Float32, 5, 5); modulus = p)
    _tiny_assert_destination_rejected_without_write(
        inverse!,
        [good[1], wrong_shape],
        sources,
        GPUFiniteFieldMatrices.CuModArraySizeMismatchException,
    )

    wrong_input_modulus = CuModMatrix{Float32}(Matrix{Float32}(I, 4, 4); modulus = 103)
    _tiny_assert_destination_rejected_without_write(
        inverse!,
        good,
        [sources[1], wrong_input_modulus],
        GPUFiniteFieldMatrices.CuModArrayModulusMismatchException,
    )

    wrong_input_type = CuModMatrix{Float64}(Matrix{Float64}(I, 4, 4); modulus = p)
    wrong_input_type_dest =
        [_tiny_blank_destination(sources[1]), _tiny_blank_destination(wrong_input_type)]
    _tiny_assert_destination_rejected_without_write(
        inverse!,
        wrong_input_type_dest,
        [sources[1], wrong_input_type],
        GPUFiniteFieldMatrices.CuModArrayModulusMismatchException,
    )

    short_input_parent = CUDA.fill(Float32(29), 3, 3)
    short_input = CuModMatrix(short_input_parent; modulus = p, new_size = (4, 4))
    full_destination = CuModMatrix(CUDA.zeros(Float32, 4, 4); modulus = p)
    _tiny_assert_destination_rejected_without_write(
        inverse!,
        [full_destination],
        [short_input],
        GPUFiniteFieldMatrices.CuModArraySizeMismatchException,
    )

    wrong_modulus = CuModMatrix{Float32}(Matrix{Float32}(I, 4, 4); modulus = 103)
    _tiny_assert_destination_rejected_without_write(
        inverse!,
        [good[1], wrong_modulus],
        sources,
        GPUFiniteFieldMatrices.CuModArrayModulusMismatchException,
    )

    wrong_type = CuModMatrix{Float64}(Matrix{Float64}(I, 4, 4); modulus = p)
    _tiny_assert_destination_rejected_without_write(
        inverse!,
        [good[1], wrong_type],
        sources,
        ArgumentError,
    )

    short_parent = CUDA.fill(Float32(31), 3, 3)
    short_capacity = CuModMatrix(short_parent; modulus = p, new_size = (4, 4))
    @test size(short_capacity) == (4, 4)
    @test size(parent(short_capacity)) == (3, 3)
    _tiny_assert_destination_rejected_without_write(
        inverse!,
        [good[1], short_capacity],
        sources,
        GPUFiniteFieldMatrices.CuModArraySizeMismatchException,
    )

    _tiny_assert_destination_rejected_without_write(
        inverse!,
        sources,
        sources,
        ArgumentError,
    )

    aliased_output = [good[1], good[1]]
    _tiny_assert_destination_rejected_without_write(
        inverse!,
        aliased_output,
        sources,
        ArgumentError,
    )

    shared_parent = parent(sources[1])
    shared_input_wrapper =
        CuModMatrix(shared_parent; modulus = p, new_size = size(sources[1]))
    cross_alias = [good[1], shared_input_wrapper]
    _tiny_assert_destination_rejected_without_write(
        inverse!,
        cross_alias,
        sources,
        ArgumentError,
    )

    duplicate_inputs = [sources[1], sources[2], sources[1]]
    _tiny_assert_inplace_rejected_without_write(inverse!, duplicate_inputs, ArgumentError)

    wrong_input_size = CuModMatrix{Float32}(Matrix{Float32}(I, 8, 8); modulus = p)
    wrong_input = [wrong_input_size]
    wrong_input_dest = [_tiny_blank_destination(wrong_input_size)]
    _tiny_assert_destination_rejected_without_write(
        inverse!,
        wrong_input_dest,
        wrong_input,
        GPUFiniteFieldMatrices.CuModArraySizeMismatchException,
    )

    composite = [CuModMatrix{Float32}(Matrix{Float32}(I, 4, 4); modulus = 15)]
    composite_dest = [_tiny_blank_destination(composite[1])]
    _tiny_assert_destination_rejected_without_write(
        inverse!,
        composite_dest,
        composite,
        GPUFiniteFieldMatrices.CuModMatrixModulusNotPrimeException;
        options = PLUQOptions(check_prime = true),
    )
    return nothing
end

function test_batched_tiny_kernels()
    rng = Random.MersenneTwister(17)
    p = 101
    for n in (4, 8, 16, 32)
        hosts = [_tiny_invertible_host(n, p, Float32, rng) for _ = 1:4]
        mats = _tiny_gpu_batch(hosts, p)
        origs = _tiny_gpu_batch(hosts, p)
        Fs = if n == 4
            GPUFiniteFieldMatrices.pluq_batched_4x4!(mats)
        elseif n == 8
            GPUFiniteFieldMatrices.pluq_batched_8x8!(mats)
        elseif n == 16
            GPUFiniteFieldMatrices.pluq_batched_16x16!(mats)
        else
            GPUFiniteFieldMatrices.pluq_batched_32x32!(mats)
        end
        @test length(Fs) == length(mats)
        for i in eachindex(mats)
            @test Fs[i].rank == n
            @test GPUFiniteFieldMatrices.pluq_check_identity(Fs[i], origs[i])
        end

        _test_tiny_inverse_destination_forms(n, hosts, p)

        singular_host = Matrix{Float32}(I, n, n)
        singular_host[2, :] .= singular_host[1, :]
        singular_hosts = [copy(hosts[1]), copy(hosts[2]), singular_host]
        inverse! = _tiny_inverse_bang(n)

        singular_sources = _tiny_gpu_batch(singular_hosts, p)
        singular_input_storage = [Array(parent(A)) for A in singular_sources]
        singular_dests = [_tiny_blank_destination(A) for A in singular_sources]
        singular_destination_storage = [Array(parent(A)) for A in singular_dests]
        @test_throws GPUFiniteFieldMatrices.InverseNotDefinedException inverse!(
            singular_dests,
            singular_sources,
        )
        @test all(
            i -> Array(parent(singular_sources[i])) == singular_input_storage[i],
            eachindex(singular_sources),
        )
        @test all(
            i -> Array(parent(singular_dests[i])) == singular_destination_storage[i],
            eachindex(singular_dests),
        )

        singular_inplace = _tiny_gpu_batch(singular_hosts, p)
        singular_inplace_storage = [Array(parent(A)) for A in singular_inplace]
        @test_throws GPUFiniteFieldMatrices.InverseNotDefinedException inverse!(
            singular_inplace,
        )
        @test all(
            i -> Array(parent(singular_inplace[i])) == singular_inplace_storage[i],
            eachindex(singular_inplace),
        )
    end

    # A fixed-size helper must honor non-augmented strategy requests through
    # the existing scalar fallback, without exposing partial writes on failure.
    p = 101
    n = 4
    hosts =
        [_tiny_invertible_host(n, p, Float32, Random.MersenneTwister(201 + i)) for i = 1:2]
    inverse! = _tiny_inverse_bang(n)
    fallback_sources = _tiny_gpu_batch(hosts, p)
    fallback_source_storage = [Array(parent(A)) for A in fallback_sources]
    fallback_dests = [_tiny_blank_destination(A) for A in fallback_sources]
    fallback_result = inverse!(
        fallback_dests,
        fallback_sources;
        options = PLUQOptions(inverse_strategy = :pluq, check_prime = true),
    )
    @test fallback_result === fallback_dests
    @test all(
        i -> Array(parent(fallback_sources[i])) == fallback_source_storage[i],
        eachindex(fallback_sources),
    )
    @test all(
        i -> _tiny_host_inverse_identity(
            hosts[i],
            _tiny_logical_host(fallback_dests[i], n),
            p,
        ),
        eachindex(fallback_dests),
    )

    fallback_inplace = _tiny_gpu_batch(hosts, p)
    fallback_inplace_result = inverse!(
        fallback_inplace;
        options = PLUQOptions(inverse_strategy = :pluq, check_prime = true),
    )
    @test fallback_inplace_result === fallback_inplace
    @test all(i -> fallback_inplace_result[i] === fallback_inplace[i], eachindex(hosts))
    @test all(
        i -> _tiny_host_inverse_identity(
            hosts[i],
            _tiny_logical_host(fallback_inplace[i], n),
            p,
        ),
        eachindex(fallback_inplace),
    )

    singular = Matrix{Float32}(I, n, n)
    singular[2, :] .= singular[1, :]
    fallback_singular_hosts = [copy(hosts[1]), copy(hosts[2]), singular]
    fallback_singular_sources = _tiny_gpu_batch(fallback_singular_hosts, p)
    fallback_singular_input_storage = [Array(parent(A)) for A in fallback_singular_sources]
    fallback_singular_dests =
        [_tiny_blank_destination(A) for A in fallback_singular_sources]
    fallback_singular_destination_storage =
        [Array(parent(A)) for A in fallback_singular_dests]
    fallback_options = PLUQOptions(inverse_strategy = :pluq)
    @test_throws GPUFiniteFieldMatrices.InverseNotDefinedException inverse!(
        fallback_singular_dests,
        fallback_singular_sources;
        options = fallback_options,
    )
    @test all(
        i ->
            Array(parent(fallback_singular_sources[i])) ==
            fallback_singular_input_storage[i],
        eachindex(fallback_singular_sources),
    )
    @test all(
        i ->
            Array(parent(fallback_singular_dests[i])) ==
            fallback_singular_destination_storage[i],
        eachindex(fallback_singular_dests),
    )

    fallback_singular_inplace = _tiny_gpu_batch(fallback_singular_hosts, p)
    fallback_singular_inplace_storage =
        [Array(parent(A)) for A in fallback_singular_inplace]
    @test_throws GPUFiniteFieldMatrices.InverseNotDefinedException inverse!(
        fallback_singular_inplace;
        options = fallback_options,
    )
    @test all(
        i ->
            Array(parent(fallback_singular_inplace[i])) ==
            fallback_singular_inplace_storage[i],
        eachindex(fallback_singular_inplace),
    )

    test_batched_tiny_destination_preflight()
    return nothing
end
