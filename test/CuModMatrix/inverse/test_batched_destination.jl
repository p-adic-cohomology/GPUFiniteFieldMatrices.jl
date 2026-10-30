function _batch_pluq_host_identity(original, packed, p, q, rank::Int, modulus::Int)
    m, n = size(original)
    @test size(packed) == (m, n)
    @test length(p) == m
    @test length(q) == n
    @test all(i -> 1 <= i <= m, p)
    @test all(j -> 1 <= j <= n, q)
    @test sort(p) == collect(1:m)
    @test sort(q) == collect(1:n)
    @test 0 <= rank <= min(m, n)
    if size(packed) != (m, n) ||
       length(p) != m ||
       length(q) != n ||
       sort(p) != collect(1:m) ||
       sort(q) != collect(1:n) ||
       !(0 <= rank <= min(m, n))
        return false
    end

    N = BigInt(modulus)
    for i = 1:m, j = 1:n
        paq = mod(BigInt(Int(original[p[i], q[j]])), N)
        lu = BigInt(0)
        for k = 1:m
            l_ik = if i == k
                1
            elseif i > k && k <= n
                packed[i, k]
            else
                0
            end
            u_kj = k <= j ? packed[k, j] : 0
            lu = mod(lu + BigInt(Int(l_ik)) * BigInt(Int(u_kj)), N)
        end
        paq == lu || return false
    end
    return true
end

function _batch_host_modular_product(A, B, modulus::Int)
    m, k = size(A)
    k == size(B, 1) || throw(DimensionMismatch("modular product dimensions do not match"))
    n = size(B, 2)
    N = BigInt(modulus)
    C = Matrix{BigInt}(undef, m, n)
    for i = 1:m, j = 1:n
        value = BigInt(0)
        for t = 1:k
            value = mod(value + BigInt(Int(A[i, t])) * BigInt(Int(B[t, j])), N)
        end
        C[i, j] = value
    end
    return C
end

function _batch_host_inverse_identity(A, B, modulus::Int)
    m, n = size(A)
    m == n && size(B) == (n, n) || return false
    product = _batch_host_modular_product(A, B, modulus)
    reverse_product = _batch_host_modular_product(B, A, modulus)
    return all(i == j ? product[i, j] == 1 : product[i, j] == 0 for i = 1:n, j = 1:n) &&
           all(
        i == j ? reverse_product[i, j] == 1 : reverse_product[i, j] == 0 for
        i = 1:n, j = 1:n
    )
end

@testset "Batched PLUQ exact host oracle" begin
    m, n, pmod = 3, 4, 101
    L = [1 0 0; 2 1 0; 3 4 1]
    U = [5 6 7 8; 0 9 10 11; 0 0 0 0]
    packed = zeros(Int, m, n)
    for i = 1:m, j = 1:n
        packed[i, j] = i > j ? L[i, j] : U[i, j]
    end
    p = [2, 1, 3]
    q = [2, 1, 4, 3]
    paq = mod.(L * U, pmod)
    original = zeros(Int, m, n)
    for i = 1:m, j = 1:n
        original[p[i], q[j]] = paq[i, j]
    end

    @test _batch_pluq_host_identity(original, packed, p, q, 2, pmod)
    corrupted = copy(original)
    corrupted[p[1], q[1]] = mod(corrupted[p[1], q[1]] + 1, pmod)
    @test !_batch_pluq_host_identity(corrupted, packed, p, q, 2, pmod)

    m_tall, n_tall = 4, 2
    L_tall = [1 0 0 0; 2 1 0 0; 3 4 1 0; 5 6 7 1]
    U_tall = [5 6; 0 9; 0 0; 0 0]
    packed_tall = zeros(Int, m_tall, n_tall)
    for i = 1:m_tall, j = 1:n_tall
        packed_tall[i, j] = i > j ? L_tall[i, j] : U_tall[i, j]
    end
    p_tall = [4, 1, 3, 2]
    q_tall = [2, 1]
    paq_tall = mod.(L_tall * U_tall, pmod)
    original_tall = zeros(Int, m_tall, n_tall)
    for i = 1:m_tall, j = 1:n_tall
        original_tall[p_tall[i], q_tall[j]] = paq_tall[i, j]
    end
    @test _batch_pluq_host_identity(original_tall, packed_tall, p_tall, q_tall, 2, pmod)
end

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

function _batch_rank_q_host(n::Int, variant::Int = 0)
    n >= 4 || throw(ArgumentError("rank/q batch fixture requires n >= 4"))
    A = zeros(Int, n, n)
    A[1:3, 2] .= [2, 5, 1]
    A[1:3, 3] .= [3, 7, 4]
    A[1:3, 4] .= A[1:3, 2] .+ (variant + 1) .* A[1:3, 3]
    return A
end

function _batch_destination_matrix(A::CuModMatrix)
    return CuModMatrix(copy(parent(A)); modulus = modulus(A), new_size = size(A))
end

function _batch_destination_blank(A::CuModMatrix; value::Float32 = 37)
    data = CUDA.fill(value, size(parent(A))...)
    return CuModMatrix(data; modulus = modulus(A), new_size = size(A))
end

function _check_batch_factor_storage(
    Fs,
    mats,
    originals;
    expected_ranks = nothing,
    require_nontrivial_q::Bool = false,
)
    @test length(Fs) == length(mats) == length(originals)
    for i in eachindex(mats)
        @test parent(Fs[i].LU) === parent(mats[i])
        m, n = size(originals[i])
        expected_rank = expected_ranks === nothing ? min(m, n) : expected_ranks[i]
        @test Fs[i].rank == expected_rank
        @test length(Fs[i].p) == m
        @test length(Fs[i].q) == n
        @test all(k -> 1 <= k <= m, Fs[i].p)
        @test all(k -> 1 <= k <= n, Fs[i].q)
        @test sort(Fs[i].p) == collect(1:m)
        @test sort(Fs[i].q) == collect(1:n)
        if require_nontrivial_q
            @test Fs[i].q != collect(1:n)
        end
        original_host = Array(parent(originals[i]))[1:m, 1:n]
        packed_host = Array(parent(Fs[i].LU))[1:m, 1:n]
        @test _batch_pluq_host_identity(
            original_host,
            packed_host,
            Fs[i].p,
            Fs[i].q,
            Fs[i].rank,
            modulus(Fs[i].LU),
        )
        if m == n
            @test GPUFiniteFieldMatrices.pluq_check_identity(Fs[i], originals[i])
        end
    end
    if length(Fs) > 1
        @test Fs[1].p !== Fs[2].p
        @test Fs[1].q !== Fs[2].q
    end
end

function _check_pluq_batch_forms(
    inputs;
    options = PLUQOptions(),
    expected_ranks = nothing,
    require_nontrivial_q::Bool = false,
)
    originals = [
        CuModMatrix(copy(parent(A)); modulus = modulus(A), new_size = size(A)) for
        A in inputs
    ]
    input_storage = [Array(parent(A)) for A in inputs]

    inplace = [_batch_destination_matrix(A) for A in inputs]
    Fs_inplace = GPUFiniteFieldMatrices.pluq_batched!(inplace; options = options)
    _check_batch_factor_storage(
        Fs_inplace,
        inplace,
        originals;
        expected_ranks = expected_ranks,
        require_nontrivial_q = require_nontrivial_q,
    )
    @test all(i -> Array(parent(inputs[i])) == input_storage[i], eachindex(inputs))

    dests = [_batch_destination_blank(A) for A in inputs]
    dest_storage = [Array(parent(A)) for A in dests]
    Fs_dest = GPUFiniteFieldMatrices.pluq_batched!(dests, inputs; options = options)
    _check_batch_factor_storage(
        Fs_dest,
        dests,
        originals;
        expected_ranks = expected_ranks,
        require_nontrivial_q = require_nontrivial_q,
    )
    @test all(i -> Array(parent(inputs[i])) == input_storage[i], eachindex(inputs))
    @test all(i -> Array(parent(dests[i])) != dest_storage[i], eachindex(dests))

    Fs_allocating = GPUFiniteFieldMatrices.pluq_batched(inputs; options = options)
    _check_batch_factor_storage(
        Fs_allocating,
        [F.LU for F in Fs_allocating],
        originals;
        expected_ranks = expected_ranks,
        require_nontrivial_q = require_nontrivial_q,
    )
    @test all(i -> Array(parent(inputs[i])) == input_storage[i], eachindex(inputs))
    @test all(i -> parent(Fs_allocating[i].LU) !== parent(inputs[i]), eachindex(inputs))
    for i in eachindex(inputs)
        @test Fs_inplace[i].p !== Fs_dest[i].p
        @test Fs_inplace[i].q !== Fs_dest[i].q
        @test Fs_inplace[i].p !== Fs_allocating[i].p
        @test Fs_inplace[i].q !== Fs_allocating[i].q
        @test Fs_dest[i].p !== Fs_allocating[i].p
        @test Fs_dest[i].q !== Fs_allocating[i].q
    end
    return nothing
end

function _check_rejected_inplace_aliases(
    n::Int;
    options = PLUQOptions(),
    use_nxn::Bool = false,
)
    p = 101
    base = CuModMatrix{Float32}(_batch_rank_q_host(n); modulus = p)
    shared_parent = copy(parent(base))
    first_wrapper = CuModMatrix(shared_parent; modulus = p, new_size = size(base))
    second_wrapper = CuModMatrix(shared_parent; modulus = p, new_size = size(base))
    clean = CuModMatrix{Float32}(_batch_destination_host(n, n, 1); modulus = p)
    shared_batch = [clean, first_wrapper, second_wrapper]
    shared_storage = [Array(parent(A)) for A in shared_batch]
    if use_nxn
        @test_throws ArgumentError GPUFiniteFieldMatrices.pluq_batched_NxN!(
            shared_batch;
            options = options,
        )
    else
        @test_throws ArgumentError GPUFiniteFieldMatrices.pluq_batched!(
            shared_batch;
            options = options,
        )
    end
    @test all(
        i -> Array(parent(shared_batch[i])) == shared_storage[i],
        eachindex(shared_batch),
    )

    duplicate_source = CuModMatrix{Float32}(_batch_rank_q_host(n, 1); modulus = p)
    duplicate_clean = CuModMatrix{Float32}(_batch_destination_host(n, n); modulus = p)
    duplicate_batch = [duplicate_clean, duplicate_source, duplicate_source]
    duplicate_storage = [Array(parent(A)) for A in duplicate_batch]
    @test_throws ArgumentError GPUFiniteFieldMatrices.pluq_batched!(
        duplicate_batch;
        options = options,
    )
    @test all(
        i -> Array(parent(duplicate_batch[i])) == duplicate_storage[i],
        eachindex(duplicate_batch),
    )
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

    # Rank two and a zero leading column force a non-identity column
    # permutation for both the tiny kernel and the scalar fallback.
    for n in (4, 33)
        rank_q_inputs = [
            CuModMatrix{Float32}(_batch_rank_q_host(n, variant); modulus = p) for
            variant = 0:1
        ]
        _check_pluq_batch_forms(
            rank_q_inputs;
            expected_ranks = [2, 2],
            require_nontrivial_q = true,
        )
    end

    # Overlapping in-place inputs are rejected before any storage is changed
    # through tiny, option-routed fallback, and scalar fallback dispatch.
    _check_rejected_inplace_aliases(4)
    _check_rejected_inplace_aliases(4; options = PLUQOptions(blocksize = 16))
    _check_rejected_inplace_aliases(33)
    _check_rejected_inplace_aliases(4; use_nxn = true)

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

    # A real CuArray wrapper can report matching logical dimensions while its
    # physical backing is too small. A later bad destination must fail during
    # preflight, before the earlier good destination or either source changes.
    capacity_inputs =
        [CuModMatrix{Float32}(_batch_destination_host(33, 33, k); modulus = p) for k = 0:1]
    capacity_good = _batch_destination_blank(capacity_inputs[1])
    capacity_bad_parent = CUDA.fill(Float32(41), 32, 32)
    capacity_bad =
        CuModMatrix(capacity_bad_parent; modulus = p, new_size = size(capacity_inputs[2]))
    @test size(capacity_bad) == size(capacity_inputs[2])
    @test size(parent(capacity_bad)) == (32, 32)
    capacity_dests = [capacity_good, capacity_bad]
    capacity_dest_storage = [Array(parent(A)) for A in capacity_dests]
    capacity_input_storage = [Array(parent(A)) for A in capacity_inputs]
    @test_throws GPUFiniteFieldMatrices.CuModArraySizeMismatchException GPUFiniteFieldMatrices.pluq_batched!(
        capacity_dests,
        capacity_inputs,
    )
    @test all(
        i -> Array(parent(capacity_dests[i])) == capacity_dest_storage[i],
        eachindex(capacity_dests),
    )
    @test all(
        i -> Array(parent(capacity_inputs[i])) == capacity_input_storage[i],
        eachindex(capacity_inputs),
    )

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

    tiny_mixed_inputs = [
        CuModMatrix{Float32}(_batch_destination_host(4, 4); modulus = p),
        CuModMatrix{Float64}(_batch_destination_host(4, 4, 1); modulus = p),
    ]
    tiny_mixed_storage = [Array(parent(A)) for A in tiny_mixed_inputs]
    @test_throws ArgumentError GPUFiniteFieldMatrices.inverse_batched(tiny_mixed_inputs)
    @test_throws ArgumentError GPUFiniteFieldMatrices.pluq_batched!(tiny_mixed_inputs)
    @test all(
        i -> Array(parent(tiny_mixed_inputs[i])) == tiny_mixed_storage[i],
        eachindex(tiny_mixed_inputs),
    )

    # Fallback inverse batches retain their per-item element types, for both
    # ordinary dispatch and the applicable multi-stream path.
    for options in (PLUQOptions(), PLUQOptions(batch_streams = 2))
        mixed_inputs = [
            CuModMatrix{Float32}(_batch_destination_host(33, 33); modulus = p),
            CuModMatrix{Float64}(_batch_destination_host(33, 33, 1); modulus = p),
        ]
        mixed_input_storage = [Array(parent(A)) for A in mixed_inputs]
        mixed_originals = [Array(parent(A))[1:rows(A), 1:cols(A)] for A in mixed_inputs]
        mixed_inverses =
            GPUFiniteFieldMatrices.inverse_batched(mixed_inputs; options = options)
        @test eltype(parent(mixed_inverses[1])) === Float32
        @test eltype(parent(mixed_inverses[2])) === Float64
        @test all(
            i -> Array(parent(mixed_inputs[i])) == mixed_input_storage[i],
            eachindex(mixed_inputs),
        )
        @test all(
            i -> _batch_host_inverse_identity(
                mixed_originals[i],
                Array(parent(mixed_inverses[i]))[1:33, 1:33],
                p,
            ),
            eachindex(mixed_inputs),
        )
    end
    return nothing
end
