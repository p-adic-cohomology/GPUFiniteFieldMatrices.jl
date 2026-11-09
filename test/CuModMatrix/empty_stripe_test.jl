@testset "Empty stripe product budgets on CPU" begin
    G = GPUFiniteFieldMatrices
    @test G.find_max_stripe_ops(Float32, 8_192) == 0
    @test G._stripe_safe_chunk(Float32, 8_192, nothing, nothing; empty_inner = true) == 1
    @test_throws ArgumentError G._stripe_safe_chunk(Float32, 8_192, nothing, nothing)
    @test_throws ArgumentError G._stripe_safe_chunk(
        Float32,
        8_192,
        nothing,
        1;
        empty_inner = true,
    )
    for bound in (1, 0, -1)
        @test_throws ArgumentError G._stripe_safe_chunk(
            Float32,
            8_192,
            bound,
            nothing;
            empty_inner = true,
        )
    end
    @test_throws ArgumentError G._stripe_safe_chunk(
        Float32,
        0,
        8_192,
        nothing;
        empty_inner = true,
    )
    C = fill(7, 2, 3)
    @test G._stripe_run!(C, zeros(Int, 2, 0), zeros(Int, 0, 3), 8_192, 1, nothing) === C
    @test C == zeros(Int, 2, 3)
    z = fill(7, 2)
    @test G._stripe_run!(z, zeros(Int, 2, 0), Int[], 8_192, 1, nothing) === z
    @test z == zeros(Int, 2)
end

@testset "Empty stripe device identity policy on CPU" begin
    # Exercise the pure identity predicate, not CUDA arrays or device execution.
    G = GPUFiniteFieldMatrices
    @test G._stripe_require_same_device(0, 0, 0) === nothing
    @test G._stripe_require_same_device(1, 1, 1) === nothing
    for devices in ((0, 1, 0), (0, 0, 1), (0, 1, 1))
        @test_throws ArgumentError G._stripe_require_same_device(devices...)
    end
end

function test_empty_stripe_products()
    G = GPUFiniteFieldMatrices
    N = 8_192
    A = CuModMatrix{Float32}(zeros(Int, 2, 0); modulus = N)
    B = CuModMatrix{Float32}(zeros(Int, 0, 3); modulus = N)
    x = CuModVector{Float32}(Int[]; modulus = N)
    C = zeros(CuModMatrix{Float32}, 2, 3; modulus = N)
    z = zeros(CuModVector{Float32}, 2; modulus = N)
    input_snapshots = (Array(parent(A)), Array(parent(B)), Array(parent(x)))

    for output in (C, z)
        fill!(parent(output), 7)
    end
    @test G.stripe_mul!(C, A, B) === C
    @test G.stripe_mul!(z, A, x) === z
    @test Array(C) == zeros(Int, 2, 3)
    @test Array(z) == zeros(Int, 2)
    @test all(iszero, Array(parent(C)))
    @test all(iszero, Array(parent(z)))
    @test Array(G.stripe_mul(A, B)) == zeros(Int, 2, 3)
    @test Array(G.stripe_mul(A, x)) == zeros(Int, 2)

    for (destination, right) in ((C, B), (z, x))
        for keywords in (
            (; max_ops = 1),
            (; max_ops = 0),
            (; max_ops = -1),
            (; input_bound = 1),
            (; input_bound = 0),
            (; modulus_override = 0),
            (; modulus_override = -1),
        )
            fill!(parent(destination), 7)
            snapshot = Array(parent(destination))
            @test_throws ArgumentError G.stripe_mul!(destination, A, right; keywords...)
            @test Array(parent(destination)) == snapshot
        end
    end
    wrong_C = zeros(CuModMatrix{Float32}, 2, 4; modulus = N)
    wrong_z = zeros(CuModVector{Float32}, 3; modulus = N)
    for (destination, right) in ((wrong_C, B), (wrong_z, x))
        fill!(parent(destination), 7)
        snapshot = Array(parent(destination))
        @test_throws DimensionMismatch G.stripe_mul!(destination, A, right)
        @test Array(parent(destination)) == snapshot
    end
    raw = CUDA.zeros(Float32, 0, 0)
    @test_throws ArgumentError G._stripe_mul!(raw, raw, raw; modulus = N)
    @test (Array(parent(A)), Array(parent(B)), Array(parent(x))) == input_snapshots
    test_empty_stripe_device_preflight()
    test_karatsuba_global_device_preflight()
end

function test_empty_stripe_device_preflight()
    G = GPUFiniteFieldMatrices
    devices = collect(CUDA.devices())
    if length(devices) < 2
        @info "Empty stripe mixed-device regressions require two CUDA devices."
        return nothing
    end
    N = 8_192
    build = function (device)
        CUDA.device!(device) do
            (
                A = CuModMatrix{Float32}(zeros(Int, 2, 0); modulus = N),
                B = CuModMatrix{Float32}(zeros(Int, 0, 3); modulus = N),
                x = CuModVector{Float32}(Int[]; modulus = N),
                C = zeros(CuModMatrix{Float32}, 2, 3; modulus = N),
                z = zeros(CuModVector{Float32}, 2; modulus = N),
            )
        end
    end
    local_arrays, other_arrays = build(devices[1]), build(devices[2])
    nested_logical = function (A)
        logical = view(parent(A), ntuple(i -> 1:size(A, i), ndims(A))...)
        view(logical, ntuple(i -> 1:size(A, i), ndims(A))...)
    end
    CUDA.device!(devices[1]) do
        for (left, right) in (
            (local_arrays, other_arrays),
            (other_arrays, local_arrays),
            (other_arrays, other_arrays),
        )
            for (destination, operand) in
                ((local_arrays.C, right.B), (local_arrays.z, right.x))
                fill!(parent(destination), 7)
                before = Array(parent(destination))
                metadata = (size(destination), G.modulus(destination), parent(destination))
                @test_throws ArgumentError G.stripe_mul!(destination, left.A, operand)
                @test Array(parent(destination)) == before
                @test (size(destination), G.modulus(destination), parent(destination)) ===
                      metadata
                @test CUDA.device() == devices[1]
                @test_throws ArgumentError G._stripe_mul!(
                    nested_logical(destination),
                    nested_logical(left.A),
                    nested_logical(operand);
                    modulus = N,
                )
                @test Array(parent(destination)) == before
                @test CUDA.device() == devices[1]
            end
        end
    end
    return nothing
end

function test_karatsuba_global_device_preflight()
    G = GPUFiniteFieldMatrices
    devices = collect(CUDA.devices())
    if length(devices) < 2
        @info "Karatsuba mixed-device regressions require two CUDA devices."
        return nothing
    end

    @testset "Karatsuba-wide device preflight" begin
        entry_device = CUDA.device()

        function padded_plane(device, dims, modulus, seed)
            rows, cols = dims
            backing = fill(Float64(700 + seed), max(rows, 1) + 2, max(cols, 1) + 2)
            if rows > 0 && cols > 0
                backing[1:rows, 1:cols] .= Float64(Base.mod(seed, 9) + 1)
            end
            return CUDA.device!(device) do
                G.CuModMatrix{Float64}(
                    CUDA.CuArray(backing);
                    modulus,
                    mod = false,
                    new_size = dims,
                )
            end
        end

        function karatsuba_matrix(dims, low_device, high_device, plan_device, seed)
            low = padded_plane(low_device, dims, 10, seed)
            high = padded_plane(high_device, dims, 10, seed + 1)
            plan = padded_plane(plan_device, dims, 10, seed + 2)
            matrix = G.KaratsubaMatrix(
                low,
                high;
                low_modulus = 10,
                high_modulus = 10,
                modulus = 100,
            )
            matrix.plan = plan
            return matrix
        end

        function plane_snapshot(plane)
            storage = parent(plane)
            return CUDA.device!(CUDA.device(storage)) do
                Array(storage)
            end
        end

        function matrix_snapshot(matrix)
            return map(
                plane_snapshot,
                (G.low_part(matrix), G.high_part(matrix), matrix.plan),
            )
        end

        function dirty_padding_values(plane)
            storage = plane_snapshot(plane)
            rows, cols = size(plane)
            logical = falses(size(storage))
            logical[1:rows, 1:cols] .= true
            return storage[.!logical]
        end

        # The empty case has a nonempty output and reaches recombination. The
        # nonempty case also covers staging; this does not assume that the
        # guarded staging kernel writes plans for an empty inner dimension.
        for (inner, case_name) in ((0, "empty inner"), (2, "nonempty inner"))
            @testset "$case_name ($inner)" begin
                # Each product group is device-homogeneous, but the groups
                # jointly use two devices.
                A = karatsuba_matrix((2, inner), devices[1], devices[1], devices[2], 10)
                B = karatsuba_matrix((inner, 3), devices[1], devices[1], devices[2], 20)
                C = karatsuba_matrix((2, 3), devices[1], devices[2], devices[1], 30)
                @test G.modulus(A) == G.modulus(B) == G.modulus(C) == 100

                data = G._karatsuba_prepare_gemm_views(A, B, C)
                @test (
                    CUDA.device(G._stripe_storage_root(data.c_low)),
                    CUDA.device(G._stripe_storage_root(data.a_low)),
                    CUDA.device(G._stripe_storage_root(data.b_low)),
                ) == (devices[1], devices[1], devices[1])
                @test (
                    CUDA.device(G._stripe_storage_root(data.c_high)),
                    CUDA.device(G._stripe_storage_root(data.a_plan)),
                    CUDA.device(G._stripe_storage_root(data.b_plan)),
                ) == (devices[2], devices[2], devices[2])
                @test (
                    CUDA.device(G._stripe_storage_root(data.c_plan)),
                    CUDA.device(G._stripe_storage_root(data.a_high)),
                    CUDA.device(G._stripe_storage_root(data.b_high)),
                ) == (devices[1], devices[1], devices[1])
                products = G._karatsuba_matmul_product_plan(
                    G.low_modulus(A),
                    G.high_modulus(A),
                    G.low_modulus(B),
                    G.high_modulus(B),
                    G.low_modulus(C),
                    G.high_modulus(C),
                )
                @test G._karatsuba_preflight_stripe_product(
                    data.c_low,
                    data.a_low,
                    data.b_low,
                    products.low,
                ) > 0
                @test G._karatsuba_preflight_stripe_product(
                    data.c_high,
                    data.a_plan,
                    data.b_plan,
                    products.middle,
                ) > 0
                @test G._karatsuba_preflight_stripe_product(
                    data.c_plan,
                    data.a_high,
                    data.b_high,
                    products.high,
                ) > 0

                matrices = (A, B, C)
                planes = Tuple(
                    plane for matrix in matrices for
                    plane in (G.low_part(matrix), G.high_part(matrix), matrix.plan)
                )
                before = map(matrix_snapshot, matrices)
                @test all(
                    plane -> all(value -> !iszero(value), plane_snapshot(plane)),
                    planes,
                )
                @test all(plane -> all(>(10.0), dirty_padding_values(plane)), planes)

                caught = try
                    LinearAlgebra.mul!(C, A, B)
                    nothing
                catch error
                    error
                end
                @test caught isa ArgumentError
                @test caught isa ArgumentError &&
                      occursin("same CUDA device", sprint(showerror, caught))
                @test map(matrix_snapshot, matrices) == before
                @test CUDA.device() == entry_device
            end
        end
    end
    return nothing
end
