using CUDA
using Random
using Statistics
using GPUFiniteFieldMatrices
using Nemo

const GFFM = GPUFiniteFieldMatrices

struct MatrixSpec
    elem_type::DataType
    p::Int
    rows::Int
    cols::Int
end

function random_matrix_from_spec(spec::MatrixSpec; rng::AbstractRNG = Random.default_rng())
    A = rand(rng, 0:(spec.p-1), spec.rows, spec.cols)
    A_typed = Matrix{spec.elem_type}(A)
    return CuModMatrix(A_typed, spec.p; elem_type = spec.elem_type)
end

function _append_time!(times::Dict{Symbol,Vector{Float64}}, key::Symbol, value::Float64)
    push!(get!(times, key, Float64[]), value)
    return nothing
end

function _append_time!(times::Dict{Symbol,Vector{Float64}}, key::Symbol, value::Real)
    push!(get!(times, key, Float64[]), Float64(value))
    return nothing
end

function _timed_find_pivot(
    d_A,
    A_rows,
    row,
    col,
    Perm_cols,
    Perm_col_idx,
    times::Dict{Symbol,Vector{Float64}},
)
    t_findmax = CUDA.@elapsed begin
        col_view = @view d_A[row:A_rows, col]
        pivot_val, pivot_idx = findmax(col_view)
    end
    _append_time!(times, :pivot_findmax, t_findmax)

    col_view = @view d_A[row:A_rows, col]
    pivot_val, pivot_idx = findmax(col_view)

    if pivot_val == 0
        t_swap_cols = CUDA.@elapsed begin
            @cuda blocks=cld(A_rows, GFFM.TILE_WIDTH) threads=GFFM.TILE_WIDTH GFFM.swap_cols(
                d_A,
                col,
                Perm_col_idx,
            )
        end
        _append_time!(times, :pivot_swap_cols, t_swap_cols)

        t_push_perm = CUDA.@elapsed begin
            push!(Perm_cols, (col, Perm_col_idx))
        end
        _append_time!(times, :pivot_push_perm_cols, t_push_perm)

        return -1, -1
    end

    return pivot_val, pivot_idx
end

function pluq_gpu_kernel_timed(A::CuModMatrix; debug::Bool = false)
    times = Dict{Symbol,Vector{Float64}}()

    function _print_plup_debug(stage)
        if debug
            println("Stage: $stage")
            println("Perm_col_idx: ", Perm_col_idx)
            println("Perm_cols: ", Perm_cols)
            println("Perm_rows: ", Perm_rows)
            println("row: ", row)
            println("col: ", col)
            println()
            println("d_A:")
            display(@view d_A[1:rows, 1:cols])
            println("d_L:")
            display(@view d_L[1:rows, 1:rows])
            println()
        end
    end

    t_init = CUDA.@elapsed begin
        N = A.N
        A_padded_rows = size(A.data, 1)
        d_A = copy(A.data)
        d_L = CUDA.zeros(eltype(A.data), (A_padded_rows, A_padded_rows))
        Perm_rows = Array{Tuple{Int,Int}}(undef, 0)
        Perm_cols = Array{Tuple{Int,Int}}(undef, 0)
    end
    _append_time!(times, :init, t_init)

    N = A.N
    A_padded_rows = size(A.data, 1)
    d_A = copy(A.data)
    d_L = CUDA.zeros(eltype(A.data), (A_padded_rows, A_padded_rows))
    Perm_rows = Array{Tuple{Int,Int}}(undef, 0)
    Perm_cols = Array{Tuple{Int,Int}}(undef, 0)

    rows, cols = size(A.data)
    rows -= GFFM.TILE_WIDTH
    cols -= GFFM.TILE_WIDTH

    row = 1
    col = 1
    Perm_col_idx = cols

    while row <= rows && col <= cols
        t_iter_total = CUDA.@elapsed begin
            pivot_val, pivot_idx = -1, -1

            _print_plup_debug("Iteration $row, $col start")

            t_find_pivot_col = CUDA.@elapsed begin
                while true
                    pivot_val, pivot_idx = _timed_find_pivot(
                        d_A,
                        rows,
                        row,
                        col,
                        Perm_cols,
                        Perm_col_idx,
                        times,
                    )
                    if pivot_val > 0
                        break
                    end
                    col += 1
                    if col > cols
                        break
                    end
                end
            end
            _append_time!(times, :find_pivot_col, t_find_pivot_col)

            if col <= cols
                t_mod_inverse = CUDA.@elapsed begin
                    pivot_val_inv = GFFM.mod_inv(pivot_val, N)
                end
                _append_time!(times, :mod_inverse, t_mod_inverse)
                pivot_val_inv = GFFM.mod_inv(pivot_val, N)

                t_swap_and_mod = CUDA.@elapsed begin
                    GFFM.swap_and_mod(
                        d_A,
                        d_L,
                        row,
                        pivot_idx + row - 1,
                        pivot_val_inv,
                        rows,
                        cols,
                        N,
                        Perm_rows,
                    )
                end
                _append_time!(times, :swap_and_mod, t_swap_and_mod)

                _print_plup_debug("Iteration $row, $col swapped and modded")

                t_move_zero = CUDA.@elapsed begin
                    GFFM.move_and_zero_out(
                        d_A,
                        d_L,
                        rows,
                        row,
                        col,
                        pivot_val_inv,
                        pivot_val,
                        N,
                    )
                end
                _append_time!(times, :move_and_zero_out, t_move_zero)

                _print_plup_debug("Iteration $row, $col moved and zeroed out")

                t_update_sub = CUDA.@elapsed begin
                    @cuda blocks=cld(cols - col + 1, GFFM.TILE_WIDTH) threads=GFFM.TILE_WIDTH shmem=GFFM.TILE_WIDTH*sizeof(
                        GFFM.DEFAULT_TYPE,
                    ) GFFM.update_sub_matrix_kernel(d_A, d_L, row, col, N, rows)
                end
                _append_time!(times, :update_sub_matrix, t_update_sub)

                _print_plup_debug("Iteration $row, $col updated sub matrix")

                t_sync = CUDA.@elapsed begin
                    CUDA.synchronize()
                end
                _append_time!(times, :synchronize, t_sync)

                row += 1
                col += 1
            end
        end
        _append_time!(times, :iteration_total, t_iter_total)

        if col > cols
            break
        end
    end

    t_finalize = CUDA.@elapsed begin
        U = CuModMatrix(d_A, N; new_size = (rows, cols))
        L = CuModMatrix(d_L, N; new_size = (rows, rows))
    end
    _append_time!(times, :finalize, t_finalize)

    U = CuModMatrix(d_A, N; new_size = (rows, cols))
    L = CuModMatrix(d_L, N; new_size = (rows, rows))

    return U, L, Perm_rows, Perm_cols, times
end

function summarize_times(times::Dict{Symbol,Vector{Float64}})
    summary = Dict{Symbol,NamedTuple}()
    for (op, vals) in times
        if !isempty(vals)
            q = quantile(vals, [0.25, 0.5, 0.75])
            summary[op] =
                (n = length(vals), mean = mean(vals), median = q[2], q1 = q[1], q3 = q[3])
        end
    end
    return summary
end

function merge_time_dicts!(
    acc::Dict{Symbol,Vector{Float64}},
    src::Dict{Symbol,Vector{Float64}},
)
    for (k, v) in src
        append!(get!(acc, k, Float64[]), v)
    end
    return acc
end

function is_invertible_with_inverse_timed(A::CuModMatrix; debug::Bool = false)
    if debug
        println("A")
        display(A)
    end

    CUDA.@time begin
        U, L, P, Q, pluq_times = pluq_gpu_kernel_timed(A, debug = debug)

        CUDA.@time begin
            invertible = GFFM._is_invertible(U)
            if !invertible
                return (
                    invertible = false,
                    pluq_times = summarize_times(pluq_times),
                    pluq_raw = pluq_times,
                )
            end
        end

        CUDA.@time begin
            U_inv = upper_triangular_inverse_no_copy(U; debug = debug)
        end

        CUDA.@time begin
            L_inv = lower_triangular_inverse_no_copy(L; debug = debug)
        end

        if debug
            println("L")
            display(L)
            println("U")
            display(U)
            println("L_inv")
            display(L_inv)
            println("U_inv")
            display(U_inv)
        end

        function _compute_A_inv(U_inv, L_inv, P, Q)
            CUDA.@time begin
                GFFM.apply_col_inv_perm!(P, L_inv)
            end

            CUDA.@time begin
                GFFM.apply_row_inv_perm!(Q, U_inv)
            end

            CUDA.@time begin
                A_inv = U_inv * L_inv
                return A_inv
            end
        end

        A_inv = _compute_A_inv(U_inv, L_inv, P, Q)

        if debug
            println("P")
            display(P)
            println("Q")
            display(Q)
            println("A_inv")
            display(A_inv)
            println("A * A_inv")
            display(A * A_inv)
        end

        return (
            invertible = true,
            pluq_times = summarize_times(pluq_times),
            pluq_raw = pluq_times,
        )
    end
end

function nemo_is_invertible_with_inverse_timed(A::CuModMatrix)

    NemoMod = getfield(Main, :Nemo)
    A_cpu = Array(A)
    nrows, ncols = size(A_cpu)
    R = NemoMod.GF(A.N)
    A_nemo = NemoMod.matrix(R, [R(x) for x in A_cpu])

    elapsed = @elapsed begin
        invertible, _ = NemoMod.is_invertible_with_inverse(A_nemo, side = :right)
    end
    invertible, _ = NemoMod.is_invertible_with_inverse(A_nemo, side = :right)

    return (elapsed = elapsed, invertible = invertible, size = (nrows, ncols))
end

function run_is_invertible_with_inverse_experiment(
    specs::Vector{MatrixSpec};
    samples_per_spec::Int = 1,
    seed::Int = 1234,
    debug::Bool = false,
    time_nemo::Bool = true,
)
    rng = Random.MersenneTwister(seed)
    results = Vector{NamedTuple}()
    all_pluq_times = Dict{Symbol,Vector{Float64}}()
    nemo_times = Float64[]

    for spec in specs
        for sample_id = 1:samples_per_spec
            A = random_matrix_from_spec(spec; rng = rng)
            gpu_elapsed =
                @elapsed metadata = is_invertible_with_inverse_timed(A; debug = debug)
            merge_time_dicts!(all_pluq_times, metadata.pluq_raw)
            nemo_result = nothing

            if time_nemo
                nemo_result = nemo_is_invertible_with_inverse_timed(A)
                push!(nemo_times, nemo_result.elapsed)
            end

            push!(
                results,
                (
                    spec = spec,
                    sample_id = sample_id,
                    invertible = metadata.invertible,
                    gpu_elapsed = gpu_elapsed,
                    pluq_summary = metadata.pluq_times,
                    nemo = nemo_result,
                ),
            )
        end
    end

    nemo_summary =
        isempty(nemo_times) ? nothing :
        (
            n = length(nemo_times),
            mean = mean(nemo_times),
            median = median(nemo_times),
            q1 = quantile(nemo_times, 0.25),
            q3 = quantile(nemo_times, 0.75),
        )

    return (
        results = results,
        pluq_summary = summarize_times(all_pluq_times),
        pluq_raw = all_pluq_times,
        nemo_summary = nemo_summary,
        nemo_raw = nemo_times,
    )
end

function _print_summary_times(summary::Dict{Symbol,NamedTuple}, title::String)
    println(title)
    for op in sort!(collect(keys(summary)); by = string)
        s = summary[op]
        println(
            "  $(op): n=$(s.n), mean=$(s.mean), median=$(s.median), q1=$(s.q1), q3=$(s.q3)",
        )
    end
    println()
    return nothing
end

function _spec_string(spec::MatrixSpec)
    return "$(spec.rows)x$(spec.cols), p=$(spec.p), type=$(spec.elem_type)"
end

function _print_per_sample_backend_times(results::Vector{NamedTuple})
    println("Per-sample backend timings (seconds)")
    for r in results
        spec_str = _spec_string(r.spec)
        gpu_text = "GPU=$(r.gpu_elapsed)"
        nemo_text = r.nemo === nothing ? "CPU Nemo=n/a" : "CPU Nemo=$(r.nemo.elapsed)"
        println(
            "  sample=$(r.sample_id), matrix=[$(spec_str)], $(gpu_text), $(nemo_text), invertible=$(r.invertible)",
        )
    end
    println()
    return nothing
end

function _ensure_nemo_loaded(time_nemo::Bool)
    if !time_nemo
        return false
    end
    if isdefined(Main, :Nemo)
        return true
    end
    try
        @eval import Nemo
        return true
    catch
        return false
    end
end

function run_default_is_invertible_with_inverse_experiment(;
    samples_per_spec::Int = 1,
    seed::Int = 1234,
    debug::Bool = false,
    time_nemo::Bool = true,
)
    specs = MatrixSpec[
        MatrixSpec(Int64, 101, 64, 64),
        MatrixSpec(Int64, 101, 96, 96),
        MatrixSpec(Int64, 101, 128, 128),
    ]

    effective_time_nemo = _ensure_nemo_loaded(time_nemo)

    out = run_is_invertible_with_inverse_experiment(
        specs;
        samples_per_spec = samples_per_spec,
        seed = seed,
        debug = debug,
        time_nemo = effective_time_nemo,
    )

    _print_per_sample_backend_times(out.results)
    _print_summary_times(out.pluq_summary, "PLUQ operation times (seconds)")
    if out.nemo_summary !== nothing
        s = out.nemo_summary
        println("CPU Nemo inverse times (seconds)")
        println("  n=$(s.n), mean=$(s.mean), median=$(s.median), q1=$(s.q1), q3=$(s.q3)")
        println()
    end

    return out
end

function run_include_experiment_succinct()
    specs = [
        MatrixSpec(Int, 11, 1000, 1000),
        MatrixSpec(Int, 11, 1000, 2000),
        MatrixSpec(Int, 11, 1000, 3000),
        MatrixSpec(Int, 11, 1000, 4000),
        MatrixSpec(Int, 11, 1000, 5000),
        MatrixSpec(Int, 11, 1000, 6000),
        MatrixSpec(Int, 11, 1000, 7000),
        MatrixSpec(Int, 11, 1000, 8000),
        MatrixSpec(Int, 11, 1000, 9000),
        MatrixSpec(Int, 11, 1000, 10000),
        MatrixSpec(Int, 11, 2000, 2000),
        MatrixSpec(Int, 11, 3000, 3000),
        MatrixSpec(Int, 11, 4000, 4000),
        MatrixSpec(Int, 11, 5000, 5000),
        MatrixSpec(Int, 11, 6000, 6000),
        MatrixSpec(Int, 11, 7000, 7000),
        MatrixSpec(Int, 11, 8000, 8000),
        MatrixSpec(Int, 11, 9000, 9000),
        MatrixSpec(Int, 11, 10000, 10000),
    ]
    effective_time_nemo = _ensure_nemo_loaded(true)
    out = run_is_invertible_with_inverse_experiment(
        specs;
        samples_per_spec = 1,
        seed = 1234,
        debug = false,
        time_nemo = effective_time_nemo,
    )
    println("samples=$(length(out.results))")
    _print_per_sample_backend_times(out.results)
    _print_summary_times(out.pluq_summary, "PLUQ operation times (seconds)")
    if out.nemo_summary !== nothing
        s = out.nemo_summary
        println("CPU Nemo inverse times (seconds)")
        println("  n=$(s.n), mean=$(s.mean), median=$(s.median), q1=$(s.q1), q3=$(s.q3)")
        println()
    end
    return out
end

run_include_experiment_succinct()
nothing
