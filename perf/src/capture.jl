const DEFAULT_SAMPLES = 20
const DEFAULT_SECONDS = 2.0
const DEFAULT_WARMUPS = 3
const DEFAULT_MODULUS = 101

_type(c::CaseSpec) = c.dtype == "Float32" ? Float32 : Float64

function _git_output(repo, args...)
    cmd = `git -C $repo $(String.(args))`
    return strip(read(cmd, String))
end

function _repo_for(path)
    try
        return _git_output(path, "rev-parse", "--show-toplevel")
    catch
        return nothing
    end
end

function _file_tree_hash(root; paths = nothing)
    selected = if isnothing(paths)
        try
            split(_git_output(root, "ls-files", "--cached", "--others", "--exclude-standard"), '\n'; keepempty = false)
        catch
            String[]
        end
    else
        String.(paths)
    end
    ctx = SHA.SHA256_CTX()
    for rel in sort!(unique(selected))
        file = joinpath(root, rel)
        isfile(file) || continue
        SHA.update!(ctx, codeunits(rel))
        SHA.update!(ctx, UInt8[0])
        SHA.update!(ctx, read(file))
        SHA.update!(ctx, UInt8[0xff])
    end
    return bytes2hex(SHA.digest!(ctx))
end

function _revision(repo)
    isnothing(repo) && return "unknown"
    try
        return _git_output(repo, "rev-parse", "HEAD")
    catch
        return "unknown"
    end
end

function _working_tree_state(repo)
    isnothing(repo) && return (dirty = nothing, hash = "unknown", status = "unknown")
    status = try
        _git_output(repo, "status", "--porcelain", "--untracked-files=all")
    catch
        "unknown"
    end
    return (dirty = !isempty(status), hash = _file_tree_hash(repo), status = status)
end

function _package_versions()
    versions = Dict{String,String}()
    for (uuid, info) in Pkg.dependencies()
        name = something(info.name, string(uuid))
        name in ("GPUFiniteFieldMatrices", "GFMPerf") && continue
        versions[name] = isnothing(info.version) ? "stdlib-or-unversioned" : string(info.version)
    end
    return Dict(k => versions[k] for k in sort!(collect(keys(versions))))
end

function _safe_call(f)
    try
        value = f()
        return value === nothing ? "unavailable" : string(value)
    catch
        return "unavailable"
    end
end

function _metadata(repo, role, suite, measurement_policy)
    package_path = pathof(GPUFiniteFieldMatrices)
    harness_path = pkgdir(@__MODULE__)
    target_repo = _repo_for(dirname(dirname(package_path)))
    state = _working_tree_state(target_repo)
    manifest_path = Base.active_project() === nothing ? nothing : joinpath(dirname(Base.active_project()), "Manifest.toml")
    prefs = Dict{String,Any}(
        "JULIA_CUDA_MEMORY_POOL" => get(ENV, "JULIA_CUDA_MEMORY_POOL", "default"),
        "JULIA_CUDA_SOFT_MEMORY_LIMIT" => get(ENV, "JULIA_CUDA_SOFT_MEMORY_LIMIT", "default"),
        "JULIA_CUDA_HARD_MEMORY_LIMIT" => get(ENV, "JULIA_CUDA_HARD_MEMORY_LIMIT", "default"),
        "CUDA_VISIBLE_DEVICES" => get(ENV, "CUDA_VISIBLE_DEVICES", "default"),
        "effective_cuda_package_preferences" => Base.get_preferences(Base.PkgId(CUDA).uuid),
        "effective_target_package_preferences" => Base.get_preferences(Base.PkgId(GPUFiniteFieldMatrices).uuid),
        "preferences_file_sha256" => isfile(joinpath(dirname(something(Base.active_project(), "")), "LocalPreferences.toml")) ?
            bytes2hex(SHA.sha256(read(joinpath(dirname(Base.active_project()), "LocalPreferences.toml")))) : "none")
    launch = Dict{String,Any}(
        "julia_threads" => get(ENV, "JULIA_NUM_THREADS", "default"),
        "blas_threads" => _safe_call(() -> LinearAlgebra.BLAS.get_num_threads()),
        "cuda_stream" => "CUDA default stream; each timed call uses CUDA.@sync")
    gpu_ok = try
        CUDA.functional()
    catch
        false
    end
    gpu_identity = gpu_ok ? _safe_call(() -> CUDA.name(CUDA.device())) : "unavailable"
    runtime = gpu_ok && isdefined(CUDA, :runtime_version) ? _safe_call(() -> getfield(CUDA, :runtime_version)()) : "unavailable"
    driver = gpu_ok && isdefined(CUDA, :driver_version) ? _safe_call(() -> getfield(CUDA, :driver_version)()) : "unavailable"
    return Dict{String,Any}(
        "created_at" => string(now(UTC)),
        "role" => String(role),
        "suite" => String(suite),
        "julia_version" => string(VERSION),
        "cuda_version" => string(Base.pkgversion(CUDA)),
        "benchmarktools_version" => string(Base.pkgversion(BenchmarkTools)),
        "cuda_runtime" => runtime,
        "cuda_driver" => driver,
        "gpu_identity" => gpu_identity,
        "cuda_functional" => gpu_ok,
        "cuda_preferences" => prefs,
        "launch_settings" => launch,
        "normalized_dependencies" => _package_versions(),
        "active_project" => something(Base.active_project(), "unknown"),
        "manifest_sha256" => !isnothing(manifest_path) && isfile(manifest_path) ? bytes2hex(SHA.sha256(read(manifest_path))) : "missing",
        "target_package_path" => something(package_path, "unknown"),
        "target_package_version" => string(Base.pkgversion(GPUFiniteFieldMatrices)),
        "target_commit" => _revision(target_repo),
        "target_dirty" => state.dirty,
        "target_source_hash_before" => state.hash,
        "target_status_before" => state.status,
        "harness_path" => harness_path,
        "harness_hash" => _file_tree_hash(harness_path; paths = [joinpath("src", f) for f in readdir(joinpath(harness_path, "src")) if endswith(f, ".jl")]),
        "measurement_policy" => measurement_policy)
end

function _cumod_type(T, D)
    constructor = D == 1 ? GPUFiniteFieldMatrices.CuModVector :
        D == 2 ? GPUFiniteFieldMatrices.CuModMatrix : GPUFiniteFieldMatrices.CuModArray
    return D in (1, 2) ? Core.apply_type(constructor, T) : Core.apply_type(constructor, T, D)
end

function _constructor_style(T, D)
    cmtype = _cumod_type(T, D)
    cpu_type = Array{T,D}
    positional = hasmethod(cmtype, Tuple{cpu_type,Int})
    keyword = hasmethod(cmtype, Tuple{cpu_type})
    positional && return :positional_modulus
    keyword && return :keyword_modulus
    return :unsupported
end

function _adapter_profile(T, D)
    cmtype = _cumod_type(T, D)
    style = _constructor_style(T, D)
    storage = hasfield(cmtype, :data) ? :data : :parent
    return (constructor_style = style, storage_field = storage)
end

function _cumod_constructor(T, D, modulus)
    cmtype = _cumod_type(T, D)
    style = _constructor_style(T, D)
    style == :positional_modulus && return data -> cmtype(data, Int(modulus))
    style == :keyword_modulus && return data -> cmtype(data; modulus = Int(modulus))
    throw(ArgumentError("no CuModArray{$T,$D} constructor accepts CPU arrays and this revision's modulus convention"))
end

_typed_cumod(T, data, modulus) = _cumod_constructor(T, ndims(data), modulus)(data)

function _storage_array(A)
    if hasfield(typeof(A), :data)
        data = getfield(A, :data)
        data isa AbstractArray || throw(ArgumentError("CuModArray storage field is not an array"))
        return data
    elseif applicable(parent, A)
        return parent(A)
    end
    throw(ArgumentError("no storage adapter is available for $(typeof(A))"))
end

function _cpu_matrix(rng, T, m, n, modulus)
    return T.(rand(rng, 0:(modulus - 1), m, n))
end

function _cpu_vector(rng, T, n, modulus)
    return T.(rand(rng, 0:(modulus - 1), n))
end

const BIGINT_ORACLE_PRODUCT_LIMIT = 250_000
const FLOAT64_EXACT_INTEGER_LIMIT = BigInt(2)^53

function _oracle_bigint_gemm(A, B, modulus)
    m, k = size(A)
    k2, n = size(B)
    k == k2 || throw(DimensionMismatch("oracle input dimensions do not match"))
    modulus > 0 || throw(ArgumentError("oracle modulus must be positive"))
    result = Matrix{Int}(undef, m, n)
    bigmodulus = BigInt(modulus)
    for j in 1:n, i in 1:m
        acc = BigInt(0)
        for q in 1:k
            acc += BigInt(A[i, q]) * BigInt(B[q, j])
        end
        result[i, j] = Int(mod(acc, bigmodulus))
    end
    return result
end

function _nonnegative_integer_max(A, name)
    eltype(A) in (Float32, Float64) || throw(ArgumentError("exact Float64 oracle requires Float32/Float64 fixture inputs"))
    all(x -> isfinite(x) && x >= 0 && isinteger(x), A) ||
        throw(ArgumentError("exact Float64 oracle requires finite nonnegative integer-valued $name"))
    maximum_value = maximum(A; init = zero(eltype(A)))
    exact_input_limit = eltype(A) === Float32 ? BigInt(2)^24 : FLOAT64_EXACT_INTEGER_LIMIT
    maximum_integer = BigInt(maximum_value)
    maximum_integer <= exact_input_limit ||
        throw(ArgumentError("exact Float64 oracle unsafe: $name values exceed exact $(eltype(A)) integer range"))
    return maximum_integer
end

function _oracle_bigint_dot(A, B, row, column, modulus)
    acc = BigInt(0)
    for q in axes(A, 2)
        acc += BigInt(A[row, q]) * BigInt(B[q, column])
    end
    return mod(acc, modulus)
end

function _oracle_float64_gemm(A, B, modulus)
    m, k = size(A)
    k2, n = size(B)
    k == k2 || throw(DimensionMismatch("oracle input dimensions do not match"))
    modulus > 0 || throw(ArgumentError("oracle modulus must be positive"))
    max_a = _nonnegative_integer_max(A, "left matrix")
    max_b = _nonnegative_integer_max(B, "right matrix")
    accumulator_bound = BigInt(k) * max_a * max_b
    accumulator_bound <= FLOAT64_EXACT_INTEGER_LIMIT ||
        throw(ArgumentError("exact Float64 oracle unsafe: dot-product bound $accumulator_bound exceeds 2^53"))

    a64 = Float64.(A)
    b64 = Float64.(B)
    products = Matrix{Float64}(undef, m, n)
    LinearAlgebra.mul!(products, a64, b64)
    result = Matrix{Int}(undef, m, n)
    float_bound = Float64(accumulator_bound)
    for index in eachindex(products)
        value = products[index]
        isfinite(value) && 0 <= value <= float_bound && isinteger(value) ||
            throw(ArgumentError("exact Float64 oracle returned an out-of-bound or non-integer result"))
        result[index] = Int(mod(round(Int64, value), Int64(modulus)))
    end

    bigmodulus = BigInt(modulus)
    rows = (1, cld(m, 2), m)
    columns = (1, cld(n, 2), n)
    checks = Set{Tuple{Int,Int}}((i, j) for i in rows for j in columns)
    for (i, j) in checks
        exact = _oracle_bigint_dot(A, B, i, j, bigmodulus)
        result[i, j] == exact || throw(ArgumentError("exact Float64 oracle failed its BigInt cross-check at ($i, $j)"))
    end
    return result
end

function _oracle_gemm(A, B, modulus)
    m, k = size(A)
    k2, n = size(B)
    k == k2 || throw(DimensionMismatch("oracle input dimensions do not match"))
    products = BigInt(m) * BigInt(k) * BigInt(n)
    if products <= BIGINT_ORACLE_PRODUCT_LIMIT
        return (values = _oracle_bigint_gemm(A, B, modulus), method = "BigInt modular dot products")
    end
    return (values = _oracle_float64_gemm(A, B, modulus),
        method = "bounded exact Float64 BLAS with BigInt corner and center cross-checks")
end

function _oracle_gemv(A, x, modulus)
    size(A, 2) == length(x) || throw(DimensionMismatch("oracle input dimensions do not match"))
    return [mod(sum(BigInt(A[i, q]) * BigInt(x[q]) for q in eachindex(x); init = BigInt(0)), BigInt(modulus)) for i in axes(A, 1)]
end

_host(A) = Array(A)

function _component(K, plane)
    accessor = plane == :low ? :low_part : :high_part
    if isdefined(GPUFiniteFieldMatrices, accessor)
        return getfield(GPUFiniteFieldMatrices, accessor)(K)
    end
    field = plane == :low ? :data1 : :data2
    hasfield(typeof(K), field) || throw(ArgumentError("Karatsuba component accessor $accessor is unavailable"))
    return getfield(K, field)
end

function _materialize(K, T)
    value = if isdefined(GPUFiniteFieldMatrices, :from_karatsuba)
        GPUFiniteFieldMatrices.from_karatsuba(T, K)
    elseif isdefined(GPUFiniteFieldMatrices, :KMatToMat)
        GPUFiniteFieldMatrices.KMatToMat(T, K)
    else
        K
    end
    return Array(value)
end

function _geometry(A)
    parent_array = _storage_array(A)
    return Dict{String,Any}(
        "logical_shape" => collect(size(A)),
        "physical_shape" => collect(size(parent_array)),
        "physical_strides" => collect(strides(parent_array)),
        "element_type" => string(eltype(A)))
end

function _unavailable(reason; status = "unsupported")
    return (status = String(status), reason = String(reason), runner = nothing,
        reset = () -> nothing, validate = () -> (false, String(reason)),
        geometry = Dict{String,Any}())
end

function _make_case(c::CaseSpec, role)
    c.target == "current-only" && !startswith(String(role), "current") &&
        return _unavailable("current-only case is not requested for this historical target")
    rng = MersenneTwister(c.seed)
    T = _type(c)
    N = c.modulus
    op = c.operation
    try
        if c.group in ("gemm", "gemm-padding", "gemm-scale")
            m, k, n = c.dims
            ah = _cpu_matrix(rng, T, m, k, N)
            bh = _cpu_matrix(rng, T, k, n, N)
            A = _typed_cumod(T, ah, N)
            B = _typed_cumod(T, bh, N)
            ch = zeros(T, m, n)
            C = _typed_cumod(T, ch, N)
            cparent = copy(_storage_array(C))
            result = Ref{Any}(nothing)
            oracle = _oracle_gemm(ah, bh, N)
            expected = oracle.values
            runner = op == "mul!" ? (() -> LinearAlgebra.mul!(C, A, B)) : (() -> (result[] = A * B))
            reset = op == "mul!" ? (() -> copyto!(_storage_array(C), cparent)) : (() -> nothing)
            output = op == "mul!" ? (() -> C) : (() -> result[])
            validate = () -> begin
                reset()
                runner()
                CUDA.synchronize()
                got = _host(output())
                ok = size(got) == size(expected) && all(BigInt(got[i]) == expected[i] for i in eachindex(expected))
                input_ok = _host(A) == ah && _host(B) == bh
                return ok && input_ok, ok ? (input_ok ? "$(oracle.method) passed" : "input preservation failed") : "$(oracle.method) mismatch"
            end
            return (status = "ready", reason = "", runner = runner, reset = reset,
                validate = validate, geometry = Dict("A" => _geometry(A), "B" => _geometry(B), "C" => _geometry(C)))
        elseif c.group == "gemv"
            m, n = c.dims
            ah = _cpu_matrix(rng, T, m, n, N)
            xh = _cpu_vector(rng, T, n, N)
            A = _typed_cumod(T, ah, N)
            x = _typed_cumod(T, xh, N)
            yh = zeros(T, m)
            y = _typed_cumod(T, yh, N)
            yparent = copy(_storage_array(y))
            result = Ref{Any}(nothing)
            expected = _oracle_gemv(ah, xh, N)
            runner = op == "mul!" ? (() -> LinearAlgebra.mul!(y, A, x)) : (() -> (result[] = A * x))
            reset = op == "mul!" ? (() -> copyto!(_storage_array(y), yparent)) : (() -> nothing)
            output = op == "mul!" ? (() -> y) : (() -> result[])
            validate = () -> begin
                reset()
                runner()
                CUDA.synchronize()
                got = _host(output())
                ok = length(got) == length(expected) && all(BigInt(got[i]) == expected[i] for i in eachindex(expected))
                input_ok = _host(A) == ah && _host(x) == xh
                return ok && input_ok, ok ? (input_ok ? "BigInt modular oracle passed" : "input preservation failed") : "BigInt modular oracle mismatch"
            end
            return (status = "ready", reason = "", runner = runner, reset = reset,
                validate = validate, geometry = Dict("A" => _geometry(A), "x" => _geometry(x), "y" => _geometry(y)))
        elseif c.group == "elementwise"
            m, n = c.dims
            ah = _cpu_matrix(rng, T, m, n, N)
            bh = _cpu_matrix(rng, T, m, n, N)
            A = _typed_cumod(T, ah, N)
            B = _typed_cumod(T, bh, N)
            ch = zeros(T, m, n)
            C = _typed_cumod(T, ch, N)
            cparent = copy(_storage_array(C))
            result = Ref{Any}(nothing)
            construct = _cumod_constructor(T, 2, N)
            runner = if op == "+"
                () -> (result[] = A + B)
            elseif op == "add!"
                isdefined(GPUFiniteFieldMatrices, :add!) || return _unavailable("add! is not defined by this package revision")
                () -> getfield(GPUFiniteFieldMatrices, :add!)(C, A, B)
            elseif op == ".*"
                () -> (result[] = A .* B)
            elseif op == "mul_elementwise!"
                isdefined(GPUFiniteFieldMatrices, :mul_elementwise!) || return _unavailable("mul_elementwise! is not defined by this package revision")
                () -> getfield(GPUFiniteFieldMatrices, :mul_elementwise!)(C, A, B)
            elseif op == "copy"
                () -> (result[] = copy(A))
            elseif op == "construct"
                () -> (result[] = construct(ah))
            else
                return _unavailable("unknown elementwise operation $op")
            end
            mutating = op in ("add!", "mul_elementwise!")
            reset = mutating ? (() -> copyto!(_storage_array(C), cparent)) : (() -> nothing)
            output = mutating ? (() -> C) : (() -> result[])
            expected = if op in ("+", "add!")
                mod.(BigInt.(ah) .+ BigInt.(bh), BigInt(N))
            elseif op in (".*", "mul_elementwise!")
                mod.(BigInt.(ah) .* BigInt.(bh), BigInt(N))
            else
                BigInt.(ah)
            end
            validate = () -> begin
                reset()
                runner()
                CUDA.synchronize()
                got = _host(output())
                ok = size(got) == size(expected) && all(BigInt(got[i]) == expected[i] for i in eachindex(expected))
                input_ok = _host(A) == ah && (op in ("+", "add!", ".*", "mul_elementwise!") ? _host(B) == bh : true)
                return ok && input_ok, ok ? (input_ok ? "elementwise oracle passed" : "input preservation failed") : "elementwise oracle mismatch"
            end
            return (status = "ready", reason = "", runner = runner, reset = reset,
                validate = validate, geometry = Dict("A" => _geometry(A), "B" => _geometry(B), "C" => _geometry(C)))
        elseif c.group in ("karatsuba-gemm", "karatsuba-gemv", "karatsuba32-current")
            return _make_karatsuba_case(c, role, rng)
        elseif c.group == "inverse-pluq"
            n, _ = c.dims
            ah = zeros(T, n, n)
            for i in 1:n
                ah[i, i] = i + 1
            end
            A = _typed_cumod(T, ah, N)
            result = Ref{Any}(nothing)
            runner = op == "inv" ? (() -> (result[] = inv(A))) : (() -> (result[] = GPUFiniteFieldMatrices.pluq(A)))
            validate = () -> begin
                runner()
                CUDA.synchronize()
                if op == "inv"
                    expected = zeros(BigInt, n, n)
                    for i in 1:n
                        expected[i, i] = invmod(BigInt(ah[i, i]), BigInt(N))
                    end
                    got = _host(result[])
                    ok = size(got) == size(expected) && all(BigInt(got[i]) == expected[i] for i in eachindex(expected))
                    return ok, ok ? "diagonal modular inverse oracle passed" : "inverse oracle mismatch"
                end
                F = result[]
                hasproperty(F, :L) && hasproperty(F, :U) || return false, "PLUQ result lacks L/U factors"
                got = _host(F.L * F.U)
                identity_p = hasproperty(F, :p) && collect(F.p) == collect(1:n)
                identity_q = hasproperty(F, :q) && collect(F.q) == collect(1:n)
                ok = identity_p && identity_q && got == ah
                return ok, ok ? "pivot-free PLUQ factorization oracle passed" : "PLUQ factorization mismatch"
            end
            return (status = "ready", reason = "", runner = runner, reset = () -> nothing,
                validate = validate, geometry = Dict("A" => _geometry(A)))
        end
        return _unavailable("unknown group $(c.group)")
    catch err
        status = err isa MethodError ? "unsupported" : "failed"
        return _unavailable("case construction failed: " * sprint(showerror, err); status = status)
    end
end

function _make_karatsuba_case(c::CaseSpec, role, rng)
    T = _type(c)
    N = c.modulus
    R = something(c.low_modulus, N)
    H = something(c.high_modulus, 1)
    op = c.operation
    M = N
    function split_k(data)
        source = _typed_cumod(T, data, N)
        if isdefined(GPUFiniteFieldMatrices, :to_karatsuba)
            return GPUFiniteFieldMatrices.to_karatsuba(T, source;
                low_modulus = R, high_modulus = H, modulus = M)
        elseif isdefined(GPUFiniteFieldMatrices, :KaratsubaMatrix)
            return GPUFiniteFieldMatrices.KaratsubaMatrix(T, source, R, H, M)
        end
        throw(ArgumentError("no compatible Karatsuba conversion adapter is available"))
    end
    function init_plan!(K)
        isdefined(GPUFiniteFieldMatrices, :initialize_plan!) ||
            throw(ArgumentError("initialize_plan! is unavailable"))
        getfield(GPUFiniteFieldMatrices, :initialize_plan!)(K)
    end
    if c.group == "karatsuba-gemv"
        m, n = c.dims
        ah = T.(rand(rng, 0:(R - 1), m, n))
        xh = T.(rand(rng, 0:(R - 1), n))
        A = split_k(ah)
        x = split_k(xh)
        yh = zeros(T, m)
        y = split_k(yh)
        foreach(init_plan!, (A, x, y))
        result = Ref{Any}(nothing)
        runner = op == "mul!" ? (() -> LinearAlgebra.mul!(y, A, x)) : (() -> (result[] = A * x))
        output = op == "mul!" ? (() -> y) : (() -> result[])
        expected = _oracle_gemv(ah, xh, N)
        validate = () -> begin
            runner()
            CUDA.synchronize()
            got = _materialize(output(), T)
            ok = length(got) == length(expected) && all(BigInt(got[i]) == expected[i] for i in eachindex(expected))
            return ok, ok ? "low-plane-only BigInt modular oracle passed" : "Karatsuba GEMV oracle mismatch"
        end
        return (status = "ready", reason = "", runner = runner, reset = () -> nothing,
            validate = validate, geometry = Dict("A" => _geometry(_component(A, :low)),
                "x" => _geometry(_component(x, :low)), "y" => _geometry(_component(y, :low))))
    end
    m, k, n = c.dims
    ah = T.(rand(rng, 0:(R - 1), m, k))
    bh = T.(rand(rng, 0:(R - 1), k, n))
    A = split_k(ah)
    B = split_k(bh)
    ch = zeros(T, m, n)
    C = split_k(ch)
    foreach(init_plan!, (A, B, C))
    result = Ref{Any}(nothing)
    runner = op == "mul!" ? (() -> LinearAlgebra.mul!(C, A, B)) : (() -> (result[] = A * B))
    output = op == "mul!" ? (() -> C) : (() -> result[])
    oracle = _oracle_gemm(ah, bh, N)
    expected = oracle.values
    validate = () -> begin
        runner()
        CUDA.synchronize()
        got = _materialize(output(), T)
        ok = size(got) == size(expected) && all(BigInt(got[i]) == expected[i] for i in eachindex(expected))
        input_ok = _materialize(A, T) == ah && _materialize(B, T) == bh
        return ok && input_ok, ok ? (input_ok ? "low-plane-only $(oracle.method) passed" : "Karatsuba input preservation failed") : "Karatsuba GEMM oracle mismatch"
    end
    reset = () -> nothing
    # Karatsuba plan and both component buffers are initialized once. Reset only output values before each mutating sample.
    if op == "mul!"
        low0 = copy(_storage_array(_component(C, :low)))
        high0 = copy(_storage_array(_component(C, :high)))
        reset = () -> begin
            copyto!(_storage_array(_component(C, :low)), low0)
            copyto!(_storage_array(_component(C, :high)), high0)
            CUDA.synchronize()
        end
    end
    return (status = "ready", reason = "", runner = runner, reset = reset,
        validate = validate, geometry = Dict("A_low" => _geometry(_component(A, :low)),
            "A_high" => _geometry(_component(A, :high)),
            "B_low" => _geometry(_component(B, :low)),
            "C_low" => _geometry(_component(C, :low))))
end

function _cpu_allocation_summary(trial)
    memory = trial.memory
    allocs = trial.allocs
    return (bytes = memory == typemax(Int) ? nothing : Int(memory),
        count = allocs == typemax(Int) ? nothing : Int(allocs))
end

function _cuda_allocation_summary(timed)
    stats = timed.gpu_memstats
    bytes = Int(timed.gpu_bytes)
    count = Int(stats.alloc_count)
    raw = Dict{String,Any}(
        "elapsed_seconds" => timed.time,
        "gpu_bytes" => bytes,
        "gpu_alloc_count" => count,
        "gpu_memtime" => timed.gpu_memtime)
    return (count = count, bytes = bytes, raw = raw)
end

function _cuda_allocations(runner, reset)
    reset()
    CUDA.synchronize()
    timed = CUDA.@timed runner()
    CUDA.synchronize()
    return _cuda_allocation_summary(timed)
end

function _display_name(c)
    case_id(c)
end

function _case_record(c, built, cuda_ok, role, samples, seconds, warmups)
    base = Dict{String,Any}("case_id" => _display_name(c), "spec" => Dict(
        "group" => c.group, "operation" => c.operation, "dtype" => c.dtype,
        "dims" => c.dims, "modulus" => c.modulus, "low_modulus" => c.low_modulus,
        "high_modulus" => c.high_modulus, "layout" => c.layout, "seed" => c.seed,
        "target" => c.target), "geometry" => built.geometry,
        "correctness" => nothing, "correctness_detail" => "not run",
        "times_ns" => Int[], "gctimes_ns" => Int[],
        "cpu_alloc_bytes" => nothing, "cpu_alloc_count" => nothing,
        "cpu_allocation_method" => "BenchmarkTools.Trial minimum across samples",
        "gpu_alloc_bytes" => nothing, "gpu_alloc_count" => nothing, "gpu_alloc_accounting" => nothing)
    if !cuda_ok
        merge!(base, Dict("status" => "unavailable", "detail" => "CUDA.functional() is false; no CPU fallback is measured"))
        return base
    elseif built.status != "ready"
        merge!(base, Dict("status" => built.status, "detail" => built.reason))
        return base
    end
    try
        ok, detail = built.validate()
        base["correctness"] = ok
        base["correctness_detail"] = detail
        if !ok
            base["status"] = "incorrect"
            base["detail"] = detail
            return base
        end
        for _ in 1:warmups
            built.reset()
            CUDA.@sync built.runner()
        end
        bench = BenchmarkTools.@benchmarkable CUDA.@sync($(built.runner)()) setup=($(built.reset)(); CUDA.synchronize()) evals=1
        trial = BenchmarkTools.run(bench; samples = samples, seconds = seconds, evals = 1, gctrial = true)
        base["times_ns"] = copy(trial.times)
        base["gctimes_ns"] = copy(trial.gctimes)
        cpu = _cpu_allocation_summary(trial)
        base["cpu_alloc_bytes"] = cpu.bytes
        base["cpu_alloc_count"] = cpu.count
        gpu = _cuda_allocations(built.runner, built.reset)
        base["gpu_alloc_bytes"] = gpu.bytes
        base["gpu_alloc_count"] = gpu.count
        base["gpu_alloc_accounting"] = gpu.raw
        base["actual_samples"] = length(trial.times)
        base["status"] = "ok"
        base["detail"] = "correctness oracle passed; public operation timed with CUDA synchronization"
    catch err
        base["status"] = err isa MethodError ? "unsupported" : "failed"
        base["detail"] = sprint(showerror, err)
        base["exception_type"] = string(typeof(err))
    end
    return base
end

function _write_csv(path, records)
    open(path, "w") do io
        write(io, csv_line(("case_id", "status", "correctness", "median_ns", "mad_ns",
            "p05_ns", "p95_ns", "samples", "cpu_alloc_bytes", "cpu_alloc_count",
            "gpu_alloc_bytes", "gpu_alloc_count", "detail")))
        for r in records
            stats = summarize_samples(r["times_ns"])
            write(io, csv_line((r["case_id"], r["status"], r["correctness"], stats["median"],
                stats["mad"], stats["p05"], stats["p95"], stats["n"], r["cpu_alloc_bytes"],
                r["cpu_alloc_count"], r["gpu_alloc_bytes"], r["gpu_alloc_count"], r["detail"])))
        end
    end
end

"""Capture a deterministic suite in the currently loaded package world.

Call from a fresh Julia process after activating that variant's benchmark environment.
"""
function capture(label; suite = :quick, groups = nothing, case_ids = nothing,
        role = "current", samples = DEFAULT_SAMPLES, seconds = DEFAULT_SECONDS,
        warmups = DEFAULT_WARMUPS, output_dir = joinpath(pwd(), "perf", "results"), overwrite = false)
    samples > 0 || throw(ArgumentError("samples must be positive"))
    seconds > 0 || throw(ArgumentError("seconds must be positive"))
    warmups >= 0 || throw(ArgumentError("warmups must be nonnegative"))
    label = String(label)
    occursin(r"^[A-Za-z0-9][A-Za-z0-9._-]*$", label) || throw(ArgumentError("label must use letters, digits, dot, underscore, or dash"))
    all_cases = suite == :quick ? quick_cases() : suite == :full ? full_cases() : throw(ArgumentError("suite must be :quick or :full"))
    if !isnothing(groups)
        wanted = Set(String.(groups))
        all_cases = filter(c -> c.group in wanted, all_cases)
    end
    if !isnothing(case_ids)
        wanted = Set(String.(case_ids))
        all_cases = filter(c -> case_id(c) in wanted, all_cases)
        found = Set(case_id.(all_cases))
        missing = setdiff(wanted, found)
        isempty(missing) || throw(ArgumentError("unknown case IDs: " * join(sort!(collect(missing)), ", ")))
    end
    isempty(all_cases) && throw(ArgumentError("selection contains no cases"))
    policy = Dict{String,Any}("samples_requested" => samples, "seconds_per_case" => seconds,
        "warmups_per_case" => warmups, "evals_per_sample" => 1,
        "timer" => "BenchmarkTools host wall time with CUDA.@sync",
        "setup" => "mutable output restore and CUDA synchronization outside timed sample")
    repo = _repo_for(dirname(dirname(pathof(GPUFiniteFieldMatrices))))
    before = _metadata(repo, role, suite, policy)
    cuda_ok = before["cuda_functional"]
    records = Dict{String,Any}[]
    for c in all_cases
        built = cuda_ok ? _make_case(c, role) : _unavailable("CUDA.functional() is false")
        push!(records, _case_record(c, built, cuda_ok, role, samples, seconds, warmups))
    end
    after_state = _working_tree_state(repo)
    valid = before["target_source_hash_before"] == after_state.hash
    metadata = copy(before)
    metadata["target_source_hash_after"] = after_state.hash
    metadata["source_changed_during_capture"] = !valid
    metadata["capture_valid"] = valid
    metadata["case_selection"] = [case_id(c) for c in all_cases]
    run = Dict{String,Any}("schema_version" => GFMPerfCore.RUN_SCHEMA,
        "run_label" => label, "run_dir" => "", "metadata" => metadata,
        "measurement_policy" => policy, "cases" => records)
    outroot = abspath(output_dir)
    mkpath(outroot)
    target = joinpath(outroot, label)
    GFMPerfCore.validate_output_path(target; overwrite = overwrite)
    tmp = joinpath(outroot, ".$(label).tmp-$(uuid4())")
    mkdir(tmp)
    try
        run["run_dir"] = target
        write_json(joinpath(tmp, "run.json"), run)
        write_json(joinpath(tmp, "metadata.json"), metadata)
        _write_csv(joinpath(tmp, "summary.csv"), records)
        if isfile(something(Base.active_project(), ""))
            cp(Base.active_project(), joinpath(tmp, "Project.toml"))
        end
        manifest = joinpath(dirname(something(Base.active_project(), "")), "Manifest.toml")
        isfile(manifest) && cp(manifest, joinpath(tmp, "Manifest.toml"))
        if ispath(target)
            rm(target; force = true, recursive = true)
        end
        mv(tmp, target)
    catch
        ispath(tmp) && rm(tmp; force = true, recursive = true)
        rethrow()
    end
    println("GFMPerf capture saved: $target")
    println("Cases: $(length(records)); CUDA functional: $cuda_ok; source snapshot valid: $valid")
    return target
end
