"""
    PLUQOptions

Configuration for the GPU PLUQ path.

Fields:
- `blocksize`: recursive panel width.
- `basecase`: switch to basecase kernel when segment length is small.
- `pivot_policy`: pivot policy tag (`:first_nonzero` currently).
- `lazy_q`: enable lazy permutation-vector composition inside basecase.
- `nftb`: tunable tiny-kernel thread grouping factor (32 * `nftb`, clamped).
- `pivot_warp_kernel`: warp pivot reduction (`:ballot` or `:shfl`).
- `trsm_mode`: triangular-solve kernel (`:auto`, `:panel`, or `:warp`).
- `trsm_warp_threshold`: largest panel depth selected by automatic warp TRSM.
- `schur_tile`: Schur-update tile width (`8`, `16`, or `32`).
- `schur_transpose_u`: transpose the shared-memory U tile.
- `mod_backend`: modular multiplication preference (`:auto`, `:baseline`, or
  `:barrett`); currently retained for API compatibility while kernels select a
  safe backend from the modulus.
- `inverse_strategy`: square inverse algorithm (`:pluq` default, `:augmented`
  reference path). A wide right inverse first tries the leading square block
  through the PLUQ path, then falls back to augmented rank-revealing
  elimination if that block is singular.
- `autotune`: choose size-dependent PLUQ parameters at the API boundary.
- `batch_streams`: number of CUDA streams used by non-tiny batch fallbacks.
- `check_prime`: validate the finite-field precondition on entry.

The constructor throws `ArgumentError` for unsupported symbols, tiles, or
non-positive size parameters.
"""
struct PLUQOptions
    blocksize::Int
    basecase::Int
    pivot_policy::Symbol
    lazy_q::Bool
    nftb::Int
    pivot_warp_kernel::Symbol
    trsm_mode::Symbol
    trsm_warp_threshold::Int
    schur_tile::Int
    schur_transpose_u::Bool
    mod_backend::Symbol
    inverse_strategy::Symbol
    autotune::Bool
    batch_streams::Int
    check_prime::Bool
end

function PLUQOptions(
    options::PLUQOptions;
    blocksize::Int = options.blocksize,
    basecase::Int = options.basecase,
    pivot_policy::Symbol = options.pivot_policy,
    lazy_q::Bool = options.lazy_q,
    nftb::Int = options.nftb,
    pivot_warp_kernel::Symbol = options.pivot_warp_kernel,
    trsm_mode::Symbol = options.trsm_mode,
    trsm_warp_threshold::Int = options.trsm_warp_threshold,
    schur_tile::Int = options.schur_tile,
    schur_transpose_u::Bool = options.schur_transpose_u,
    mod_backend::Symbol = options.mod_backend,
    inverse_strategy::Symbol = options.inverse_strategy,
    autotune::Bool = options.autotune,
    batch_streams::Int = options.batch_streams,
    check_prime::Bool = options.check_prime,
)
    return PLUQOptions(;
        blocksize,
        basecase,
        pivot_policy,
        lazy_q,
        nftb,
        pivot_warp_kernel,
        trsm_mode,
        trsm_warp_threshold,
        schur_tile,
        schur_transpose_u,
        mod_backend,
        inverse_strategy,
        autotune,
        batch_streams,
        check_prime,
    )
end

"""
    PLUQOptions(; blocksize=64, basecase=32, pivot_policy=:first_nonzero, lazy_q=true, nftb=8, check_prime=false)

Construct `PLUQOptions` with validated positive block sizes.
"""
function PLUQOptions(;
    blocksize::Int = 64,
    basecase::Int = 32,
    pivot_policy::Symbol = :first_nonzero,
    lazy_q::Bool = true,
    nftb::Int = 8,
    pivot_warp_kernel::Symbol = :ballot,
    trsm_mode::Symbol = :auto,
    trsm_warp_threshold::Int = 32,
    schur_tile::Int = 16,
    schur_transpose_u::Bool = false,
    mod_backend::Symbol = :auto,
    inverse_strategy::Symbol = :pluq,
    autotune::Bool = false,
    batch_streams::Int = 1,
    check_prime::Bool = false,
)
    if blocksize < 1 ||
       basecase < 1 ||
       nftb < 1 ||
       trsm_warp_threshold < 1 ||
       batch_streams < 1
        throw(
            ArgumentError(
                "blocksize, basecase, nftb, trsm_warp_threshold, and batch_streams must be positive",
            ),
        )
    end
    if !(pivot_warp_kernel in (:ballot, :shfl))
        throw(ArgumentError("pivot_warp_kernel must be :ballot or :shfl"))
    end
    if !(trsm_mode in (:auto, :panel, :warp))
        throw(ArgumentError("trsm_mode must be :auto, :panel, or :warp"))
    end
    if !(schur_tile in (8, 16, 32))
        throw(ArgumentError("schur_tile must be one of 8, 16, or 32"))
    end
    if !(mod_backend in (:auto, :baseline, :barrett))
        throw(ArgumentError("mod_backend must be :auto, :baseline, or :barrett"))
    end
    if !(inverse_strategy in (:augmented, :pluq))
        throw(ArgumentError("inverse_strategy must be :augmented or :pluq"))
    end
    return PLUQOptions(
        blocksize,
        basecase,
        pivot_policy,
        lazy_q,
        nftb,
        pivot_warp_kernel,
        trsm_mode,
        trsm_warp_threshold,
        schur_tile,
        schur_transpose_u,
        mod_backend,
        inverse_strategy,
        autotune,
        batch_streams,
        check_prime,
    )
end

"""
    PLUQFactorization{T}

Result container for GPU PLUQ:
- `LU`: packed in-place factor storage
- `p`: row permutation vector
- `q`: column permutation vector
- `rank`: computed rank over `mod N`
"""
struct PLUQFactorization{T}
    LU::CuModMatrix{T}
    p::Vector{Int}
    q::Vector{Int}
    rank::Int
end
