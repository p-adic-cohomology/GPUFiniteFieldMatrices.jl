# CuModMatrix Inverse Algorithms

This directory contains GPU implementations for PLUQ factorization and matrix
inversion over finite fields.

All algorithms in this directory assume that the modulus `N` is prime. Pass
`PLUQOptions(check_prime=true)` to public entry points to validate that
precondition on the host before GPU kernels launch. The current CUDA kernels pass
`N` as `Int32`, so inverse routines reject moduli larger than `typemax(Int32)`.

## Source Papers

- Jingen Xiang, Huangdong Meng, and Ashraf Aboulnaga, "Scalable Matrix
  Inversion Using MapReduce", HPDC 2014 ([local PDF](HPDC.pdf)).
- Ahmad Abdelfattah, Azzam Haidar, Stanimire Tomov, and Jack Dongarra,
  "Factorization and Inversion of a Million Matrices using GPUs: Challenges and
  Countermeasures", ICCS 2017 ([local PDF](ICCS.pdf)).

## Relationship to the Papers

- **HPDC block LU is the main large-matrix template.** Our recursive PLUQ path
  factors a diagonal panel, computes the off-diagonal blocks with left and
  right triangular solves, applies the Schur update `A22 -= L21*U12`, and
  recurses. It then forms the inverse from triangular inverses and
  permutations. We extend the paper's row-pivoted real LU to row-and-column
  pivoted PLUQ over a finite field. We do not implement its MapReduce/Hadoop
  pipeline, distributed storage, or I/O optimizations.
- **ICCS supplies the tiny-batch workload and fused-augmentation idea.** Our
  4/8/16/32 batch API assigns one matrix to each CUDA block, and the tiny
  inverse keeps `[A I]` in shared memory inside one kernel. The current kernel
  is a correctness baseline: one lane performs the elimination. It does not
  yet implement the paper's cooperative 1D register kernel, warp-shuffle pivot
  search, tunable multiple matrices per block, or delayed row swaps. The
  general augmented inverse also uses `[A I]`, but launches multiple kernels
  per pivot and therefore is not the paper's fused tiny-matrix design.

## Algorithms Implemented

- Blocked PLUQ factorization: adapts the HPDC block-recursive LU structure to
  finite-field PLUQ by factoring a panel, applying row and column permutations,
  solving the off-diagonal triangular systems, updating the Schur complement on
  the GPU, and recursing on the trailing block.
- Square inverse: defaults to `inverse_strategy = :pluq`, which routes through
  PLUQ factorization, triangular inverses, modular matrix multiplication, and
  permutation application. `inverse_strategy = :augmented` is kept as a
  reference baseline using GPU Gauss-Jordan elimination over `[A I]`.
- Triangular-inverse leaf kernels write into caller-provided storage through
  internal destination-first helpers. The qualified
  `upper_triangular_inverse_no_copy` and `lower_triangular_inverse_no_copy`
  entry points allocate their result buffers; recursive paths keep the source
  factors separate from those destinations.
- Rectangular one-sided inverses: use rank-revealing row and column pivoting over
  augmented systems to construct right inverses for full row-rank matrices, with
  left inverses handled through transposition. The square `inverse_strategy`
  option does not change these rectangular algorithms.
- Tiny batched PLUQ and inverse kernels: specialize the ICCS-style workload of
  many small matrices for fixed square sizes 4, 8, 16, and 32.

## Large-Matrix CUDA Optimizations

- A panel is factored by one cooperative CUDA block. Pivot searches, global
  row and column swaps, multiplier scaling, and rank-one panel updates remain
  on the device; the host reads one rank value per panel instead of one pivot
  value per column. The same kernel records permutations and diagonal inverses
  in device memory.
- Triangular panel solves delay modular reduction until the end of a dot
  product whenever the complete integer result fits exactly in the element
  type. Right solves reuse the diagonal inverses recorded during factorization.
- Safe Schur complements use cuBLAS GEMM for `A22 -= L21*U12`, followed by one
  canonical modular reduction per output. The tiled Barrett-reduction kernel
  remains the fallback when a complete panel dot product is not exactly
  representable.
- General modular products split the inner dimension into exact chunks and
  reduce between chunks. For IEEE floating-point storage the chunk bound uses
  the full integer significand (24 bits for `Float32`, 53 for `Float64`). This
  keeps triangular inversion and final inverse composition correct when the
  matrix dimension exceeds a single exact GEMM.
- Exact `Float32` paths reject CUDA `FAST_MATH`, because TF32 input truncation
  does not preserve finite-field representatives. CUDA `DEFAULT_MATH` and
  `PEDANTIC_MATH` use the required FP32 inputs.

The kernel choices follow the official [CUDA.jl kernel-programming
guide](https://cuda.juliagpu.org/stable/development/kernel/), its
[driver/occupancy documentation](https://cuda.juliagpu.org/stable/lib/cudadrv/),
and NVIDIA's [CUDA C++ Best Practices
Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/). In
particular, the implementation uses warp-multiple blocks, coalesced
column-major traversal where dependencies permit it, shared-memory reduction,
warp shuffle reduction, fewer host/device synchronization points, and cuBLAS
for the large matrix products.
