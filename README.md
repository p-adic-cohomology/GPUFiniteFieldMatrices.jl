# GPUFiniteFieldMatrices.jl

[![CI](https://github.com/UCSD-computational-number-theory/GPUFiniteFieldMatrices.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/UCSD-computational-number-theory/GPUFiniteFieldMatrices.jl/actions/workflows/CI.yml)

This package aims to implement fast finite field matrix operations on the GPU.

**This package is work-in-progress research software.** It may have rough edges. PRs and contributions are welcome.

## Current functionality

We currently implement:

- `CuModMatrix`, a `CuArray` wrapper which represents a matrix modulo `N`. `CuModMatrix` pads elements to 32 bits, and has various matrix operations implemented as GPU kernels. This includes:
  - matrix multiplication (for `N < ~2^26`)
  - if `N` is prime, RREF, LU decomposition, PLUQ decomposition, square
    inversion, and rectangular one-sided inverses. See
    [`src/CuModMatrix/inverse/README.md`](src/CuModMatrix/inverse/README.md) for
    the inverse algorithms and their modulus contract.
- `KaratsubaMatrix`, a matrix type implementing Karatsuba matrix multiplication. This also supports addition, subtraction, and other basic operations.

## Getting Started

#### Installing for the first time

First, install Julia 1.12. If you're new to Julia, you can find a tutorial for non-experts [here](https://jjgarzella.github.io/blog/how-to-install-julia/). In short:

- On Windows, install Ubuntu through WSL and use the Ubuntu terminal.
- On Mac, open Terminal and run `xcode-select --install`.
- On Linux, open your usual terminal.

Then install Julia with `curl -fsSL https://install.julialang.org | sh`, restart your terminal if necessary, and open a Julia REPL with `julia`.

This package uses CUDA. You'll need to install the CUDA driver, see [the instructions in the CUDA.jl docs](https://cuda.juliagpu.org/stable/installation/overview/).

Then, clone this repo with `git clone https://github.com/UCSD-computational-number-theory/GPUFiniteFieldMatrices.jl.git`.

`Revise.jl` is recommended for hot reloading. From a new Julia REPL, run
`using Pkg; Pkg.add("Revise")` (or, using Julia's package prompt, run `] add Revise`) once, and then run `using Revise` upon opening every new REPL.

The first time you install GPUFiniteFieldMatrices.jl, it will take a while for Julia to download and compile all of the dependencies. To do this, make sure your Julia REPL is in the GPUFiniteFieldMatrices.jl project folder. Then, in the package prompt (i.e. after pressing `]`), run

```
activate .
instantiate
```

The `instantiate` step will take a while as Julia's package manager downloads and compiles all dependencies.

Then, press backspace to go back to a julia prompt and run

```
using GPUFiniteFieldMatrices
```

#### Starting a new session after installation is complete

After starting a new Julia REPL, run

```
using Revise
] activate .
```

Then, press backspace to go back to a julia prompt and run

```
using GPUFiniteFieldMatrices
```

#### Troubleshooting

Please ensure that you're using the latest version of Julia and that your CUDA driver is installed. If the package manager gives you errors, try deleting Manifest.toml, upgrading Julia, updating the package registry, and updating CUDA.jl.

## Developer checks

With Julia 1.12 and Python 3.12 available, install the pinned pre-commit version and its Git hook:

```sh
python -m pip install pre-commit==4.6.2
pre-commit install
pre-commit run --all-files --show-diff-on-failure
```

JuliaFormatter 2.14.0 is installed into the isolated `@gpuffm_formatter` Julia environment by the hook. ReLint remains an optional manual check and does not run during normal commits or CI:

```sh
pre-commit run --hook-stage manual lint-checks --all-files
```

## Sample code

#### Matrices modulo N

Execute the following lines in Julia REPL:

```julia
using GPUFiniteFieldMatrices

A = CuModMatrix([1 2; 3 4], 5)
B = CuModMatrix([2 1; 0 3], 5)
A * B
```

#### Karatsuba matrices

```julia
using GPUFiniteFieldMatrices

A = KaratsubaMatrix([1 2; 3 4], [0 1; 1 0])
B = KaratsubaMatrix([2 1; 0 3], [1 0; 2 1])
A * B
```

## License

GPUFiniteFieldMatrices.jl is distributed under the MIT license.
