# Template Vector Library (TVL)

The Template Vector Library (TVL) offers hardware-oblivious SIMD programming. The Technische Universität Dresden originally developed the TVL using C++-templates and their work is still ongoing (see their [GitHub repository](https://github.com/MorphStore/TVLLib)).

As C++-templates hinder optimization at compile time, this TVL project tries to move from a template-driven design to a domain-specific language (DSL). It uses LLVM's subproject [MLIR](https://mlir.llvm.org/) (Multi-Level Intermediate Representation) to build the compiler and reuse common compiler infrastructure.

## TVL MLIR Operations

For the various operations TVL offers in MLIR have a look at the appropriate [GitHub Wiki Page](https://github.com/BrandonSchmitt/TVL/wiki/TvlOps).

## Building

This setup assumes that you have built LLVM and MLIR in `$BUILD_DIR` and installed them to `$PREFIX` (for further information see the [MLIR Getting Started documentation](https://mlir.llvm.org/getting_started/)).

Prepare the cmake enviroment
```sh
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=$PREFIX/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=$BUILD_DIR/bin/llvm-lit
```

To build the compiler, run
```sh
cmake --build . --target tvlc
```

To build and launch the tests, run
```sh
cmake --build . --target check-tvl
```
To build the documentation from the TableGen description of the dialect operations, run
```sh
cmake --build . --target mlir-doc
```
**Note**: Make sure to pass `-DLLVM_INSTALL_UTILS=ON` when building LLVM with CMake in order to install `FileCheck` to the chosen installation prefix.

