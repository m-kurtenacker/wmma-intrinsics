# Fast matrix multiplication in AnyDSL

Tests and Benchmarking repository for accelerated matrix multiplications using hardware accelerated matrix multiplication primitives.
Currently supported / planning:

- [X] PTX WMMA through mainline LLVM intrinsics
- [ ] AMD MFMA compiler intrinsics
- [ ] Whatever Intel has?

# Building

You need AnyDSL -- any version should be fine -- as well as support for the particular backend enabled in the runtime.
This project can then be build using cmake:

```
cmake -DAnyDSL_DIR=<AnyDSL cmake prefix path>
```

# Testing

A small set of tests is included in the test directory.
The generated folder contains a script that can be used to test a variety of different combinations of parameters, such as dimensionalities and memory layouts.
This script will generate hundreds if not thousands of tests, use at your own discretion.
