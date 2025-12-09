# tachyon · GPU Accelerated DataFrame Library

**tachyon** is a fast, GPU-native DataFrame engine for analytics and machine-learning workloads. It uses a columnar execution model with runtime-compiled (JIT) GPU kernels to deliver high throughput and optimized execution.

## Features
- JIT-compiled GPU kernels for dynamic, hardware-specialized execution.
- Half-precision numeric support (f16, bf16) for reduced memory footprint and improved performance.
- Columnar DataFrame engine inspired by Apache Arrow.
- Vectorized math operations, filtering, and expression evaluation.

## Setup, Build, Test
```bash
git clone https://github.com/neocrafttech/tachyon.git
cd tachyon
./bolt.sh setup
./bolt.sh build
./bolt.sh test
