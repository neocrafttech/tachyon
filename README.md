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
```

The default setup and CPU tests do not require CUDA. For GPU tests, run
`./bolt.sh setup --cuda` on a machine with a working NVIDIA driver. This installs
the CUDA compiler and runtime/NVRTC development libraries without the full
toolkit or driver bundle. `./bolt.sh test` enables GPU tests when `nvidia-smi`
reports a working GPU. If NVIDIA hardware is present but the driver is unavailable,
tests stop with a diagnostic. Use `./bolt.sh test --cpu` to explicitly run CPU tests,
or `./bolt.sh test --gpu` to require GPU tests. The `nouveau` driver does not provide
CUDA support; the host needs an NVIDIA compute driver as well as the toolkit.

## Sample Usage
### Parse and inspect an expression
```rust
use compute::data_type::DataType;
use compute::error::ErrorMode;
use compute::expr::SchemaContext;
use compute::parser::parse_scheme_expr;

let expr = parse_scheme_expr("(+, a, 10)")?;
let schema = SchemaContext::new()
    .with_column("a", DataType::I32)
    .with_error_mode(ErrorMode::Tachyon);

assert_eq!(expr.infer_type(&schema)?, DataType::I32);
```

### Evaluate on GPU
```rust
use std::sync::Arc;

use compute::column::{Column, VecArray};
use compute::data_type::DataType;
use compute::error::ErrorMode;
use compute::evaluate::{Device, evaluate};
use compute::expr::Expr;
use compute::operator::Operator;

// Requires the `gpu` feature and a CUDA-capable environment.
let input = Column::<u64>::new(
    "a",
    Arc::new(VecArray { data: vec![1_i32, 2, 3], datatype: DataType::I32 }),
    None,
);
let expr = Expr::binary(Operator::Add, Expr::col("a"), Expr::i32(5));
let output = evaluate(Device::GPU, ErrorMode::Tachyon, &expr, &[input]).await?;

assert_eq!(output[0].data_as_slice::<i32>(), Some(&[6, 7, 8][..]));
```
