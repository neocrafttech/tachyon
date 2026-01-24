mod test_utils;
use compute::data_type::DataType;
use compute::error::ErrorMode;
use compute::operator::Operator;
use half::f16;

use crate::test_utils::CastTo;

macro_rules! test_eval_aggregate_matrix {
    (
        $operator:expr,
        $error_mode:expr,
        $size_min:expr,
        $size_max:expr,
        [
            $(
                ( $test_name:ident, $native_type:ident, $data_type:expr )
            ),* $(,)?
        ]
    ) => {
        $(
            test_eval_aggregate_fn!(
                $test_name,
                $operator,
                $error_mode,
                $native_type,
                $data_type,
                $size_min,
                $size_max,
            );
        )*
    };
}

trait FromUsize {
    fn from_usize(v: usize) -> Self;
}

macro_rules! impl_from_usize {
    ($($t:ty),* $(,)?) => {
        $(
            impl FromUsize for $t {
                fn from_usize(v: usize) -> Self {
                    v as $t
                }
            }
        )*
    };
}

impl_from_usize!(i8, i16, i32, i64, u8, u16, u32, u64, f32, f64);

impl FromUsize for f16 {
    fn from_usize(v: usize) -> Self {
        f16::from_f32(v as f32)
    }
}

trait AggAddOps: Sized {
    fn add_tachyon(self, rhs: Self) -> Self;
    fn add_ansi(self, rhs: Self) -> Option<Self>;
}

macro_rules! impl_agg_add_int {
    ($($t:ty),* $(,)?) => {
        $(
            impl AggAddOps for $t {
                fn add_tachyon(self, rhs: Self) -> Self {
                    self.wrapping_add(rhs)
                }

                fn add_ansi(self, rhs: Self) -> Option<Self> {
                    self.checked_add(rhs)
                }
            }
        )*
    };
}

impl_agg_add_int!(i8, i16, i32, i64, u8, u16, u32, u64);

impl AggAddOps for f16 {
    fn add_tachyon(self, rhs: Self) -> Self {
        self + rhs
    }

    fn add_ansi(self, rhs: Self) -> Option<Self> {
        Some(self + rhs)
    }
}

impl AggAddOps for f32 {
    fn add_tachyon(self, rhs: Self) -> Self {
        self + rhs
    }

    fn add_ansi(self, rhs: Self) -> Option<Self> {
        Some(self + rhs)
    }
}

impl AggAddOps for f64 {
    fn add_tachyon(self, rhs: Self) -> Self {
        self + rhs
    }

    fn add_ansi(self, rhs: Self) -> Option<Self> {
        Some(self + rhs)
    }
}

macro_rules! random_aggregate_vec {
    ($size:expr, i8) => {
        random_vec!($size, i8, -64i8, 64i8)
    };
    ($size:expr, i16) => {
        random_vec!($size, i16, -1024i16, 1024i16)
    };
    ($size:expr, i32) => {
        random_vec!($size, i32, -10_000i32, 10_000i32)
    };
    ($size:expr, i64) => {
        random_vec!($size, i64, -100_000i64, 100_000i64)
    };
    ($size:expr, u8) => {
        random_vec!($size, u8, 0u8, 128u8)
    };
    ($size:expr, u16) => {
        random_vec!($size, u16, 0u16, 1024u16)
    };
    ($size:expr, u32) => {
        random_vec!($size, u32, 0u32, 10_000u32)
    };
    ($size:expr, u64) => {
        random_vec!($size, u64, 0u64, 100_000u64)
    };
    ($size:expr, f16) => {{
        use rand::Rng;
        let mut rng = rand::rng();
        (0..$size)
            .map(|_| f16::from_f32(rng.random_range(-1000.0f32..1000.0f32)))
            .collect::<Vec<f16>>()
    }};
    ($size:expr, f32) => {
        random_vec!($size, f32, -1000.0f32, 1000.0f32)
    };
    ($size:expr, f64) => {
        random_vec!($size, f64, -1000.0f64, 1000.0f64)
    };
}

macro_rules! test_eval_aggregate_fn {
    (
        $test_name:ident,
        $operator:expr,
        $error_mode:expr,
        $native_type:ident,
        $data_type:expr,
        $size_min:expr,
        $size_max:expr,
    ) => {
        #[cfg(feature = "gpu")]
        #[tokio::test]
        async fn $test_name() {
            use compute::data_type::DataType;
            use compute::error::ErrorMode;
            use compute::evaluate::{Device, evaluate};
            use compute::expr::Expr;
            use compute::operator::Operator;

            use crate::create_column;
            use crate::test_utils::init_tracing;

            init_tracing();
            let size = random_num!($size_min, $size_max);
            let a_vec: Vec<$native_type> = random_aggregate_vec!(size, $native_type);
            let a_bit_vec = random_bit_vec!(size, u64);
            let col_a = create_column!(a_vec, Some(a_bit_vec.clone()), "a", $data_type);

            let expr = Expr::aggregate($operator, Expr::col("a"), false);
            let result = evaluate(Device::GPU, $error_mode, &expr, &[col_a]).await;

            let valid_values: Vec<$native_type> = a_vec
                .iter()
                .enumerate()
                .filter_map(|(i, v)| if a_bit_vec.is_valid(i) { Some(*v) } else { None })
                .collect();
            let error_mode = $error_mode;

            match $operator {
                Operator::Count => {
                    assert!(
                        result.is_ok(),
                        "aggregate {:?} for {:?} failed: {:?}",
                        $operator,
                        $data_type,
                        result.as_ref().err()
                    );
                    let result = result.unwrap();
                    let out_bits = result[0].null_bits_as_slice().unwrap();
                    let output = result[0].data_as_slice::<u64>().unwrap();
                    assert_eq!(output.len(), 1);
                    assert!(out_bits.is_valid(0));
                    assert_eq!(output[0], valid_values.len() as u64);
                }
                Operator::Min => {
                    assert!(
                        result.is_ok(),
                        "aggregate {:?} for {:?} failed: {:?}",
                        $operator,
                        $data_type,
                        result.as_ref().err()
                    );
                    let result = result.unwrap();
                    let out_bits = result[0].null_bits_as_slice().unwrap();
                    let output = result[0].data_as_slice::<$native_type>().unwrap();
                    assert_eq!(output.len(), 1);
                    if valid_values.is_empty() {
                        assert!(out_bits.is_null(0));
                    } else {
                        let expected = valid_values
                            .iter()
                            .copied()
                            .reduce(|a, b| if a < b { a } else { b })
                            .unwrap();
                        if $data_type.is_float() {
                            let expected_f64: f64 = expected.cast();
                            let actual_f64: f64 = output[0].cast();
                            let eps: f64 = if $data_type == DataType::F16 {
                                1e-2_f64.max(expected_f64.abs() * 1e-2)
                            } else {
                                1e-6_f64.max(expected_f64.abs() * 1e-6)
                            };
                            assert!((expected_f64 - actual_f64).abs() <= eps);
                        } else {
                            assert_eq!(output[0], expected);
                        }
                    }
                }
                Operator::Max => {
                    assert!(
                        result.is_ok(),
                        "aggregate {:?} for {:?} failed: {:?}",
                        $operator,
                        $data_type,
                        result.as_ref().err()
                    );
                    let result = result.unwrap();
                    let out_bits = result[0].null_bits_as_slice().unwrap();
                    let output = result[0].data_as_slice::<$native_type>().unwrap();
                    assert_eq!(output.len(), 1);
                    if valid_values.is_empty() {
                        assert!(out_bits.is_null(0));
                    } else {
                        let expected = valid_values
                            .iter()
                            .copied()
                            .reduce(|a, b| if a > b { a } else { b })
                            .unwrap();
                        if $data_type.is_float() {
                            let expected_f64: f64 = expected.cast();
                            let actual_f64: f64 = output[0].cast();
                            let eps: f64 = if $data_type == DataType::F16 {
                                1e-2_f64.max(expected_f64.abs() * 1e-2)
                            } else {
                                1e-6_f64.max(expected_f64.abs() * 1e-6)
                            };
                            assert!((expected_f64 - actual_f64).abs() <= eps);
                        } else {
                            assert_eq!(output[0], expected);
                        }
                    }
                }
                Operator::Sum => {
                    let mut overflowed = false;
                    let expected = valid_values.iter().copied().fold(
                        <$native_type as Default>::default(),
                        |acc, v| match error_mode {
                            ErrorMode::Ansi => match acc.add_ansi(v) {
                                Some(next) => next,
                                None => {
                                    overflowed = true;
                                    acc.add_tachyon(v)
                                }
                            },
                            ErrorMode::Tachyon => acc.add_tachyon(v),
                        },
                    );

                    if overflowed {
                        assert!(result.is_err());
                    } else {
                        assert!(
                            result.is_ok(),
                            "aggregate {:?} for {:?} failed: {:?}",
                            $operator,
                            $data_type,
                            result.as_ref().err()
                        );
                        let result = result.unwrap();
                        let out_bits = result[0].null_bits_as_slice().unwrap();
                        let output = result[0].data_as_slice::<$native_type>().unwrap();
                        assert_eq!(output.len(), 1);
                        if valid_values.is_empty() {
                            assert!(out_bits.is_null(0));
                        } else {
                            if $data_type.is_float() {
                                let expected_f64: f64 = expected.cast();
                                let actual_f64: f64 = output[0].cast();
                                let sum_abs: f64 = valid_values
                                    .iter()
                                    .map(|v| {
                                        let x: f64 = (*v).cast();
                                        x.abs()
                                    })
                                    .sum();
                                let eps: f64 = if $data_type == DataType::F16 {
                                    1e-1_f64.max(sum_abs * 5e-2)
                                } else {
                                    1e-2_f64.max(sum_abs * 1e-4)
                                };
                                assert!((expected_f64 - actual_f64).abs() <= eps);
                            } else {
                                assert_eq!(output[0], expected);
                            }
                        }
                    }
                }
                Operator::Avg => {
                    let mut overflowed = false;
                    let sum = valid_values.iter().copied().fold(
                        <$native_type as Default>::default(),
                        |acc, v| match error_mode {
                            ErrorMode::Ansi => match acc.add_ansi(v) {
                                Some(next) => next,
                                None => {
                                    overflowed = true;
                                    acc.add_tachyon(v)
                                }
                            },
                            ErrorMode::Tachyon => acc.add_tachyon(v),
                        },
                    );

                    if overflowed {
                        assert!(result.is_err());
                    } else {
                        assert!(
                            result.is_ok(),
                            "aggregate {:?} for {:?} failed: {:?}",
                            $operator,
                            $data_type,
                            result.as_ref().err()
                        );
                        let result = result.unwrap();
                        let out_bits = result[0].null_bits_as_slice().unwrap();
                        let output = result[0].data_as_slice::<$native_type>().unwrap();
                        assert_eq!(output.len(), 1);
                        if valid_values.is_empty() {
                            assert!(out_bits.is_null(0));
                        } else {
                            let cnt = <$native_type as FromUsize>::from_usize(valid_values.len());
                            let expected = sum / cnt;
                            if $data_type.is_float() {
                                let expected_f64: f64 = expected.cast();
                                let actual_f64: f64 = output[0].cast();
                                let sum_abs: f64 = valid_values
                                    .iter()
                                    .map(|v| {
                                        let x: f64 = (*v).cast();
                                        x.abs()
                                    })
                                    .sum();
                                let mean_abs = sum_abs / valid_values.len() as f64;
                                let eps: f64 = if $data_type == DataType::F16 {
                                    1e-2_f64.max(mean_abs * 5e-2)
                                } else {
                                    1e-3_f64.max(mean_abs * 1e-4)
                                };
                                assert!((expected_f64 - actual_f64).abs() <= eps);
                            } else {
                                assert_eq!(output[0], expected);
                            }
                        }
                    }
                }
                _ => unreachable!("unsupported aggregate operator"),
            }
        }
    };
}

test_eval_aggregate_matrix!(
    Operator::Min,
    ErrorMode::Tachyon,
    256,
    4096,
    [
        (test_agg_min_i8, i8, DataType::I8),
        (test_agg_min_i16, i16, DataType::I16),
        (test_agg_min_i32, i32, DataType::I32),
        (test_agg_min_i64, i64, DataType::I64),
        (test_agg_min_u8, u8, DataType::U8),
        (test_agg_min_u16, u16, DataType::U16),
        (test_agg_min_u32, u32, DataType::U32),
        (test_agg_min_u64, u64, DataType::U64),
        (test_agg_min_f16, f16, DataType::F16),
        (test_agg_min_f32, f32, DataType::F32),
        (test_agg_min_f64, f64, DataType::F64),
    ]
);

test_eval_aggregate_matrix!(
    Operator::Max,
    ErrorMode::Tachyon,
    256,
    4096,
    [
        (test_agg_max_i8, i8, DataType::I8),
        (test_agg_max_i16, i16, DataType::I16),
        (test_agg_max_i32, i32, DataType::I32),
        (test_agg_max_i64, i64, DataType::I64),
        (test_agg_max_u8, u8, DataType::U8),
        (test_agg_max_u16, u16, DataType::U16),
        (test_agg_max_u32, u32, DataType::U32),
        (test_agg_max_u64, u64, DataType::U64),
        (test_agg_max_f16, f16, DataType::F16),
        (test_agg_max_f32, f32, DataType::F32),
        (test_agg_max_f64, f64, DataType::F64),
    ]
);

test_eval_aggregate_matrix!(
    Operator::Sum,
    ErrorMode::Tachyon,
    256,
    4096,
    [
        (test_agg_sum_i8, i8, DataType::I8),
        (test_agg_sum_i16, i16, DataType::I16),
        (test_agg_sum_i32, i32, DataType::I32),
        (test_agg_sum_i64, i64, DataType::I64),
        (test_agg_sum_u8, u8, DataType::U8),
        (test_agg_sum_u16, u16, DataType::U16),
        (test_agg_sum_u32, u32, DataType::U32),
        (test_agg_sum_u64, u64, DataType::U64),
        (test_agg_sum_f16, f16, DataType::F16),
        (test_agg_sum_f32, f32, DataType::F32),
        (test_agg_sum_f64, f64, DataType::F64),
    ]
);

test_eval_aggregate_matrix!(
    Operator::Avg,
    ErrorMode::Tachyon,
    256,
    4096,
    [
        (test_agg_avg_i8, i8, DataType::I8),
        (test_agg_avg_i16, i16, DataType::I16),
        (test_agg_avg_i32, i32, DataType::I32),
        (test_agg_avg_i64, i64, DataType::I64),
        (test_agg_avg_u8, u8, DataType::U8),
        (test_agg_avg_u16, u16, DataType::U16),
        (test_agg_avg_u32, u32, DataType::U32),
        (test_agg_avg_u64, u64, DataType::U64),
        (test_agg_avg_f16, f16, DataType::F16),
        (test_agg_avg_f32, f32, DataType::F32),
        (test_agg_avg_f64, f64, DataType::F64),
    ]
);

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_aggregate_sum_ansi_i16_overflow_returns_error() {
    use compute::bit_vector::BitVector;
    use compute::data_type::DataType;
    use compute::error::ErrorMode;
    use compute::evaluate::{Device, evaluate};
    use compute::expr::Expr;
    use compute::operator::Operator;

    use crate::create_column;

    let a_vec: Vec<i16> = vec![30_000, 30_000];
    let a_bits = BitVector::<u64>::new_all_valid(a_vec.len());
    let col_a = create_column!(a_vec, Some(a_bits), "a", DataType::I16);
    let expr = Expr::aggregate(Operator::Sum, Expr::col("a"), false);
    let result = evaluate(Device::GPU, ErrorMode::Ansi, &expr, &[col_a]).await;
    assert!(result.is_err());
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_aggregate_avg_ansi_i16_overflow_returns_error() {
    use compute::bit_vector::BitVector;
    use compute::data_type::DataType;
    use compute::error::ErrorMode;
    use compute::evaluate::{Device, evaluate};
    use compute::expr::Expr;
    use compute::operator::Operator;

    use crate::create_column;

    let a_vec: Vec<i16> = vec![30_000, 30_000, 30_000];
    let a_bits = BitVector::<u64>::new_all_valid(a_vec.len());
    let col_a = create_column!(a_vec, Some(a_bits), "a", DataType::I16);
    let expr = Expr::aggregate(Operator::Avg, Expr::col("a"), false);
    let result = evaluate(Device::GPU, ErrorMode::Ansi, &expr, &[col_a]).await;
    assert!(result.is_err());
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_aggregate_sum_ansi_i16_no_overflow_ok() {
    use compute::bit_vector::BitVector;
    use compute::data_type::DataType;
    use compute::error::ErrorMode;
    use compute::evaluate::{Device, evaluate};
    use compute::expr::Expr;
    use compute::operator::Operator;

    use crate::create_column;

    let a_vec: Vec<i16> = vec![100, -10, 20];
    let a_bits = BitVector::<u64>::new_all_valid(a_vec.len());
    let col_a = create_column!(a_vec, Some(a_bits), "a", DataType::I16);
    let expr = Expr::aggregate(Operator::Sum, Expr::col("a"), false);
    let result = evaluate(Device::GPU, ErrorMode::Ansi, &expr, &[col_a]).await.unwrap();
    let out = result[0].data_as_slice::<i16>().unwrap();
    assert_eq!(out[0], 110);
}

test_eval_aggregate_matrix!(
    Operator::Count,
    ErrorMode::Tachyon,
    256,
    4096,
    [
        (test_agg_count_i8, i8, DataType::I8),
        (test_agg_count_i16, i16, DataType::I16),
        (test_agg_count_i32, i32, DataType::I32),
        (test_agg_count_i64, i64, DataType::I64),
        (test_agg_count_u8, u8, DataType::U8),
        (test_agg_count_u16, u16, DataType::U16),
        (test_agg_count_u32, u32, DataType::U32),
        (test_agg_count_u64, u64, DataType::U64),
        (test_agg_count_f16, f16, DataType::F16),
        (test_agg_count_f32, f32, DataType::F32),
        (test_agg_count_f64, f64, DataType::F64),
    ]
);

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_aggregate_min_i32() {
    use compute::bit_vector::BitVector;
    use compute::data_type::DataType;
    use compute::error::ErrorMode;
    use compute::evaluate::{Device, evaluate};
    use compute::expr::Expr;
    use compute::operator::Operator;

    use crate::create_column;
    use crate::test_utils::init_tracing;
    init_tracing();

    let a_vec: Vec<i32> = vec![11, -7, 3, 42, 0];
    let a_bit_vec = BitVector::<u64>::new_all_valid(a_vec.len());
    let col_a = create_column!(a_vec, Some(a_bit_vec), "a", DataType::I32);

    let expr = Expr::aggregate(Operator::Min, Expr::col("a"), false);
    let result = evaluate(Device::GPU, ErrorMode::Tachyon, &expr, &[col_a]).await.unwrap();

    let output = result[0].data_as_slice::<i32>().unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0], -7);
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_aggregate_sum_i32() {
    use compute::bit_vector::BitVector;
    use compute::data_type::DataType;
    use compute::error::ErrorMode;
    use compute::evaluate::{Device, evaluate};
    use compute::expr::Expr;
    use compute::operator::Operator;

    use crate::create_column;
    use crate::test_utils::init_tracing;
    init_tracing();

    let a_vec: Vec<i32> = vec![10, 20, -5, 1];
    let a_bit_vec = BitVector::<u64>::new_all_valid(a_vec.len());
    let col_a = create_column!(a_vec, Some(a_bit_vec), "a", DataType::I32);

    let expr = Expr::aggregate(Operator::Sum, Expr::col("a"), false);
    let result = evaluate(Device::GPU, ErrorMode::Tachyon, &expr, &[col_a]).await.unwrap();

    let output = result[0].data_as_slice::<i32>().unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0], 26);
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_aggregate_max_i32() {
    use compute::bit_vector::BitVector;
    use compute::data_type::DataType;
    use compute::error::ErrorMode;
    use compute::evaluate::{Device, evaluate};
    use compute::expr::Expr;
    use compute::operator::Operator;

    use crate::create_column;
    use crate::test_utils::init_tracing;
    init_tracing();

    let a_vec: Vec<i32> = vec![10, 20, -5, 1];
    let a_bit_vec = BitVector::<u64>::new_all_valid(a_vec.len());
    let col_a = create_column!(a_vec, Some(a_bit_vec), "a", DataType::I32);

    let expr = Expr::aggregate(Operator::Max, Expr::col("a"), false);
    let result = evaluate(Device::GPU, ErrorMode::Tachyon, &expr, &[col_a]).await.unwrap();

    let output = result[0].data_as_slice::<i32>().unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0], 20);
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_aggregate_avg_f64() {
    use compute::bit_vector::BitVector;
    use compute::data_type::DataType;
    use compute::error::ErrorMode;
    use compute::evaluate::{Device, evaluate};
    use compute::expr::Expr;
    use compute::operator::Operator;

    use crate::create_column;
    use crate::test_utils::init_tracing;
    init_tracing();

    let a_vec: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0];
    let a_bit_vec = BitVector::<u64>::new_all_valid(a_vec.len());
    let col_a = create_column!(a_vec, Some(a_bit_vec), "a", DataType::F64);

    let expr = Expr::aggregate(Operator::Avg, Expr::col("a"), false);
    let result = evaluate(Device::GPU, ErrorMode::Tachyon, &expr, &[col_a]).await.unwrap();

    let output = result[0].data_as_slice::<f64>().unwrap();
    assert_eq!(output.len(), 1);
    assert!((output[0] - 25.0).abs() < 1e-9);
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_aggregate_count_i32_with_nulls() {
    use compute::bit_vector::BitVector;
    use compute::data_type::DataType;
    use compute::error::ErrorMode;
    use compute::evaluate::{Device, evaluate};
    use compute::expr::Expr;
    use compute::operator::Operator;

    use crate::create_column;
    use crate::test_utils::init_tracing;
    init_tracing();

    let a_vec: Vec<i32> = vec![1, 2, 3, 4, 5];
    let mut a_bit_vec = BitVector::<u64>::new_all_valid(a_vec.len());
    a_bit_vec.set_null(1);
    a_bit_vec.set_null(4);

    let col_a = create_column!(a_vec, Some(a_bit_vec), "a", DataType::I32);

    let expr = Expr::aggregate(Operator::Count, Expr::col("a"), false);
    let result = evaluate(Device::GPU, ErrorMode::Tachyon, &expr, &[col_a]).await.unwrap();

    let output = result[0].data_as_slice::<u64>().unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0], 3);
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_aggregate_max_i32_with_nulls() {
    use compute::bit_vector::BitVector;
    use compute::data_type::DataType;
    use compute::error::ErrorMode;
    use compute::evaluate::{Device, evaluate};
    use compute::expr::Expr;
    use compute::operator::Operator;

    use crate::create_column;
    use crate::test_utils::init_tracing;
    init_tracing();

    let a_vec: Vec<i32> = vec![1, 100, 2, 99, 3];
    let mut a_bit_vec = BitVector::<u64>::new_all_valid(a_vec.len());
    a_bit_vec.set_null(1);
    a_bit_vec.set_null(3);
    let col_a = create_column!(a_vec, Some(a_bit_vec), "a", DataType::I32);

    let expr = Expr::aggregate(Operator::Max, Expr::col("a"), false);
    let result = evaluate(Device::GPU, ErrorMode::Tachyon, &expr, &[col_a]).await.unwrap();

    let output = result[0].data_as_slice::<i32>().unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0], 3);
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_aggregate_avg_f64_with_nulls() {
    use compute::bit_vector::BitVector;
    use compute::data_type::DataType;
    use compute::error::ErrorMode;
    use compute::evaluate::{Device, evaluate};
    use compute::expr::Expr;
    use compute::operator::Operator;

    use crate::create_column;
    use crate::test_utils::init_tracing;
    init_tracing();

    let a_vec: Vec<f64> = vec![10.0, 20.0, 30.0, 40.0];
    let mut a_bit_vec = BitVector::<u64>::new_all_valid(a_vec.len());
    a_bit_vec.set_null(1);
    a_bit_vec.set_null(3);
    let col_a = create_column!(a_vec, Some(a_bit_vec), "a", DataType::F64);

    let expr = Expr::aggregate(Operator::Avg, Expr::col("a"), false);
    let result = evaluate(Device::GPU, ErrorMode::Tachyon, &expr, &[col_a]).await.unwrap();

    let output = result[0].data_as_slice::<f64>().unwrap();
    assert_eq!(output.len(), 1);
    assert!((output[0] - 20.0).abs() < 1e-9);
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_aggregate_min_all_null_returns_null() {
    use compute::bit_vector::BitVector;
    use compute::data_type::DataType;
    use compute::error::ErrorMode;
    use compute::evaluate::{Device, evaluate};
    use compute::expr::Expr;
    use compute::operator::Operator;

    use crate::create_column;
    use crate::test_utils::init_tracing;
    init_tracing();

    let a_vec: Vec<i32> = vec![7, 8, 9];
    let a_bit_vec = BitVector::<u64>::new_all_null(a_vec.len());
    let col_a = create_column!(a_vec, Some(a_bit_vec), "a", DataType::I32);

    let expr = Expr::aggregate(Operator::Min, Expr::col("a"), false);
    let result = evaluate(Device::GPU, ErrorMode::Tachyon, &expr, &[col_a]).await.unwrap();

    let bit_vec = result[0].null_bits_as_slice().unwrap();
    assert!(bit_vec.is_null(0));
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_aggregate_count_all_null_returns_zero() {
    use compute::bit_vector::BitVector;
    use compute::data_type::DataType;
    use compute::error::ErrorMode;
    use compute::evaluate::{Device, evaluate};
    use compute::expr::Expr;
    use compute::operator::Operator;

    use crate::create_column;
    use crate::test_utils::init_tracing;
    init_tracing();

    let a_vec: Vec<i32> = vec![7, 8, 9];
    let a_bit_vec = BitVector::<u64>::new_all_null(a_vec.len());
    let col_a = create_column!(a_vec, Some(a_bit_vec), "a", DataType::I32);

    let expr = Expr::aggregate(Operator::Count, Expr::col("a"), false);
    let result = evaluate(Device::GPU, ErrorMode::Tachyon, &expr, &[col_a]).await.unwrap();

    let output = result[0].data_as_slice::<u64>().unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0], 0);
    let bit_vec = result[0].null_bits_as_slice().unwrap();
    assert!(bit_vec.is_valid(0));
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_aggregate_min_f32_ignores_nan_when_non_nan_exists() {
    use compute::bit_vector::BitVector;
    use compute::data_type::DataType;
    use compute::error::ErrorMode;
    use compute::evaluate::{Device, evaluate};
    use compute::expr::Expr;
    use compute::operator::Operator;

    use crate::create_column;
    use crate::test_utils::init_tracing;
    init_tracing();

    let a_vec: Vec<f32> = vec![f32::NAN, 4.0, -3.0, f32::NAN, 2.0];
    let a_bit_vec = BitVector::<u64>::new_all_valid(a_vec.len());
    let col_a = create_column!(a_vec, Some(a_bit_vec), "a", DataType::F32);

    let expr = Expr::aggregate(Operator::Min, Expr::col("a"), false);
    let result = evaluate(Device::GPU, ErrorMode::Tachyon, &expr, &[col_a]).await.unwrap();

    let output = result[0].data_as_slice::<f32>().unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0], -3.0);
}
