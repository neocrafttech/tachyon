use arrow::datatypes::{
    ArrowPrimitiveType, Float16Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use half::f16;

pub trait ArrowMapper {
    type ArrowType: ArrowPrimitiveType;
}

impl ArrowMapper for i8 {
    type ArrowType = Int8Type;
}
impl ArrowMapper for i16 {
    type ArrowType = Int16Type;
}
impl ArrowMapper for i32 {
    type ArrowType = Int32Type;
}
impl ArrowMapper for i64 {
    type ArrowType = Int64Type;
}
impl ArrowMapper for u8 {
    type ArrowType = UInt8Type;
}
impl ArrowMapper for u16 {
    type ArrowType = UInt16Type;
}
impl ArrowMapper for u32 {
    type ArrowType = UInt32Type;
}
impl ArrowMapper for u64 {
    type ArrowType = UInt64Type;
}
impl ArrowMapper for f16 {
    type ArrowType = Float16Type;
}
impl ArrowMapper for f32 {
    type ArrowType = Float32Type;
}
impl ArrowMapper for f64 {
    type ArrowType = Float64Type;
}

#[allow(dead_code)]
trait ToF64 {
    fn to_f64(self) -> f64;
}

macro_rules! impl_to_f64 {
    ($($t:ty)*) => {
        $(
            impl ToF64 for $t {
                fn to_f64(self) -> f64 {
                    self as f64
                }
            }
        )*
    };
}

impl_to_f64!(u8 u16 u32 u64 usize i8 i16 i32 i64 isize f32 f64);

impl ToF64 for f16 {
    fn to_f64(self) -> f64 {
        f64::from(f32::from(self))
    }
}

#[macro_export]
macro_rules! random_num {
    ($min:expr, $max:expr) => {{
        use rand;
        use rand::Rng;
        let mut rng = rand::rng();
        let num: usize = rng.random_range($min..$max);
        num
    }};
}

#[macro_export]
macro_rules! random_vec {
    ($size:expr, $ty:ty, $min:expr, $max:expr) => {{
        use rand::Rng;
        let mut rng = rand::rng();
        (0..$size).map(|_| rng.random_range($min..$max)).collect::<Vec<$ty>>()
    }};
}

#[macro_export]
macro_rules! random_bit_vec {
    ($size:expr, $ty:ty) => {{
        use compute::bit_vector::BitVector;
        use rand;
        use rand::Rng;
        let mut rng = rand::rng();
        const BITS: usize = std::mem::size_of::<$ty>() * 8;
        let num_blocks = $size.div_ceil(BITS);
        let mut bits: Vec<$ty> = Vec::with_capacity(num_blocks);

        for _ in 0..(num_blocks.saturating_sub(1)) {
            let random_block: $ty = rng.random_range(0..=<$ty>::MAX);
            bits.push(random_block);
        }

        if num_blocks > 0 {
            let last_idx = num_blocks - 1;
            let total_used_bits = last_idx * BITS;
            let valid_bits_in_last_block = $size - total_used_bits;
            let last_block: $ty = rng.random_range(0..=<$ty>::MAX);

            if valid_bits_in_last_block < BITS {
                let low_bits_mask = !(<$ty>::MAX << valid_bits_in_last_block);
                bits.push(last_block & low_bits_mask);
            } else {
                bits.push(last_block);
            }
        }

        BitVector::new(bits, $size)
    }};
}

#[macro_export]
macro_rules! create_arrow_array {
    ($vec:expr, $bit_vec:expr, $native_type:ty) => {{
        let arrow_vec: Vec<Option<$native_type>> = $vec
            .iter()
            .enumerate()
            .map(|(i, &x)| if $bit_vec.is_valid(i) { Some(x) } else { None })
            .collect();
        PrimitiveArray::<<$native_type as ArrowMapper>::ArrowType>::from(arrow_vec)
    }};
}

#[macro_export]
macro_rules! create_column {
    ($vec:expr, $bit_vec:expr, $name:expr, $data_type:expr) => {{
        use std::sync::Arc;

        use compute::column::{Column, VecArray};
        let arr = Arc::new(VecArray { data: $vec.clone(), datatype: $data_type });
        Column::new($name, arr, $bit_vec)
    }};
}

macro_rules! test_eval_binary_matrix {
    (
        $verify_arrow_fn:expr,
        $operator:expr,
        $error_mode:expr,
        [
            $(
                ( $test_name:ident, $native_type:ty, $data_type:expr, $result_type:ty, $size_min:expr, $size_max:expr, $value_min:expr, $value_max:expr)
            ),* $(,)?
        ]
    ) => {
        $(
            test_eval_binary_fn!(
                $verify_arrow_fn,
                $test_name,
                $operator,
                $error_mode,
                $native_type,
                $data_type,
                $result_type,
                $size_min,
                $size_max,
                $value_min,
                $value_max
            );
        )*
    };
}

macro_rules! test_eval_binary_fn {
    (
        $verify_arrow_fn:expr,
        $test_name:ident,
        $operator:expr,
        $error_mode:expr,
        $native_type:ty,
        $data_type:expr,
        $result_type:ty,
        $size_min:expr,
        $size_max:expr,
        $value_min:expr,
        $value_max:expr
    ) => {
        #[cfg(feature = "gpu")]
        #[tokio::test]
        async fn $test_name() {
            use arrow::array::{Array, PrimitiveArray};
            use compute::data_type::DataType;
            use compute::error::ErrorMode;
            use compute::evaluate::{Device, evaluate};
            use compute::expr::Expr;
            use compute::operator::Operator;
            let size = random_num!($size_min, $size_max);
            let a_vec: Vec<$native_type> = random_vec!(size, $native_type, $value_min, $value_max);
            let b_vec: Vec<$native_type> = random_vec!(size, $native_type, $value_min, $value_max);

            let a_bit_vec = random_bit_vec!(size, u64);
            let b_bit_vec = random_bit_vec!(size, u64);

            let col_a = create_column!(a_vec, Some(a_bit_vec.clone()), "a", $data_type);
            let col_b = create_column!(b_vec, Some(b_bit_vec.clone()), "b", $data_type);

            let expr = Expr::binary($operator, Expr::col("a"), Expr::col("b"));

            let result = evaluate(Device::GPU, $error_mode, &expr, &vec![col_a, col_b]).await;

            let epsilon = if $data_type.is_float() { 1e-6 } else { 0.0 };

            let arrow_a = create_arrow_array!(a_vec, a_bit_vec, $native_type);
            let arrow_b = create_arrow_array!(b_vec, b_bit_vec, $native_type);

            let arrow_result = $verify_arrow_fn(&arrow_a, &arrow_b);
            match arrow_result {
                Ok(arrow_result) => {
                    let arrow_output = arrow_result
                        .as_any()
                        .downcast_ref::<PrimitiveArray<<$native_type as ArrowMapper>::ArrowType>>()
                        .unwrap();
                    assert!(result.is_ok());
                    let result = result.unwrap();

                    assert!(result[0].data_as_slice::<$result_type>().is_some());
                    let output = result[0].data_as_slice::<$result_type>().unwrap();
                    let bit_vec = result[0].null_bits_as_slice().unwrap();
                    for i in 0..size {
                        if a_bit_vec.is_null(i) || b_bit_vec.is_null(i) {
                            assert!(bit_vec.is_null(i));
                        } else {
                            assert!(bit_vec.is_valid(i));
                            let expected = arrow_output.value(i).to_f64();
                            let actual = output[i].to_f64();
                            let diff = match expected {
                                f64::INFINITY
                                    if actual.is_infinite() && actual.is_sign_positive() =>
                                {
                                    0.0
                                }
                                f64::NEG_INFINITY
                                    if actual.is_infinite() && actual.is_sign_negative() =>
                                {
                                    0.0
                                }
                                _ => {
                                    if expected > actual {
                                        expected - actual
                                    } else {
                                        actual - expected
                                    }
                                }
                            };
                            assert!(
                                diff <= epsilon as f64,
                                "Mismatch at index {}: expected {} op {} = {}, got {}, diff {}",
                                i,
                                &a_vec[i],
                                &b_vec[i],
                                expected,
                                actual,
                                diff
                            );
                        }
                    }
                }
                _ => assert!(!result.is_ok()),
            }
        }
    };
}

macro_rules! test_eval_binary_cmp_matrix {
    (
        $op:tt,
        $operator:expr,
        $error_mode:expr,
        [
            $(
                ( $test_name:ident, $native_type:ty, $data_type:expr, $size_min:expr, $size_max:expr, $value_min:expr, $value_max:expr)
            ),* $(,)?
        ]
    ) => {
        $(
            test_eval_binary_cmp_fn!(
                $op,
                $test_name,
                $operator,
                $error_mode,
                $native_type,
                $data_type,
                $size_min,
                $size_max,
                $value_min,
                $value_max
            );
        )*
    };
}

macro_rules! test_eval_binary_cmp_fn {
    (
        $op:tt,
        $test_name:ident,
        $operator:expr,
        $error_mode:expr,
        $native_type:ty,
        $data_type:expr,
        $size_min:expr,
        $size_max:expr,
        $value_min:expr,
        $value_max:expr
    ) => {
        #[cfg(feature = "gpu")]
        #[tokio::test]
        async fn $test_name() {
            use compute::data_type::DataType;
            use compute::error::ErrorMode;
            use compute::evaluate::{Device, evaluate};
            use compute::expr::Expr;
            use compute::operator::Operator;
            let size = random_num!($size_min, $size_max);
            let a_vec: Vec<$native_type> = random_vec!(size, $native_type, $value_min, $value_max);
            let b_vec: Vec<$native_type> = random_vec!(size, $native_type, $value_min, $value_max);

            let a_bit_vec = random_bit_vec!(size, u32);
            let b_bit_vec = random_bit_vec!(size, u32);

            let col_a = create_column!(a_vec, Some(a_bit_vec.clone()), "a", $data_type);
            let col_b = create_column!(b_vec, Some(b_bit_vec.clone()), "b", $data_type);

            let expr = Expr::binary($operator, Expr::col("a"), Expr::col("b"));

            let result = evaluate(Device::GPU, $error_mode, &expr, &vec![col_a, col_b]).await;
            assert!(result.is_ok());
            let result = result.unwrap();
            assert!(result[0].data_as_slice::<bool>().is_some());
            let output = result[0].data_as_slice::<bool>().unwrap();
            let bit_vec = result[0].null_bits_as_slice().unwrap();
            for i in 0..size {
                if a_bit_vec.is_null(i) || b_bit_vec.is_null(i) {
                    assert!(bit_vec.is_null(i));
                } else {
                let actual = output[i];
                let expected = &a_vec[i] $op &b_vec[i];
                assert_eq!(
                    actual, expected,
                    "Mismatch at index {}: expected {} op {} = {}, got {}",
                    i, &a_vec[i], &b_vec[i], expected, actual,
                );
            }
            }
        }
    };
}

test_eval_binary_matrix!(
    arrow::compute::kernels::numeric::add_wrapping,
    Operator::Add,
    ErrorMode::Tachyon,
    [
        (test_add_i8, i8, DataType::I8, i8, 100, 100_000, i8::MIN, i8::MAX),
        (test_add_u8, u8, DataType::U8, u8, 100, 100_000, u8::MIN, u8::MAX),
        (test_add_i16, i16, DataType::I16, i16, 100, 100_000, i16::MIN, i16::MAX),
        (test_add_u16, u16, DataType::U16, u16, 500, 500_000, u16::MIN, u16::MAX),
        (test_add_i32, i32, DataType::I32, i32, 100, 100_000, i32::MIN / 2, i32::MAX / 2),
        (test_add_u32, u32, DataType::U32, u32, 512, 512_000, u32::MIN / 2, u32::MAX / 2),
        (test_add_i64, i64, DataType::I64, i64, 1024, 1024_000, i64::MIN / 2, i64::MAX / 2),
        (test_add_u64, u64, DataType::U64, u64, 512, 512_000, u64::MIN / 2, u64::MAX / 2),
        (test_add_f16, f16, DataType::F16, f16, 100, 100_000, f16::MIN, f16::MAX),
        (test_add_f32, f32, DataType::F32, f32, 100, 100_000, f32::MIN / 2.0, f32::MAX / 2.0),
        (test_add_f64, f64, DataType::F64, f64, 100, 100_000, f64::MIN / 2.0, f64::MAX / 2.0),
    ]
);

test_eval_binary_matrix!(
    arrow::compute::kernels::numeric::sub_wrapping,
    Operator::Sub,
    ErrorMode::Tachyon,
    [
        (test_sub_i8, i8, DataType::I8, i8, 100, 100_000, i8::MIN, i8::MAX),
        (test_sub_u8, u8, DataType::U8, u8, 100, 100_000, u8::MIN, u8::MAX),
        (test_sub_i16, i16, DataType::I16, i16, 500, 500_000, i16::MIN, i16::MAX),
        (test_sub_u16, u16, DataType::U16, u16, 500, 500_000, u16::MIN, u16::MAX),
        (test_sub_i32, i32, DataType::I32, i32, 1024, 1024_000, i32::MIN, i32::MAX),
        (test_sub_u32, u32, DataType::U32, u32, 512, 512_000, u32::MIN, u32::MAX),
        (test_sub_i64, i64, DataType::I64, i64, 1024, 1024_000, i64::MIN, i64::MAX),
        (test_sub_u64, u64, DataType::U64, u64, 512, 512_000, u64::MIN, u64::MAX),
        (test_sub_f16, f16, DataType::F16, f16, 100, 100_000, f16::MIN, f16::MAX),
        (test_sub_f32, f32, DataType::F32, f32, 1024, 1024_000, f32::MIN / 2.0, f32::MAX / 2.0),
        (test_sub_f64, f64, DataType::F64, f64, 1000, 1000_000, f64::MIN / 2.0, f64::MAX / 2.0),
    ]
);

test_eval_binary_matrix!(
    arrow::compute::kernels::numeric::mul_wrapping,
    Operator::Mul,
    ErrorMode::Tachyon,
    [
        (test_mul_i8, i8, DataType::I8, i8, 100, 100_000, i8::MIN, i8::MAX),
        (test_mul_u8, u8, DataType::U8, u8, 100, 100_000, u8::MIN, u8::MAX),
        (test_mul_i16, i16, DataType::I16, i16, 500, 500_000, i16::MIN, i16::MAX),
        (test_mul_u16, u16, DataType::U16, u16, 500, 500_000, u16::MIN / 2_000, u16::MAX / 2_000),
        (test_mul_i32, i32, DataType::I32, i32, 1024, 1024_000, i32::MIN / 2_000, i32::MAX / 2_000),
        (test_mul_u32, u32, DataType::U32, u32, 512, 512_000, u32::MIN / 2_000, u32::MAX / 2_000),
        (test_mul_i64, i64, DataType::I64, i64, 1024, 1024_000, i64::MIN / 2_000, i64::MAX / 2_000),
        (test_mul_u64, u64, DataType::U64, u64, 512, 512_000, u64::MIN / 2_000, u64::MAX / 2_000),
        (test_mul_f16, f16, DataType::F16, f16, 100, 100_000, f16::MIN, f16::MAX),
        (test_mul_f32, f32, DataType::F32, f32, 10, 10_000, f32::MIN / 2_000.0, f32::MAX / 2_000.0),
        (test_mul_f64, f64, DataType::F64, f64, 10, 10_000, f64::MIN / 2_000.0, f64::MAX / 2_000.0),
    ]
);

test_eval_binary_matrix!(
    arrow::compute::kernels::numeric::div,
    Operator::Div,
    ErrorMode::Tachyon,
    [
        (test_div_i8, i8, DataType::I8, i8, 100, 100_000, i8::MIN, i8::MAX),
        (test_div_u8, u8, DataType::U8, u8, 100, 100_000, u8::MIN, u8::MAX),
        (test_div_i16, i16, DataType::I16, i16, 500, 500_000, i16::MIN, i16::MAX),
        (test_div_u16, u16, DataType::U16, u16, 500, 500_000, u16::MIN, u16::MAX),
        (test_div_i32, i32, DataType::I32, i32, 1024, 1024_000, i32::MIN, i32::MAX),
        (test_div_u32, u32, DataType::U32, u32, 512, 512_000, u32::MIN, u32::MAX),
        (test_div_i64, i64, DataType::I64, i64, 1024, 1024_000, i64::MIN, i64::MAX),
        (test_div_u64, u64, DataType::U64, u64, 512, 512_000, u64::MIN, u64::MAX),
        (test_div_f16, f16, DataType::F16, f16, 100, 100_000, f16::MIN, f16::MAX),
        (test_div_f32, f32, DataType::F32, f32, 1024, 1024_000, f32::MIN / 2.0, f32::MAX / 2.0),
        (test_div_f64, f64, DataType::F64, f64, 1000, 1000_000, f64::MIN / 2.0, f64::MAX / 2.0),
    ]
);

test_eval_binary_matrix!(
    arrow::compute::kernels::numeric::add,
    Operator::Add,
    ErrorMode::Ansi,
    [
        (test_add_ansi_i8, i8, DataType::I8, i8, 100, 100_000, i8::MIN, i8::MAX),
        (test_add_ansi_u8, u8, DataType::U8, u8, 100, 100_000, u8::MIN, u8::MAX),
        (test_add_ansi_i16, i16, DataType::I16, i16, 500, 500_000, i16::MIN, i16::MAX),
        (test_add_ansi_u16, u16, DataType::U16, u16, 500, 500_000, u16::MIN, u16::MAX),
        (test_add_ansi_i32, i32, DataType::I32, i32, 1024, 1024_000, i32::MIN, i32::MAX),
        (test_add_ansi_u32, u32, DataType::U32, u32, 512, 512_000, u32::MIN, u32::MAX),
        (test_add_ansi_i64, i64, DataType::I64, i64, 1024, 1024_000, i64::MIN, i64::MAX),
        (test_add_ansi_u64, u64, DataType::U64, u64, 512, 512_000, u64::MIN, u64::MAX),
        (test_add_ansi_f16, f16, DataType::F16, f16, 100, 100_000, f16::MIN, f16::MAX),
        (test_add_ansi_f32, f32, DataType::F32, f32, 1, 10_000, f32::MIN / 2.0, f32::MAX / 2.0),
        (test_add_ansi_f64, f64, DataType::F64, f64, 1, 10_000, f64::MIN / 2.0, f64::MAX / 2.0),
    ]
);

test_eval_binary_matrix!(
    arrow::compute::kernels::numeric::sub,
    Operator::Sub,
    ErrorMode::Ansi,
    [
        (test_sub_ansi_i8, i8, DataType::I8, i8, 100, 100_000, i8::MIN, i8::MAX),
        (test_sub_ansi_u8, u8, DataType::U8, u8, 100, 100_000, u8::MIN, u8::MAX),
        (test_sub_ansi_i16, i16, DataType::I16, i16, 500, 500_000, i16::MIN, i16::MAX),
        (test_sub_ansi_u16, u16, DataType::U16, u16, 500, 500_000, u16::MIN, u16::MAX),
        (test_sub_ansi_i32, i32, DataType::I32, i32, 1024, 1024_000, i32::MIN, i32::MAX),
        (test_sub_ansi_u32, u32, DataType::U32, u32, 512, 512_000, u32::MIN, u32::MAX),
        (test_sub_ansi_i64, i64, DataType::I64, i64, 1024, 1024_000, i64::MIN, i64::MAX),
        (test_sub_ansi_u64, u64, DataType::U64, u64, 512, 512_000, u64::MIN, u64::MAX),
        (test_sub_ansi_f16, f16, DataType::F16, f16, 100, 100_000, f16::MIN, f16::MAX),
        (test_sub_ansi_f32, f32, DataType::F32, f32, 1024, 10_000, f32::MIN / 2.0, f32::MAX / 2.0),
        (test_sub_ansi_f64, f64, DataType::F64, f64, 1000, 10_000, f64::MIN / 2.0, f64::MAX / 2.0),
    ]
);

test_eval_binary_matrix!(
    arrow::compute::kernels::numeric::mul,
    Operator::Mul,
    ErrorMode::Ansi,
    [
        (test_mul_ansi_i8, i8, DataType::I8, i8, 100, 100_000, i8::MIN, i8::MAX),
        (test_mul_ansi_u8, u8, DataType::U8, u8, 100, 100_000, u8::MIN, u8::MAX),
        (test_mul_ansi_i16, i16, DataType::I16, i16, 500, 500_000, i16::MIN, i16::MAX),
        (test_mul_ansi_u16, u16, DataType::U16, u16, 500, 500_000, u16::MIN, u16::MAX),
        (test_mul_ansi_i32, i32, DataType::I32, i32, 1024, 1024_000, i32::MIN, i32::MAX),
        (test_mul_ansi_u32, u32, DataType::U32, u32, 512, 512_000, u32::MIN, u32::MAX),
        (test_mul_ansi_i64, i64, DataType::I64, i64, 1024, 1024_000, i64::MIN, i64::MAX),
        (test_mul_ansi_u64, u64, DataType::U64, u64, 512, 512_000, u64::MIN, u64::MAX),
        (test_mul_ansi_f16, f16, DataType::F16, f16, 100, 100_000, f16::MIN, f16::MAX),
        (test_mul_ansi_f32, f32, DataType::F32, f32, 1024, 10_000, f32::MIN / 2.0, f32::MAX / 2.0),
        (test_mul_ansi_f64, f64, DataType::F64, f64, 1000, 10_000, f64::MIN / 2.0, f64::MAX / 2.0),
    ]
);

test_eval_binary_cmp_matrix!(
    ==,
    Operator::Eq,
    ErrorMode::Tachyon,
    [
        (test_eq_i8, i8, DataType::I8, 100, 500_000, i8::MIN, i8::MAX),
        (test_eq_u8, u8, DataType::U8, 100,100_000, u8::MIN, u8::MAX),
        (test_eq_i16, i16, DataType::I16, 500, 500_000, i16::MIN, i16::MAX),
        (test_eq_u16, u16, DataType::U16, 500, 200_000, u16::MIN, u16::MAX),
        (test_eq_i32, i32, DataType::I32, 1024, 500_000, i32::MIN, i32::MAX),
        (test_eq_u32, u32, DataType::U32, 512, 400_000, u32::MIN, u32::MAX),
        (test_eq_i64, i64, DataType::I64, 1024, 500_000, i64::MIN, i64::MAX),
        (test_eq_u64, u64, DataType::U64, 512, 500_000, u64::MIN, u64::MAX),
        (test_eq_f32, f32, DataType::F32, 1024, 500_000, f32::MIN / 2.0, f32::MAX / 2.0),
        (test_eq_f64, f64, DataType::F64, 1000, 500_000, f64::MIN / 2.0, f64::MAX / 2.0),
    ]
);

test_eval_binary_cmp_matrix!(
    !=,
    Operator::NotEq,
    ErrorMode::Tachyon,
    [
        (test_neq_i8, i8, DataType::I8, 100, 400_000, i8::MIN, i8::MAX),
        (test_neq_u8, u8, DataType::U8, 100, 400_000, u8::MIN, u8::MAX),
        (test_neq_i16, i16, DataType::I16, 500, 400_000, i16::MIN, i16::MAX),
        (test_neq_u16, u16, DataType::U16, 500, 500_000, u16::MIN, u16::MAX),
        (test_neq_i32, i32, DataType::I32, 1024, 400_000, i32::MIN, i32::MAX),
        (test_neq_u32, u32, DataType::U32, 512, 600_000, u32::MIN, u32::MAX),
        (test_neq_i64, i64, DataType::I64, 1024, 400_000, i64::MIN, i64::MAX),
        (test_neq_u64, u64, DataType::U64, 512, 200_000, u64::MIN, u64::MAX),
        (test_neq_f32, f32, DataType::F32, 1024, 400_000, f32::MIN / 2.0, f32::MAX / 2.0),
        (test_neq_f64, f64, DataType::F64, 1000, 300_000, f64::MIN / 2.0, f64::MAX / 2.0),
    ]
);

test_eval_binary_cmp_matrix!(
    >,
    Operator::Gt,
    ErrorMode::Tachyon,
    [
        (test_gt_i8, i8, DataType::I8, 10, 100_000, i8::MIN, i8::MAX),
        (test_gt_u8, u8, DataType::U8, 100, 200_000, u8::MIN, u8::MAX),
        (test_gt_i16, i16, DataType::I16, 500, 100_000, i16::MIN, i16::MAX),
        (test_gt_u16, u16, DataType::U16, 500, 500_000, u16::MIN, u16::MAX),
        (test_gt_i32, i32, DataType::I32, 1024, 500_000, i32::MIN, i32::MAX),
        (test_gt_u32, u32, DataType::U32, 512, 100_000, u32::MIN, u32::MAX),
        (test_gt_i64, i64, DataType::I64, 1024, 100_000, i64::MIN, i64::MAX),
        (test_gt_u64, u64, DataType::U64, 512, 100_000, u64::MIN, u64::MAX),
        (test_gt_f32, f32, DataType::F32, 1024, 400_000, f32::MIN / 2.0, f32::MAX / 2.0),
        (test_gt_f64, f64, DataType::F64, 1000, 100_000, f64::MIN / 2.0, f64::MAX / 2.0),
    ]
);

test_eval_binary_cmp_matrix!(
    >=,
    Operator::GtEq,
    ErrorMode::Tachyon,
    [
        (test_gteq_i8, i8, DataType::I8,  100, 100_000, i8::MIN, i8::MAX),
        (test_gteq_u8, u8, DataType::U8, 100, 100_000, u8::MIN, u8::MAX),
        (test_gteq_i16, i16, DataType::I16, 500, 100_000, i16::MIN, i16::MAX),
        (test_gteq_u16, u16, DataType::U16, 500, 100_000, u16::MIN, u16::MAX),
        (test_gteq_i32, i32, DataType::I32, 1024, 100_000, i32::MIN, i32::MAX),
        (test_gteq_u32, u32, DataType::U32, 512, 100_000, u32::MIN, u32::MAX),
        (test_gteq_i64, i64, DataType::I64, 1024, 100_000, i64::MIN, i64::MAX),
        (test_gteq_u64, u64, DataType::U64, 512, 100_000, u64::MIN, u64::MAX),
        (test_gteq_f32, f32, DataType::F32, 1024, 100_000, f32::MIN / 2.0, f32::MAX / 2.0),
        (test_gteq_f64, f64, DataType::F64, 1000, 100_000, f64::MIN / 2.0, f64::MAX / 2.0),
    ]
);

test_eval_binary_cmp_matrix!(
    <,
    Operator::Lt,
    ErrorMode::Tachyon,
    [
        (test_lt_i8, i8, DataType::I8, 100, 500_000, i8::MIN, i8::MAX),
        (test_lt_u8, u8, DataType::U8, 100, 500_000, u8::MIN, u8::MAX),
        (test_lt_i16, i16, DataType::I16, 500, 500_000, i16::MIN, i16::MAX),
        (test_lt_u16, u16, DataType::U16, 500, 500_000, u16::MIN, u16::MAX),
        (test_lt_i32, i32, DataType::I32, 1024, 500_000, i32::MIN, i32::MAX),
        (test_lt_u32, u32, DataType::U32, 512, 500_000, u32::MIN, u32::MAX),
        (test_lt_i64, i64, DataType::I64, 1024, 500_000, i64::MIN, i64::MAX),
        (test_lt_u64, u64, DataType::U64, 100, 500_000, u64::MIN, u64::MAX),
        (test_lt_f32, f32, DataType::F32, 1024, 500_000, f32::MIN / 2.0, f32::MAX / 2.0),
        (test_lt_f64, f64, DataType::F64, 100, 500_000, f64::MIN / 2.0, f64::MAX / 2.0),
    ]
);

test_eval_binary_cmp_matrix!(
    <=,
    Operator::LtEq,
    ErrorMode::Tachyon,
    [
        (test_lteq_i8, i8, DataType::I8,  100, 500_000, i8::MIN, i8::MAX),
        (test_lteq_u8, u8, DataType::U8, 100, 500_000, u8::MIN, u8::MAX),
        (test_lteq_i16, i16, DataType::I16, 500, 500_000, i16::MIN, i16::MAX),
        (test_lteq_u16, u16, DataType::U16, 500, 500_000, u16::MIN, u16::MAX),
        (test_lteq_i32, i32, DataType::I32, 1024, 500_000, i32::MIN, i32::MAX),
        (test_lteq_u32, u32, DataType::U32, 512, 500_000, u32::MIN, u32::MAX),
        (test_lteq_i64, i64, DataType::I64, 1024, 500_000, i64::MIN, i64::MAX),
        (test_lteq_u64, u64, DataType::U64, 512, 500_000, u64::MIN, u64::MAX),
        (test_lteq_f32, f32, DataType::F32, 1024, 500_000, f32::MIN / 2.0, f32::MAX / 2.0),
        (test_lteq_f64, f64, DataType::F64, 1000, 500_000, f64::MIN / 2.0, f64::MAX / 2.0),
    ]
);

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_div_by_zero() {
    use compute::bit_vector::BitVector;
    use compute::data_type::DataType;
    use compute::error::ErrorMode;
    use compute::evaluate::{Device, evaluate};
    use compute::expr::Expr;
    use compute::operator::Operator;
    use half::bf16;

    let a_vec: Vec<bf16> = vec![bf16::from_f32(1.0), bf16::from_f32(2.0), bf16::from_f32(3.0)];
    let b_vec: Vec<bf16> = vec![bf16::from_f32(1.0), bf16::from_f32(0.0), bf16::from_f32(2.0)];

    let a_bit_vec = BitVector::<u64>::new_all_valid(a_vec.len());
    let b_bit_vec = BitVector::<u64>::new_all_valid(b_vec.len());

    let col_a = create_column!(a_vec, Some(a_bit_vec), "a", DataType::BF16);
    let col_b = create_column!(b_vec, Some(b_bit_vec), "b", DataType::BF16);

    let expr = Expr::binary(Operator::Div, Expr::col("a"), Expr::col("b"));

    let result = evaluate(Device::GPU, ErrorMode::Tachyon, &expr, &vec![col_a, col_b]).await;

    assert!(result.is_err());
    assert_eq!(result.unwrap_err().to_string(), "CUDA error: Kernel Error: Division by zero");
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_add_with_null() {
    use compute::bit_vector::BitVector;
    use compute::data_type::DataType;
    use compute::error::ErrorMode;
    use compute::evaluate::{Device, evaluate};
    use compute::expr::Expr;
    use compute::operator::Operator;
    use half::bf16;

    let a_vec: Vec<bf16> =
        vec![bf16::from_f32(1.0), bf16::from_f32(2.0), bf16::from_f32(3.0), bf16::from_f32(4.0)];
    let b_vec: Vec<bf16> =
        vec![bf16::from_f32(3.0), bf16::from_f32(5.0), bf16::from_f32(2.0), bf16::from_f32(1.0)];

    let mut a_bit_vec = BitVector::<u64>::new_all_valid(a_vec.len());
    let mut b_bit_vec = BitVector::<u64>::new_all_valid(b_vec.len());

    a_bit_vec.set_null(1);
    a_bit_vec.set_null(2);
    b_bit_vec.set_null(2);
    let col_a = create_column!(a_vec, Some(a_bit_vec), "a", DataType::BF16);
    let col_b = create_column!(b_vec, Some(b_bit_vec), "b", DataType::BF16);

    let expr = Expr::binary(Operator::Add, Expr::col("a"), Expr::col("b"));

    let result = evaluate(Device::GPU, ErrorMode::Tachyon, &expr, &vec![col_a, col_b]).await;
    let result = result.unwrap();
    assert!(result[0].data_as_slice::<bf16>().is_some());
    let output = result[0].data_as_slice::<bf16>().unwrap();
    let bit_vec = result[0].null_bits_as_slice().unwrap();
    assert!(bit_vec.is_valid(0));
    assert!(!bit_vec.is_valid(1));
    assert!(!bit_vec.is_valid(2));
    assert!(bit_vec.is_valid(3));
    assert_eq!(output[0], bf16::from_f32(4.0));
    assert_eq!(output[3], bf16::from_f32(5.0));
}
