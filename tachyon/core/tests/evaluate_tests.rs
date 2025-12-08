use std::i16;

use arrow::datatypes::{
    ArrowPrimitiveType, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type,
    UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};

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
impl ArrowMapper for f32 {
    type ArrowType = Float32Type;
}
impl ArrowMapper for f64 {
    type ArrowType = Float64Type;
}

#[macro_export]
macro_rules! random_vec {
    ($size:expr, $native_type:ty) => {{
        use rand;
        use rand::Rng;
        let mut rng = rand::rng();
        (0..$size)
            .map(|_| {
                rng.random_range(
                    <$native_type>::MIN / (2 as $native_type)
                        ..<$native_type>::MAX / (2 as $native_type),
                )
            })
            .collect::<Vec<$native_type>>()
    }};
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
macro_rules! create_column {
    ($vec:expr, $name:expr, $data_type:expr) => {{
        use core::column::{Column, VecArray};
        use std::sync::Arc;
        let arr = Arc::new(VecArray { data: $vec.clone(), datatype: $data_type });
        Column::new($name, arr, None)
    }};
}

macro_rules! test_eval_binary_matrix {
    (
        $verify_arrow_fn:expr,
        $operator:expr,
        $error_mode:expr,
        [
            $(
                ( $test_name:ident, $native_type:ty, $data_type:expr, $result_type:ty, $size_min:expr, $size_max:expr)
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
                $size_max
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
        $size_max:expr
    ) => {
        #[cfg(feature = "gpu")]
        #[test]
        fn $test_name() {
            use core::data_type::DataType;
            use core::error::ErrorMode;
            use core::evaluate::{Device, evaluate};
            use core::expr::Expr;
            use core::operator::Operator;

            use arrow::array::{Array, PrimitiveArray};
            let size = random_num!($size_min, $size_max);
            let a_vec: Vec<$native_type> = random_vec!(size, $native_type);
            let b_vec: Vec<$native_type> = random_vec!(size, $native_type);

            let col_a = create_column!(a_vec, "a", $data_type);
            let col_b = create_column!(b_vec, "b", $data_type);

            let expr = Expr::binary($operator, Expr::col("a"), Expr::col("b"));

            let result = evaluate(Device::GPU, $error_mode, &expr, &vec![col_a, col_b]);

            let epsilon = if $data_type.is_float() { 1e-6 } else { 0.0 };
            let arrow_a =
                PrimitiveArray::<<$native_type as ArrowMapper>::ArrowType>::from(a_vec.clone());
            let arrow_b =
                PrimitiveArray::<<$native_type as ArrowMapper>::ArrowType>::from(b_vec.clone());

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
                    for i in 0..size {
                        let expected: $result_type = arrow_output.value(i).into();
                        let actual = output[i];
                        let expected = expected as f64;
                        let actual = actual as f64;
                        let diff = match expected {
                            f64::INFINITY if actual.is_infinite() && actual.is_sign_positive() => {
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
                ( $test_name:ident, $native_type:ty, $data_type:expr, $size_min:expr, $size_max:expr)
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
                $size_max
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
        $size_max:expr
    ) => {
        #[cfg(feature = "gpu")]
        #[test]
        fn $test_name() {
            use core::data_type::DataType;
            use core::error::ErrorMode;
            use core::evaluate::{Device, evaluate};
            use core::expr::Expr;
            use core::operator::Operator;
            let size = random_num!($size_min, $size_max);
            let a_vec: Vec<$native_type> = random_vec!(size, $native_type);
            let b_vec: Vec<$native_type> = random_vec!(size, $native_type);

            let col_a = create_column!(a_vec, "a", $data_type);
            let col_b = create_column!(b_vec, "b", $data_type);

            let expr = Expr::binary($operator, Expr::col("a"), Expr::col("b"));

            let result = evaluate(Device::GPU, $error_mode, &expr, &vec![col_a, col_b]);
            assert!(result.is_ok());
            let result = result.unwrap();
            assert!(result[0].data_as_slice::<bool>().is_some());
            let output = result[0].data_as_slice::<bool>().unwrap();
            for i in 0..size {
                let actual = output[i];
                let expected = &a_vec[i] $op &b_vec[i];
                assert_eq!(
                    actual, expected,
                    "Mismatch at index {}: expected {} op {} = {}, got {}",
                    i, &a_vec[i], &b_vec[i], expected, actual,
                );
            }
        }
    };
}

test_eval_binary_matrix!(
    arrow::compute::kernels::numeric::add_wrapping,
    Operator::Add,
    ErrorMode::Tachyon,
    [
        (test_add_i8, i8, DataType::I8, i8, 100, 100_000),
        (test_add_u8, u8, DataType::U8, u8, 100, 100_000),
        (test_add_i16, i16, DataType::I16, i16, 100, 100_000),
        (test_add_u16, u16, DataType::U16, u16, 500, 500_000),
        (test_add_i32, i32, DataType::I32, i32, 100, 100_000),
        (test_add_u32, u32, DataType::U32, u32, 512, 512_000),
        (test_add_i64, i64, DataType::I64, i64, 1024, 1024_000),
        (test_add_u64, u64, DataType::U64, u64, 512, 512_000),
        (test_add_f32, f32, DataType::F32, f32, 100, 100_000),
        (test_add_f64, f64, DataType::F64, f64, 100, 100_000),
    ]
);

test_eval_binary_matrix!(
    arrow::compute::kernels::numeric::sub_wrapping,
    Operator::Sub,
    ErrorMode::Tachyon,
    [
        (test_sub_i8, i8, DataType::I8, i8, 100, 100_000),
        (test_sub_u8, u8, DataType::U8, u8, 100, 100_000),
        (test_sub_i16, i16, DataType::I16, i16, 500, 500_000),
        (test_sub_u16, u16, DataType::U16, u16, 500, 500_000),
        (test_sub_i32, i32, DataType::I32, i32, 1024, 1024_000),
        (test_sub_u32, u32, DataType::U32, u32, 512, 512_000),
        (test_sub_i64, i64, DataType::I64, i64, 1024, 1024_000),
        (test_sub_u64, u64, DataType::U64, u64, 512, 512_000),
        (test_sub_f32, f32, DataType::F32, f32, 1024, 1024_000),
        (test_sub_f64, f64, DataType::F64, f64, 1000, 1000_000),
    ]
);

test_eval_binary_matrix!(
    arrow::compute::kernels::numeric::mul_wrapping,
    Operator::Mul,
    ErrorMode::Tachyon,
    [
        (test_mul_i8, i8, DataType::I8, i8, 100, 100_000),
        (test_mul_u8, u8, DataType::U8, u8, 100, 100_000),
        (test_mul_i16, i16, DataType::I16, i16, 500, 500_000),
        (test_mul_u16, u16, DataType::U16, u16, 500, 500_000),
        (test_mul_i32, i32, DataType::I32, i32, 1024, 1024_000),
        (test_mul_u32, u32, DataType::U32, u32, 512, 512_000),
        (test_mul_i64, i64, DataType::I64, i64, 1024, 1024_000),
        (test_mul_u64, u64, DataType::U64, u64, 512, 512_000),
        (test_mul_f32, f32, DataType::F32, f32, 1024, 1024_000),
        (test_mul_f64, f64, DataType::F64, f64, 1000, 1000_000),
    ]
);

test_eval_binary_matrix!(
    arrow::compute::kernels::numeric::div,
    Operator::Div,
    ErrorMode::Tachyon,
    [
        (test_div_i8, i8, DataType::I8, i8, 100, 100_000),
        (test_div_u8, u8, DataType::U8, u8, 100, 100_000),
        (test_div_i16, i16, DataType::I16, i16, 500, 500_000),
        (test_div_u16, u16, DataType::U16, u16, 500, 500_000),
        (test_div_i32, i32, DataType::I32, i32, 1024, 1024_000),
        (test_div_u32, u32, DataType::U32, u32, 512, 512_000),
        (test_div_i64, i64, DataType::I64, i64, 1024, 1024_000),
        (test_div_u64, u64, DataType::U64, u64, 512, 512_000),
        (test_div_f32, f32, DataType::F32, f32, 1024, 1024_000),
        (test_div_f64, f64, DataType::F64, f64, 1000, 1000_000),
    ]
);

test_eval_binary_matrix!(
    arrow::compute::kernels::numeric::add,
    Operator::Add,
    ErrorMode::Ansi,
    [
        (test_add_ansi_i8, i8, DataType::I8, i8, 100, 100_000),
        (test_add_ansi_u8, u8, DataType::U8, u8, 100, 100_000),
        (test_add_ansi_i16, i16, DataType::I16, i16, 500, 500_000),
        (test_add_ansi_u16, u16, DataType::U16, u16, 500, 500_000),
        (test_add_ansi_i32, i32, DataType::I32, i32, 1024, 1024_000),
        (test_add_ansi_u32, u32, DataType::U32, u32, 512, 512_000),
        (test_add_ansi_i64, i64, DataType::I64, i64, 1024, 1024_000),
        (test_add_ansi_u64, u64, DataType::U64, u64, 512, 512_000),
        (test_add_ansi_f32, f32, DataType::F32, f32, 1024, 1024_000),
        (test_add_ansi_f64, f64, DataType::F64, f64, 1000, 1000_000),
    ]
);

test_eval_binary_matrix!(
    arrow::compute::kernels::numeric::sub,
    Operator::Sub,
    ErrorMode::Ansi,
    [
        (test_sub_ansi_i8, i8, DataType::I8, i8, 100, 100_000),
        (test_sub_ansi_u8, u8, DataType::U8, u8, 100, 100_000),
        (test_sub_ansi_i16, i16, DataType::I16, i16, 500, 500_000),
        (test_sub_ansi_u16, u16, DataType::U16, u16, 500, 500_000),
        (test_sub_ansi_i32, i32, DataType::I32, i32, 1024, 1024_000),
        (test_sub_ansi_u32, u32, DataType::U32, u32, 512, 512_000),
        (test_sub_ansi_i64, i64, DataType::I64, i64, 1024, 1024_000),
        (test_sub_ansi_u64, u64, DataType::U64, u64, 512, 512_000),
        (test_sub_ansi_f32, f32, DataType::F32, f32, 1024, 1024_000),
        (test_sub_ansi_f64, f64, DataType::F64, f64, 1000, 1000_000),
    ]
);

test_eval_binary_matrix!(
    arrow::compute::kernels::numeric::mul,
    Operator::Mul,
    ErrorMode::Ansi,
    [
        (test_mul_ansi_i8, i8, DataType::I8, i8, 100, 100_000),
        (test_mul_ansi_u8, u8, DataType::U8, u8, 100, 100_000),
        (test_mul_ansi_i16, i16, DataType::I16, i16, 500, 500_000),
        (test_mul_ansi_u16, u16, DataType::U16, u16, 500, 500_000),
        (test_mul_ansi_i32, i32, DataType::I32, i32, 1024, 1024_000),
        (test_mul_ansi_u32, u32, DataType::U32, u32, 512, 512_000),
        (test_mul_ansi_i64, i64, DataType::I64, i64, 1024, 1024_000),
        (test_mul_ansi_u64, u64, DataType::U64, u64, 512, 512_000),
        (test_mul_ansi_f32, f32, DataType::F32, f32, 1024, 1024_000),
        (test_mul_ansi_f64, f64, DataType::F64, f64, 1000, 1000_000),
    ]
);

test_eval_binary_cmp_matrix!(
    ==,
    Operator::Eq,
    ErrorMode::Tachyon,
    [
        (test_eq_i8, i8, DataType::I8, 100, 500_000),
        (test_eq_u8, u8, DataType::U8, 100,100_000),
        (test_eq_i16, i16, DataType::I16, 500, 500_000),
        (test_eq_u16, u16, DataType::U16, 500, 200_000),
        (test_eq_i32, i32, DataType::I32, 1024, 500_000),
        (test_eq_u32, u32, DataType::U32, 512, 400_000),
        (test_eq_i64, i64, DataType::I64, 1024, 500_000),
        (test_eq_u64, u64, DataType::U64, 512, 500_000),
        (test_eq_f32, f32, DataType::F32, 1024, 500_000),
        (test_eq_f64, f64, DataType::F64, 1000, 500_000),
    ]
);

test_eval_binary_cmp_matrix!(
    !=,
    Operator::NotEq,
    ErrorMode::Tachyon,
    [
        (test_neq_i8, i8, DataType::I8, 100, 400_000),
        (test_neq_u8, u8, DataType::U8, 100, 400_000),
        (test_neq_i16, i16, DataType::I16, 500, 400_000),
        (test_neq_u16, u16, DataType::U16, 500, 500_000),
        (test_neq_i32, i32, DataType::I32, 1024, 400_000),
        (test_neq_u32, u32, DataType::U32, 512, 600_000),
        (test_neq_i64, i64, DataType::I64, 1024, 400_000),
        (test_neq_u64, u64, DataType::U64, 512, 200_000),
        (test_neq_f32, f32, DataType::F32, 1024, 400_000),
        (test_neq_f64, f64, DataType::F64, 1000, 300_000),
    ]
);

test_eval_binary_cmp_matrix!(
    >,
    Operator::Gt,
    ErrorMode::Tachyon,
    [
        (test_gt_i8, i8, DataType::I8, 10, 100_000),
        (test_gt_u8, u8, DataType::U8, 100, 200_000),
        (test_gt_i16, i16, DataType::I16, 500, 100_000),
        (test_gt_u16, u16, DataType::U16, 500, 500_000),
        (test_gt_i32, i32, DataType::I32, 1024, 500_000),
        (test_gt_u32, u32, DataType::U32, 512, 100_000),
        (test_gt_i64, i64, DataType::I64, 1024, 100_000),
        (test_gt_u64, u64, DataType::U64, 512, 100_000),
        (test_gt_f32, f32, DataType::F32, 1024, 400_000),
        (test_gt_f64, f64, DataType::F64, 1000, 100_000),
    ]
);

test_eval_binary_cmp_matrix!(
    >=,
    Operator::GtEq,
    ErrorMode::Tachyon,
    [
        (test_gteq_i8, i8, DataType::I8,  100, 100_000),
        (test_gteq_u8, u8, DataType::U8, 100, 100_000),
        (test_gteq_i16, i16, DataType::I16, 500, 100_000),
        (test_gteq_u16, u16, DataType::U16, 500, 100_000),
        (test_gteq_i32, i32, DataType::I32, 1024, 100_000),
        (test_gteq_u32, u32, DataType::U32, 512, 100_000),
        (test_gteq_i64, i64, DataType::I64, 1024, 100_000),
        (test_gteq_u64, u64, DataType::U64, 512, 100_000),
        (test_gteq_f32, f32, DataType::F32, 1024, 100_000),
        (test_gteq_f64, f64, DataType::F64, 1000, 100_000),
    ]
);

test_eval_binary_cmp_matrix!(
    <,
    Operator::Lt,
    ErrorMode::Tachyon,
    [
        (test_lt_i8, i8, DataType::I8, 100, 500_000),
        (test_lt_u8, u8, DataType::U8, 100, 500_000),
        (test_lt_i16, i16, DataType::I16, 500, 500_000),
        (test_lt_u16, u16, DataType::U16, 500, 500_000),
        (test_lt_i32, i32, DataType::I32, 1024, 500_000),
        (test_lt_u32, u32, DataType::U32, 512, 500_000),
        (test_lt_i64, i64, DataType::I64, 1024, 500_000),
        (test_lt_u64, u64, DataType::U64, 100, 500_000),
        (test_lt_f32, f32, DataType::F32, 1024, 500_000),
        (test_lt_f64, f64, DataType::F64, 100, 500_000),
    ]
);

test_eval_binary_cmp_matrix!(
    <=,
    Operator::LtEq,
    ErrorMode::Tachyon,
    [
        (test_lteq_i8, i8, DataType::I8,  100, 500_000),
        (test_lteq_u8, u8, DataType::U8, 100, 500_000),
        (test_lteq_i16, i16, DataType::I16, 500, 500_000),
        (test_lteq_u16, u16, DataType::U16, 500, 500_000),
        (test_lteq_i32, i32, DataType::I32, 1024, 500_000),
        (test_lteq_u32, u32, DataType::U32, 512, 500_000),
        (test_lteq_i64, i64, DataType::I64, 1024, 500_000),
        (test_lteq_u64, u64, DataType::U64, 512, 500_000),
        (test_lteq_f32, f32, DataType::F32, 1024, 500_000),
        (test_lteq_f64, f64, DataType::F64, 1000, 500_000),
    ]
);
