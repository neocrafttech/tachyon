mod test_utils;
use compute::operator::Operator;
use half::f16;

use crate::test_utils::{ArrowMapper, CastTo, TypeTestRange, init_tracing};

macro_rules! test_eval_binary_matrix {
    (
        $verify_arrow_fn:expr,
        $operator:expr,
        $error_mode:expr,
        $size_min:expr,
        $size_max:expr,
        [
            $(
                ( $test_name:ident, $native_type1:ty, $data_type1:expr, $native_type2:ty, $data_type2:expr, $result_type:ty, $res_data_type:expr)
            ),* $(,)?
        ]
    ) => {
        $(
            test_eval_binary_fn!(
                $verify_arrow_fn,
                $test_name,
                $operator,
                $error_mode,
                $native_type1,
                $data_type1,
                $native_type2,
                $data_type2,
                $result_type,
                $res_data_type,
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
        $native_type1:ty,
        $data_type1:expr,
        $native_type2:ty,
        $data_type2:expr,
        $result_type:ty,
        $res_data_type:expr,
        $size_min:expr,
        $size_max:expr
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
            init_tracing();
            let size = random_num!($size_min, $size_max);
            let value_range1 = <$native_type1>::test_range();
            let value_range2 = <$native_type2>::test_range();
            let a_vec: Vec<$native_type1> =
                random_vec!(size, $native_type1, value_range1.0, value_range1.1);
            let b_vec: Vec<$native_type2> =
                random_vec!(size, $native_type2, value_range2.0, value_range2.1);

            let a_bit_vec = random_bit_vec!(size, u64);
            let b_bit_vec = random_bit_vec!(size, u64);

            let col_a = create_column!(a_vec, Some(a_bit_vec.clone()), "a", $data_type1);
            let col_b = create_column!(b_vec, Some(b_bit_vec.clone()), "b", $data_type2);

            let expr = Expr::binary($operator, Expr::col("a"), Expr::col("b"));

            let result = evaluate(Device::GPU, $error_mode, &expr, &vec![col_a, col_b]).await;

            let epsilon = if $res_data_type.is_float() { 1e-6 } else { 0.0 };

            let arrow_a = create_arrow_array!(a_vec, a_bit_vec, $result_type);
            let arrow_b = create_arrow_array!(b_vec, b_bit_vec, $result_type);

            let arrow_result = $verify_arrow_fn(&arrow_a, &arrow_b);
            match arrow_result {
                Ok(arrow_result) => {
                    let arrow_output = arrow_result
                        .as_any()
                        .downcast_ref::<PrimitiveArray<<$result_type as ArrowMapper>::ArrowType>>()
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
                            let expected: f64 = arrow_output.value(i).cast();
                            let actual: f64 = output[i].cast();
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
                            if actual.is_nan() && expected.is_nan() {
                                //Treat it as equal
                                continue;
                            }
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
        $operator:expr,
        $error_mode:expr,
        $size_min:expr,
        $size_max:expr,
        [
            $(
                ( $test_name:ident, $native_type1:ty, $data_type1:expr, $native_type2:ty, $data_type2:expr)
            ),* $(,)?
        ]
    ) => {
        $(
            test_eval_binary_cmp_fn!(
                $test_name,
                $operator,
                $error_mode,
                $native_type1,
                $data_type1,
                $native_type2,
                $data_type2,
                $size_min,
                $size_max,
            );
        )*
    };
}

macro_rules! test_eval_binary_cmp_fn {
    (
        $test_name:ident,
        $operator:expr,
        $error_mode:expr,
        $native_type1:ty,
        $data_type1:expr,
        $native_type2:ty,
        $data_type2:expr,
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

            use crate::test_utils::compare_numeric_arrays;
            init_tracing();
            let size = random_num!($size_min, $size_max);
            let value_range1 = <$native_type1>::test_range();
            let value_range2 = <$native_type2>::test_range();
            let a_vec: Vec<$native_type1> =
                random_vec!(size, $native_type1, value_range1.0, value_range1.1);
            let b_vec: Vec<$native_type2> =
                random_vec!(size, $native_type2, value_range2.0, value_range2.1);

            let a_bit_vec = random_bit_vec!(size, u32);
            let b_bit_vec = random_bit_vec!(size, u32);

            let col_a = create_column!(a_vec, Some(a_bit_vec.clone()), "a", $data_type1);
            let col_b = create_column!(b_vec, Some(b_bit_vec.clone()), "b", $data_type2);

            let arrow_a = create_arrow_array!(a_vec, a_bit_vec, $native_type1);
            let arrow_b = create_arrow_array!(b_vec, b_bit_vec, $native_type2);

            let expr = Expr::binary($operator, Expr::col("a"), Expr::col("b"));

            let result = evaluate(Device::GPU, $error_mode, &expr, &vec![col_a, col_b]).await;
            assert!(result.is_ok());
            let result = result.unwrap();
            assert!(result[0].data_as_slice::<bool>().is_some());
            let output = result[0].data_as_slice::<bool>().unwrap();
            let bit_vec = result[0].null_bits_as_slice().unwrap();

            let arrow_result = compare_numeric_arrays(&arrow_a, &arrow_b, $operator).unwrap();
            for i in 0..size {
                if a_bit_vec.is_null(i) || b_bit_vec.is_null(i) {
                    assert!(bit_vec.is_null(i));
                } else {
                    let actual = output[i];
                    let expected = arrow_result.value(i);
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
    100,
    500_000,
    [
        // I8 combinations (10 cases - excluding U64)
        (test_add_i8_i8, i8, DataType::I8, i8, DataType::I8, i8, DataType::I8),
        (test_add_i8_i16, i8, DataType::I8, i16, DataType::I16, i16, DataType::I16),
        (test_add_i8_i32, i8, DataType::I8, i32, DataType::I32, i32, DataType::I32),
        (test_add_i8_i64, i8, DataType::I8, i64, DataType::I64, i64, DataType::I64),
        (test_add_i8_u8, i8, DataType::I8, u8, DataType::U8, i16, DataType::I16),
        (test_add_i8_u16, i8, DataType::I8, u16, DataType::U16, i32, DataType::I32),
        (test_add_i8_u32, i8, DataType::I8, u32, DataType::U32, i64, DataType::I64),
        // I8 + U64 omitted (error case)
        (test_add_i8_f16, i8, DataType::I8, f16, DataType::F16, f16, DataType::F16),
        (test_add_i8_f32, i8, DataType::I8, f32, DataType::F32, f32, DataType::F32),
        (test_add_i8_f64, i8, DataType::I8, f64, DataType::F64, f64, DataType::F64),
        // I16 combinations (10 cases - excluding U64)
        (test_add_i16_i8, i16, DataType::I16, i8, DataType::I8, i16, DataType::I16),
        (test_add_i16_i16, i16, DataType::I16, i16, DataType::I16, i16, DataType::I16),
        (test_add_i16_i32, i16, DataType::I16, i32, DataType::I32, i32, DataType::I32),
        (test_add_i16_i64, i16, DataType::I16, i64, DataType::I64, i64, DataType::I64),
        (test_add_i16_u8, i16, DataType::I16, u8, DataType::U8, i16, DataType::I16),
        (test_add_i16_u16, i16, DataType::I16, u16, DataType::U16, i32, DataType::I32),
        (test_add_i16_u32, i16, DataType::I16, u32, DataType::U32, i64, DataType::I64),
        // I16 + U64 omitted (error case)
        (test_add_i16_f16, i16, DataType::I16, f16, DataType::F16, f16, DataType::F16),
        (test_add_i16_f32, i16, DataType::I16, f32, DataType::F32, f32, DataType::F32),
        (test_add_i16_f64, i16, DataType::I16, f64, DataType::F64, f64, DataType::F64),
        // I32 combinations (10 cases - excluding u64)
        (test_add_i32_i8, i32, DataType::I32, i8, DataType::I8, i32, DataType::I32),
        (test_add_i32_i16, i32, DataType::I32, i16, DataType::I16, i32, DataType::I32),
        (test_add_i32_i32, i32, DataType::I32, i32, DataType::I32, i32, DataType::I32),
        (test_add_i32_i64, i32, DataType::I32, i64, DataType::I64, i64, DataType::I64),
        (test_add_i32_u8, i32, DataType::I32, u8, DataType::U8, i32, DataType::I32),
        (test_add_i32_u16, i32, DataType::I32, u16, DataType::U16, i32, DataType::I32),
        (test_add_i32_u32, i32, DataType::I32, u32, DataType::U32, i64, DataType::I64),
        // I32 + U64 omitted (error case)
        (test_add_i32_f16, i32, DataType::I32, f16, DataType::F16, f16, DataType::F16),
        (test_add_i32_f32, i32, DataType::I32, f32, DataType::F32, f32, DataType::F32),
        (test_add_i32_f64, i32, DataType::I32, f64, DataType::F64, f64, DataType::F64),
        // I64 combinations (10 cases - excluding U64)
        (test_add_i64_i8, i64, DataType::I64, i8, DataType::I8, i64, DataType::I64),
        (test_add_i64_i16, i64, DataType::I64, i16, DataType::I16, i64, DataType::I64),
        (test_add_i64_i32, i64, DataType::I64, i32, DataType::I32, i64, DataType::I64),
        (test_add_i64_i64, i64, DataType::I64, i64, DataType::I64, i64, DataType::I64),
        (test_add_i64_u8, i64, DataType::I64, u8, DataType::U8, i64, DataType::I64),
        (test_add_i64_u16, i64, DataType::I64, u16, DataType::U16, i64, DataType::I64),
        (test_add_i64_u32, i64, DataType::I64, u32, DataType::U32, i64, DataType::I64),
        // I64 + U64 omitted (error case)
        (test_add_i64_f16, i64, DataType::I64, f16, DataType::F16, f16, DataType::F16),
        (test_add_i64_f32, i64, DataType::I64, f32, DataType::F32, f64, DataType::F64),
        (test_add_i64_f64, i64, DataType::I64, f64, DataType::F64, f64, DataType::F64),
        // U8 combinations (11 cases)
        (test_add_u8_i8, u8, DataType::U8, i8, DataType::I8, i16, DataType::I16),
        (test_add_u8_i16, u8, DataType::U8, i16, DataType::I16, i16, DataType::I16),
        (test_add_u8_i32, u8, DataType::U8, i32, DataType::I32, i32, DataType::I32),
        (test_add_u8_i64, u8, DataType::U8, i64, DataType::I64, i64, DataType::I64),
        (test_add_u8_u8, u8, DataType::U8, u8, DataType::U8, u8, DataType::U8),
        (test_add_u8_u16, u8, DataType::U8, u16, DataType::U16, u16, DataType::U16),
        (test_add_u8_u32, u8, DataType::U8, u32, DataType::U32, u32, DataType::U32),
        (test_add_u8_u64, u8, DataType::U8, u64, DataType::U64, u64, DataType::U64),
        (test_add_u8_f16, u8, DataType::U8, f16, DataType::F16, f16, DataType::F16),
        (test_add_u8_f32, u8, DataType::U8, f32, DataType::F32, f32, DataType::F32),
        (test_add_u8_f64, u8, DataType::U8, f64, DataType::F64, f64, DataType::F64),
        // U16 combinations (11 cases)
        (test_add_u16_i8, u16, DataType::U16, i8, DataType::I8, i32, DataType::I32),
        (test_add_u16_i16, u16, DataType::U16, i16, DataType::I16, i32, DataType::I32),
        (test_add_u16_i32, u16, DataType::U16, i32, DataType::I32, i32, DataType::I32),
        (test_add_u16_i64, u16, DataType::U16, i64, DataType::I64, i64, DataType::I64),
        (test_add_u16_u8, u16, DataType::U16, u8, DataType::U8, u16, DataType::U16),
        (test_add_u16_u16, u16, DataType::U16, u16, DataType::U16, u16, DataType::U16),
        (test_add_u16_u32, u16, DataType::U16, u32, DataType::U32, u32, DataType::U32),
        (test_add_u16_u64, u16, DataType::U16, u64, DataType::U64, u64, DataType::U64),
        (test_add_u16_f16, u16, DataType::U16, f16, DataType::F16, f16, DataType::F16),
        (test_add_u16_f32, u16, DataType::U16, f32, DataType::F32, f32, DataType::F32),
        (test_add_u16_f64, u16, DataType::U16, f64, DataType::F64, f64, DataType::F64),
        // U32 combinations (11 cases)
        (test_add_u32_i8, u32, DataType::U32, i8, DataType::I8, i64, DataType::I64),
        (test_add_u32_i16, u32, DataType::U32, i16, DataType::I16, i64, DataType::I64),
        (test_add_u32_i32, u32, DataType::U32, i32, DataType::I32, i64, DataType::I64),
        (test_add_u32_i64, u32, DataType::U32, i64, DataType::I64, i64, DataType::I64),
        (test_add_u32_u8, u32, DataType::U32, u8, DataType::U8, u32, DataType::U32),
        (test_add_u32_u16, u32, DataType::U32, u16, DataType::U16, u32, DataType::U32),
        (test_add_u32_u32, u32, DataType::U32, u32, DataType::U32, u32, DataType::U32),
        (test_add_u32_u64, u32, DataType::U32, u64, DataType::U64, u64, DataType::U64),
        (test_add_u32_f16, u32, DataType::U32, f16, DataType::F16, f16, DataType::F16),
        (test_add_u32_f32, u32, DataType::U32, f32, DataType::F32, f32, DataType::F32),
        (test_add_u32_f64, u32, DataType::U32, f64, DataType::F64, f64, DataType::F64),
        // U64 combinations (7 cases - excluding Ix)
        (test_add_u64_u8, u64, DataType::U64, u8, DataType::U8, u64, DataType::U64),
        (test_add_u64_u16, u64, DataType::U64, u16, DataType::U16, u64, DataType::U64),
        (test_add_u64_u32, u64, DataType::U64, u32, DataType::U32, u64, DataType::U64),
        (test_add_u64_u64, u64, DataType::U64, u64, DataType::U64, u64, DataType::U64),
        (test_add_u64_f16, u64, DataType::U64, f16, DataType::F16, f16, DataType::F16),
        (test_add_u64_f32, u64, DataType::U64, f32, DataType::F32, f64, DataType::F64),
        (test_add_u64_f64, u64, DataType::U64, f64, DataType::F64, f64, DataType::F64),
        // F16 combinations (11 cases)
        (test_add_f16_i8, f16, DataType::F16, i8, DataType::I8, f16, DataType::F16),
        (test_add_f16_i16, f16, DataType::F16, i16, DataType::I16, f16, DataType::F16),
        (test_add_f16_i32, f16, DataType::F16, i32, DataType::I32, f16, DataType::F16),
        (test_add_f16_i64, f16, DataType::F16, i64, DataType::I64, f16, DataType::F16),
        (test_add_f16_u8, f16, DataType::F16, u8, DataType::U8, f16, DataType::F16),
        (test_add_f16_u16, f16, DataType::F16, u16, DataType::U16, f16, DataType::F16),
        (test_add_f16_u32, f16, DataType::F16, u32, DataType::U32, f16, DataType::F16),
        (test_add_f16_u64, f16, DataType::F16, u64, DataType::U64, f16, DataType::F16),
        (test_add_f16_f16, f16, DataType::F16, f16, DataType::F16, f16, DataType::F16),
        (test_add_f16_f32, f16, DataType::F16, f32, DataType::F32, f32, DataType::F32),
        (test_add_f16_f64, f16, DataType::F16, f64, DataType::F64, f64, DataType::F64),
        // F32 combinations (11 cases)
        (test_add_f32_i8, f32, DataType::F32, i8, DataType::I8, f32, DataType::F32),
        (test_add_f32_i16, f32, DataType::F32, i16, DataType::I16, f32, DataType::F32),
        (test_add_f32_i32, f32, DataType::F32, i32, DataType::I32, f32, DataType::F32),
        (test_add_f32_i64, f32, DataType::F32, i64, DataType::I64, f64, DataType::F64),
        (test_add_f32_u8, f32, DataType::F32, u8, DataType::U8, f32, DataType::F32),
        (test_add_f32_u16, f32, DataType::F32, u16, DataType::U16, f32, DataType::F32),
        (test_add_f32_u32, f32, DataType::F32, u32, DataType::U32, f32, DataType::F32),
        (test_add_f32_u64, f32, DataType::F32, u64, DataType::U64, f64, DataType::F64),
        (test_add_f32_f16, f32, DataType::F32, f16, DataType::F16, f32, DataType::F32),
        (test_add_f32_f32, f32, DataType::F32, f32, DataType::F32, f32, DataType::F32),
        (test_add_f32_f64, f32, DataType::F32, f64, DataType::F64, f64, DataType::F64),
        // F64 combinations (11 cases)
        (test_add_f64_i8, f64, DataType::F64, i8, DataType::I8, f64, DataType::F64),
        (test_add_f64_i16, f64, DataType::F64, i16, DataType::I16, f64, DataType::F64),
        (test_add_f64_i32, f64, DataType::F64, i32, DataType::I32, f64, DataType::F64),
        (test_add_f64_i64, f64, DataType::F64, i64, DataType::I64, f64, DataType::F64),
        (test_add_f64_u8, f64, DataType::F64, u8, DataType::U8, f64, DataType::F64),
        (test_add_f64_u16, f64, DataType::F64, u16, DataType::U16, f64, DataType::F64),
        (test_add_f64_u32, f64, DataType::F64, u32, DataType::U32, f64, DataType::F64),
        (test_add_f64_u64, f64, DataType::F64, u64, DataType::U64, f64, DataType::F64),
        (test_add_f64_f16, f64, DataType::F64, f16, DataType::F16, f64, DataType::F64),
        (test_add_f64_f32, f64, DataType::F64, f32, DataType::F32, f64, DataType::F64),
        (test_add_f64_f64, f64, DataType::F64, f64, DataType::F64, f64, DataType::F64),
    ]
);

test_eval_binary_matrix!(
    arrow::compute::kernels::numeric::sub_wrapping,
    Operator::Sub,
    ErrorMode::Tachyon,
    10,
    400_000,
    [
        // I8 combinations (10 cases - excluding U64)
        (test_sub_i8_i8, i8, DataType::I8, i8, DataType::I8, i8, DataType::I8),
        (test_sub_i8_i16, i8, DataType::I8, i16, DataType::I16, i16, DataType::I16),
        (test_sub_i8_i32, i8, DataType::I8, i32, DataType::I32, i32, DataType::I32),
        (test_sub_i8_i64, i8, DataType::I8, i64, DataType::I64, i64, DataType::I64),
        (test_sub_i8_u8, i8, DataType::I8, u8, DataType::U8, i16, DataType::I16),
        (test_sub_i8_u16, i8, DataType::I8, u16, DataType::U16, i32, DataType::I32),
        (test_sub_i8_u32, i8, DataType::I8, u32, DataType::U32, i64, DataType::I64),
        // I8 + U64 omitted (error case)
        (test_sub_i8_f16, i8, DataType::I8, f16, DataType::F16, f16, DataType::F16),
        (test_sub_i8_f32, i8, DataType::I8, f32, DataType::F32, f32, DataType::F32),
        (test_sub_i8_f64, i8, DataType::I8, f64, DataType::F64, f64, DataType::F64),
        // I16 combinations (10 cases - excluding U64)
        (test_sub_i16_i8, i16, DataType::I16, i8, DataType::I8, i16, DataType::I16),
        (test_sub_i16_i16, i16, DataType::I16, i16, DataType::I16, i16, DataType::I16),
        (test_sub_i16_i32, i16, DataType::I16, i32, DataType::I32, i32, DataType::I32),
        (test_sub_i16_i64, i16, DataType::I16, i64, DataType::I64, i64, DataType::I64),
        (test_sub_i16_u8, i16, DataType::I16, u8, DataType::U8, i16, DataType::I16),
        (test_sub_i16_u16, i16, DataType::I16, u16, DataType::U16, i32, DataType::I32),
        (test_sub_i16_u32, i16, DataType::I16, u32, DataType::U32, i64, DataType::I64),
        // I16 + U64 omitted (error case)
        (test_sub_i16_f16, i16, DataType::I16, f16, DataType::F16, f16, DataType::F16),
        (test_sub_i16_f32, i16, DataType::I16, f32, DataType::F32, f32, DataType::F32),
        (test_sub_i16_f64, i16, DataType::I16, f64, DataType::F64, f64, DataType::F64),
        // I32 combinations (10 cases - excluding u64)
        (test_sub_i32_i8, i32, DataType::I32, i8, DataType::I8, i32, DataType::I32),
        (test_sub_i32_i16, i32, DataType::I32, i16, DataType::I16, i32, DataType::I32),
        (test_sub_i32_i32, i32, DataType::I32, i32, DataType::I32, i32, DataType::I32),
        (test_sub_i32_i64, i32, DataType::I32, i64, DataType::I64, i64, DataType::I64),
        (test_sub_i32_u8, i32, DataType::I32, u8, DataType::U8, i32, DataType::I32),
        (test_sub_i32_u16, i32, DataType::I32, u16, DataType::U16, i32, DataType::I32),
        (test_sub_i32_u32, i32, DataType::I32, u32, DataType::U32, i64, DataType::I64),
        // I32 + U64 omitted (error case)
        (test_sub_i32_f16, i32, DataType::I32, f16, DataType::F16, f16, DataType::F16),
        (test_sub_i32_f32, i32, DataType::I32, f32, DataType::F32, f32, DataType::F32),
        (test_sub_i32_f64, i32, DataType::I32, f64, DataType::F64, f64, DataType::F64),
        // I64 combinations (10 cases - excluding U64)
        (test_sub_i64_i8, i64, DataType::I64, i8, DataType::I8, i64, DataType::I64),
        (test_sub_i64_i16, i64, DataType::I64, i16, DataType::I16, i64, DataType::I64),
        (test_sub_i64_i32, i64, DataType::I64, i32, DataType::I32, i64, DataType::I64),
        (test_sub_i64_i64, i64, DataType::I64, i64, DataType::I64, i64, DataType::I64),
        (test_sub_i64_u8, i64, DataType::I64, u8, DataType::U8, i64, DataType::I64),
        (test_sub_i64_u16, i64, DataType::I64, u16, DataType::U16, i64, DataType::I64),
        (test_sub_i64_u32, i64, DataType::I64, u32, DataType::U32, i64, DataType::I64),
        // I64 + U64 omitted (error case)
        (test_sub_i64_f16, i64, DataType::I64, f16, DataType::F16, f16, DataType::F16),
        (test_sub_i64_f32, i64, DataType::I64, f32, DataType::F32, f64, DataType::F64),
        (test_sub_i64_f64, i64, DataType::I64, f64, DataType::F64, f64, DataType::F64),
        // U8 combinations (11 cases)
        (test_sub_u8_i8, u8, DataType::U8, i8, DataType::I8, i16, DataType::I16),
        (test_sub_u8_i16, u8, DataType::U8, i16, DataType::I16, i16, DataType::I16),
        (test_sub_u8_i32, u8, DataType::U8, i32, DataType::I32, i32, DataType::I32),
        (test_sub_u8_i64, u8, DataType::U8, i64, DataType::I64, i64, DataType::I64),
        (test_sub_u8_u8, u8, DataType::U8, u8, DataType::U8, u8, DataType::U8),
        (test_sub_u8_u16, u8, DataType::U8, u16, DataType::U16, u16, DataType::U16),
        (test_sub_u8_u32, u8, DataType::U8, u32, DataType::U32, u32, DataType::U32),
        (test_sub_u8_u64, u8, DataType::U8, u64, DataType::U64, u64, DataType::U64),
        (test_sub_u8_f16, u8, DataType::U8, f16, DataType::F16, f16, DataType::F16),
        (test_sub_u8_f32, u8, DataType::U8, f32, DataType::F32, f32, DataType::F32),
        (test_sub_u8_f64, u8, DataType::U8, f64, DataType::F64, f64, DataType::F64),
        // U16 combinations (11 cases)
        (test_sub_u16_i8, u16, DataType::U16, i8, DataType::I8, i32, DataType::I32),
        (test_sub_u16_i16, u16, DataType::U16, i16, DataType::I16, i32, DataType::I32),
        (test_sub_u16_i32, u16, DataType::U16, i32, DataType::I32, i32, DataType::I32),
        (test_sub_u16_i64, u16, DataType::U16, i64, DataType::I64, i64, DataType::I64),
        (test_sub_u16_u8, u16, DataType::U16, u8, DataType::U8, u16, DataType::U16),
        (test_sub_u16_u16, u16, DataType::U16, u16, DataType::U16, u16, DataType::U16),
        (test_sub_u16_u32, u16, DataType::U16, u32, DataType::U32, u32, DataType::U32),
        (test_sub_u16_u64, u16, DataType::U16, u64, DataType::U64, u64, DataType::U64),
        (test_sub_u16_f16, u16, DataType::U16, f16, DataType::F16, f16, DataType::F16),
        (test_sub_u16_f32, u16, DataType::U16, f32, DataType::F32, f32, DataType::F32),
        (test_sub_u16_f64, u16, DataType::U16, f64, DataType::F64, f64, DataType::F64),
        // U32 combinations (11 cases)
        (test_sub_u32_i8, u32, DataType::U32, i8, DataType::I8, i64, DataType::I64),
        (test_sub_u32_i16, u32, DataType::U32, i16, DataType::I16, i64, DataType::I64),
        (test_sub_u32_i32, u32, DataType::U32, i32, DataType::I32, i64, DataType::I64),
        (test_sub_u32_i64, u32, DataType::U32, i64, DataType::I64, i64, DataType::I64),
        (test_sub_u32_u8, u32, DataType::U32, u8, DataType::U8, u32, DataType::U32),
        (test_sub_u32_u16, u32, DataType::U32, u16, DataType::U16, u32, DataType::U32),
        (test_sub_u32_u32, u32, DataType::U32, u32, DataType::U32, u32, DataType::U32),
        (test_sub_u32_u64, u32, DataType::U32, u64, DataType::U64, u64, DataType::U64),
        (test_sub_u32_f16, u32, DataType::U32, f16, DataType::F16, f16, DataType::F16),
        (test_sub_u32_f32, u32, DataType::U32, f32, DataType::F32, f32, DataType::F32),
        (test_sub_u32_f64, u32, DataType::U32, f64, DataType::F64, f64, DataType::F64),
        // U64 combinations (7 cases - excluding Ix)
        (test_sub_u64_u8, u64, DataType::U64, u8, DataType::U8, u64, DataType::U64),
        (test_sub_u64_u16, u64, DataType::U64, u16, DataType::U16, u64, DataType::U64),
        (test_sub_u64_u32, u64, DataType::U64, u32, DataType::U32, u64, DataType::U64),
        (test_sub_u64_u64, u64, DataType::U64, u64, DataType::U64, u64, DataType::U64),
        (test_sub_u64_f16, u64, DataType::U64, f16, DataType::F16, f16, DataType::F16),
        (test_sub_u64_f32, u64, DataType::U64, f32, DataType::F32, f64, DataType::F64),
        (test_sub_u64_f64, u64, DataType::U64, f64, DataType::F64, f64, DataType::F64),
        // F16 combinations (11 cases)
        (test_sub_f16_i8, f16, DataType::F16, i8, DataType::I8, f16, DataType::F16),
        (test_sub_f16_i16, f16, DataType::F16, i16, DataType::I16, f16, DataType::F16),
        (test_sub_f16_i32, f16, DataType::F16, i32, DataType::I32, f16, DataType::F16),
        (test_sub_f16_i64, f16, DataType::F16, i64, DataType::I64, f16, DataType::F16),
        (test_sub_f16_u8, f16, DataType::F16, u8, DataType::U8, f16, DataType::F16),
        (test_sub_f16_u16, f16, DataType::F16, u16, DataType::U16, f16, DataType::F16),
        (test_sub_f16_u32, f16, DataType::F16, u32, DataType::U32, f16, DataType::F16),
        (test_sub_f16_u64, f16, DataType::F16, u64, DataType::U64, f16, DataType::F16),
        (test_sub_f16_f16, f16, DataType::F16, f16, DataType::F16, f16, DataType::F16),
        (test_sub_f16_f32, f16, DataType::F16, f32, DataType::F32, f32, DataType::F32),
        (test_sub_f16_f64, f16, DataType::F16, f64, DataType::F64, f64, DataType::F64),
        // F32 combinations (11 cases)
        (test_sub_f32_i8, f32, DataType::F32, i8, DataType::I8, f32, DataType::F32),
        (test_sub_f32_i16, f32, DataType::F32, i16, DataType::I16, f32, DataType::F32),
        (test_sub_f32_i32, f32, DataType::F32, i32, DataType::I32, f32, DataType::F32),
        (test_sub_f32_i64, f32, DataType::F32, i64, DataType::I64, f64, DataType::F64),
        (test_sub_f32_u8, f32, DataType::F32, u8, DataType::U8, f32, DataType::F32),
        (test_sub_f32_u16, f32, DataType::F32, u16, DataType::U16, f32, DataType::F32),
        (test_sub_f32_u32, f32, DataType::F32, u32, DataType::U32, f32, DataType::F32),
        (test_sub_f32_u64, f32, DataType::F32, u64, DataType::U64, f64, DataType::F64),
        (test_sub_f32_f16, f32, DataType::F32, f16, DataType::F16, f32, DataType::F32),
        (test_sub_f32_f32, f32, DataType::F32, f32, DataType::F32, f32, DataType::F32),
        (test_sub_f32_f64, f32, DataType::F32, f64, DataType::F64, f64, DataType::F64),
        // F64 combinations (11 cases)
        (test_sub_f64_i8, f64, DataType::F64, i8, DataType::I8, f64, DataType::F64),
        (test_sub_f64_i16, f64, DataType::F64, i16, DataType::I16, f64, DataType::F64),
        (test_sub_f64_i32, f64, DataType::F64, i32, DataType::I32, f64, DataType::F64),
        (test_sub_f64_i64, f64, DataType::F64, i64, DataType::I64, f64, DataType::F64),
        (test_sub_f64_u8, f64, DataType::F64, u8, DataType::U8, f64, DataType::F64),
        (test_sub_f64_u16, f64, DataType::F64, u16, DataType::U16, f64, DataType::F64),
        (test_sub_f64_u32, f64, DataType::F64, u32, DataType::U32, f64, DataType::F64),
        (test_sub_f64_u64, f64, DataType::F64, u64, DataType::U64, f64, DataType::F64),
        (test_sub_f64_f16, f64, DataType::F64, f16, DataType::F16, f64, DataType::F64),
        (test_sub_f64_f32, f64, DataType::F64, f32, DataType::F32, f64, DataType::F64),
        (test_sub_f64_f64, f64, DataType::F64, f64, DataType::F64, f64, DataType::F64),
    ]
);

test_eval_binary_matrix!(
    arrow::compute::kernels::numeric::mul_wrapping,
    Operator::Mul,
    ErrorMode::Tachyon,
    100,
    200_000,
    [
        // I8 combinations (10 cases - excluding U64)
        (test_mul_i8_i8, i8, DataType::I8, i8, DataType::I8, i8, DataType::I8),
        (test_mul_i8_i16, i8, DataType::I8, i16, DataType::I16, i16, DataType::I16),
        (test_mul_i8_i32, i8, DataType::I8, i32, DataType::I32, i32, DataType::I32),
        (test_mul_i8_i64, i8, DataType::I8, i64, DataType::I64, i64, DataType::I64),
        (test_mul_i8_u8, i8, DataType::I8, u8, DataType::U8, i16, DataType::I16),
        (test_mul_i8_u16, i8, DataType::I8, u16, DataType::U16, i32, DataType::I32),
        (test_mul_i8_u32, i8, DataType::I8, u32, DataType::U32, i64, DataType::I64),
        // I8 + U64 omitted (error case)
        (test_mul_i8_f16, i8, DataType::I8, f16, DataType::F16, f16, DataType::F16),
        (test_mul_i8_f32, i8, DataType::I8, f32, DataType::F32, f32, DataType::F32),
        (test_mul_i8_f64, i8, DataType::I8, f64, DataType::F64, f64, DataType::F64),
        // I16 combinations (10 cases - excluding U64)
        (test_mul_i16_i8, i16, DataType::I16, i8, DataType::I8, i16, DataType::I16),
        (test_mul_i16_i16, i16, DataType::I16, i16, DataType::I16, i16, DataType::I16),
        (test_mul_i16_i32, i16, DataType::I16, i32, DataType::I32, i32, DataType::I32),
        (test_mul_i16_i64, i16, DataType::I16, i64, DataType::I64, i64, DataType::I64),
        (test_mul_i16_u8, i16, DataType::I16, u8, DataType::U8, i16, DataType::I16),
        (test_mul_i16_u16, i16, DataType::I16, u16, DataType::U16, i32, DataType::I32),
        (test_mul_i16_u32, i16, DataType::I16, u32, DataType::U32, i64, DataType::I64),
        // I16 + U64 omitted (error case)
        (test_mul_i16_f16, i16, DataType::I16, f16, DataType::F16, f16, DataType::F16),
        (test_mul_i16_f32, i16, DataType::I16, f32, DataType::F32, f32, DataType::F32),
        (test_mul_i16_f64, i16, DataType::I16, f64, DataType::F64, f64, DataType::F64),
        // I32 combinations (10 cases - excluding u64)
        (test_mul_i32_i8, i32, DataType::I32, i8, DataType::I8, i32, DataType::I32),
        (test_mul_i32_i16, i32, DataType::I32, i16, DataType::I16, i32, DataType::I32),
        (test_mul_i32_i32, i32, DataType::I32, i32, DataType::I32, i32, DataType::I32),
        (test_mul_i32_i64, i32, DataType::I32, i64, DataType::I64, i64, DataType::I64),
        (test_mul_i32_u8, i32, DataType::I32, u8, DataType::U8, i32, DataType::I32),
        (test_mul_i32_u16, i32, DataType::I32, u16, DataType::U16, i32, DataType::I32),
        (test_mul_i32_u32, i32, DataType::I32, u32, DataType::U32, i64, DataType::I64),
        // I32 + U64 omitted (error case)
        (test_mul_i32_f16, i32, DataType::I32, f16, DataType::F16, f16, DataType::F16),
        (test_mul_i32_f32, i32, DataType::I32, f32, DataType::F32, f32, DataType::F32),
        (test_mul_i32_f64, i32, DataType::I32, f64, DataType::F64, f64, DataType::F64),
        // I64 combinations (10 cases - excluding U64)
        (test_mul_i64_i8, i64, DataType::I64, i8, DataType::I8, i64, DataType::I64),
        (test_mul_i64_i16, i64, DataType::I64, i16, DataType::I16, i64, DataType::I64),
        (test_mul_i64_i32, i64, DataType::I64, i32, DataType::I32, i64, DataType::I64),
        (test_mul_i64_i64, i64, DataType::I64, i64, DataType::I64, i64, DataType::I64),
        (test_mul_i64_u8, i64, DataType::I64, u8, DataType::U8, i64, DataType::I64),
        (test_mul_i64_u16, i64, DataType::I64, u16, DataType::U16, i64, DataType::I64),
        (test_mul_i64_u32, i64, DataType::I64, u32, DataType::U32, i64, DataType::I64),
        // I64 + U64 omitted (error case)
        (test_mul_i64_f16, i64, DataType::I64, f16, DataType::F16, f16, DataType::F16),
        (test_mul_i64_f32, i64, DataType::I64, f32, DataType::F32, f64, DataType::F64),
        (test_mul_i64_f64, i64, DataType::I64, f64, DataType::F64, f64, DataType::F64),
        // U8 combinations (11 cases)
        (test_mul_u8_i8, u8, DataType::U8, i8, DataType::I8, i16, DataType::I16),
        (test_mul_u8_i16, u8, DataType::U8, i16, DataType::I16, i16, DataType::I16),
        (test_mul_u8_i32, u8, DataType::U8, i32, DataType::I32, i32, DataType::I32),
        (test_mul_u8_i64, u8, DataType::U8, i64, DataType::I64, i64, DataType::I64),
        (test_mul_u8_u8, u8, DataType::U8, u8, DataType::U8, u8, DataType::U8),
        (test_mul_u8_u16, u8, DataType::U8, u16, DataType::U16, u16, DataType::U16),
        (test_mul_u8_u32, u8, DataType::U8, u32, DataType::U32, u32, DataType::U32),
        (test_mul_u8_u64, u8, DataType::U8, u64, DataType::U64, u64, DataType::U64),
        (test_mul_u8_f16, u8, DataType::U8, f16, DataType::F16, f16, DataType::F16),
        (test_mul_u8_f32, u8, DataType::U8, f32, DataType::F32, f32, DataType::F32),
        (test_mul_u8_f64, u8, DataType::U8, f64, DataType::F64, f64, DataType::F64),
        // U16 combinations (11 cases)
        (test_mul_u16_i8, u16, DataType::U16, i8, DataType::I8, i32, DataType::I32),
        (test_mul_u16_i16, u16, DataType::U16, i16, DataType::I16, i32, DataType::I32),
        (test_mul_u16_i32, u16, DataType::U16, i32, DataType::I32, i32, DataType::I32),
        (test_mul_u16_i64, u16, DataType::U16, i64, DataType::I64, i64, DataType::I64),
        (test_mul_u16_u8, u16, DataType::U16, u8, DataType::U8, u16, DataType::U16),
        (test_mul_u16_u16, u16, DataType::U16, u16, DataType::U16, u16, DataType::U16),
        (test_mul_u16_u32, u16, DataType::U16, u32, DataType::U32, u32, DataType::U32),
        (test_mul_u16_u64, u16, DataType::U16, u64, DataType::U64, u64, DataType::U64),
        (test_mul_u16_f16, u16, DataType::U16, f16, DataType::F16, f16, DataType::F16),
        (test_mul_u16_f32, u16, DataType::U16, f32, DataType::F32, f32, DataType::F32),
        (test_mul_u16_f64, u16, DataType::U16, f64, DataType::F64, f64, DataType::F64),
        // U32 combinations (11 cases)
        (test_mul_u32_i8, u32, DataType::U32, i8, DataType::I8, i64, DataType::I64),
        (test_mul_u32_i16, u32, DataType::U32, i16, DataType::I16, i64, DataType::I64),
        (test_mul_u32_i32, u32, DataType::U32, i32, DataType::I32, i64, DataType::I64),
        (test_mul_u32_i64, u32, DataType::U32, i64, DataType::I64, i64, DataType::I64),
        (test_mul_u32_u8, u32, DataType::U32, u8, DataType::U8, u32, DataType::U32),
        (test_mul_u32_u16, u32, DataType::U32, u16, DataType::U16, u32, DataType::U32),
        (test_mul_u32_u32, u32, DataType::U32, u32, DataType::U32, u32, DataType::U32),
        (test_mul_u32_u64, u32, DataType::U32, u64, DataType::U64, u64, DataType::U64),
        (test_mul_u32_f16, u32, DataType::U32, f16, DataType::F16, f16, DataType::F16),
        (test_mul_u32_f32, u32, DataType::U32, f32, DataType::F32, f32, DataType::F32),
        (test_mul_u32_f64, u32, DataType::U32, f64, DataType::F64, f64, DataType::F64),
        // U64 combinations (7 cases - excluding Ix)
        (test_mul_u64_u8, u64, DataType::U64, u8, DataType::U8, u64, DataType::U64),
        (test_mul_u64_u16, u64, DataType::U64, u16, DataType::U16, u64, DataType::U64),
        (test_mul_u64_u32, u64, DataType::U64, u32, DataType::U32, u64, DataType::U64),
        (test_mul_u64_u64, u64, DataType::U64, u64, DataType::U64, u64, DataType::U64),
        (test_mul_u64_f16, u64, DataType::U64, f16, DataType::F16, f16, DataType::F16),
        (test_mul_u64_f32, u64, DataType::U64, f32, DataType::F32, f64, DataType::F64),
        (test_mul_u64_f64, u64, DataType::U64, f64, DataType::F64, f64, DataType::F64),
        // F16 combinations (11 cases)
        (test_mul_f16_i8, f16, DataType::F16, i8, DataType::I8, f16, DataType::F16),
        (test_mul_f16_i16, f16, DataType::F16, i16, DataType::I16, f16, DataType::F16),
        (test_mul_f16_i32, f16, DataType::F16, i32, DataType::I32, f16, DataType::F16),
        (test_mul_f16_i64, f16, DataType::F16, i64, DataType::I64, f16, DataType::F16),
        (test_mul_f16_u8, f16, DataType::F16, u8, DataType::U8, f16, DataType::F16),
        (test_mul_f16_u16, f16, DataType::F16, u16, DataType::U16, f16, DataType::F16),
        (test_mul_f16_u32, f16, DataType::F16, u32, DataType::U32, f16, DataType::F16),
        (test_mul_f16_u64, f16, DataType::F16, u64, DataType::U64, f16, DataType::F16),
        (test_mul_f16_f16, f16, DataType::F16, f16, DataType::F16, f16, DataType::F16),
        (test_mul_f16_f32, f16, DataType::F16, f32, DataType::F32, f32, DataType::F32),
        (test_mul_f16_f64, f16, DataType::F16, f64, DataType::F64, f64, DataType::F64),
        // F32 combinations (11 cases)
        (test_mul_f32_i8, f32, DataType::F32, i8, DataType::I8, f32, DataType::F32),
        (test_mul_f32_i16, f32, DataType::F32, i16, DataType::I16, f32, DataType::F32),
        (test_mul_f32_i32, f32, DataType::F32, i32, DataType::I32, f32, DataType::F32),
        (test_mul_f32_i64, f32, DataType::F32, i64, DataType::I64, f64, DataType::F64),
        (test_mul_f32_u8, f32, DataType::F32, u8, DataType::U8, f32, DataType::F32),
        (test_mul_f32_u16, f32, DataType::F32, u16, DataType::U16, f32, DataType::F32),
        (test_mul_f32_u32, f32, DataType::F32, u32, DataType::U32, f32, DataType::F32),
        (test_mul_f32_u64, f32, DataType::F32, u64, DataType::U64, f64, DataType::F64),
        (test_mul_f32_f16, f32, DataType::F32, f16, DataType::F16, f32, DataType::F32),
        (test_mul_f32_f32, f32, DataType::F32, f32, DataType::F32, f32, DataType::F32),
        (test_mul_f32_f64, f32, DataType::F32, f64, DataType::F64, f64, DataType::F64),
        // F64 combinations (11 cases)
        (test_mul_f64_i8, f64, DataType::F64, i8, DataType::I8, f64, DataType::F64),
        (test_mul_f64_i16, f64, DataType::F64, i16, DataType::I16, f64, DataType::F64),
        (test_mul_f64_i32, f64, DataType::F64, i32, DataType::I32, f64, DataType::F64),
        (test_mul_f64_i64, f64, DataType::F64, i64, DataType::I64, f64, DataType::F64),
        (test_mul_f64_u8, f64, DataType::F64, u8, DataType::U8, f64, DataType::F64),
        (test_mul_f64_u16, f64, DataType::F64, u16, DataType::U16, f64, DataType::F64),
        (test_mul_f64_u32, f64, DataType::F64, u32, DataType::U32, f64, DataType::F64),
        (test_mul_f64_u64, f64, DataType::F64, u64, DataType::U64, f64, DataType::F64),
        (test_mul_f64_f16, f64, DataType::F64, f16, DataType::F16, f64, DataType::F64),
        (test_mul_f64_f32, f64, DataType::F64, f32, DataType::F32, f64, DataType::F64),
        (test_mul_f64_f64, f64, DataType::F64, f64, DataType::F64, f64, DataType::F64),
    ]
);

test_eval_binary_matrix!(
    arrow::compute::kernels::numeric::div,
    Operator::Div,
    ErrorMode::Tachyon,
    200,
    400_000,
    [
        // I8 combinations (10 cases - excluding U64)
        (test_div_i8_i8, i8, DataType::I8, i8, DataType::I8, i8, DataType::I8),
        (test_div_i8_i16, i8, DataType::I8, i16, DataType::I16, i16, DataType::I16),
        (test_div_i8_i32, i8, DataType::I8, i32, DataType::I32, i32, DataType::I32),
        (test_div_i8_i64, i8, DataType::I8, i64, DataType::I64, i64, DataType::I64),
        (test_div_i8_u8, i8, DataType::I8, u8, DataType::U8, i16, DataType::I16),
        (test_div_i8_u16, i8, DataType::I8, u16, DataType::U16, i32, DataType::I32),
        (test_div_i8_u32, i8, DataType::I8, u32, DataType::U32, i64, DataType::I64),
        // I8 + U64 omitted (error case)
        (test_div_i8_f16, i8, DataType::I8, f16, DataType::F16, f16, DataType::F16),
        (test_div_i8_f32, i8, DataType::I8, f32, DataType::F32, f32, DataType::F32),
        (test_div_i8_f64, i8, DataType::I8, f64, DataType::F64, f64, DataType::F64),
        // I16 combinations (10 cases - excluding U64)
        (test_div_i16_i8, i16, DataType::I16, i8, DataType::I8, i16, DataType::I16),
        (test_div_i16_i16, i16, DataType::I16, i16, DataType::I16, i16, DataType::I16),
        (test_div_i16_i32, i16, DataType::I16, i32, DataType::I32, i32, DataType::I32),
        (test_div_i16_i64, i16, DataType::I16, i64, DataType::I64, i64, DataType::I64),
        (test_div_i16_u8, i16, DataType::I16, u8, DataType::U8, i16, DataType::I16),
        (test_div_i16_u16, i16, DataType::I16, u16, DataType::U16, i32, DataType::I32),
        (test_div_i16_u32, i16, DataType::I16, u32, DataType::U32, i64, DataType::I64),
        // I16 + U64 omitted (error case)
        (test_div_i16_f16, i16, DataType::I16, f16, DataType::F16, f16, DataType::F16),
        (test_div_i16_f32, i16, DataType::I16, f32, DataType::F32, f32, DataType::F32),
        (test_div_i16_f64, i16, DataType::I16, f64, DataType::F64, f64, DataType::F64),
        // I32 combinations (10 cases - excluding u64)
        (test_div_i32_i8, i32, DataType::I32, i8, DataType::I8, i32, DataType::I32),
        (test_div_i32_i16, i32, DataType::I32, i16, DataType::I16, i32, DataType::I32),
        (test_div_i32_i32, i32, DataType::I32, i32, DataType::I32, i32, DataType::I32),
        (test_div_i32_i64, i32, DataType::I32, i64, DataType::I64, i64, DataType::I64),
        (test_div_i32_u8, i32, DataType::I32, u8, DataType::U8, i32, DataType::I32),
        (test_div_i32_u16, i32, DataType::I32, u16, DataType::U16, i32, DataType::I32),
        (test_div_i32_u32, i32, DataType::I32, u32, DataType::U32, i64, DataType::I64),
        // I32 + U64 omitted (error case)
        (test_div_i32_f16, i32, DataType::I32, f16, DataType::F16, f16, DataType::F16),
        (test_div_i32_f32, i32, DataType::I32, f32, DataType::F32, f32, DataType::F32),
        (test_div_i32_f64, i32, DataType::I32, f64, DataType::F64, f64, DataType::F64),
        // I64 combinations (10 cases - excluding U64)
        (test_div_i64_i8, i64, DataType::I64, i8, DataType::I8, i64, DataType::I64),
        (test_div_i64_i16, i64, DataType::I64, i16, DataType::I16, i64, DataType::I64),
        (test_div_i64_i32, i64, DataType::I64, i32, DataType::I32, i64, DataType::I64),
        (test_div_i64_i64, i64, DataType::I64, i64, DataType::I64, i64, DataType::I64),
        (test_div_i64_u8, i64, DataType::I64, u8, DataType::U8, i64, DataType::I64),
        (test_div_i64_u16, i64, DataType::I64, u16, DataType::U16, i64, DataType::I64),
        (test_div_i64_u32, i64, DataType::I64, u32, DataType::U32, i64, DataType::I64),
        // I64 + U64 omitted (error case)
        (test_div_i64_f16, i64, DataType::I64, f16, DataType::F16, f16, DataType::F16),
        (test_div_i64_f32, i64, DataType::I64, f32, DataType::F32, f64, DataType::F64),
        (test_div_i64_f64, i64, DataType::I64, f64, DataType::F64, f64, DataType::F64),
        // U8 combinations (11 cases)
        (test_div_u8_i8, u8, DataType::U8, i8, DataType::I8, i16, DataType::I16),
        (test_div_u8_i16, u8, DataType::U8, i16, DataType::I16, i16, DataType::I16),
        (test_div_u8_i32, u8, DataType::U8, i32, DataType::I32, i32, DataType::I32),
        (test_div_u8_i64, u8, DataType::U8, i64, DataType::I64, i64, DataType::I64),
        (test_div_u8_u8, u8, DataType::U8, u8, DataType::U8, u8, DataType::U8),
        (test_div_u8_u16, u8, DataType::U8, u16, DataType::U16, u16, DataType::U16),
        (test_div_u8_u32, u8, DataType::U8, u32, DataType::U32, u32, DataType::U32),
        (test_div_u8_u64, u8, DataType::U8, u64, DataType::U64, u64, DataType::U64),
        (test_div_u8_f16, u8, DataType::U8, f16, DataType::F16, f16, DataType::F16),
        (test_div_u8_f32, u8, DataType::U8, f32, DataType::F32, f32, DataType::F32),
        (test_div_u8_f64, u8, DataType::U8, f64, DataType::F64, f64, DataType::F64),
        // U16 combinations (11 cases)
        (test_div_u16_i8, u16, DataType::U16, i8, DataType::I8, i32, DataType::I32),
        (test_div_u16_i16, u16, DataType::U16, i16, DataType::I16, i32, DataType::I32),
        (test_div_u16_i32, u16, DataType::U16, i32, DataType::I32, i32, DataType::I32),
        (test_div_u16_i64, u16, DataType::U16, i64, DataType::I64, i64, DataType::I64),
        (test_div_u16_u8, u16, DataType::U16, u8, DataType::U8, u16, DataType::U16),
        (test_div_u16_u16, u16, DataType::U16, u16, DataType::U16, u16, DataType::U16),
        (test_div_u16_u32, u16, DataType::U16, u32, DataType::U32, u32, DataType::U32),
        (test_div_u16_u64, u16, DataType::U16, u64, DataType::U64, u64, DataType::U64),
        (test_div_u16_f16, u16, DataType::U16, f16, DataType::F16, f16, DataType::F16),
        (test_div_u16_f32, u16, DataType::U16, f32, DataType::F32, f32, DataType::F32),
        (test_div_u16_f64, u16, DataType::U16, f64, DataType::F64, f64, DataType::F64),
        // U32 combinations (11 cases)
        (test_div_u32_i8, u32, DataType::U32, i8, DataType::I8, i64, DataType::I64),
        (test_div_u32_i16, u32, DataType::U32, i16, DataType::I16, i64, DataType::I64),
        (test_div_u32_i32, u32, DataType::U32, i32, DataType::I32, i64, DataType::I64),
        (test_div_u32_i64, u32, DataType::U32, i64, DataType::I64, i64, DataType::I64),
        (test_div_u32_u8, u32, DataType::U32, u8, DataType::U8, u32, DataType::U32),
        (test_div_u32_u16, u32, DataType::U32, u16, DataType::U16, u32, DataType::U32),
        (test_div_u32_u32, u32, DataType::U32, u32, DataType::U32, u32, DataType::U32),
        (test_div_u32_u64, u32, DataType::U32, u64, DataType::U64, u64, DataType::U64),
        (test_div_u32_f16, u32, DataType::U32, f16, DataType::F16, f16, DataType::F16),
        (test_div_u32_f32, u32, DataType::U32, f32, DataType::F32, f32, DataType::F32),
        (test_div_u32_f64, u32, DataType::U32, f64, DataType::F64, f64, DataType::F64),
        // U64 combinations (7 cases - excluding Ix)
        (test_div_u64_u8, u64, DataType::U64, u8, DataType::U8, u64, DataType::U64),
        (test_div_u64_u16, u64, DataType::U64, u16, DataType::U16, u64, DataType::U64),
        (test_div_u64_u32, u64, DataType::U64, u32, DataType::U32, u64, DataType::U64),
        (test_div_u64_u64, u64, DataType::U64, u64, DataType::U64, u64, DataType::U64),
        (test_div_u64_f16, u64, DataType::U64, f16, DataType::F16, f16, DataType::F16),
        (test_div_u64_f32, u64, DataType::U64, f32, DataType::F32, f64, DataType::F64),
        (test_div_u64_f64, u64, DataType::U64, f64, DataType::F64, f64, DataType::F64),
        // F16 combinations (11 cases)
        (test_div_f16_i8, f16, DataType::F16, i8, DataType::I8, f16, DataType::F16),
        (test_div_f16_i16, f16, DataType::F16, i16, DataType::I16, f16, DataType::F16),
        (test_div_f16_i32, f16, DataType::F16, i32, DataType::I32, f16, DataType::F16),
        (test_div_f16_i64, f16, DataType::F16, i64, DataType::I64, f16, DataType::F16),
        (test_div_f16_u8, f16, DataType::F16, u8, DataType::U8, f16, DataType::F16),
        (test_div_f16_u16, f16, DataType::F16, u16, DataType::U16, f16, DataType::F16),
        (test_div_f16_u32, f16, DataType::F16, u32, DataType::U32, f16, DataType::F16),
        (test_div_f16_u64, f16, DataType::F16, u64, DataType::U64, f16, DataType::F16),
        (test_div_f16_f16, f16, DataType::F16, f16, DataType::F16, f16, DataType::F16),
        (test_div_f16_f32, f16, DataType::F16, f32, DataType::F32, f32, DataType::F32),
        (test_div_f16_f64, f16, DataType::F16, f64, DataType::F64, f64, DataType::F64),
        // F32 combinations (11 cases)
        (test_div_f32_i8, f32, DataType::F32, i8, DataType::I8, f32, DataType::F32),
        (test_div_f32_i16, f32, DataType::F32, i16, DataType::I16, f32, DataType::F32),
        (test_div_f32_i32, f32, DataType::F32, i32, DataType::I32, f32, DataType::F32),
        (test_div_f32_i64, f32, DataType::F32, i64, DataType::I64, f64, DataType::F64),
        (test_div_f32_u8, f32, DataType::F32, u8, DataType::U8, f32, DataType::F32),
        (test_div_f32_u16, f32, DataType::F32, u16, DataType::U16, f32, DataType::F32),
        (test_div_f32_u32, f32, DataType::F32, u32, DataType::U32, f32, DataType::F32),
        (test_div_f32_u64, f32, DataType::F32, u64, DataType::U64, f64, DataType::F64),
        (test_div_f32_f16, f32, DataType::F32, f16, DataType::F16, f32, DataType::F32),
        (test_div_f32_f32, f32, DataType::F32, f32, DataType::F32, f32, DataType::F32),
        (test_div_f32_f64, f32, DataType::F32, f64, DataType::F64, f64, DataType::F64),
        // F64 combinations (11 cases)
        (test_div_f64_i8, f64, DataType::F64, i8, DataType::I8, f64, DataType::F64),
        (test_div_f64_i16, f64, DataType::F64, i16, DataType::I16, f64, DataType::F64),
        (test_div_f64_i32, f64, DataType::F64, i32, DataType::I32, f64, DataType::F64),
        (test_div_f64_i64, f64, DataType::F64, i64, DataType::I64, f64, DataType::F64),
        (test_div_f64_u8, f64, DataType::F64, u8, DataType::U8, f64, DataType::F64),
        (test_div_f64_u16, f64, DataType::F64, u16, DataType::U16, f64, DataType::F64),
        (test_div_f64_u32, f64, DataType::F64, u32, DataType::U32, f64, DataType::F64),
        (test_div_f64_u64, f64, DataType::F64, u64, DataType::U64, f64, DataType::F64),
        (test_div_f64_f16, f64, DataType::F64, f16, DataType::F16, f64, DataType::F64),
        (test_div_f64_f32, f64, DataType::F64, f32, DataType::F32, f64, DataType::F64),
        (test_div_f64_f64, f64, DataType::F64, f64, DataType::F64, f64, DataType::F64),
    ]
);

test_eval_binary_matrix!(
    arrow::compute::kernels::numeric::add,
    Operator::Add,
    ErrorMode::Ansi,
    100,
    500_000,
    [
        // I8 combinations (10 cases - excluding U64)
        (test_add_ansi_i8_i8, i8, DataType::I8, i8, DataType::I8, i8, DataType::I8),
        (test_add_ansi_i8_i16, i8, DataType::I8, i16, DataType::I16, i16, DataType::I16),
        (test_add_ansi_i8_i32, i8, DataType::I8, i32, DataType::I32, i32, DataType::I32),
        (test_add_ansi_i8_i64, i8, DataType::I8, i64, DataType::I64, i64, DataType::I64),
        (test_add_ansi_i8_u8, i8, DataType::I8, u8, DataType::U8, i16, DataType::I16),
        (test_add_ansi_i8_u16, i8, DataType::I8, u16, DataType::U16, i32, DataType::I32),
        (test_add_ansi_i8_u32, i8, DataType::I8, u32, DataType::U32, i64, DataType::I64),
        // I8 + U64 omitted (error case)
        (test_add_ansi_i8_f16, i8, DataType::I8, f16, DataType::F16, f16, DataType::F16),
        (test_add_ansi_i8_f32, i8, DataType::I8, f32, DataType::F32, f32, DataType::F32),
        (test_add_ansi_i8_f64, i8, DataType::I8, f64, DataType::F64, f64, DataType::F64),
        // I16 combinations (10 cases - excluding U64)
        (test_add_ansi_i16_i8, i16, DataType::I16, i8, DataType::I8, i16, DataType::I16),
        (test_add_ansi_i16_i16, i16, DataType::I16, i16, DataType::I16, i16, DataType::I16),
        (test_add_ansi_i16_i32, i16, DataType::I16, i32, DataType::I32, i32, DataType::I32),
        (test_add_ansi_i16_i64, i16, DataType::I16, i64, DataType::I64, i64, DataType::I64),
        (test_add_ansi_i16_u8, i16, DataType::I16, u8, DataType::U8, i16, DataType::I16),
        (test_add_ansi_i16_u16, i16, DataType::I16, u16, DataType::U16, i32, DataType::I32),
        (test_add_ansi_i16_u32, i16, DataType::I16, u32, DataType::U32, i64, DataType::I64),
        // I16 + U64 omitted (error case)
        (test_add_ansi_i16_f16, i16, DataType::I16, f16, DataType::F16, f16, DataType::F16),
        (test_add_ansi_i16_f32, i16, DataType::I16, f32, DataType::F32, f32, DataType::F32),
        (test_add_ansi_i16_f64, i16, DataType::I16, f64, DataType::F64, f64, DataType::F64),
        // I32 combinations (10 cases - excluding u64)
        (test_add_ansi_i32_i8, i32, DataType::I32, i8, DataType::I8, i32, DataType::I32),
        (test_add_ansi_i32_i16, i32, DataType::I32, i16, DataType::I16, i32, DataType::I32),
        (test_add_ansi_i32_i32, i32, DataType::I32, i32, DataType::I32, i32, DataType::I32),
        (test_add_ansi_i32_i64, i32, DataType::I32, i64, DataType::I64, i64, DataType::I64),
        (test_add_ansi_i32_u8, i32, DataType::I32, u8, DataType::U8, i32, DataType::I32),
        (test_add_ansi_i32_u16, i32, DataType::I32, u16, DataType::U16, i32, DataType::I32),
        (test_add_ansi_i32_u32, i32, DataType::I32, u32, DataType::U32, i64, DataType::I64),
        // I32 + U64 omitted (error case)
        (test_add_ansi_i32_f16, i32, DataType::I32, f16, DataType::F16, f16, DataType::F16),
        (test_add_ansi_i32_f32, i32, DataType::I32, f32, DataType::F32, f32, DataType::F32),
        (test_add_ansi_i32_f64, i32, DataType::I32, f64, DataType::F64, f64, DataType::F64),
        // I64 combinations (10 cases - excluding U64)
        (test_add_ansi_i64_i8, i64, DataType::I64, i8, DataType::I8, i64, DataType::I64),
        (test_add_ansi_i64_i16, i64, DataType::I64, i16, DataType::I16, i64, DataType::I64),
        (test_add_ansi_i64_i32, i64, DataType::I64, i32, DataType::I32, i64, DataType::I64),
        (test_add_ansi_i64_i64, i64, DataType::I64, i64, DataType::I64, i64, DataType::I64),
        (test_add_ansi_i64_u8, i64, DataType::I64, u8, DataType::U8, i64, DataType::I64),
        (test_add_ansi_i64_u16, i64, DataType::I64, u16, DataType::U16, i64, DataType::I64),
        (test_add_ansi_i64_u32, i64, DataType::I64, u32, DataType::U32, i64, DataType::I64),
        // I64 + U64 omitted (error case)
        (test_add_ansi_i64_f16, i64, DataType::I64, f16, DataType::F16, f16, DataType::F16),
        (test_add_ansi_i64_f32, i64, DataType::I64, f32, DataType::F32, f64, DataType::F64),
        (test_add_ansi_i64_f64, i64, DataType::I64, f64, DataType::F64, f64, DataType::F64),
        // U8 combinations (11 cases)
        (test_add_ansi_u8_i8, u8, DataType::U8, i8, DataType::I8, i16, DataType::I16),
        (test_add_ansi_u8_i16, u8, DataType::U8, i16, DataType::I16, i16, DataType::I16),
        (test_add_ansi_u8_i32, u8, DataType::U8, i32, DataType::I32, i32, DataType::I32),
        (test_add_ansi_u8_i64, u8, DataType::U8, i64, DataType::I64, i64, DataType::I64),
        (test_add_ansi_u8_u8, u8, DataType::U8, u8, DataType::U8, u8, DataType::U8),
        (test_add_ansi_u8_u16, u8, DataType::U8, u16, DataType::U16, u16, DataType::U16),
        (test_add_ansi_u8_u32, u8, DataType::U8, u32, DataType::U32, u32, DataType::U32),
        (test_add_ansi_u8_u64, u8, DataType::U8, u64, DataType::U64, u64, DataType::U64),
        (test_add_ansi_u8_f16, u8, DataType::U8, f16, DataType::F16, f16, DataType::F16),
        (test_add_ansi_u8_f32, u8, DataType::U8, f32, DataType::F32, f32, DataType::F32),
        (test_add_ansi_u8_f64, u8, DataType::U8, f64, DataType::F64, f64, DataType::F64),
        // U16 combinations (11 cases)
        (test_add_ansi_u16_i8, u16, DataType::U16, i8, DataType::I8, i32, DataType::I32),
        (test_add_ansi_u16_i16, u16, DataType::U16, i16, DataType::I16, i32, DataType::I32),
        (test_add_ansi_u16_i32, u16, DataType::U16, i32, DataType::I32, i32, DataType::I32),
        (test_add_ansi_u16_i64, u16, DataType::U16, i64, DataType::I64, i64, DataType::I64),
        (test_add_ansi_u16_u8, u16, DataType::U16, u8, DataType::U8, u16, DataType::U16),
        (test_add_ansi_u16_u16, u16, DataType::U16, u16, DataType::U16, u16, DataType::U16),
        (test_add_ansi_u16_u32, u16, DataType::U16, u32, DataType::U32, u32, DataType::U32),
        (test_add_ansi_u16_u64, u16, DataType::U16, u64, DataType::U64, u64, DataType::U64),
        (test_add_ansi_u16_f16, u16, DataType::U16, f16, DataType::F16, f16, DataType::F16),
        (test_add_ansi_u16_f32, u16, DataType::U16, f32, DataType::F32, f32, DataType::F32),
        (test_add_ansi_u16_f64, u16, DataType::U16, f64, DataType::F64, f64, DataType::F64),
        // U32 combinations (11 cases)
        (test_add_ansi_u32_i8, u32, DataType::U32, i8, DataType::I8, i64, DataType::I64),
        (test_add_ansi_u32_i16, u32, DataType::U32, i16, DataType::I16, i64, DataType::I64),
        (test_add_ansi_u32_i32, u32, DataType::U32, i32, DataType::I32, i64, DataType::I64),
        (test_add_ansi_u32_i64, u32, DataType::U32, i64, DataType::I64, i64, DataType::I64),
        (test_add_ansi_u32_u8, u32, DataType::U32, u8, DataType::U8, u32, DataType::U32),
        (test_add_ansi_u32_u16, u32, DataType::U32, u16, DataType::U16, u32, DataType::U32),
        (test_add_ansi_u32_u32, u32, DataType::U32, u32, DataType::U32, u32, DataType::U32),
        (test_add_ansi_u32_u64, u32, DataType::U32, u64, DataType::U64, u64, DataType::U64),
        (test_add_ansi_u32_f16, u32, DataType::U32, f16, DataType::F16, f16, DataType::F16),
        (test_add_ansi_u32_f32, u32, DataType::U32, f32, DataType::F32, f32, DataType::F32),
        (test_add_ansi_u32_f64, u32, DataType::U32, f64, DataType::F64, f64, DataType::F64),
        // U64 combinations (7 cases - excluding Ix)
        (test_add_ansi_u64_u8, u64, DataType::U64, u8, DataType::U8, u64, DataType::U64),
        (test_add_ansi_u64_u16, u64, DataType::U64, u16, DataType::U16, u64, DataType::U64),
        (test_add_ansi_u64_u32, u64, DataType::U64, u32, DataType::U32, u64, DataType::U64),
        (test_add_ansi_u64_u64, u64, DataType::U64, u64, DataType::U64, u64, DataType::U64),
        (test_add_ansi_u64_f16, u64, DataType::U64, f16, DataType::F16, f16, DataType::F16),
        (test_add_ansi_u64_f32, u64, DataType::U64, f32, DataType::F32, f64, DataType::F64),
        (test_add_ansi_u64_f64, u64, DataType::U64, f64, DataType::F64, f64, DataType::F64),
        // F16 combinations (11 cases)
        (test_add_ansi_f16_i8, f16, DataType::F16, i8, DataType::I8, f16, DataType::F16),
        (test_add_ansi_f16_i16, f16, DataType::F16, i16, DataType::I16, f16, DataType::F16),
        (test_add_ansi_f16_i32, f16, DataType::F16, i32, DataType::I32, f16, DataType::F16),
        (test_add_ansi_f16_i64, f16, DataType::F16, i64, DataType::I64, f16, DataType::F16),
        (test_add_ansi_f16_u8, f16, DataType::F16, u8, DataType::U8, f16, DataType::F16),
        (test_add_ansi_f16_u16, f16, DataType::F16, u16, DataType::U16, f16, DataType::F16),
        (test_add_ansi_f16_u32, f16, DataType::F16, u32, DataType::U32, f16, DataType::F16),
        (test_add_ansi_f16_u64, f16, DataType::F16, u64, DataType::U64, f16, DataType::F16),
        (test_add_ansi_f16_f16, f16, DataType::F16, f16, DataType::F16, f16, DataType::F16),
        (test_add_ansi_f16_f32, f16, DataType::F16, f32, DataType::F32, f32, DataType::F32),
        (test_add_ansi_f16_f64, f16, DataType::F16, f64, DataType::F64, f64, DataType::F64),
        // F32 combinations (11 cases)
        (test_add_ansi_f32_i8, f32, DataType::F32, i8, DataType::I8, f32, DataType::F32),
        (test_add_ansi_f32_i16, f32, DataType::F32, i16, DataType::I16, f32, DataType::F32),
        (test_add_ansi_f32_i32, f32, DataType::F32, i32, DataType::I32, f32, DataType::F32),
        (test_add_ansi_f32_i64, f32, DataType::F32, i64, DataType::I64, f64, DataType::F64),
        (test_add_ansi_f32_u8, f32, DataType::F32, u8, DataType::U8, f32, DataType::F32),
        (test_add_ansi_f32_u16, f32, DataType::F32, u16, DataType::U16, f32, DataType::F32),
        (test_add_ansi_f32_u32, f32, DataType::F32, u32, DataType::U32, f32, DataType::F32),
        (test_add_ansi_f32_u64, f32, DataType::F32, u64, DataType::U64, f64, DataType::F64),
        (test_add_ansi_f32_f16, f32, DataType::F32, f16, DataType::F16, f32, DataType::F32),
        (test_add_ansi_f32_f32, f32, DataType::F32, f32, DataType::F32, f32, DataType::F32),
        (test_add_ansi_f32_f64, f32, DataType::F32, f64, DataType::F64, f64, DataType::F64),
        // F64 combinations (11 cases)
        (test_add_ansi_f64_i8, f64, DataType::F64, i8, DataType::I8, f64, DataType::F64),
        (test_add_ansi_f64_i16, f64, DataType::F64, i16, DataType::I16, f64, DataType::F64),
        (test_add_ansi_f64_i32, f64, DataType::F64, i32, DataType::I32, f64, DataType::F64),
        (test_add_ansi_f64_i64, f64, DataType::F64, i64, DataType::I64, f64, DataType::F64),
        (test_add_ansi_f64_u8, f64, DataType::F64, u8, DataType::U8, f64, DataType::F64),
        (test_add_ansi_f64_u16, f64, DataType::F64, u16, DataType::U16, f64, DataType::F64),
        (test_add_ansi_f64_u32, f64, DataType::F64, u32, DataType::U32, f64, DataType::F64),
        (test_add_ansi_f64_u64, f64, DataType::F64, u64, DataType::U64, f64, DataType::F64),
        (test_add_ansi_f64_f16, f64, DataType::F64, f16, DataType::F16, f64, DataType::F64),
        (test_add_ansi_f64_f32, f64, DataType::F64, f32, DataType::F32, f64, DataType::F64),
        (test_add_ansi_f64_f64, f64, DataType::F64, f64, DataType::F64, f64, DataType::F64),
    ]
);

test_eval_binary_matrix!(
    arrow::compute::kernels::numeric::sub,
    Operator::Sub,
    ErrorMode::Ansi,
    100,
    500_000,
    [
        // I8 combinations (10 cases - excluding U64)
        (test_sub_ansi_i8_i8, i8, DataType::I8, i8, DataType::I8, i8, DataType::I8),
        (test_sub_ansi_i8_i16, i8, DataType::I8, i16, DataType::I16, i16, DataType::I16),
        (test_sub_ansi_i8_i32, i8, DataType::I8, i32, DataType::I32, i32, DataType::I32),
        (test_sub_ansi_i8_i64, i8, DataType::I8, i64, DataType::I64, i64, DataType::I64),
        (test_sub_ansi_i8_u8, i8, DataType::I8, u8, DataType::U8, i16, DataType::I16),
        (test_sub_ansi_i8_u16, i8, DataType::I8, u16, DataType::U16, i32, DataType::I32),
        (test_sub_ansi_i8_u32, i8, DataType::I8, u32, DataType::U32, i64, DataType::I64),
        // I8 + U64 omitted (error case)
        (test_sub_ansi_i8_f16, i8, DataType::I8, f16, DataType::F16, f16, DataType::F16),
        (test_sub_ansi_i8_f32, i8, DataType::I8, f32, DataType::F32, f32, DataType::F32),
        (test_sub_ansi_i8_f64, i8, DataType::I8, f64, DataType::F64, f64, DataType::F64),
        // I16 combinations (10 cases - excluding U64)
        (test_sub_ansi_i16_i8, i16, DataType::I16, i8, DataType::I8, i16, DataType::I16),
        (test_sub_ansi_i16_i16, i16, DataType::I16, i16, DataType::I16, i16, DataType::I16),
        (test_sub_ansi_i16_i32, i16, DataType::I16, i32, DataType::I32, i32, DataType::I32),
        (test_sub_ansi_i16_i64, i16, DataType::I16, i64, DataType::I64, i64, DataType::I64),
        (test_sub_ansi_i16_u8, i16, DataType::I16, u8, DataType::U8, i16, DataType::I16),
        (test_sub_ansi_i16_u16, i16, DataType::I16, u16, DataType::U16, i32, DataType::I32),
        (test_sub_ansi_i16_u32, i16, DataType::I16, u32, DataType::U32, i64, DataType::I64),
        // I16 + U64 omitted (error case)
        (test_sub_ansi_i16_f16, i16, DataType::I16, f16, DataType::F16, f16, DataType::F16),
        (test_sub_ansi_i16_f32, i16, DataType::I16, f32, DataType::F32, f32, DataType::F32),
        (test_sub_ansi_i16_f64, i16, DataType::I16, f64, DataType::F64, f64, DataType::F64),
        // I32 combinations (10 cases - excluding u64)
        (test_sub_ansi_i32_i8, i32, DataType::I32, i8, DataType::I8, i32, DataType::I32),
        (test_sub_ansi_i32_i16, i32, DataType::I32, i16, DataType::I16, i32, DataType::I32),
        (test_sub_ansi_i32_i32, i32, DataType::I32, i32, DataType::I32, i32, DataType::I32),
        (test_sub_ansi_i32_i64, i32, DataType::I32, i64, DataType::I64, i64, DataType::I64),
        (test_sub_ansi_i32_u8, i32, DataType::I32, u8, DataType::U8, i32, DataType::I32),
        (test_sub_ansi_i32_u16, i32, DataType::I32, u16, DataType::U16, i32, DataType::I32),
        (test_sub_ansi_i32_u32, i32, DataType::I32, u32, DataType::U32, i64, DataType::I64),
        // I32 + U64 omitted (error case)
        (test_sub_ansi_i32_f16, i32, DataType::I32, f16, DataType::F16, f16, DataType::F16),
        (test_sub_ansi_i32_f32, i32, DataType::I32, f32, DataType::F32, f32, DataType::F32),
        (test_sub_ansi_i32_f64, i32, DataType::I32, f64, DataType::F64, f64, DataType::F64),
        // I64 combinations (10 cases - excluding U64)
        (test_sub_ansi_i64_i8, i64, DataType::I64, i8, DataType::I8, i64, DataType::I64),
        (test_sub_ansi_i64_i16, i64, DataType::I64, i16, DataType::I16, i64, DataType::I64),
        (test_sub_ansi_i64_i32, i64, DataType::I64, i32, DataType::I32, i64, DataType::I64),
        (test_sub_ansi_i64_i64, i64, DataType::I64, i64, DataType::I64, i64, DataType::I64),
        (test_sub_ansi_i64_u8, i64, DataType::I64, u8, DataType::U8, i64, DataType::I64),
        (test_sub_ansi_i64_u16, i64, DataType::I64, u16, DataType::U16, i64, DataType::I64),
        (test_sub_ansi_i64_u32, i64, DataType::I64, u32, DataType::U32, i64, DataType::I64),
        // I64 + U64 omitted (error case)
        (test_sub_ansi_i64_f16, i64, DataType::I64, f16, DataType::F16, f16, DataType::F16),
        (test_sub_ansi_i64_f32, i64, DataType::I64, f32, DataType::F32, f64, DataType::F64),
        (test_sub_ansi_i64_f64, i64, DataType::I64, f64, DataType::F64, f64, DataType::F64),
        // U8 combinations (11 cases)
        (test_sub_ansi_u8_i8, u8, DataType::U8, i8, DataType::I8, i16, DataType::I16),
        (test_sub_ansi_u8_i16, u8, DataType::U8, i16, DataType::I16, i16, DataType::I16),
        (test_sub_ansi_u8_i32, u8, DataType::U8, i32, DataType::I32, i32, DataType::I32),
        (test_sub_ansi_u8_i64, u8, DataType::U8, i64, DataType::I64, i64, DataType::I64),
        (test_sub_ansi_u8_u8, u8, DataType::U8, u8, DataType::U8, u8, DataType::U8),
        (test_sub_ansi_u8_u16, u8, DataType::U8, u16, DataType::U16, u16, DataType::U16),
        (test_sub_ansi_u8_u32, u8, DataType::U8, u32, DataType::U32, u32, DataType::U32),
        (test_sub_ansi_u8_u64, u8, DataType::U8, u64, DataType::U64, u64, DataType::U64),
        (test_sub_ansi_u8_f16, u8, DataType::U8, f16, DataType::F16, f16, DataType::F16),
        (test_sub_ansi_u8_f32, u8, DataType::U8, f32, DataType::F32, f32, DataType::F32),
        (test_sub_ansi_u8_f64, u8, DataType::U8, f64, DataType::F64, f64, DataType::F64),
        // U16 combinations (11 cases)
        (test_sub_ansi_u16_i8, u16, DataType::U16, i8, DataType::I8, i32, DataType::I32),
        (test_sub_ansi_u16_i16, u16, DataType::U16, i16, DataType::I16, i32, DataType::I32),
        (test_sub_ansi_u16_i32, u16, DataType::U16, i32, DataType::I32, i32, DataType::I32),
        (test_sub_ansi_u16_i64, u16, DataType::U16, i64, DataType::I64, i64, DataType::I64),
        (test_sub_ansi_u16_u8, u16, DataType::U16, u8, DataType::U8, u16, DataType::U16),
        (test_sub_ansi_u16_u16, u16, DataType::U16, u16, DataType::U16, u16, DataType::U16),
        (test_sub_ansi_u16_u32, u16, DataType::U16, u32, DataType::U32, u32, DataType::U32),
        (test_sub_ansi_u16_u64, u16, DataType::U16, u64, DataType::U64, u64, DataType::U64),
        (test_sub_ansi_u16_f16, u16, DataType::U16, f16, DataType::F16, f16, DataType::F16),
        (test_sub_ansi_u16_f32, u16, DataType::U16, f32, DataType::F32, f32, DataType::F32),
        (test_sub_ansi_u16_f64, u16, DataType::U16, f64, DataType::F64, f64, DataType::F64),
        // U32 combinations (11 cases)
        (test_sub_ansi_u32_i8, u32, DataType::U32, i8, DataType::I8, i64, DataType::I64),
        (test_sub_ansi_u32_i16, u32, DataType::U32, i16, DataType::I16, i64, DataType::I64),
        (test_sub_ansi_u32_i32, u32, DataType::U32, i32, DataType::I32, i64, DataType::I64),
        (test_sub_ansi_u32_i64, u32, DataType::U32, i64, DataType::I64, i64, DataType::I64),
        (test_sub_ansi_u32_u8, u32, DataType::U32, u8, DataType::U8, u32, DataType::U32),
        (test_sub_ansi_u32_u16, u32, DataType::U32, u16, DataType::U16, u32, DataType::U32),
        (test_sub_ansi_u32_u32, u32, DataType::U32, u32, DataType::U32, u32, DataType::U32),
        (test_sub_ansi_u32_u64, u32, DataType::U32, u64, DataType::U64, u64, DataType::U64),
        (test_sub_ansi_u32_f16, u32, DataType::U32, f16, DataType::F16, f16, DataType::F16),
        (test_sub_ansi_u32_f32, u32, DataType::U32, f32, DataType::F32, f32, DataType::F32),
        (test_sub_ansi_u32_f64, u32, DataType::U32, f64, DataType::F64, f64, DataType::F64),
        // U64 combinations (7 cases - excluding Ix)
        (test_sub_ansi_u64_u8, u64, DataType::U64, u8, DataType::U8, u64, DataType::U64),
        (test_sub_ansi_u64_u16, u64, DataType::U64, u16, DataType::U16, u64, DataType::U64),
        (test_sub_ansi_u64_u32, u64, DataType::U64, u32, DataType::U32, u64, DataType::U64),
        (test_sub_ansi_u64_u64, u64, DataType::U64, u64, DataType::U64, u64, DataType::U64),
        (test_sub_ansi_u64_f16, u64, DataType::U64, f16, DataType::F16, f16, DataType::F16),
        (test_sub_ansi_u64_f32, u64, DataType::U64, f32, DataType::F32, f64, DataType::F64),
        (test_sub_ansi_u64_f64, u64, DataType::U64, f64, DataType::F64, f64, DataType::F64),
        // F16 combinations (11 cases)
        (test_sub_ansi_f16_i8, f16, DataType::F16, i8, DataType::I8, f16, DataType::F16),
        (test_sub_ansi_f16_i16, f16, DataType::F16, i16, DataType::I16, f16, DataType::F16),
        (test_sub_ansi_f16_i32, f16, DataType::F16, i32, DataType::I32, f16, DataType::F16),
        (test_sub_ansi_f16_i64, f16, DataType::F16, i64, DataType::I64, f16, DataType::F16),
        (test_sub_ansi_f16_u8, f16, DataType::F16, u8, DataType::U8, f16, DataType::F16),
        (test_sub_ansi_f16_u16, f16, DataType::F16, u16, DataType::U16, f16, DataType::F16),
        (test_sub_ansi_f16_u32, f16, DataType::F16, u32, DataType::U32, f16, DataType::F16),
        (test_sub_ansi_f16_u64, f16, DataType::F16, u64, DataType::U64, f16, DataType::F16),
        (test_sub_ansi_f16_f16, f16, DataType::F16, f16, DataType::F16, f16, DataType::F16),
        (test_sub_ansi_f16_f32, f16, DataType::F16, f32, DataType::F32, f32, DataType::F32),
        (test_sub_ansi_f16_f64, f16, DataType::F16, f64, DataType::F64, f64, DataType::F64),
        // F32 combinations (11 cases)
        (test_sub_ansi_f32_i8, f32, DataType::F32, i8, DataType::I8, f32, DataType::F32),
        (test_sub_ansi_f32_i16, f32, DataType::F32, i16, DataType::I16, f32, DataType::F32),
        (test_sub_ansi_f32_i32, f32, DataType::F32, i32, DataType::I32, f32, DataType::F32),
        (test_sub_ansi_f32_i64, f32, DataType::F32, i64, DataType::I64, f64, DataType::F64),
        (test_sub_ansi_f32_u8, f32, DataType::F32, u8, DataType::U8, f32, DataType::F32),
        (test_sub_ansi_f32_u16, f32, DataType::F32, u16, DataType::U16, f32, DataType::F32),
        (test_sub_ansi_f32_u32, f32, DataType::F32, u32, DataType::U32, f32, DataType::F32),
        (test_sub_ansi_f32_u64, f32, DataType::F32, u64, DataType::U64, f64, DataType::F64),
        (test_sub_ansi_f32_f16, f32, DataType::F32, f16, DataType::F16, f32, DataType::F32),
        (test_sub_ansi_f32_f32, f32, DataType::F32, f32, DataType::F32, f32, DataType::F32),
        (test_sub_ansi_f32_f64, f32, DataType::F32, f64, DataType::F64, f64, DataType::F64),
        // F64 combinations (11 cases)
        (test_sub_ansi_f64_i8, f64, DataType::F64, i8, DataType::I8, f64, DataType::F64),
        (test_sub_ansi_f64_i16, f64, DataType::F64, i16, DataType::I16, f64, DataType::F64),
        (test_sub_ansi_f64_i32, f64, DataType::F64, i32, DataType::I32, f64, DataType::F64),
        (test_sub_ansi_f64_i64, f64, DataType::F64, i64, DataType::I64, f64, DataType::F64),
        (test_sub_ansi_f64_u8, f64, DataType::F64, u8, DataType::U8, f64, DataType::F64),
        (test_sub_ansi_f64_u16, f64, DataType::F64, u16, DataType::U16, f64, DataType::F64),
        (test_sub_ansi_f64_u32, f64, DataType::F64, u32, DataType::U32, f64, DataType::F64),
        (test_sub_ansi_f64_u64, f64, DataType::F64, u64, DataType::U64, f64, DataType::F64),
        (test_sub_ansi_f64_f16, f64, DataType::F64, f16, DataType::F16, f64, DataType::F64),
        (test_sub_ansi_f64_f32, f64, DataType::F64, f32, DataType::F32, f64, DataType::F64),
        (test_sub_ansi_f64_f64, f64, DataType::F64, f64, DataType::F64, f64, DataType::F64),
    ]
);

test_eval_binary_matrix!(
    arrow::compute::kernels::numeric::mul,
    Operator::Mul,
    ErrorMode::Ansi,
    200,
    400_000,
    [
        // I8 combinations (10 cases - excluding U64)
        (test_mul_ansi_i8_i8, i8, DataType::I8, i8, DataType::I8, i8, DataType::I8),
        (test_mul_ansi_i8_i16, i8, DataType::I8, i16, DataType::I16, i16, DataType::I16),
        (test_mul_ansi_i8_i32, i8, DataType::I8, i32, DataType::I32, i32, DataType::I32),
        (test_mul_ansi_i8_i64, i8, DataType::I8, i64, DataType::I64, i64, DataType::I64),
        (test_mul_ansi_i8_u8, i8, DataType::I8, u8, DataType::U8, i16, DataType::I16),
        (test_mul_ansi_i8_u16, i8, DataType::I8, u16, DataType::U16, i32, DataType::I32),
        (test_mul_ansi_i8_u32, i8, DataType::I8, u32, DataType::U32, i64, DataType::I64),
        // I8 + U64 omitted (error case)
        (test_mul_ansi_i8_f16, i8, DataType::I8, f16, DataType::F16, f16, DataType::F16),
        (test_mul_ansi_i8_f32, i8, DataType::I8, f32, DataType::F32, f32, DataType::F32),
        (test_mul_ansi_i8_f64, i8, DataType::I8, f64, DataType::F64, f64, DataType::F64),
        // I16 combinations (10 cases - excluding U64)
        (test_mul_ansi_i16_i8, i16, DataType::I16, i8, DataType::I8, i16, DataType::I16),
        (test_mul_ansi_i16_i16, i16, DataType::I16, i16, DataType::I16, i16, DataType::I16),
        (test_mul_ansi_i16_i32, i16, DataType::I16, i32, DataType::I32, i32, DataType::I32),
        (test_mul_ansi_i16_i64, i16, DataType::I16, i64, DataType::I64, i64, DataType::I64),
        (test_mul_ansi_i16_u8, i16, DataType::I16, u8, DataType::U8, i16, DataType::I16),
        (test_mul_ansi_i16_u16, i16, DataType::I16, u16, DataType::U16, i32, DataType::I32),
        (test_mul_ansi_i16_u32, i16, DataType::I16, u32, DataType::U32, i64, DataType::I64),
        // I16 + U64 omitted (error case)
        (test_mul_ansi_i16_f16, i16, DataType::I16, f16, DataType::F16, f16, DataType::F16),
        (test_mul_ansi_i16_f32, i16, DataType::I16, f32, DataType::F32, f32, DataType::F32),
        (test_mul_ansi_i16_f64, i16, DataType::I16, f64, DataType::F64, f64, DataType::F64),
        // I32 combinations (10 cases - excluding u64)
        (test_mul_ansi_i32_i8, i32, DataType::I32, i8, DataType::I8, i32, DataType::I32),
        (test_mul_ansi_i32_i16, i32, DataType::I32, i16, DataType::I16, i32, DataType::I32),
        (test_mul_ansi_i32_i32, i32, DataType::I32, i32, DataType::I32, i32, DataType::I32),
        (test_mul_ansi_i32_i64, i32, DataType::I32, i64, DataType::I64, i64, DataType::I64),
        (test_mul_ansi_i32_u8, i32, DataType::I32, u8, DataType::U8, i32, DataType::I32),
        (test_mul_ansi_i32_u16, i32, DataType::I32, u16, DataType::U16, i32, DataType::I32),
        (test_mul_ansi_i32_u32, i32, DataType::I32, u32, DataType::U32, i64, DataType::I64),
        // I32 + U64 omitted (error case)
        (test_mul_ansi_i32_f16, i32, DataType::I32, f16, DataType::F16, f16, DataType::F16),
        (test_mul_ansi_i32_f32, i32, DataType::I32, f32, DataType::F32, f32, DataType::F32),
        (test_mul_ansi_i32_f64, i32, DataType::I32, f64, DataType::F64, f64, DataType::F64),
        // I64 combinations (10 cases - excluding U64)
        (test_mul_ansi_i64_i8, i64, DataType::I64, i8, DataType::I8, i64, DataType::I64),
        (test_mul_ansi_i64_i16, i64, DataType::I64, i16, DataType::I16, i64, DataType::I64),
        (test_mul_ansi_i64_i32, i64, DataType::I64, i32, DataType::I32, i64, DataType::I64),
        (test_mul_ansi_i64_i64, i64, DataType::I64, i64, DataType::I64, i64, DataType::I64),
        (test_mul_ansi_i64_u8, i64, DataType::I64, u8, DataType::U8, i64, DataType::I64),
        (test_mul_ansi_i64_u16, i64, DataType::I64, u16, DataType::U16, i64, DataType::I64),
        (test_mul_ansi_i64_u32, i64, DataType::I64, u32, DataType::U32, i64, DataType::I64),
        // I64 + U64 omitted (error case)
        (test_mul_ansi_i64_f16, i64, DataType::I64, f16, DataType::F16, f16, DataType::F16),
        (test_mul_ansi_i64_f32, i64, DataType::I64, f32, DataType::F32, f64, DataType::F64),
        (test_mul_ansi_i64_f64, i64, DataType::I64, f64, DataType::F64, f64, DataType::F64),
        // U8 combinations (11 cases)
        (test_mul_ansi_u8_i8, u8, DataType::U8, i8, DataType::I8, i16, DataType::I16),
        (test_mul_ansi_u8_i16, u8, DataType::U8, i16, DataType::I16, i16, DataType::I16),
        (test_mul_ansi_u8_i32, u8, DataType::U8, i32, DataType::I32, i32, DataType::I32),
        (test_mul_ansi_u8_i64, u8, DataType::U8, i64, DataType::I64, i64, DataType::I64),
        (test_mul_ansi_u8_u8, u8, DataType::U8, u8, DataType::U8, u8, DataType::U8),
        (test_mul_ansi_u8_u16, u8, DataType::U8, u16, DataType::U16, u16, DataType::U16),
        (test_mul_ansi_u8_u32, u8, DataType::U8, u32, DataType::U32, u32, DataType::U32),
        (test_mul_ansi_u8_u64, u8, DataType::U8, u64, DataType::U64, u64, DataType::U64),
        (test_mul_ansi_u8_f16, u8, DataType::U8, f16, DataType::F16, f16, DataType::F16),
        (test_mul_ansi_u8_f32, u8, DataType::U8, f32, DataType::F32, f32, DataType::F32),
        (test_mul_ansi_u8_f64, u8, DataType::U8, f64, DataType::F64, f64, DataType::F64),
        // U16 combinations (11 cases)
        (test_mul_ansi_u16_i8, u16, DataType::U16, i8, DataType::I8, i32, DataType::I32),
        (test_mul_ansi_u16_i16, u16, DataType::U16, i16, DataType::I16, i32, DataType::I32),
        (test_mul_ansi_u16_i32, u16, DataType::U16, i32, DataType::I32, i32, DataType::I32),
        (test_mul_ansi_u16_i64, u16, DataType::U16, i64, DataType::I64, i64, DataType::I64),
        (test_mul_ansi_u16_u8, u16, DataType::U16, u8, DataType::U8, u16, DataType::U16),
        (test_mul_ansi_u16_u16, u16, DataType::U16, u16, DataType::U16, u16, DataType::U16),
        (test_mul_ansi_u16_u32, u16, DataType::U16, u32, DataType::U32, u32, DataType::U32),
        (test_mul_ansi_u16_u64, u16, DataType::U16, u64, DataType::U64, u64, DataType::U64),
        (test_mul_ansi_u16_f16, u16, DataType::U16, f16, DataType::F16, f16, DataType::F16),
        (test_mul_ansi_u16_f32, u16, DataType::U16, f32, DataType::F32, f32, DataType::F32),
        (test_mul_ansi_u16_f64, u16, DataType::U16, f64, DataType::F64, f64, DataType::F64),
        // U32 combinations (11 cases)
        (test_mul_ansi_u32_i8, u32, DataType::U32, i8, DataType::I8, i64, DataType::I64),
        (test_mul_ansi_u32_i16, u32, DataType::U32, i16, DataType::I16, i64, DataType::I64),
        (test_mul_ansi_u32_i32, u32, DataType::U32, i32, DataType::I32, i64, DataType::I64),
        (test_mul_ansi_u32_i64, u32, DataType::U32, i64, DataType::I64, i64, DataType::I64),
        (test_mul_ansi_u32_u8, u32, DataType::U32, u8, DataType::U8, u32, DataType::U32),
        (test_mul_ansi_u32_u16, u32, DataType::U32, u16, DataType::U16, u32, DataType::U32),
        (test_mul_ansi_u32_u32, u32, DataType::U32, u32, DataType::U32, u32, DataType::U32),
        (test_mul_ansi_u32_u64, u32, DataType::U32, u64, DataType::U64, u64, DataType::U64),
        (test_mul_ansi_u32_f16, u32, DataType::U32, f16, DataType::F16, f16, DataType::F16),
        (test_mul_ansi_u32_f32, u32, DataType::U32, f32, DataType::F32, f32, DataType::F32),
        (test_mul_ansi_u32_f64, u32, DataType::U32, f64, DataType::F64, f64, DataType::F64),
        // U64 combinations (7 cases - excluding Ix)
        (test_mul_ansi_u64_u8, u64, DataType::U64, u8, DataType::U8, u64, DataType::U64),
        (test_mul_ansi_u64_u16, u64, DataType::U64, u16, DataType::U16, u64, DataType::U64),
        (test_mul_ansi_u64_u32, u64, DataType::U64, u32, DataType::U32, u64, DataType::U64),
        (test_mul_ansi_u64_u64, u64, DataType::U64, u64, DataType::U64, u64, DataType::U64),
        (test_mul_ansi_u64_f16, u64, DataType::U64, f16, DataType::F16, f16, DataType::F16),
        (test_mul_ansi_u64_f32, u64, DataType::U64, f32, DataType::F32, f64, DataType::F64),
        (test_mul_ansi_u64_f64, u64, DataType::U64, f64, DataType::F64, f64, DataType::F64),
        // F16 combinations (11 cases)
        (test_mul_ansi_f16_i8, f16, DataType::F16, i8, DataType::I8, f16, DataType::F16),
        (test_mul_ansi_f16_i16, f16, DataType::F16, i16, DataType::I16, f16, DataType::F16),
        (test_mul_ansi_f16_i32, f16, DataType::F16, i32, DataType::I32, f16, DataType::F16),
        (test_mul_ansi_f16_i64, f16, DataType::F16, i64, DataType::I64, f16, DataType::F16),
        (test_mul_ansi_f16_u8, f16, DataType::F16, u8, DataType::U8, f16, DataType::F16),
        (test_mul_ansi_f16_u16, f16, DataType::F16, u16, DataType::U16, f16, DataType::F16),
        (test_mul_ansi_f16_u32, f16, DataType::F16, u32, DataType::U32, f16, DataType::F16),
        (test_mul_ansi_f16_u64, f16, DataType::F16, u64, DataType::U64, f16, DataType::F16),
        (test_mul_ansi_f16_f16, f16, DataType::F16, f16, DataType::F16, f16, DataType::F16),
        (test_mul_ansi_f16_f32, f16, DataType::F16, f32, DataType::F32, f32, DataType::F32),
        (test_mul_ansi_f16_f64, f16, DataType::F16, f64, DataType::F64, f64, DataType::F64),
        // F32 combinations (11 cases)
        (test_mul_ansi_f32_i8, f32, DataType::F32, i8, DataType::I8, f32, DataType::F32),
        (test_mul_ansi_f32_i16, f32, DataType::F32, i16, DataType::I16, f32, DataType::F32),
        (test_mul_ansi_f32_i32, f32, DataType::F32, i32, DataType::I32, f32, DataType::F32),
        (test_mul_ansi_f32_i64, f32, DataType::F32, i64, DataType::I64, f64, DataType::F64),
        (test_mul_ansi_f32_u8, f32, DataType::F32, u8, DataType::U8, f32, DataType::F32),
        (test_mul_ansi_f32_u16, f32, DataType::F32, u16, DataType::U16, f32, DataType::F32),
        (test_mul_ansi_f32_u32, f32, DataType::F32, u32, DataType::U32, f32, DataType::F32),
        (test_mul_ansi_f32_u64, f32, DataType::F32, u64, DataType::U64, f64, DataType::F64),
        (test_mul_ansi_f32_f16, f32, DataType::F32, f16, DataType::F16, f32, DataType::F32),
        (test_mul_ansi_f32_f32, f32, DataType::F32, f32, DataType::F32, f32, DataType::F32),
        (test_mul_ansi_f32_f64, f32, DataType::F32, f64, DataType::F64, f64, DataType::F64),
        // F64 combinations (11 cases)
        (test_mul_ansi_f64_i8, f64, DataType::F64, i8, DataType::I8, f64, DataType::F64),
        (test_mul_ansi_f64_i16, f64, DataType::F64, i16, DataType::I16, f64, DataType::F64),
        (test_mul_ansi_f64_i32, f64, DataType::F64, i32, DataType::I32, f64, DataType::F64),
        (test_mul_ansi_f64_i64, f64, DataType::F64, i64, DataType::I64, f64, DataType::F64),
        (test_mul_ansi_f64_u8, f64, DataType::F64, u8, DataType::U8, f64, DataType::F64),
        (test_mul_ansi_f64_u16, f64, DataType::F64, u16, DataType::U16, f64, DataType::F64),
        (test_mul_ansi_f64_u32, f64, DataType::F64, u32, DataType::U32, f64, DataType::F64),
        (test_mul_ansi_f64_u64, f64, DataType::F64, u64, DataType::U64, f64, DataType::F64),
        (test_mul_ansi_f64_f16, f64, DataType::F64, f16, DataType::F16, f64, DataType::F64),
        (test_mul_ansi_f64_f32, f64, DataType::F64, f32, DataType::F32, f64, DataType::F64),
        (test_mul_ansi_f64_f64, f64, DataType::F64, f64, DataType::F64, f64, DataType::F64),
    ]
);

test_eval_binary_cmp_matrix!(
    Operator::Eq,
    ErrorMode::Tachyon,
    100,
    200_000,
    [
        (test_eq_i8_i8, i8, DataType::I8, i8, DataType::I8),
        (test_eq_i8_i16, i8, DataType::I8, i16, DataType::I16),
        (test_eq_i8_i32, i8, DataType::I8, i32, DataType::I32),
        (test_eq_i8_i64, i8, DataType::I8, i64, DataType::I64),
        (test_eq_i8_u8, i8, DataType::I8, u8, DataType::U8),
        (test_eq_i8_u16, i8, DataType::I8, u16, DataType::U16),
        (test_eq_i8_f16, i8, DataType::I8, f16, DataType::F16),
        (test_eq_i8_u32, i8, DataType::I8, u32, DataType::U32),
        (test_eq_i8_u64, i8, DataType::I8, u64, DataType::U64),
        (test_eq_i8_f32, i8, DataType::I8, f32, DataType::F32),
        (test_eq_i8_f64, i8, DataType::I8, f64, DataType::F64),
        (test_eq_i16_i8, i16, DataType::I16, i8, DataType::I8),
        (test_eq_i16_i16, i16, DataType::I16, i16, DataType::I16),
        (test_eq_i16_i32, i16, DataType::I16, i32, DataType::I32),
        (test_eq_i16_i64, i16, DataType::I16, i64, DataType::I64),
        (test_eq_i16_u8, i16, DataType::I16, u8, DataType::U8),
        (test_eq_i16_u16, i16, DataType::I16, u16, DataType::U16),
        (test_eq_i16_u32, i16, DataType::I16, u32, DataType::U32),
        (test_eq_i16_u64, i16, DataType::I16, u64, DataType::U64),
        (test_eq_i16_f16, i16, DataType::I16, f16, DataType::F16),
        (test_eq_i16_f32, i16, DataType::I16, f32, DataType::F32),
        (test_eq_i16_f64, i16, DataType::I16, f64, DataType::F64),
        (test_eq_i32_i8, i32, DataType::I32, i8, DataType::I8),
        (test_eq_i32_i16, i32, DataType::I32, i16, DataType::I16),
        (test_eq_i32_i32, i32, DataType::I32, i32, DataType::I32),
        (test_eq_i32_i64, i32, DataType::I32, i64, DataType::I64),
        (test_eq_i32_u8, i32, DataType::I32, u8, DataType::U8),
        (test_eq_i32_u16, i32, DataType::I32, u16, DataType::U16),
        (test_eq_i32_u32, i32, DataType::I32, u32, DataType::U32),
        (test_eq_i32_u64, i32, DataType::I32, u64, DataType::U64),
        (test_eq_i32_f16, i32, DataType::I32, f16, DataType::F16),
        (test_eq_i32_f32, i32, DataType::I32, f32, DataType::F32),
        (test_eq_i32_f64, i32, DataType::I32, f64, DataType::F64),
        (test_eq_i64_i8, i64, DataType::I64, i8, DataType::I8),
        (test_eq_i64_i16, i64, DataType::I64, i16, DataType::I16),
        (test_eq_i64_i32, i64, DataType::I64, i32, DataType::I32),
        (test_eq_i64_i64, i64, DataType::I64, i64, DataType::I64),
        (test_eq_i64_u8, i64, DataType::I64, u8, DataType::U8),
        (test_eq_i64_u16, i64, DataType::I64, u16, DataType::U16),
        (test_eq_i64_u32, i64, DataType::I64, u32, DataType::U32),
        (test_eq_i64_u64, i64, DataType::I64, u64, DataType::U64),
        (test_eq_i64_f16, i64, DataType::I64, f16, DataType::F16),
        (test_eq_i64_f32, i64, DataType::I64, f32, DataType::F32),
        (test_eq_i64_f64, i64, DataType::I64, f64, DataType::F64),
        (test_eq_u8_i8, u8, DataType::U8, i8, DataType::I8),
        (test_eq_u8_i16, u8, DataType::U8, i16, DataType::I16),
        (test_eq_u8_i32, u8, DataType::U8, i32, DataType::I32),
        (test_eq_u8_i64, u8, DataType::U8, i64, DataType::I64),
        (test_eq_u8_u8, u8, DataType::U8, u8, DataType::U8),
        (test_eq_u8_u16, u8, DataType::U8, u16, DataType::U16),
        (test_eq_u8_u32, u8, DataType::U8, u32, DataType::U32),
        (test_eq_u8_u64, u8, DataType::U8, u64, DataType::U64),
        (test_eq_u8_f16, u8, DataType::U8, f16, DataType::F16),
        (test_eq_u8_f32, u8, DataType::U8, f32, DataType::F32),
        (test_eq_u8_f64, u8, DataType::U8, f64, DataType::F64),
        (test_eq_u16_i8, u16, DataType::U16, i8, DataType::I8),
        (test_eq_u16_i16, u16, DataType::U16, i16, DataType::I16),
        (test_eq_u16_i32, u16, DataType::U16, i32, DataType::I32),
        (test_eq_u16_i64, u16, DataType::U16, i64, DataType::I64),
        (test_eq_u16_u8, u16, DataType::U16, u8, DataType::U8),
        (test_eq_u16_u16, u16, DataType::U16, u16, DataType::U16),
        (test_eq_u16_u32, u16, DataType::U16, u32, DataType::U32),
        (test_eq_u16_u64, u16, DataType::U16, u64, DataType::U64),
        (test_eq_u16_f16, u16, DataType::U16, f16, DataType::F16),
        (test_eq_u16_f32, u16, DataType::U16, f32, DataType::F32),
        (test_eq_u16_f64, u16, DataType::U16, f64, DataType::F64),
        (test_eq_u32_i8, u32, DataType::U32, i8, DataType::I8),
        (test_eq_u32_i16, u32, DataType::U32, i16, DataType::I16),
        (test_eq_u32_i32, u32, DataType::U32, i32, DataType::I32),
        (test_eq_u32_i64, u32, DataType::U32, i64, DataType::I64),
        (test_eq_u32_u8, u32, DataType::U32, u8, DataType::U8),
        (test_eq_u32_u16, u32, DataType::U32, u16, DataType::U16),
        (test_eq_u32_u32, u32, DataType::U32, u32, DataType::U32),
        (test_eq_u32_u64, u32, DataType::U32, u64, DataType::U64),
        (test_eq_u32_f16, u32, DataType::U32, f16, DataType::F16),
        (test_eq_u32_f32, u32, DataType::U32, f32, DataType::F32),
        (test_eq_u32_f64, u32, DataType::U32, f64, DataType::F64),
        (test_eq_u64_i8, u64, DataType::U64, i8, DataType::I8),
        (test_eq_u64_i16, u64, DataType::U64, i16, DataType::I16),
        (test_eq_u64_i32, u64, DataType::U64, i32, DataType::I32),
        (test_eq_u64_i64, u64, DataType::U64, i64, DataType::I64),
        (test_eq_u64_u8, u64, DataType::U64, u8, DataType::U8),
        (test_eq_u64_u16, u64, DataType::U64, u16, DataType::U16),
        (test_eq_u64_u32, u64, DataType::U64, u32, DataType::U32),
        (test_eq_u64_u64, u64, DataType::U64, u64, DataType::U64),
        (test_eq_u64_f16, u64, DataType::U64, f16, DataType::F16),
        (test_eq_u64_f32, u64, DataType::U64, f32, DataType::F32),
        (test_eq_u64_f64, u64, DataType::U64, f64, DataType::F64),
        (test_eq_f16_i8, f16, DataType::F16, i8, DataType::I8),
        (test_eq_f16_i16, f16, DataType::F16, i16, DataType::I16),
        (test_eq_f16_i32, f16, DataType::F16, i32, DataType::I32),
        (test_eq_f16_i64, f16, DataType::F16, i64, DataType::I64),
        (test_eq_f16_u8, f16, DataType::F16, u8, DataType::U8),
        (test_eq_f16_u16, f16, DataType::F16, u16, DataType::U16),
        (test_eq_f16_u32, f16, DataType::F16, u32, DataType::U32),
        (test_eq_f16_u64, f16, DataType::F16, u64, DataType::U64),
        (test_eq_f16_f16, f16, DataType::F16, f16, DataType::F16),
        (test_eq_f16_f32, f16, DataType::F16, f32, DataType::F32),
        (test_eq_f16_f64, f16, DataType::F16, f64, DataType::F64),
        (test_eq_f32_i8, f32, DataType::F32, i8, DataType::I8),
        (test_eq_f32_i16, f32, DataType::F32, i16, DataType::I16),
        (test_eq_f32_i32, f32, DataType::F32, i32, DataType::I32),
        (test_eq_f32_i64, f32, DataType::F32, i64, DataType::I64),
        (test_eq_f32_u8, f32, DataType::F32, u8, DataType::U8),
        (test_eq_f32_u16, f32, DataType::F32, u16, DataType::U16),
        (test_eq_f32_u32, f32, DataType::F32, u32, DataType::U32),
        (test_eq_f32_u64, f32, DataType::F32, u64, DataType::U64),
        (test_eq_f32_f16, f32, DataType::F32, f16, DataType::F16),
        (test_eq_f32_f32, f32, DataType::F32, f32, DataType::F32),
        (test_eq_f32_f64, f32, DataType::F32, f64, DataType::F64),
        (test_eq_f64_i8, f64, DataType::F64, i8, DataType::I8),
        (test_eq_f64_i16, f64, DataType::F64, i16, DataType::I16),
        (test_eq_f64_i32, f64, DataType::F64, i32, DataType::I32),
        (test_eq_f64_i64, f64, DataType::F64, i64, DataType::I64),
        (test_eq_f64_u8, f64, DataType::F64, u8, DataType::U8),
        (test_eq_f64_u16, f64, DataType::F64, u16, DataType::U16),
        (test_eq_f64_u32, f64, DataType::F64, u32, DataType::U32),
        (test_eq_f64_u64, f64, DataType::F64, u64, DataType::U64),
        (test_eq_f64_f16, f64, DataType::F64, f16, DataType::F16),
        (test_eq_f64_f32, f64, DataType::F64, f32, DataType::F32),
        (test_eq_f64_f64, f64, DataType::F64, f64, DataType::F64),
    ]
);

test_eval_binary_cmp_matrix!(
    Operator::NotEq,
    ErrorMode::Tachyon,
    100,
    400_000,
    [
        (test_neq_i8_i8, i8, DataType::I8, i8, DataType::I8),
        (test_neq_i8_i16, i8, DataType::I8, i16, DataType::I16),
        (test_neq_i8_i32, i8, DataType::I8, i32, DataType::I32),
        (test_neq_i8_i64, i8, DataType::I8, i64, DataType::I64),
        (test_neq_i8_u8, i8, DataType::I8, u8, DataType::U8),
        (test_neq_i8_u16, i8, DataType::I8, u16, DataType::U16),
        (test_neq_i8_f16, i8, DataType::I8, f16, DataType::F16),
        (test_neq_i8_u32, i8, DataType::I8, u32, DataType::U32),
        (test_neq_i8_u64, i8, DataType::I8, u64, DataType::U64),
        (test_neq_i8_f32, i8, DataType::I8, f32, DataType::F32),
        (test_neq_i8_f64, i8, DataType::I8, f64, DataType::F64),
        (test_neq_i16_i8, i16, DataType::I16, i8, DataType::I8),
        (test_neq_i16_i16, i16, DataType::I16, i16, DataType::I16),
        (test_neq_i16_i32, i16, DataType::I16, i32, DataType::I32),
        (test_neq_i16_i64, i16, DataType::I16, i64, DataType::I64),
        (test_neq_i16_u8, i16, DataType::I16, u8, DataType::U8),
        (test_neq_i16_u16, i16, DataType::I16, u16, DataType::U16),
        (test_neq_i16_u32, i16, DataType::I16, u32, DataType::U32),
        (test_neq_i16_u64, i16, DataType::I16, u64, DataType::U64),
        (test_neq_i16_f16, i16, DataType::I16, f16, DataType::F16),
        (test_neq_i16_f32, i16, DataType::I16, f32, DataType::F32),
        (test_neq_i16_f64, i16, DataType::I16, f64, DataType::F64),
        (test_neq_i32_i8, i32, DataType::I32, i8, DataType::I8),
        (test_neq_i32_i16, i32, DataType::I32, i16, DataType::I16),
        (test_neq_i32_i32, i32, DataType::I32, i32, DataType::I32),
        (test_neq_i32_i64, i32, DataType::I32, i64, DataType::I64),
        (test_neq_i32_u8, i32, DataType::I32, u8, DataType::U8),
        (test_neq_i32_u16, i32, DataType::I32, u16, DataType::U16),
        (test_neq_i32_u32, i32, DataType::I32, u32, DataType::U32),
        (test_neq_i32_u64, i32, DataType::I32, u64, DataType::U64),
        (test_neq_i32_f16, i32, DataType::I32, f16, DataType::F16),
        (test_neq_i32_f32, i32, DataType::I32, f32, DataType::F32),
        (test_neq_i32_f64, i32, DataType::I32, f64, DataType::F64),
        (test_neq_i64_i8, i64, DataType::I64, i8, DataType::I8),
        (test_neq_i64_i16, i64, DataType::I64, i16, DataType::I16),
        (test_neq_i64_i32, i64, DataType::I64, i32, DataType::I32),
        (test_neq_i64_i64, i64, DataType::I64, i64, DataType::I64),
        (test_neq_i64_u8, i64, DataType::I64, u8, DataType::U8),
        (test_neq_i64_u16, i64, DataType::I64, u16, DataType::U16),
        (test_neq_i64_u32, i64, DataType::I64, u32, DataType::U32),
        (test_neq_i64_u64, i64, DataType::I64, u64, DataType::U64),
        (test_neq_i64_f16, i64, DataType::I64, f16, DataType::F16),
        (test_neq_i64_f32, i64, DataType::I64, f32, DataType::F32),
        (test_neq_i64_f64, i64, DataType::I64, f64, DataType::F64),
        (test_neq_u8_i8, u8, DataType::U8, i8, DataType::I8),
        (test_neq_u8_i16, u8, DataType::U8, i16, DataType::I16),
        (test_neq_u8_i32, u8, DataType::U8, i32, DataType::I32),
        (test_neq_u8_i64, u8, DataType::U8, i64, DataType::I64),
        (test_neq_u8_u8, u8, DataType::U8, u8, DataType::U8),
        (test_neq_u8_u16, u8, DataType::U8, u16, DataType::U16),
        (test_neq_u8_u32, u8, DataType::U8, u32, DataType::U32),
        (test_neq_u8_u64, u8, DataType::U8, u64, DataType::U64),
        (test_neq_u8_f16, u8, DataType::U8, f16, DataType::F16),
        (test_neq_u8_f32, u8, DataType::U8, f32, DataType::F32),
        (test_neq_u8_f64, u8, DataType::U8, f64, DataType::F64),
        (test_neq_u16_i8, u16, DataType::U16, i8, DataType::I8),
        (test_neq_u16_i16, u16, DataType::U16, i16, DataType::I16),
        (test_neq_u16_i32, u16, DataType::U16, i32, DataType::I32),
        (test_neq_u16_i64, u16, DataType::U16, i64, DataType::I64),
        (test_neq_u16_u8, u16, DataType::U16, u8, DataType::U8),
        (test_neq_u16_u16, u16, DataType::U16, u16, DataType::U16),
        (test_neq_u16_u32, u16, DataType::U16, u32, DataType::U32),
        (test_neq_u16_u64, u16, DataType::U16, u64, DataType::U64),
        (test_neq_u16_f16, u16, DataType::U16, f16, DataType::F16),
        (test_neq_u16_f32, u16, DataType::U16, f32, DataType::F32),
        (test_neq_u16_f64, u16, DataType::U16, f64, DataType::F64),
        (test_neq_u32_i8, u32, DataType::U32, i8, DataType::I8),
        (test_neq_u32_i16, u32, DataType::U32, i16, DataType::I16),
        (test_neq_u32_i32, u32, DataType::U32, i32, DataType::I32),
        (test_neq_u32_i64, u32, DataType::U32, i64, DataType::I64),
        (test_neq_u32_u8, u32, DataType::U32, u8, DataType::U8),
        (test_neq_u32_u16, u32, DataType::U32, u16, DataType::U16),
        (test_neq_u32_u32, u32, DataType::U32, u32, DataType::U32),
        (test_neq_u32_u64, u32, DataType::U32, u64, DataType::U64),
        (test_neq_u32_f16, u32, DataType::U32, f16, DataType::F16),
        (test_neq_u32_f32, u32, DataType::U32, f32, DataType::F32),
        (test_neq_u32_f64, u32, DataType::U32, f64, DataType::F64),
        (test_neq_u64_i8, u64, DataType::U64, i8, DataType::I8),
        (test_neq_u64_i16, u64, DataType::U64, i16, DataType::I16),
        (test_neq_u64_i32, u64, DataType::U64, i32, DataType::I32),
        (test_neq_u64_i64, u64, DataType::U64, i64, DataType::I64),
        (test_neq_u64_u8, u64, DataType::U64, u8, DataType::U8),
        (test_neq_u64_u16, u64, DataType::U64, u16, DataType::U16),
        (test_neq_u64_u32, u64, DataType::U64, u32, DataType::U32),
        (test_neq_u64_u64, u64, DataType::U64, u64, DataType::U64),
        (test_neq_u64_f16, u64, DataType::U64, f16, DataType::F16),
        (test_neq_u64_f32, u64, DataType::U64, f32, DataType::F32),
        (test_neq_u64_f64, u64, DataType::U64, f64, DataType::F64),
        (test_neq_f16_i8, f16, DataType::F16, i8, DataType::I8),
        (test_neq_f16_i16, f16, DataType::F16, i16, DataType::I16),
        (test_neq_f16_i32, f16, DataType::F16, i32, DataType::I32),
        (test_neq_f16_i64, f16, DataType::F16, i64, DataType::I64),
        (test_neq_f16_u8, f16, DataType::F16, u8, DataType::U8),
        (test_neq_f16_u16, f16, DataType::F16, u16, DataType::U16),
        (test_neq_f16_u32, f16, DataType::F16, u32, DataType::U32),
        (test_neq_f16_u64, f16, DataType::F16, u64, DataType::U64),
        (test_neq_f16_f16, f16, DataType::F16, f16, DataType::F16),
        (test_neq_f16_f32, f16, DataType::F16, f32, DataType::F32),
        (test_neq_f16_f64, f16, DataType::F16, f64, DataType::F64),
        (test_neq_f32_i8, f32, DataType::F32, i8, DataType::I8),
        (test_neq_f32_i16, f32, DataType::F32, i16, DataType::I16),
        (test_neq_f32_i32, f32, DataType::F32, i32, DataType::I32),
        (test_neq_f32_i64, f32, DataType::F32, i64, DataType::I64),
        (test_neq_f32_u8, f32, DataType::F32, u8, DataType::U8),
        (test_neq_f32_u16, f32, DataType::F32, u16, DataType::U16),
        (test_neq_f32_u32, f32, DataType::F32, u32, DataType::U32),
        (test_neq_f32_u64, f32, DataType::F32, u64, DataType::U64),
        (test_neq_f32_f16, f32, DataType::F32, f16, DataType::F16),
        (test_neq_f32_f32, f32, DataType::F32, f32, DataType::F32),
        (test_neq_f32_f64, f32, DataType::F32, f64, DataType::F64),
        (test_neq_f64_i8, f64, DataType::F64, i8, DataType::I8),
        (test_neq_f64_i16, f64, DataType::F64, i16, DataType::I16),
        (test_neq_f64_i32, f64, DataType::F64, i32, DataType::I32),
        (test_neq_f64_i64, f64, DataType::F64, i64, DataType::I64),
        (test_neq_f64_u8, f64, DataType::F64, u8, DataType::U8),
        (test_neq_f64_u16, f64, DataType::F64, u16, DataType::U16),
        (test_neq_f64_u32, f64, DataType::F64, u32, DataType::U32),
        (test_neq_f64_u64, f64, DataType::F64, u64, DataType::U64),
        (test_neq_f64_f16, f64, DataType::F64, f16, DataType::F16),
        (test_neq_f64_f32, f64, DataType::F64, f32, DataType::F32),
        (test_neq_f64_f64, f64, DataType::F64, f64, DataType::F64),
    ]
);

test_eval_binary_cmp_matrix!(
    Operator::Gt,
    ErrorMode::Tachyon,
    200,
    500_000,
    [
        (test_gt_i8_i8, i8, DataType::I8, i8, DataType::I8),
        (test_gt_i8_i16, i8, DataType::I8, i16, DataType::I16),
        (test_gt_i8_i32, i8, DataType::I8, i32, DataType::I32),
        (test_gt_i8_i64, i8, DataType::I8, i64, DataType::I64),
        (test_gt_i8_u8, i8, DataType::I8, u8, DataType::U8),
        (test_gt_i8_u16, i8, DataType::I8, u16, DataType::U16),
        (test_gt_i8_f16, i8, DataType::I8, f16, DataType::F16),
        (test_gt_i8_u32, i8, DataType::I8, u32, DataType::U32),
        (test_gt_i8_u64, i8, DataType::I8, u64, DataType::U64),
        (test_gt_i8_f32, i8, DataType::I8, f32, DataType::F32),
        (test_gt_i8_f64, i8, DataType::I8, f64, DataType::F64),
        (test_gt_i16_i8, i16, DataType::I16, i8, DataType::I8),
        (test_gt_i16_i16, i16, DataType::I16, i16, DataType::I16),
        (test_gt_i16_i32, i16, DataType::I16, i32, DataType::I32),
        (test_gt_i16_i64, i16, DataType::I16, i64, DataType::I64),
        (test_gt_i16_u8, i16, DataType::I16, u8, DataType::U8),
        (test_gt_i16_u16, i16, DataType::I16, u16, DataType::U16),
        (test_gt_i16_u32, i16, DataType::I16, u32, DataType::U32),
        (test_gt_i16_u64, i16, DataType::I16, u64, DataType::U64),
        (test_gt_i16_f16, i16, DataType::I16, f16, DataType::F16),
        (test_gt_i16_f32, i16, DataType::I16, f32, DataType::F32),
        (test_gt_i16_f64, i16, DataType::I16, f64, DataType::F64),
        (test_gt_i32_i8, i32, DataType::I32, i8, DataType::I8),
        (test_gt_i32_i16, i32, DataType::I32, i16, DataType::I16),
        (test_gt_i32_i32, i32, DataType::I32, i32, DataType::I32),
        (test_gt_i32_i64, i32, DataType::I32, i64, DataType::I64),
        (test_gt_i32_u8, i32, DataType::I32, u8, DataType::U8),
        (test_gt_i32_u16, i32, DataType::I32, u16, DataType::U16),
        (test_gt_i32_u32, i32, DataType::I32, u32, DataType::U32),
        (test_gt_i32_u64, i32, DataType::I32, u64, DataType::U64),
        (test_gt_i32_f16, i32, DataType::I32, f16, DataType::F16),
        (test_gt_i32_f32, i32, DataType::I32, f32, DataType::F32),
        (test_gt_i32_f64, i32, DataType::I32, f64, DataType::F64),
        (test_gt_i64_i8, i64, DataType::I64, i8, DataType::I8),
        (test_gt_i64_i16, i64, DataType::I64, i16, DataType::I16),
        (test_gt_i64_i32, i64, DataType::I64, i32, DataType::I32),
        (test_gt_i64_i64, i64, DataType::I64, i64, DataType::I64),
        (test_gt_i64_u8, i64, DataType::I64, u8, DataType::U8),
        (test_gt_i64_u16, i64, DataType::I64, u16, DataType::U16),
        (test_gt_i64_u32, i64, DataType::I64, u32, DataType::U32),
        (test_gt_i64_u64, i64, DataType::I64, u64, DataType::U64),
        (test_gt_i64_f16, i64, DataType::I64, f16, DataType::F16),
        (test_gt_i64_f32, i64, DataType::I64, f32, DataType::F32),
        (test_gt_i64_f64, i64, DataType::I64, f64, DataType::F64),
        (test_gt_u8_i8, u8, DataType::U8, i8, DataType::I8),
        (test_gt_u8_i16, u8, DataType::U8, i16, DataType::I16),
        (test_gt_u8_i32, u8, DataType::U8, i32, DataType::I32),
        (test_gt_u8_i64, u8, DataType::U8, i64, DataType::I64),
        (test_gt_u8_u8, u8, DataType::U8, u8, DataType::U8),
        (test_gt_u8_u16, u8, DataType::U8, u16, DataType::U16),
        (test_gt_u8_u32, u8, DataType::U8, u32, DataType::U32),
        (test_gt_u8_u64, u8, DataType::U8, u64, DataType::U64),
        (test_gt_u8_f16, u8, DataType::U8, f16, DataType::F16),
        (test_gt_u8_f32, u8, DataType::U8, f32, DataType::F32),
        (test_gt_u8_f64, u8, DataType::U8, f64, DataType::F64),
        (test_gt_u16_i8, u16, DataType::U16, i8, DataType::I8),
        (test_gt_u16_i16, u16, DataType::U16, i16, DataType::I16),
        (test_gt_u16_i32, u16, DataType::U16, i32, DataType::I32),
        (test_gt_u16_i64, u16, DataType::U16, i64, DataType::I64),
        (test_gt_u16_u8, u16, DataType::U16, u8, DataType::U8),
        (test_gt_u16_u16, u16, DataType::U16, u16, DataType::U16),
        (test_gt_u16_u32, u16, DataType::U16, u32, DataType::U32),
        (test_gt_u16_u64, u16, DataType::U16, u64, DataType::U64),
        (test_gt_u16_f16, u16, DataType::U16, f16, DataType::F16),
        (test_gt_u16_f32, u16, DataType::U16, f32, DataType::F32),
        (test_gt_u16_f64, u16, DataType::U16, f64, DataType::F64),
        (test_gt_u32_i8, u32, DataType::U32, i8, DataType::I8),
        (test_gt_u32_i16, u32, DataType::U32, i16, DataType::I16),
        (test_gt_u32_i32, u32, DataType::U32, i32, DataType::I32),
        (test_gt_u32_i64, u32, DataType::U32, i64, DataType::I64),
        (test_gt_u32_u8, u32, DataType::U32, u8, DataType::U8),
        (test_gt_u32_u16, u32, DataType::U32, u16, DataType::U16),
        (test_gt_u32_u32, u32, DataType::U32, u32, DataType::U32),
        (test_gt_u32_u64, u32, DataType::U32, u64, DataType::U64),
        (test_gt_u32_f16, u32, DataType::U32, f16, DataType::F16),
        (test_gt_u32_f32, u32, DataType::U32, f32, DataType::F32),
        (test_gt_u32_f64, u32, DataType::U32, f64, DataType::F64),
        (test_gt_u64_i8, u64, DataType::U64, i8, DataType::I8),
        (test_gt_u64_i16, u64, DataType::U64, i16, DataType::I16),
        (test_gt_u64_i32, u64, DataType::U64, i32, DataType::I32),
        (test_gt_u64_i64, u64, DataType::U64, i64, DataType::I64),
        (test_gt_u64_u8, u64, DataType::U64, u8, DataType::U8),
        (test_gt_u64_u16, u64, DataType::U64, u16, DataType::U16),
        (test_gt_u64_u32, u64, DataType::U64, u32, DataType::U32),
        (test_gt_u64_u64, u64, DataType::U64, u64, DataType::U64),
        (test_gt_u64_f16, u64, DataType::U64, f16, DataType::F16),
        (test_gt_u64_f32, u64, DataType::U64, f32, DataType::F32),
        (test_gt_u64_f64, u64, DataType::U64, f64, DataType::F64),
        (test_gt_f16_i8, f16, DataType::F16, i8, DataType::I8),
        (test_gt_f16_i16, f16, DataType::F16, i16, DataType::I16),
        (test_gt_f16_i32, f16, DataType::F16, i32, DataType::I32),
        (test_gt_f16_i64, f16, DataType::F16, i64, DataType::I64),
        (test_gt_f16_u8, f16, DataType::F16, u8, DataType::U8),
        (test_gt_f16_u16, f16, DataType::F16, u16, DataType::U16),
        (test_gt_f16_u32, f16, DataType::F16, u32, DataType::U32),
        (test_gt_f16_u64, f16, DataType::F16, u64, DataType::U64),
        (test_gt_f16_f16, f16, DataType::F16, f16, DataType::F16),
        (test_gt_f16_f32, f16, DataType::F16, f32, DataType::F32),
        (test_gt_f16_f64, f16, DataType::F16, f64, DataType::F64),
        (test_gt_f32_i8, f32, DataType::F32, i8, DataType::I8),
        (test_gt_f32_i16, f32, DataType::F32, i16, DataType::I16),
        (test_gt_f32_i32, f32, DataType::F32, i32, DataType::I32),
        (test_gt_f32_i64, f32, DataType::F32, i64, DataType::I64),
        (test_gt_f32_u8, f32, DataType::F32, u8, DataType::U8),
        (test_gt_f32_u16, f32, DataType::F32, u16, DataType::U16),
        (test_gt_f32_u32, f32, DataType::F32, u32, DataType::U32),
        (test_gt_f32_u64, f32, DataType::F32, u64, DataType::U64),
        (test_gt_f32_f16, f32, DataType::F32, f16, DataType::F16),
        (test_gt_f32_f32, f32, DataType::F32, f32, DataType::F32),
        (test_gt_f32_f64, f32, DataType::F32, f64, DataType::F64),
        (test_gt_f64_i8, f64, DataType::F64, i8, DataType::I8),
        (test_gt_f64_i16, f64, DataType::F64, i16, DataType::I16),
        (test_gt_f64_i32, f64, DataType::F64, i32, DataType::I32),
        (test_gt_f64_i64, f64, DataType::F64, i64, DataType::I64),
        (test_gt_f64_u8, f64, DataType::F64, u8, DataType::U8),
        (test_gt_f64_u16, f64, DataType::F64, u16, DataType::U16),
        (test_gt_f64_u32, f64, DataType::F64, u32, DataType::U32),
        (test_gt_f64_u64, f64, DataType::F64, u64, DataType::U64),
        (test_gt_f64_f16, f64, DataType::F64, f16, DataType::F16),
        (test_gt_f64_f32, f64, DataType::F64, f32, DataType::F32),
        (test_gt_f64_f64, f64, DataType::F64, f64, DataType::F64),
    ]
);

test_eval_binary_cmp_matrix!(
    Operator::GtEq,
    ErrorMode::Tachyon,
    128,
    512_000,
    [
        (test_gteq_i8_i8, i8, DataType::I8, i8, DataType::I8),
        (test_gteq_i8_i16, i8, DataType::I8, i16, DataType::I16),
        (test_gteq_i8_i32, i8, DataType::I8, i32, DataType::I32),
        (test_gteq_i8_i64, i8, DataType::I8, i64, DataType::I64),
        (test_gteq_i8_u8, i8, DataType::I8, u8, DataType::U8),
        (test_gteq_i8_u16, i8, DataType::I8, u16, DataType::U16),
        (test_gteq_i8_f16, i8, DataType::I8, f16, DataType::F16),
        (test_gteq_i8_u32, i8, DataType::I8, u32, DataType::U32),
        (test_gteq_i8_u64, i8, DataType::I8, u64, DataType::U64),
        (test_gteq_i8_f32, i8, DataType::I8, f32, DataType::F32),
        (test_gteq_i8_f64, i8, DataType::I8, f64, DataType::F64),
        (test_gteq_i16_i8, i16, DataType::I16, i8, DataType::I8),
        (test_gteq_i16_i16, i16, DataType::I16, i16, DataType::I16),
        (test_gteq_i16_i32, i16, DataType::I16, i32, DataType::I32),
        (test_gteq_i16_i64, i16, DataType::I16, i64, DataType::I64),
        (test_gteq_i16_u8, i16, DataType::I16, u8, DataType::U8),
        (test_gteq_i16_u16, i16, DataType::I16, u16, DataType::U16),
        (test_gteq_i16_u32, i16, DataType::I16, u32, DataType::U32),
        (test_gteq_i16_u64, i16, DataType::I16, u64, DataType::U64),
        (test_gteq_i16_f16, i16, DataType::I16, f16, DataType::F16),
        (test_gteq_i16_f32, i16, DataType::I16, f32, DataType::F32),
        (test_gteq_i16_f64, i16, DataType::I16, f64, DataType::F64),
        (test_gteq_i32_i8, i32, DataType::I32, i8, DataType::I8),
        (test_gteq_i32_i16, i32, DataType::I32, i16, DataType::I16),
        (test_gteq_i32_i32, i32, DataType::I32, i32, DataType::I32),
        (test_gteq_i32_i64, i32, DataType::I32, i64, DataType::I64),
        (test_gteq_i32_u8, i32, DataType::I32, u8, DataType::U8),
        (test_gteq_i32_u16, i32, DataType::I32, u16, DataType::U16),
        (test_gteq_i32_u32, i32, DataType::I32, u32, DataType::U32),
        (test_gteq_i32_u64, i32, DataType::I32, u64, DataType::U64),
        (test_gteq_i32_f16, i32, DataType::I32, f16, DataType::F16),
        (test_gteq_i32_f32, i32, DataType::I32, f32, DataType::F32),
        (test_gteq_i32_f64, i32, DataType::I32, f64, DataType::F64),
        (test_gteq_i64_i8, i64, DataType::I64, i8, DataType::I8),
        (test_gteq_i64_i16, i64, DataType::I64, i16, DataType::I16),
        (test_gteq_i64_i32, i64, DataType::I64, i32, DataType::I32),
        (test_gteq_i64_i64, i64, DataType::I64, i64, DataType::I64),
        (test_gteq_i64_u8, i64, DataType::I64, u8, DataType::U8),
        (test_gteq_i64_u16, i64, DataType::I64, u16, DataType::U16),
        (test_gteq_i64_u32, i64, DataType::I64, u32, DataType::U32),
        (test_gteq_i64_u64, i64, DataType::I64, u64, DataType::U64),
        (test_gteq_i64_f16, i64, DataType::I64, f16, DataType::F16),
        (test_gteq_i64_f32, i64, DataType::I64, f32, DataType::F32),
        (test_gteq_i64_f64, i64, DataType::I64, f64, DataType::F64),
        (test_gteq_u8_i8, u8, DataType::U8, i8, DataType::I8),
        (test_gteq_u8_i16, u8, DataType::U8, i16, DataType::I16),
        (test_gteq_u8_i32, u8, DataType::U8, i32, DataType::I32),
        (test_gteq_u8_i64, u8, DataType::U8, i64, DataType::I64),
        (test_gteq_u8_u8, u8, DataType::U8, u8, DataType::U8),
        (test_gteq_u8_u16, u8, DataType::U8, u16, DataType::U16),
        (test_gteq_u8_u32, u8, DataType::U8, u32, DataType::U32),
        (test_gteq_u8_u64, u8, DataType::U8, u64, DataType::U64),
        (test_gteq_u8_f16, u8, DataType::U8, f16, DataType::F16),
        (test_gteq_u8_f32, u8, DataType::U8, f32, DataType::F32),
        (test_gteq_u8_f64, u8, DataType::U8, f64, DataType::F64),
        (test_gteq_u16_i8, u16, DataType::U16, i8, DataType::I8),
        (test_gteq_u16_i16, u16, DataType::U16, i16, DataType::I16),
        (test_gteq_u16_i32, u16, DataType::U16, i32, DataType::I32),
        (test_gteq_u16_i64, u16, DataType::U16, i64, DataType::I64),
        (test_gteq_u16_u8, u16, DataType::U16, u8, DataType::U8),
        (test_gteq_u16_u16, u16, DataType::U16, u16, DataType::U16),
        (test_gteq_u16_u32, u16, DataType::U16, u32, DataType::U32),
        (test_gteq_u16_u64, u16, DataType::U16, u64, DataType::U64),
        (test_gteq_u16_f16, u16, DataType::U16, f16, DataType::F16),
        (test_gteq_u16_f32, u16, DataType::U16, f32, DataType::F32),
        (test_gteq_u16_f64, u16, DataType::U16, f64, DataType::F64),
        (test_gteq_u32_i8, u32, DataType::U32, i8, DataType::I8),
        (test_gteq_u32_i16, u32, DataType::U32, i16, DataType::I16),
        (test_gteq_u32_i32, u32, DataType::U32, i32, DataType::I32),
        (test_gteq_u32_i64, u32, DataType::U32, i64, DataType::I64),
        (test_gteq_u32_u8, u32, DataType::U32, u8, DataType::U8),
        (test_gteq_u32_u16, u32, DataType::U32, u16, DataType::U16),
        (test_gteq_u32_u32, u32, DataType::U32, u32, DataType::U32),
        (test_gteq_u32_u64, u32, DataType::U32, u64, DataType::U64),
        (test_gteq_u32_f16, u32, DataType::U32, f16, DataType::F16),
        (test_gteq_u32_f32, u32, DataType::U32, f32, DataType::F32),
        (test_gteq_u32_f64, u32, DataType::U32, f64, DataType::F64),
        (test_gteq_u64_i8, u64, DataType::U64, i8, DataType::I8),
        (test_gteq_u64_i16, u64, DataType::U64, i16, DataType::I16),
        (test_gteq_u64_i32, u64, DataType::U64, i32, DataType::I32),
        (test_gteq_u64_i64, u64, DataType::U64, i64, DataType::I64),
        (test_gteq_u64_u8, u64, DataType::U64, u8, DataType::U8),
        (test_gteq_u64_u16, u64, DataType::U64, u16, DataType::U16),
        (test_gteq_u64_u32, u64, DataType::U64, u32, DataType::U32),
        (test_gteq_u64_u64, u64, DataType::U64, u64, DataType::U64),
        (test_gteq_u64_f16, u64, DataType::U64, f16, DataType::F16),
        (test_gteq_u64_f32, u64, DataType::U64, f32, DataType::F32),
        (test_gteq_u64_f64, u64, DataType::U64, f64, DataType::F64),
        (test_gteq_f16_i8, f16, DataType::F16, i8, DataType::I8),
        (test_gteq_f16_i16, f16, DataType::F16, i16, DataType::I16),
        (test_gteq_f16_i32, f16, DataType::F16, i32, DataType::I32),
        (test_gteq_f16_i64, f16, DataType::F16, i64, DataType::I64),
        (test_gteq_f16_u8, f16, DataType::F16, u8, DataType::U8),
        (test_gteq_f16_u16, f16, DataType::F16, u16, DataType::U16),
        (test_gteq_f16_u32, f16, DataType::F16, u32, DataType::U32),
        (test_gteq_f16_u64, f16, DataType::F16, u64, DataType::U64),
        (test_gteq_f16_f16, f16, DataType::F16, f16, DataType::F16),
        (test_gteq_f16_f32, f16, DataType::F16, f32, DataType::F32),
        (test_gteq_f16_f64, f16, DataType::F16, f64, DataType::F64),
        (test_gteq_f32_i8, f32, DataType::F32, i8, DataType::I8),
        (test_gteq_f32_i16, f32, DataType::F32, i16, DataType::I16),
        (test_gteq_f32_i32, f32, DataType::F32, i32, DataType::I32),
        (test_gteq_f32_i64, f32, DataType::F32, i64, DataType::I64),
        (test_gteq_f32_u8, f32, DataType::F32, u8, DataType::U8),
        (test_gteq_f32_u16, f32, DataType::F32, u16, DataType::U16),
        (test_gteq_f32_u32, f32, DataType::F32, u32, DataType::U32),
        (test_gteq_f32_u64, f32, DataType::F32, u64, DataType::U64),
        (test_gteq_f32_f16, f32, DataType::F32, f16, DataType::F16),
        (test_gteq_f32_f32, f32, DataType::F32, f32, DataType::F32),
        (test_gteq_f32_f64, f32, DataType::F32, f64, DataType::F64),
        (test_gteq_f64_i8, f64, DataType::F64, i8, DataType::I8),
        (test_gteq_f64_i16, f64, DataType::F64, i16, DataType::I16),
        (test_gteq_f64_i32, f64, DataType::F64, i32, DataType::I32),
        (test_gteq_f64_i64, f64, DataType::F64, i64, DataType::I64),
        (test_gteq_f64_u8, f64, DataType::F64, u8, DataType::U8),
        (test_gteq_f64_u16, f64, DataType::F64, u16, DataType::U16),
        (test_gteq_f64_u32, f64, DataType::F64, u32, DataType::U32),
        (test_gteq_f64_u64, f64, DataType::F64, u64, DataType::U64),
        (test_gteq_f64_f16, f64, DataType::F64, f16, DataType::F16),
        (test_gteq_f64_f32, f64, DataType::F64, f32, DataType::F32),
        (test_gteq_f64_f64, f64, DataType::F64, f64, DataType::F64),
    ]
);

test_eval_binary_cmp_matrix!(
    Operator::Lt,
    ErrorMode::Tachyon,
    10,
    100_000,
    [
        (test_lt_i8_i8, i8, DataType::I8, i8, DataType::I8),
        (test_lt_i8_i16, i8, DataType::I8, i16, DataType::I16),
        (test_lt_i8_i32, i8, DataType::I8, i32, DataType::I32),
        (test_lt_i8_i64, i8, DataType::I8, i64, DataType::I64),
        (test_lt_i8_u8, i8, DataType::I8, u8, DataType::U8),
        (test_lt_i8_u16, i8, DataType::I8, u16, DataType::U16),
        (test_lt_i8_f16, i8, DataType::I8, f16, DataType::F16),
        (test_lt_i8_u32, i8, DataType::I8, u32, DataType::U32),
        (test_lt_i8_u64, i8, DataType::I8, u64, DataType::U64),
        (test_lt_i8_f32, i8, DataType::I8, f32, DataType::F32),
        (test_lt_i8_f64, i8, DataType::I8, f64, DataType::F64),
        (test_lt_i16_i8, i16, DataType::I16, i8, DataType::I8),
        (test_lt_i16_i16, i16, DataType::I16, i16, DataType::I16),
        (test_lt_i16_i32, i16, DataType::I16, i32, DataType::I32),
        (test_lt_i16_i64, i16, DataType::I16, i64, DataType::I64),
        (test_lt_i16_u8, i16, DataType::I16, u8, DataType::U8),
        (test_lt_i16_u16, i16, DataType::I16, u16, DataType::U16),
        (test_lt_i16_u32, i16, DataType::I16, u32, DataType::U32),
        (test_lt_i16_u64, i16, DataType::I16, u64, DataType::U64),
        (test_lt_i16_f16, i16, DataType::I16, f16, DataType::F16),
        (test_lt_i16_f32, i16, DataType::I16, f32, DataType::F32),
        (test_lt_i16_f64, i16, DataType::I16, f64, DataType::F64),
        (test_lt_i32_i8, i32, DataType::I32, i8, DataType::I8),
        (test_lt_i32_i16, i32, DataType::I32, i16, DataType::I16),
        (test_lt_i32_i32, i32, DataType::I32, i32, DataType::I32),
        (test_lt_i32_i64, i32, DataType::I32, i64, DataType::I64),
        (test_lt_i32_u8, i32, DataType::I32, u8, DataType::U8),
        (test_lt_i32_u16, i32, DataType::I32, u16, DataType::U16),
        (test_lt_i32_u32, i32, DataType::I32, u32, DataType::U32),
        (test_lt_i32_u64, i32, DataType::I32, u64, DataType::U64),
        (test_lt_i32_f16, i32, DataType::I32, f16, DataType::F16),
        (test_lt_i32_f32, i32, DataType::I32, f32, DataType::F32),
        (test_lt_i32_f64, i32, DataType::I32, f64, DataType::F64),
        (test_lt_i64_i8, i64, DataType::I64, i8, DataType::I8),
        (test_lt_i64_i16, i64, DataType::I64, i16, DataType::I16),
        (test_lt_i64_i32, i64, DataType::I64, i32, DataType::I32),
        (test_lt_i64_i64, i64, DataType::I64, i64, DataType::I64),
        (test_lt_i64_u8, i64, DataType::I64, u8, DataType::U8),
        (test_lt_i64_u16, i64, DataType::I64, u16, DataType::U16),
        (test_lt_i64_u32, i64, DataType::I64, u32, DataType::U32),
        (test_lt_i64_u64, i64, DataType::I64, u64, DataType::U64),
        (test_lt_i64_f16, i64, DataType::I64, f16, DataType::F16),
        (test_lt_i64_f32, i64, DataType::I64, f32, DataType::F32),
        (test_lt_i64_f64, i64, DataType::I64, f64, DataType::F64),
        (test_lt_u8_i8, u8, DataType::U8, i8, DataType::I8),
        (test_lt_u8_i16, u8, DataType::U8, i16, DataType::I16),
        (test_lt_u8_i32, u8, DataType::U8, i32, DataType::I32),
        (test_lt_u8_i64, u8, DataType::U8, i64, DataType::I64),
        (test_lt_u8_u8, u8, DataType::U8, u8, DataType::U8),
        (test_lt_u8_u16, u8, DataType::U8, u16, DataType::U16),
        (test_lt_u8_u32, u8, DataType::U8, u32, DataType::U32),
        (test_lt_u8_u64, u8, DataType::U8, u64, DataType::U64),
        (test_lt_u8_f16, u8, DataType::U8, f16, DataType::F16),
        (test_lt_u8_f32, u8, DataType::U8, f32, DataType::F32),
        (test_lt_u8_f64, u8, DataType::U8, f64, DataType::F64),
        (test_lt_u16_i8, u16, DataType::U16, i8, DataType::I8),
        (test_lt_u16_i16, u16, DataType::U16, i16, DataType::I16),
        (test_lt_u16_i32, u16, DataType::U16, i32, DataType::I32),
        (test_lt_u16_i64, u16, DataType::U16, i64, DataType::I64),
        (test_lt_u16_u8, u16, DataType::U16, u8, DataType::U8),
        (test_lt_u16_u16, u16, DataType::U16, u16, DataType::U16),
        (test_lt_u16_u32, u16, DataType::U16, u32, DataType::U32),
        (test_lt_u16_u64, u16, DataType::U16, u64, DataType::U64),
        (test_lt_u16_f16, u16, DataType::U16, f16, DataType::F16),
        (test_lt_u16_f32, u16, DataType::U16, f32, DataType::F32),
        (test_lt_u16_f64, u16, DataType::U16, f64, DataType::F64),
        (test_lt_u32_i8, u32, DataType::U32, i8, DataType::I8),
        (test_lt_u32_i16, u32, DataType::U32, i16, DataType::I16),
        (test_lt_u32_i32, u32, DataType::U32, i32, DataType::I32),
        (test_lt_u32_i64, u32, DataType::U32, i64, DataType::I64),
        (test_lt_u32_u8, u32, DataType::U32, u8, DataType::U8),
        (test_lt_u32_u16, u32, DataType::U32, u16, DataType::U16),
        (test_lt_u32_u32, u32, DataType::U32, u32, DataType::U32),
        (test_lt_u32_u64, u32, DataType::U32, u64, DataType::U64),
        (test_lt_u32_f16, u32, DataType::U32, f16, DataType::F16),
        (test_lt_u32_f32, u32, DataType::U32, f32, DataType::F32),
        (test_lt_u32_f64, u32, DataType::U32, f64, DataType::F64),
        (test_lt_u64_i8, u64, DataType::U64, i8, DataType::I8),
        (test_lt_u64_i16, u64, DataType::U64, i16, DataType::I16),
        (test_lt_u64_i32, u64, DataType::U64, i32, DataType::I32),
        (test_lt_u64_i64, u64, DataType::U64, i64, DataType::I64),
        (test_lt_u64_u8, u64, DataType::U64, u8, DataType::U8),
        (test_lt_u64_u16, u64, DataType::U64, u16, DataType::U16),
        (test_lt_u64_u32, u64, DataType::U64, u32, DataType::U32),
        (test_lt_u64_u64, u64, DataType::U64, u64, DataType::U64),
        (test_lt_u64_f16, u64, DataType::U64, f16, DataType::F16),
        (test_lt_u64_f32, u64, DataType::U64, f32, DataType::F32),
        (test_lt_u64_f64, u64, DataType::U64, f64, DataType::F64),
        (test_lt_f16_i8, f16, DataType::F16, i8, DataType::I8),
        (test_lt_f16_i16, f16, DataType::F16, i16, DataType::I16),
        (test_lt_f16_i32, f16, DataType::F16, i32, DataType::I32),
        (test_lt_f16_i64, f16, DataType::F16, i64, DataType::I64),
        (test_lt_f16_u8, f16, DataType::F16, u8, DataType::U8),
        (test_lt_f16_u16, f16, DataType::F16, u16, DataType::U16),
        (test_lt_f16_u32, f16, DataType::F16, u32, DataType::U32),
        (test_lt_f16_u64, f16, DataType::F16, u64, DataType::U64),
        (test_lt_f16_f16, f16, DataType::F16, f16, DataType::F16),
        (test_lt_f16_f32, f16, DataType::F16, f32, DataType::F32),
        (test_lt_f16_f64, f16, DataType::F16, f64, DataType::F64),
        (test_lt_f32_i8, f32, DataType::F32, i8, DataType::I8),
        (test_lt_f32_i16, f32, DataType::F32, i16, DataType::I16),
        (test_lt_f32_i32, f32, DataType::F32, i32, DataType::I32),
        (test_lt_f32_i64, f32, DataType::F32, i64, DataType::I64),
        (test_lt_f32_u8, f32, DataType::F32, u8, DataType::U8),
        (test_lt_f32_u16, f32, DataType::F32, u16, DataType::U16),
        (test_lt_f32_u32, f32, DataType::F32, u32, DataType::U32),
        (test_lt_f32_u64, f32, DataType::F32, u64, DataType::U64),
        (test_lt_f32_f16, f32, DataType::F32, f16, DataType::F16),
        (test_lt_f32_f32, f32, DataType::F32, f32, DataType::F32),
        (test_lt_f32_f64, f32, DataType::F32, f64, DataType::F64),
        (test_lt_f64_i8, f64, DataType::F64, i8, DataType::I8),
        (test_lt_f64_i16, f64, DataType::F64, i16, DataType::I16),
        (test_lt_f64_i32, f64, DataType::F64, i32, DataType::I32),
        (test_lt_f64_i64, f64, DataType::F64, i64, DataType::I64),
        (test_lt_f64_u8, f64, DataType::F64, u8, DataType::U8),
        (test_lt_f64_u16, f64, DataType::F64, u16, DataType::U16),
        (test_lt_f64_u32, f64, DataType::F64, u32, DataType::U32),
        (test_lt_f64_u64, f64, DataType::F64, u64, DataType::U64),
        (test_lt_f64_f16, f64, DataType::F64, f16, DataType::F16),
        (test_lt_f64_f32, f64, DataType::F64, f32, DataType::F32),
        (test_lt_f64_f64, f64, DataType::F64, f64, DataType::F64),
    ]
);

test_eval_binary_cmp_matrix!(
    Operator::LtEq,
    ErrorMode::Tachyon,
    100,
    250_000,
    [
        (test_lteq_i8_i8, i8, DataType::I8, i8, DataType::I8),
        (test_lteq_i8_i16, i8, DataType::I8, i16, DataType::I16),
        (test_lteq_i8_i32, i8, DataType::I8, i32, DataType::I32),
        (test_lteq_i8_i64, i8, DataType::I8, i64, DataType::I64),
        (test_lteq_i8_u8, i8, DataType::I8, u8, DataType::U8),
        (test_lteq_i8_u16, i8, DataType::I8, u16, DataType::U16),
        (test_lteq_i8_f16, i8, DataType::I8, f16, DataType::F16),
        (test_lteq_i8_u32, i8, DataType::I8, u32, DataType::U32),
        (test_lteq_i8_u64, i8, DataType::I8, u64, DataType::U64),
        (test_lteq_i8_f32, i8, DataType::I8, f32, DataType::F32),
        (test_lteq_i8_f64, i8, DataType::I8, f64, DataType::F64),
        (test_lteq_i16_i8, i16, DataType::I16, i8, DataType::I8),
        (test_lteq_i16_i16, i16, DataType::I16, i16, DataType::I16),
        (test_lteq_i16_i32, i16, DataType::I16, i32, DataType::I32),
        (test_lteq_i16_i64, i16, DataType::I16, i64, DataType::I64),
        (test_lteq_i16_u8, i16, DataType::I16, u8, DataType::U8),
        (test_lteq_i16_u16, i16, DataType::I16, u16, DataType::U16),
        (test_lteq_i16_u32, i16, DataType::I16, u32, DataType::U32),
        (test_lteq_i16_u64, i16, DataType::I16, u64, DataType::U64),
        (test_lteq_i16_f16, i16, DataType::I16, f16, DataType::F16),
        (test_lteq_i16_f32, i16, DataType::I16, f32, DataType::F32),
        (test_lteq_i16_f64, i16, DataType::I16, f64, DataType::F64),
        (test_lteq_i32_i8, i32, DataType::I32, i8, DataType::I8),
        (test_lteq_i32_i16, i32, DataType::I32, i16, DataType::I16),
        (test_lteq_i32_i32, i32, DataType::I32, i32, DataType::I32),
        (test_lteq_i32_i64, i32, DataType::I32, i64, DataType::I64),
        (test_lteq_i32_u8, i32, DataType::I32, u8, DataType::U8),
        (test_lteq_i32_u16, i32, DataType::I32, u16, DataType::U16),
        (test_lteq_i32_u32, i32, DataType::I32, u32, DataType::U32),
        (test_lteq_i32_u64, i32, DataType::I32, u64, DataType::U64),
        (test_lteq_i32_f16, i32, DataType::I32, f16, DataType::F16),
        (test_lteq_i32_f32, i32, DataType::I32, f32, DataType::F32),
        (test_lteq_i32_f64, i32, DataType::I32, f64, DataType::F64),
        (test_lteq_i64_i8, i64, DataType::I64, i8, DataType::I8),
        (test_lteq_i64_i16, i64, DataType::I64, i16, DataType::I16),
        (test_lteq_i64_i32, i64, DataType::I64, i32, DataType::I32),
        (test_lteq_i64_i64, i64, DataType::I64, i64, DataType::I64),
        (test_lteq_i64_u8, i64, DataType::I64, u8, DataType::U8),
        (test_lteq_i64_u16, i64, DataType::I64, u16, DataType::U16),
        (test_lteq_i64_u32, i64, DataType::I64, u32, DataType::U32),
        (test_lteq_i64_u64, i64, DataType::I64, u64, DataType::U64),
        (test_lteq_i64_f16, i64, DataType::I64, f16, DataType::F16),
        (test_lteq_i64_f32, i64, DataType::I64, f32, DataType::F32),
        (test_lteq_i64_f64, i64, DataType::I64, f64, DataType::F64),
        (test_lteq_u8_i8, u8, DataType::U8, i8, DataType::I8),
        (test_lteq_u8_i16, u8, DataType::U8, i16, DataType::I16),
        (test_lteq_u8_i32, u8, DataType::U8, i32, DataType::I32),
        (test_lteq_u8_i64, u8, DataType::U8, i64, DataType::I64),
        (test_lteq_u8_u8, u8, DataType::U8, u8, DataType::U8),
        (test_lteq_u8_u16, u8, DataType::U8, u16, DataType::U16),
        (test_lteq_u8_u32, u8, DataType::U8, u32, DataType::U32),
        (test_lteq_u8_u64, u8, DataType::U8, u64, DataType::U64),
        (test_lteq_u8_f16, u8, DataType::U8, f16, DataType::F16),
        (test_lteq_u8_f32, u8, DataType::U8, f32, DataType::F32),
        (test_lteq_u8_f64, u8, DataType::U8, f64, DataType::F64),
        (test_lteq_u16_i8, u16, DataType::U16, i8, DataType::I8),
        (test_lteq_u16_i16, u16, DataType::U16, i16, DataType::I16),
        (test_lteq_u16_i32, u16, DataType::U16, i32, DataType::I32),
        (test_lteq_u16_i64, u16, DataType::U16, i64, DataType::I64),
        (test_lteq_u16_u8, u16, DataType::U16, u8, DataType::U8),
        (test_lteq_u16_u16, u16, DataType::U16, u16, DataType::U16),
        (test_lteq_u16_u32, u16, DataType::U16, u32, DataType::U32),
        (test_lteq_u16_u64, u16, DataType::U16, u64, DataType::U64),
        (test_lteq_u16_f16, u16, DataType::U16, f16, DataType::F16),
        (test_lteq_u16_f32, u16, DataType::U16, f32, DataType::F32),
        (test_lteq_u16_f64, u16, DataType::U16, f64, DataType::F64),
        (test_lteq_u32_i8, u32, DataType::U32, i8, DataType::I8),
        (test_lteq_u32_i16, u32, DataType::U32, i16, DataType::I16),
        (test_lteq_u32_i32, u32, DataType::U32, i32, DataType::I32),
        (test_lteq_u32_i64, u32, DataType::U32, i64, DataType::I64),
        (test_lteq_u32_u8, u32, DataType::U32, u8, DataType::U8),
        (test_lteq_u32_u16, u32, DataType::U32, u16, DataType::U16),
        (test_lteq_u32_u32, u32, DataType::U32, u32, DataType::U32),
        (test_lteq_u32_u64, u32, DataType::U32, u64, DataType::U64),
        (test_lteq_u32_f16, u32, DataType::U32, f16, DataType::F16),
        (test_lteq_u32_f32, u32, DataType::U32, f32, DataType::F32),
        (test_lteq_u32_f64, u32, DataType::U32, f64, DataType::F64),
        (test_lteq_u64_i8, u64, DataType::U64, i8, DataType::I8),
        (test_lteq_u64_i16, u64, DataType::U64, i16, DataType::I16),
        (test_lteq_u64_i32, u64, DataType::U64, i32, DataType::I32),
        (test_lteq_u64_i64, u64, DataType::U64, i64, DataType::I64),
        (test_lteq_u64_u8, u64, DataType::U64, u8, DataType::U8),
        (test_lteq_u64_u16, u64, DataType::U64, u16, DataType::U16),
        (test_lteq_u64_u32, u64, DataType::U64, u32, DataType::U32),
        (test_lteq_u64_u64, u64, DataType::U64, u64, DataType::U64),
        (test_lteq_u64_f16, u64, DataType::U64, f16, DataType::F16),
        (test_lteq_u64_f32, u64, DataType::U64, f32, DataType::F32),
        (test_lteq_u64_f64, u64, DataType::U64, f64, DataType::F64),
        (test_lteq_f16_i8, f16, DataType::F16, i8, DataType::I8),
        (test_lteq_f16_i16, f16, DataType::F16, i16, DataType::I16),
        (test_lteq_f16_i32, f16, DataType::F16, i32, DataType::I32),
        (test_lteq_f16_i64, f16, DataType::F16, i64, DataType::I64),
        (test_lteq_f16_u8, f16, DataType::F16, u8, DataType::U8),
        (test_lteq_f16_u16, f16, DataType::F16, u16, DataType::U16),
        (test_lteq_f16_u32, f16, DataType::F16, u32, DataType::U32),
        (test_lteq_f16_u64, f16, DataType::F16, u64, DataType::U64),
        (test_lteq_f16_f16, f16, DataType::F16, f16, DataType::F16),
        (test_lteq_f16_f32, f16, DataType::F16, f32, DataType::F32),
        (test_lteq_f16_f64, f16, DataType::F16, f64, DataType::F64),
        (test_lteq_f32_i8, f32, DataType::F32, i8, DataType::I8),
        (test_lteq_f32_i16, f32, DataType::F32, i16, DataType::I16),
        (test_lteq_f32_i32, f32, DataType::F32, i32, DataType::I32),
        (test_lteq_f32_i64, f32, DataType::F32, i64, DataType::I64),
        (test_lteq_f32_u8, f32, DataType::F32, u8, DataType::U8),
        (test_lteq_f32_u16, f32, DataType::F32, u16, DataType::U16),
        (test_lteq_f32_u32, f32, DataType::F32, u32, DataType::U32),
        (test_lteq_f32_u64, f32, DataType::F32, u64, DataType::U64),
        (test_lteq_f32_f16, f32, DataType::F32, f16, DataType::F16),
        (test_lteq_f32_f32, f32, DataType::F32, f32, DataType::F32),
        (test_lteq_f32_f64, f32, DataType::F32, f64, DataType::F64),
        (test_lteq_f64_i8, f64, DataType::F64, i8, DataType::I8),
        (test_lteq_f64_i16, f64, DataType::F64, i16, DataType::I16),
        (test_lteq_f64_i32, f64, DataType::F64, i32, DataType::I32),
        (test_lteq_f64_i64, f64, DataType::F64, i64, DataType::I64),
        (test_lteq_f64_u8, f64, DataType::F64, u8, DataType::U8),
        (test_lteq_f64_u16, f64, DataType::F64, u16, DataType::U16),
        (test_lteq_f64_u32, f64, DataType::F64, u32, DataType::U32),
        (test_lteq_f64_u64, f64, DataType::F64, u64, DataType::U64),
        (test_lteq_f64_f16, f64, DataType::F64, f16, DataType::F16),
        (test_lteq_f64_f32, f64, DataType::F64, f32, DataType::F32),
        (test_lteq_f64_f64, f64, DataType::F64, f64, DataType::F64),
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
    let result = result.unwrap();
    assert!(result[0].data_as_slice::<bf16>().is_some());
    assert!(
        result[0].data_as_slice::<bf16>().unwrap()
            == [bf16::from_f32(1.0), bf16::from_f32(f32::INFINITY), bf16::from_f32(1.5)]
    );
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_div_zero_by_zero() {
    use compute::bit_vector::BitVector;
    use compute::data_type::DataType;
    use compute::error::ErrorMode;
    use compute::evaluate::{Device, evaluate};
    use compute::expr::Expr;
    use compute::operator::Operator;

    let a_vec: Vec<f32> = vec![1.0, 0.0, 3.0];
    let b_vec: Vec<f32> = vec![1.0, 0.0, 2.0];

    let a_bit_vec = BitVector::<u64>::new_all_valid(a_vec.len());
    let b_bit_vec = BitVector::<u64>::new_all_valid(b_vec.len());

    let col_a = create_column!(a_vec, Some(a_bit_vec), "a", DataType::F32);
    let col_b = create_column!(b_vec, Some(b_bit_vec), "b", DataType::F32);

    let expr = Expr::binary(Operator::Div, Expr::col("a"), Expr::col("b"));

    let result = evaluate(Device::GPU, ErrorMode::Tachyon, &expr, &vec![col_a, col_b]).await;
    let result = result.unwrap();
    assert!(result[0].data_as_slice::<f32>().is_some());
    let result = result[0].data_as_slice::<f32>().unwrap();
    assert_eq!(result[0], 1.0);
    assert!(result[1].is_nan());
    assert_eq!(result[2], 1.5);
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
    assert!(bit_vec.is_null(1));
    assert!(bit_vec.is_null(2));
    assert!(bit_vec.is_valid(3));
    assert_eq!(output[0], bf16::from_f32(4.0));
    assert_eq!(output[3], bf16::from_f32(5.0));
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_add_literal() {
    use compute::bit_vector::BitVector;
    use compute::data_type::DataType;
    use compute::error::ErrorMode;
    use compute::evaluate::{Device, evaluate};
    use compute::expr::{Expr, Literal};
    use compute::operator::Operator;
    use half::bf16;

    let a_vec: Vec<bf16> =
        vec![bf16::from_f32(1.0), bf16::from_f32(2.0), bf16::from_f32(3.0), bf16::from_f32(11.0)];

    let mut a_bit_vec = BitVector::<u64>::new_all_valid(a_vec.len());
    a_bit_vec.set_null(1);
    a_bit_vec.set_null(2);
    let col_a = create_column!(a_vec, Some(a_bit_vec), "a", DataType::BF16);

    let expr =
        Expr::binary(Operator::Add, Expr::col("a"), Expr::lit(Literal::BF16(bf16::from_f32(-5.0))));

    let result = evaluate(Device::GPU, ErrorMode::Tachyon, &expr, &vec![col_a]).await;
    let result = result.unwrap();
    assert!(result[0].data_as_slice::<bf16>().is_some());
    let output = result[0].data_as_slice::<bf16>().unwrap();
    let bit_vec = result[0].null_bits_as_slice().unwrap();
    assert!(bit_vec.is_valid(0));
    assert!(bit_vec.is_null(1));
    assert!(bit_vec.is_null(2));
    assert!(bit_vec.is_valid(3));
    assert_eq!(output[0], bf16::from_f32(-4.0));
    assert_eq!(output[3], bf16::from_f32(6.0));
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_add_different_types() {
    use compute::bit_vector::BitVector;
    use compute::data_type::DataType;
    use compute::error::ErrorMode;
    use compute::evaluate::{Device, evaluate};
    use compute::expr::Expr;
    use compute::operator::Operator;
    use half::bf16;

    let a_vec: Vec<bf16> =
        vec![bf16::from_f32(-1.0), bf16::from_f32(2.0), bf16::from_f32(3.0), bf16::from_f32(-4.0)];
    let b_vec: Vec<f32> = vec![3.0, 5.0, 2.0, 1.0];

    let mut a_bit_vec = BitVector::<u64>::new_all_valid(a_vec.len());
    let mut b_bit_vec = BitVector::<u64>::new_all_valid(b_vec.len());

    a_bit_vec.set_null(1);
    a_bit_vec.set_null(2);
    b_bit_vec.set_null(2);
    let col_a = create_column!(a_vec, Some(a_bit_vec), "a", DataType::BF16);
    let col_b = create_column!(b_vec, Some(b_bit_vec), "b", DataType::F32);

    let expr = Expr::binary(Operator::Add, Expr::col("a"), Expr::col("b"));

    let result = evaluate(Device::GPU, ErrorMode::Tachyon, &expr, &vec![col_a, col_b]).await;
    let result = result.unwrap();
    assert!(result[0].data_as_slice::<f32>().is_some());
    let output = result[0].data_as_slice::<f32>().unwrap();
    let bit_vec = result[0].null_bits_as_slice().unwrap();
    assert!(bit_vec.is_valid(0));
    assert!(bit_vec.is_null(1));
    assert!(bit_vec.is_null(2));
    assert!(bit_vec.is_valid(3));
    assert_eq!(output[0], 2.0);
    assert_eq!(output[3], -3.0);
}

#[cfg(feature = "gpu")]
#[tokio::test]
#[should_panic]
async fn test_add_i64_u64_should_error() {
    use compute::expr::{Expr, Literal, SchemaContext};
    use compute::operator::Operator;
    let schema = SchemaContext::new();
    let left = Expr::Literal(Literal::I64(1));
    let right = Expr::Literal(Literal::U64(1));
    let expr = Expr::Binary { op: Operator::Add, left: Box::new(left), right: Box::new(right) };
    expr.infer_type(&schema).unwrap();
}

#[cfg(feature = "gpu")]
#[tokio::test]
#[should_panic]
async fn test_add_u64_i64_should_error() {
    use compute::expr::{Expr, Literal, SchemaContext};
    use compute::operator::Operator;
    let schema = SchemaContext::new();
    let left = Expr::Literal(Literal::U64(1));
    let right = Expr::Literal(Literal::I64(1));
    let expr = Expr::Binary { op: Operator::Add, left: Box::new(left), right: Box::new(right) };
    expr.infer_type(&schema).unwrap();
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_cmp_different_types() {
    use compute::bit_vector::BitVector;
    use compute::data_type::DataType;
    use compute::error::ErrorMode;
    use compute::evaluate::{Device, evaluate};
    use compute::expr::Expr;
    use compute::operator::Operator;
    init_tracing();
    let a_vec: Vec<i32> = vec![1, 2, 3, 4, 5];
    let b_vec: Vec<i64> = vec![1, 5, 2, 4, 6];

    let mut a_bit_vec = BitVector::<u64>::new_all_valid(a_vec.len());
    let b_bit_vec = BitVector::<u64>::new_all_valid(b_vec.len());

    a_bit_vec.set_null(4);
    let col_a = create_column!(a_vec, Some(a_bit_vec), "a", DataType::I32);
    let col_b = create_column!(b_vec, Some(b_bit_vec), "b", DataType::I64);

    let expr = Expr::binary(Operator::Eq, Expr::col("a"), Expr::col("b"));

    let result = evaluate(Device::GPU, ErrorMode::Tachyon, &expr, &vec![col_a, col_b]).await;
    let result = result.unwrap();
    assert!(result[0].data_as_slice::<bool>().is_some());
    let output = result[0].data_as_slice::<bool>().unwrap();
    let bit_vec = result[0].null_bits_as_slice().unwrap();
    assert!(bit_vec.is_valid(0));
    assert!(bit_vec.is_null(4));
    assert_eq!(output[0], true);
    assert_eq!(output[1], false);
    assert_eq!(output[2], false);
    assert_eq!(output[3], true);
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_cmp_nan_eq() {
    let output = evaluate_cmp(Operator::Eq).await;
    assert_eq!(output, vec![false, false, true, true, false, false, false, true]); //Nan == Nan for Databases/Dataframe, different than language
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_cmp_nan_neq() {
    let output = evaluate_cmp(Operator::NotEq).await;
    assert_eq!(output, vec![true, true, false, false, true, true, true, false]); //Nan == Nan for Databases/Dataframe, different than language
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_cmp_nan_lt() {
    let output = evaluate_cmp(Operator::Lt).await;
    assert_eq!(output, vec![false, true, false, false, true, false, true, false]); //Nan < Nan for Databases/Dataframe, different than language
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_cmp_nan_lteq() {
    let output = evaluate_cmp(Operator::LtEq).await;
    assert_eq!(output, vec![false, true, true, true, true, false, true, true]); // any_number <= Nan for Databases/Dataframe, different than language
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_cmp_nan_gt() {
    let output = evaluate_cmp(Operator::Gt).await;
    assert_eq!(output, vec![true, false, false, false, false, true, false, false]); //Nan > Nan for Databases/Dataframe, different than language
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_cmp_nan_gteq() {
    let output = evaluate_cmp(Operator::GtEq).await;
    assert_eq!(output, vec![true, false, true, true, false, true, false, true]); //Nan >= Nan for Databases/Dataframe, different than language
}

async fn evaluate_cmp(op: Operator) -> Vec<bool> {
    use compute::bit_vector::BitVector;
    use compute::data_type::DataType;
    use compute::error::ErrorMode;
    use compute::evaluate::{Device, evaluate};
    use compute::expr::Expr;

    init_tracing();
    let a_vec: Vec<f32> =
        vec![f32::NAN, 2.0, 3.0, f32::NAN, 5.0, f32::NAN, f32::NEG_INFINITY, f32::INFINITY];
    let b_vec: Vec<f64> =
        vec![1.0, f64::NAN, 3.0, f64::NAN, 6.0, f64::INFINITY, f64::NAN, f64::INFINITY];

    let a_bit_vec = BitVector::<u64>::new_all_valid(a_vec.len());
    let b_bit_vec = BitVector::<u64>::new_all_valid(b_vec.len());

    let col_a = create_column!(a_vec, Some(a_bit_vec), "a", DataType::F32);
    let col_b = create_column!(b_vec, Some(b_bit_vec), "b", DataType::F64);

    let expr = Expr::binary(op, Expr::col("a"), Expr::col("b"));

    let result = evaluate(Device::GPU, ErrorMode::Tachyon, &expr, &vec![col_a, col_b]).await;
    let result = result.unwrap();
    assert!(result[0].data_as_slice::<bool>().is_some());
    let output = result[0].data_as_slice::<bool>().unwrap();

    return output.to_vec();
}
