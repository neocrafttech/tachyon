use core::codegen::{CodeBlock, CodeGen, float_literal_to_str};
use core::data_type::DataType;
use core::expr::{Expr, SchemaContext};
use core::operator::Operator;

use half::{bf16, f16};

macro_rules! define_type_test {
    ($test_name:ident, $col_name:expr, $data_type:expr) => {
        #[test]
        fn $test_name() {
            let schema = SchemaContext::new().with_column($col_name, $data_type);

            let expr = Expr::col($col_name);
            assert_eq!(
                expr.infer_type(&schema).unwrap(),
                $data_type,
                "Type inference failed for column '{}'",
                $col_name
            );
        }
    };
}

define_type_test!(test_type_inference_i8, "i8_col", DataType::I8);
define_type_test!(test_type_inference_i16, "i16_col", DataType::I16);
define_type_test!(test_type_inference_i32, "i32_col", DataType::I32);
define_type_test!(test_type_inference_i64, "i64_col", DataType::I64);
define_type_test!(test_type_inference_u8, "u8_col", DataType::U8);
define_type_test!(test_type_inference_u16, "u16_col", DataType::U16);
define_type_test!(test_type_inference_u32, "u32_col", DataType::U32);
define_type_test!(test_type_inference_u64, "u64_col", DataType::U64);
define_type_test!(test_type_inference_f32, "f32_col", DataType::F32);
define_type_test!(test_type_inference_f64, "f64_col", DataType::F64);
define_type_test!(test_type_inference_bool, "bool_col", DataType::BOOL);
define_type_test!(test_type_inference_str, "str_col", DataType::STR);

#[test]
fn test_type_inference_unary_neg() {
    let expr_neg = Expr::unary(Operator::Neg, Expr::i32(10));
    let schema = SchemaContext::new();
    let inferred = expr_neg.infer_type(&schema).unwrap();
    assert_eq!(inferred, DataType::I32);
}

#[test]
fn test_type_inference_unary_not() {
    let schema = SchemaContext::new().with_column("flag", DataType::BOOL);
    let expr_not = Expr::unary(Operator::Not, Expr::col("flag"));
    let inferred = expr_not.infer_type(&schema).unwrap();
    assert_eq!(inferred, DataType::BOOL);
}

fn normalize_code(code: &str) -> String {
    code.lines()
        .map(|line| line.trim())
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

#[macro_export]
macro_rules! test_codegen_literal {
    (
        $name:ident,
        rust_lit = $rust_lit:expr,
        expr_ctor = $expr_ctor:expr,
        datatype = $datatype:expr,
        expected = $expected:expr
    ) => {
        #[test]
        fn $name() {
            let schema = SchemaContext::new();
            let expr = $expr_ctor($rust_lit);

            // Type inference
            let ty = expr.infer_type(&schema).expect("type infers");
            assert_eq!(ty, $datatype);

            // Codegen
            let mut code_block = CodeBlock::default();
            expr.to_nvrtc(&schema, &mut code_block).expect("codegen");

            println!("Generated Code:\n{}", code_block.code());

            assert_eq!(normalize_code(code_block.code()), normalize_code($expected));
        }
    };
}
test_codegen_literal!(
    test_codegen_literal_i8,
    rust_lit = 10,
    expr_ctor = Expr::i8,
    datatype = DataType::I8,
    expected = r#"bool valid0 = true;
INT8 value0;
value0.value = (int8_t)10;"#
);

test_codegen_literal!(
    test_codegen_literal_i16,
    rust_lit = 1000,
    expr_ctor = Expr::i16,
    datatype = DataType::I16,
    expected = r#"bool valid0 = true;
INT16 value0;
value0.value = (int16_t)1000;"#
);

test_codegen_literal!(
    test_codegen_literal_i32,
    rust_lit = -123,
    expr_ctor = Expr::i32,
    datatype = DataType::I32,
    expected = r#"bool valid0 = true;
INT32 value0;
value0.value = (int32_t)-123;"#
);

test_codegen_literal!(
    test_codegen_literal_i64,
    rust_lit = 12334444,
    expr_ctor = Expr::i64,
    datatype = DataType::I64,
    expected = r#"bool valid0 = true;
INT64 value0;
value0.value = (int64_t)12334444ll;"#
);

test_codegen_literal!(
    test_codegen_literal_u8,
    rust_lit = 10,
    expr_ctor = Expr::u8,
    datatype = DataType::U8,
    expected = r#"bool valid0 = true;
UINT8 value0;
value0.value = (uint8_t)10;"#
);

test_codegen_literal!(
    test_codegen_literal_u16,
    rust_lit = 1000,
    expr_ctor = Expr::u16,
    datatype = DataType::U16,
    expected = r#"bool valid0 = true;
UINT16 value0;
value0.value = (uint16_t)1000;"#
);

test_codegen_literal!(
    test_codegen_literal_u32,
    rust_lit = 5667777,
    expr_ctor = Expr::u32,
    datatype = DataType::U32,
    expected = r#"bool valid0 = true;
UINT32 value0;
value0.value = (uint32_t)5667777u;"#
);

test_codegen_literal!(
    test_codegen_literal_u64,
    rust_lit = 100_000_000,
    expr_ctor = Expr::u64,
    datatype = DataType::U64,
    expected = r#"bool valid0 = true;
UINT64 value0;
value0.value = (uint64_t)100000000ull;"#
);

test_codegen_literal!(
    test_codegen_literal_bf16,
    rust_lit = bf16::from_f32(2.0),
    expr_ctor = Expr::bf16,
    datatype = DataType::BF16,
    expected = r#"bool valid0 = true;
BFLOAT16 value0;
value0.value = (bfloat16)(__float2bfloat16(2.0f));"#
);

test_codegen_literal!(
    test_codegen_literal_f16,
    rust_lit = f16::from_f32(1.5),
    expr_ctor = Expr::f16,
    datatype = DataType::F16,
    expected = r#"bool valid0 = true;
FLOAT16 value0;
value0.value = (float16)(__float2half(1.5f));"#
);

test_codegen_literal!(
    test_codegen_literal_f32,
    rust_lit = 1.5f32,
    expr_ctor = Expr::f32,
    datatype = DataType::F32,
    expected = r#"bool valid0 = true;
FLOAT32 value0;
value0.value = (float)1.5f;"#
);

test_codegen_literal!(
    test_codegen_literal_f64,
    rust_lit = 1.5e26,
    expr_ctor = Expr::f64,
    datatype = DataType::F64,
    expected = r#"bool valid0 = true;
FLOAT64 value0;
value0.value = (double)150000000000000000000000000.0;"#
);

test_codegen_literal!(
    test_codegen_literal_bool,
    rust_lit = false,
    expr_ctor = Expr::bool_lit,
    datatype = DataType::BOOL,
    expected = r#"bool valid0 = true;
BOOL value0;
value0.value = (bool)false;"#
);

#[test]
fn test_codegen_unary() {
    let schema = SchemaContext::new().with_column("a", DataType::F64);
    let expr = Expr::unary(Operator::Neg, Expr::col("a"));

    let ty = expr.infer_type(&schema).expect("type infers");
    assert_eq!(ty, DataType::F64);

    let mut code_block = CodeBlock::default();
    let _ = expr.to_nvrtc(&schema, &mut code_block).expect("codegen");
    println!("Code:");
    println!("{}", code_block.code());
    let expected = r#"FLOAT64 value0;
    bool valid0 = columns[0].load(row_idx, &value0);
    bool valid1 = valid0;
    if (valid1) {
	value1.value = (double)((-(value0)).value)
    }"#;
    assert_eq!(normalize_code(code_block.code()), normalize_code(expected))
}

#[test]
fn test_codegen_binary_same_type_cast() {
    let schema = SchemaContext::new()
        .with_column("a", DataType::F64)
        .with_column("b", DataType::F64)
        .with_column("flag", DataType::BOOL);

    let expr = Expr::binary(
        Operator::Add,
        Expr::binary(Operator::Mul, Expr::col("a"), Expr::f32(2.5)),
        Expr::col("b").cast(DataType::F64),
    );

    let ty = expr.infer_type(&schema).expect("type infers");
    assert_eq!(ty, DataType::F64);

    let mut code_block = CodeBlock::default();
    let _ = expr.to_nvrtc(&schema, &mut code_block).expect("codegen");
    println!("Code:");
    println!("{}", code_block.code());
    let expected = r#"FLOAT64 value0;
    bool valid0 = columns[0].load(row_idx, &value0);
    bool valid1 = true;
    FLOAT32 value1;
    value1.value = (float)2.5f;
    bool valid2 = valid0 & valid1;
    FLOAT64 value2;
    if (valid2) {
	value2.value = (double)(value0.value * value1.value)
    }
    FLOAT64 value3;
    bool valid3 = columns[1].load(row_idx, &value3);
    bool valid4 = valid2 & valid3;
    FLOAT64 value4;
    if (valid4) {
	value4.value = (double)(value2.value + value3.value)
    }"#;
    assert_eq!(normalize_code(code_block.code()), normalize_code(expected))
}

#[test]
fn test_codegen_binary_different_type_cast() {
    let schema = SchemaContext::new()
        .with_column("a", DataType::F64)
        .with_column("b", DataType::I64)
        .with_column("flag", DataType::BOOL);

    let expr = Expr::binary(
        Operator::Add,
        Expr::binary(Operator::Mul, Expr::col("a"), Expr::f32(2.5)),
        Expr::col("b").cast(DataType::F32),
    );

    let ty = expr.infer_type(&schema).expect("type infers");
    assert_eq!(ty, DataType::F64);

    let mut code_block = CodeBlock::default();
    let _ = expr.to_nvrtc(&schema, &mut code_block).expect("codegen");
    println!("Code:");
    println!("{}", code_block.code());
    let expected = r#"FLOAT64 value0;
    bool valid0 = columns[0].load(row_idx, &value0);
    bool valid1 = true;
    FLOAT32 value1;
    value1.value = (float)2.5f;
    bool valid2 = valid0 & valid1;
    FLOAT64 value2;
    if (valid2) {
	value2.value = (double)(value0.value * value1.value)
    }
    INT64 value3;
    bool valid3 = columns[1].load(row_idx, &value3);
    bool valid4 = valid3;
    FLOAT32 value4;
    if (valid4) {
	value4.value = (float)(value3.value)
    }
    bool valid5 = valid2 & valid4;
    FLOAT64 value5;
    if (valid5) {
	value5.value = (double)(value2.value + value4.value)
    }"#;
    assert_eq!(normalize_code(code_block.code()), normalize_code(expected))
}

#[test]
fn test_float_literal_str() {
    assert_eq!(float_literal_to_str(3.0_f64), "3.0");
    assert_eq!(float_literal_to_str(2.5_f64), "2.5");
    assert_eq!(float_literal_to_str(4.0_f32), "4.0");
    assert_eq!(float_literal_to_str(7.75_f32), "7.75");
}

#[test]
fn test_bool_ops() {
    let schema = SchemaContext::new().with_column("flag", DataType::BOOL);
    let e = Expr::binary(Operator::And, Expr::col("flag"), Expr::bool_lit(true));
    let ty = e.infer_type(&schema).unwrap();
    assert_eq!(ty, DataType::BOOL);
}
