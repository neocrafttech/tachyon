use core::data_type::DataType;
use core::expr::BinaryOp;
use core::expr::Expr;
use core::expr::SchemaContext;
use core::expr::ToNvrtc;
use core::expr::UnaryOp;
use core::expr::float_literal_to_str;

#[test]
fn infer_and_codegen_simple() {
    let schema = SchemaContext::new()
        .with_column("a", DataType::F64)
        .with_column("b", DataType::I64)
        .with_column("flag", DataType::Bool);

    // expr: (a * 2.5) + (b as double)
    let expr = Expr::binary(
        BinaryOp::Add,
        Expr::binary(BinaryOp::Mul, Expr::col("a"), Expr::f32(2.5)),
        Expr::col("b").cast(DataType::F64),
    );

    let ty = expr.infer_type(&schema).expect("type infers");
    assert_eq!(ty, DataType::F64);

    let nvrtc = expr.to_nvrtc(&schema).expect("codegen");
    // example output: ((col_a[i] * 2.5) + ((double)(col_b[i])))
    println!("nvrtc: {}", nvrtc);
}

#[test]
fn c_type_mapping() {
    assert_eq!(DataType::I8.c_type(), "int8_t");
    assert_eq!(DataType::U64.c_type(), "uint64_t");
    assert_eq!(DataType::F32.c_type(), "float");
    assert_eq!(DataType::Utf8.c_type(), "uint8_t");
}

#[test]
fn unary_neg_and_not() {
    let expr_neg = Expr::unary(UnaryOp::Neg, Expr::i32(10));
    let schema = SchemaContext::new();
    let inferred = expr_neg.infer_type(&schema).unwrap();
    assert_eq!(inferred, DataType::I32);

    let schema = SchemaContext::new().with_column("flag", DataType::Bool);
    let expr_not = Expr::unary(UnaryOp::Not, Expr::col("flag"));
    let inferred = expr_not.infer_type(&schema).unwrap();
    assert_eq!(inferred, DataType::Bool);
}

#[test]
fn type_inference_for_more_data_types() {
    let schema = SchemaContext::new()
        .with_column("i16_col", DataType::I16)
        .with_column("u32_col", DataType::U32)
        .with_column("f32_col", DataType::F32);

    let e1 = Expr::col("i16_col");
    assert_eq!(e1.infer_type(&schema).unwrap(), DataType::I16);

    let e2 = Expr::col("u32_col");
    assert_eq!(e2.infer_type(&schema).unwrap(), DataType::U32);

    let e3 = Expr::col("f32_col");
    assert_eq!(e3.infer_type(&schema).unwrap(), DataType::F32);
}

#[test]
fn float_literal_str() {
    assert_eq!(float_literal_to_str(3.0_f64), "3.0");
    assert_eq!(float_literal_to_str(2.5_f64), "2.5");
    assert_eq!(float_literal_to_str(4.0_f32), "4.0");
    assert_eq!(float_literal_to_str(7.75_f32), "7.75");
}

#[test]
fn bool_ops() {
    let schema = SchemaContext::new().with_column("flag", DataType::Bool);
    let e = Expr::binary(BinaryOp::And, Expr::col("flag"), Expr::bool_lit(true));
    let ty = e.infer_type(&schema).unwrap();
    assert_eq!(ty, DataType::Bool);
}
