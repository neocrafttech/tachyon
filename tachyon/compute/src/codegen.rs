/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use tracing::debug;

use crate::bit_vector::BitBlock;
use crate::data_type::DataType;
use crate::error::ErrorMode;
use crate::expr::{Expr, Literal, SchemaContext, TypeError};
use crate::operator::Operator;

#[derive(Debug, Default)]
/// Mutable code-generation buffer for building kernel snippets.
pub(crate) struct CodeBlock {
    code: String,
    var_counter: u16,
    column_var_map: HashMap<String, String>,
}

impl CodeBlock {
    fn add_code(&mut self, code: &str) -> &mut Self {
        self.code.push_str(code);
        self
    }

    fn add_variable_decl(&mut self, ty: &str, var: &str) -> &mut Self {
        self.add_code(&format!("\t{} {};\n", ty, var))
    }

    fn add_validity_check(&mut self, var: &str, operands: &[&str]) -> &mut Self {
        assert!(!operands.is_empty(), "add_validity_check requires at least one operand");

        let expression = match operands.len() {
            1 => operands[0].to_string(),
            _ => operands.join(" & "),
        };

        self.add_code(&format!("\t{}.valid = {};\n", var, expression))
    }

    fn add_conditional<F>(&mut self, condition: &str, body: F) -> &mut Self
    where
        F: FnOnce(&mut Self),
    {
        self.add_code(&format!("\tif ({}) {{\n", condition));
        body(self);
        self.add_code("\t}\n")
    }

    /// Emits code to load a source column value into a temporary variable.
    pub(crate) fn add_load_column<'a, B: BitBlock>(
        &'a mut self, col_name: &str, col_idx: u16, col_type: &DataType,
    ) -> &'a str {
        if !self.column_var_map.contains_key(col_name) {
            let var = self.next_var();
            let kernel_type = col_type.kernel_type();
            let bits_type = B::C_TYPE;
            let code = format!(
                "\t{kernel_type} {var} = input[{col_idx}].load<TypeKind::{kernel_type}, {bits_type}>(row_idx);\n"
            );
            self.add_code(&code);
            self.column_var_map.insert(col_name.to_string(), var);
        }

        self.column_var_map.get(col_name).unwrap()
    }

    /// Emits code to store a temporary value into an output column.
    pub(crate) fn add_store_column<B: BitBlock>(
        &mut self, col_idx: u16, col_type: &DataType, var: &str,
    ) {
        let kernel_type = col_type.kernel_type();
        let bits_type = B::C_TYPE;
        let code = format!(
            "\toutput[{col_idx}].store<TypeKind::{kernel_type}, {bits_type}>(row_idx, {var});\n"
        );
        self.add_code(&code);
    }

    pub(crate) fn next_var(&mut self) -> String {
        let var = format!("var{}", self.var_counter);
        self.var_counter += 1;
        var
    }

    /// Returns the generated code buffer.
    pub(crate) fn code(&self) -> &str {
        &self.code
    }
}

/// Trait for converting expression nodes to NVRTC kernel code.
pub(crate) trait CodeGen {
    /// Appends expression evaluation statements into `code_block`.
    fn to_nvrtc<B: BitBlock>(
        &self, schema: &SchemaContext, code_block: &mut CodeBlock,
    ) -> Result<(), TypeError>;
    /// Builds an expression value and returns the temporary variable name.
    fn build_nvrtc_code<B: BitBlock>(
        &self, schema: &SchemaContext, code_block: &mut CodeBlock,
    ) -> Result<String, TypeError>;
}

impl CodeGen for Expr {
    fn to_nvrtc<B: BitBlock>(
        &self, schema: &SchemaContext, code_block: &mut CodeBlock,
    ) -> Result<(), TypeError> {
        let expr = self.simplify(schema)?;
        let result_type = expr.infer_type(schema)?;
        let res = expr.build_nvrtc_code::<B>(schema, code_block)?;
        code_block.add_store_column::<B>(0, &result_type, &res);
        Ok(())
    }

    fn build_nvrtc_code<B: BitBlock>(
        &self, schema: &SchemaContext, code_block: &mut CodeBlock,
    ) -> Result<String, TypeError> {
        let result_type = self.infer_type(schema)?;
        debug!("Result Type: {:?}", result_type);
        let error_mode = schema.error_mode() == ErrorMode::Ansi;

        let var = match self {
            Expr::Column(col_name) => {
                let (col_idx, col_type) = match schema.lookup(col_name) {
                    Some(pair) => pair,
                    None => Err(TypeError::Unsupported(col_name.to_string()))?,
                };
                let var = code_block.add_load_column::<B>(col_name, *col_idx, col_type);
                var.to_string()
            }
            Expr::Literal(l) => {
                let value = match l {
                    Literal::I8(i) => format!("{}", i),
                    Literal::I16(i) => format!("{}", i),
                    Literal::I32(i) => format!("{}", i),
                    Literal::I64(i) => format!("{}ll", i),
                    Literal::U8(i) => format!("{}", i),
                    Literal::U16(i) => format!("{}", i),
                    Literal::U32(i) => format!("{}u", i),
                    Literal::U64(i) => format!("{}ull", i),
                    Literal::BF16(f) => {
                        format!("(__float2bfloat16({}f))", float_literal_to_str(*f))
                    }
                    Literal::F16(f) => format!("(__float2half({}f))", float_literal_to_str(*f)),
                    Literal::F32(f) => format!("{}f", float_literal_to_str(*f)),
                    Literal::F64(f) => float_literal_to_str(*f).to_string(),
                    Literal::Bool(b) => (if *b { "true" } else { "false" }).to_string(),
                    Literal::Str(s) => format!("\"{}\"", escape_c_string(s)),
                };
                let var = code_block.next_var();
                let ty_c = result_type.c_type();
                code_block
                    .add_variable_decl(result_type.kernel_type(), &var)
                    .add_validity_check(&var, &["true"])
                    .add_code(&format!("\t{var}.value = ({ty_c}){value};\n"));
                var
            }
            Expr::Unary { op, expr } => {
                let e_var = expr.build_nvrtc_code::<B>(schema, code_block)?;
                let value = match op {
                    Operator::Neg => format!("(-({}.value))", e_var),
                    Operator::Not => format!("(!({}.value))", e_var),
                    _ => Err(TypeError::Unsupported(format!("Not supported unary op {}", op)))?,
                };
                let var = code_block.next_var();
                code_block
                    .add_validity_check(&var, &[&format!("{}.valid", e_var)])
                    .add_conditional(&format!("{}.valid", var), |block| {
                        block.add_code(&format!(
                            "\t{}.value = ({})({}.value);\n",
                            var,
                            result_type.c_type(),
                            value
                        ));
                    });
                var
            }
            Expr::Binary { op, left, right } => {
                let l_var = left.build_nvrtc_code::<B>(schema, code_block)?;
                let r_var = right.build_nvrtc_code::<B>(schema, code_block)?;
                if op.is_binary() {
                    let var = code_block.next_var();
                    let kernel_fn = op_kernel_fn(*op);
                    code_block.add_code(&format!(
                        "\t{} {} = {}<{}>(ctx, {}, {});\n",
                        result_type.kernel_type(),
                        var,
                        kernel_fn,
                        error_mode,
                        l_var,
                        r_var
                    ));

                    var
                } else {
                    Err(TypeError::Unsupported(format!("Not supported binary op {}", op)))?
                }
            }
            Expr::Nary { op: _, args: _ } => unimplemented!(),
            Expr::Call { name, args } => {
                let mut arg_strs = Vec::with_capacity(args.len());
                for a in args {
                    arg_strs.push(a.build_nvrtc_code::<B>(schema, code_block)?);
                }
                let var = code_block.next_var();
                code_block.add_code(&format!("{}({})", name, arg_strs.join(", ")));
                var
            }
            Expr::Cast { expr, to } => {
                let e_var = expr.build_nvrtc_code::<B>(schema, code_block)?;
                let from = expr.infer_type(schema)?;
                if *to == from {
                    return Ok(e_var);
                }
                let var = code_block.next_var();
                let cast_fn = match (from, to) {
                    //(DataType::I8, DataType::F16) => "__ushort2half_rn",
                    (DataType::I16, DataType::F16) => "__short2half_rn",
                    (DataType::I32, DataType::F16) => "__int2half_rn",
                    (DataType::I64, DataType::F16) => "__ll2half_rn",
                    //(DataType::U8, DataType::F16) => "__ushort2half_rn",
                    (DataType::U16, DataType::F16) => "__ushort2half_rn",
                    (DataType::U32, DataType::F16) => "__uint2half_rn",
                    (DataType::U64, DataType::F16) => "__ull2half_rn",
                    _ => &format!("({})", to.c_type()),
                };
                code_block
                    .add_variable_decl(result_type.kernel_type(), &var)
                    .add_validity_check(&var, &[&format!("{}.valid", e_var)])
                    .add_conditional(&format!("{}.valid", var), |block| {
                        block.add_code(&format!(
                            "\t{}.value = {}({}.value);\n",
                            var, cast_fn, e_var,
                        ));
                    });
                var
            }
            Expr::Aggregate { op, .. } => {
                return Err(TypeError::Unsupported(format!(
                    "Aggregate {:?} is not supported in codegen yet",
                    op
                )));
            }
        };

        Ok(var)
    }
}

fn op_kernel_fn(op: Operator) -> String {
    let kernel_fn = match op {
        Operator::Add => "math::add",
        Operator::Sub => "math::sub",
        Operator::Mul => "math::mul",
        Operator::Div => "math::div",
        Operator::Eq => "math::eq",
        Operator::NotEq => "math::neq",
        Operator::Lt => "math::lt",
        Operator::LtEq => "math::lteq",
        Operator::Gt => "math::gt",
        Operator::GtEq => "math::gteq",
        Operator::And => "math::bit_and",
        Operator::Or => "math::bit_or",
        _ => unimplemented!("Unsupported operator: {:?}", op),
    };
    kernel_fn.to_string()
}

fn escape_c_string(s: &str) -> String {
    s.replace('"', "\\\"")
}
/// Formats floating-point literals for CUDA code generation.
pub(crate) fn float_literal_to_str<T: Into<f64> + Copy + PartialEq>(f: T) -> String {
    let f64_val = f.into();
    if f64_val.fract() == 0.0 { format!("{}.0", f64_val) } else { format!("{}", f64_val) }
}

#[cfg(test)]
mod tests {
    use half::{bf16, f16};

    use crate::codegen::{CodeBlock, CodeGen, float_literal_to_str};
    use crate::data_type::DataType;
    use crate::expr::{Expr, SchemaContext};
    use crate::operator::Operator;

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
    define_type_test!(test_type_inference_bool, "bool_col", DataType::Bool);
    define_type_test!(test_type_inference_str, "str_col", DataType::Str);

    #[test]
    fn test_type_inference_unary_neg() {
        let expr_neg = Expr::unary(Operator::Neg, Expr::i32(10));
        let schema = SchemaContext::new();
        let inferred = expr_neg.infer_type(&schema).unwrap();
        assert_eq!(inferred, DataType::I32);
    }

    #[test]
    fn test_type_inference_unary_not() {
        let schema = SchemaContext::new().with_column("flag", DataType::Bool);
        let expr_not = Expr::unary(Operator::Not, Expr::col("flag"));
        let inferred = expr_not.infer_type(&schema).unwrap();
        assert_eq!(inferred, DataType::Bool);
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

                let ty = expr.infer_type(&schema).expect("type infers");
                assert_eq!(ty, $datatype);

                let mut code_block = CodeBlock::default();
                expr.to_nvrtc::<u64>(&schema, &mut code_block).expect("codegen");

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
        expected = r#"Int8 var0;
            	var0.valid = true;
            	var0.value = (int8_t)10;
            	output[0].store<TypeKind::Int8, uint64_t>(row_idx, var0);"#
    );

    test_codegen_literal!(
        test_codegen_literal_i16,
        rust_lit = 1000,
        expr_ctor = Expr::i16,
        datatype = DataType::I16,
        expected = r#"Int16 var0;
            	var0.valid = true;
            	var0.value = (int16_t)1000;
            	output[0].store<TypeKind::Int16, uint64_t>(row_idx, var0);"#
    );

    test_codegen_literal!(
        test_codegen_literal_i32,
        rust_lit = -123,
        expr_ctor = Expr::i32,
        datatype = DataType::I32,
        expected = r#"Int32 var0;
            	var0.valid = true;
            	var0.value = (int32_t)-123;
            	output[0].store<TypeKind::Int32, uint64_t>(row_idx, var0);"#
    );

    test_codegen_literal!(
        test_codegen_literal_i64,
        rust_lit = 12334444,
        expr_ctor = Expr::i64,
        datatype = DataType::I64,
        expected = r#"Int64 var0;
            	var0.valid = true;
            	var0.value = (int64_t)12334444ll;
            	output[0].store<TypeKind::Int64, uint64_t>(row_idx, var0);"#
    );

    test_codegen_literal!(
        test_codegen_literal_u8,
        rust_lit = 10,
        expr_ctor = Expr::u8,
        datatype = DataType::U8,
        expected = r#"	UInt8 var0;
           	var0.valid = true;
           	var0.value = (uint8_t)10;
           	output[0].store<TypeKind::UInt8, uint64_t>(row_idx, var0);"#
    );

    test_codegen_literal!(
        test_codegen_literal_u16,
        rust_lit = 1000,
        expr_ctor = Expr::u16,
        datatype = DataType::U16,
        expected = r#"UInt16 var0;
            	var0.valid = true;
            	var0.value = (uint16_t)1000;
            	output[0].store<TypeKind::UInt16, uint64_t>(row_idx, var0);"#
    );

    test_codegen_literal!(
        test_codegen_literal_u32,
        rust_lit = 5667777,
        expr_ctor = Expr::u32,
        datatype = DataType::U32,
        expected = r#"UInt32 var0;
            	var0.valid = true;
            	var0.value = (uint32_t)5667777u;
            	output[0].store<TypeKind::UInt32, uint64_t>(row_idx, var0);"#
    );

    test_codegen_literal!(
        test_codegen_literal_u64,
        rust_lit = 100_000_000,
        expr_ctor = Expr::u64,
        datatype = DataType::U64,
        expected = r#"UInt64 var0;
            	var0.valid = true;
            	var0.value = (uint64_t)100000000ull;
            	output[0].store<TypeKind::UInt64, uint64_t>(row_idx, var0);"#
    );

    test_codegen_literal!(
        test_codegen_literal_bf16,
        rust_lit = bf16::from_f32(2.0),
        expr_ctor = Expr::bf16,
        datatype = DataType::BF16,
        expected = r#"BFloat16 var0;
           	var0.valid = true;
           	var0.value = (bfloat16)(__float2bfloat16(2.0f));
           	output[0].store<TypeKind::BFloat16, uint64_t>(row_idx, var0);"#
    );

    test_codegen_literal!(
        test_codegen_literal_f16,
        rust_lit = f16::from_f32(1.5),
        expr_ctor = Expr::f16,
        datatype = DataType::F16,
        expected = r#"Float16 var0;
            	var0.valid = true;
            	var0.value = (float16)(__float2half(1.5f));
            	output[0].store<TypeKind::Float16, uint64_t>(row_idx, var0);"#
    );

    test_codegen_literal!(
        test_codegen_literal_f32,
        rust_lit = 1.5f32,
        expr_ctor = Expr::f32,
        datatype = DataType::F32,
        expected = r#"Float32 var0;
            	var0.valid = true;
            	var0.value = (float)1.5f;
            	output[0].store<TypeKind::Float32, uint64_t>(row_idx, var0);"#
    );

    test_codegen_literal!(
        test_codegen_literal_f64,
        rust_lit = 1.5e26,
        expr_ctor = Expr::f64,
        datatype = DataType::F64,
        expected = r#"Float64 var0;
            	var0.valid = true;
            	var0.value = (double)150000000000000000000000000.0;
            	output[0].store<TypeKind::Float64, uint64_t>(row_idx, var0);"#
    );

    test_codegen_literal!(
        test_codegen_literal_bool,
        rust_lit = false,
        expr_ctor = Expr::bool_lit,
        datatype = DataType::Bool,
        expected = r#"Bool var0;
            	var0.valid = true;
            	var0.value = (bool)false;
            	output[0].store<TypeKind::Bool, uint64_t>(row_idx, var0);"#
    );

    #[test]
    fn test_codegen_unary() {
        let schema = SchemaContext::new().with_column("a", DataType::F64);
        let expr = Expr::unary(Operator::Neg, Expr::col("a"));

        let ty = expr.infer_type(&schema).expect("type infers");
        assert_eq!(ty, DataType::F64);

        let mut code_block = CodeBlock::default();
        let _ = expr.to_nvrtc::<u64>(&schema, &mut code_block).expect("codegen");
        println!("Code:");
        println!("{}", code_block.code());
        let expected = r#"Float64 var0 = input[0].load<TypeKind::Float64, uint64_t>(row_idx);
            	var1.valid = var0.valid;
            	if (var1.valid) {
            	var1.value = (double)((-(var0.value)).value);
            	}
            	output[0].store<TypeKind::Float64, uint64_t>(row_idx, var1);"#;
        assert_eq!(normalize_code(code_block.code()), normalize_code(expected))
    }

    #[test]
    fn test_codegen_binary_same_type_cast() {
        let schema = SchemaContext::new()
            .with_column("a", DataType::F64)
            .with_column("b", DataType::F64)
            .with_column("flag", DataType::Bool);

        let expr = Expr::binary(
            Operator::Add,
            Expr::binary(Operator::Mul, Expr::col("a"), Expr::f32(2.5)),
            Expr::col("b").cast(DataType::F64),
        );

        let ty = expr.infer_type(&schema).expect("type infers");
        assert_eq!(ty, DataType::F64);

        let mut code_block = CodeBlock::default();
        let _ = expr.to_nvrtc::<u64>(&schema, &mut code_block).expect("codegen");
        println!("Code:");
        println!("{}", code_block.code());
        let expected = r#"	Float64 var0 = input[0].load<TypeKind::Float64, uint64_t>(row_idx);
    	Float32 var1;
    	var1.valid = true;
    	var1.value = (float)2.5f;
    	Float64 var2;
    	var2.valid = var1.valid;
    	if (var2.valid) {
    	var2.value = (double)(var1.value);
    	}
    	Float64 var3 = math::mul<false>(ctx, var0, var2);
    	Float64 var4 = input[1].load<TypeKind::Float64, uint64_t>(row_idx);
    	Float64 var5 = math::add<false>(ctx, var3, var4);
    	output[0].store<TypeKind::Float64, uint64_t>(row_idx, var5);"#;
        assert_eq!(normalize_code(code_block.code()), normalize_code(expected))
    }

    #[test]
    fn test_codegen_binary_different_type_cast() {
        let schema = SchemaContext::new()
            .with_column("a", DataType::F64)
            .with_column("b", DataType::I64)
            .with_column("flag", DataType::Bool);

        let expr = Expr::binary(
            Operator::Add,
            Expr::binary(Operator::Mul, Expr::col("a"), Expr::f32(2.5)),
            Expr::col("b").cast(DataType::F32),
        );

        let ty = expr.infer_type(&schema).expect("type infers");
        assert_eq!(ty, DataType::F64);

        let mut code_block = CodeBlock::default();
        let _ = expr.to_nvrtc::<u64>(&schema, &mut code_block).expect("codegen");
        println!("Code:");
        println!("{}", code_block.code());
        let expected = r#" Float64 var0 = input[0].load<TypeKind::Float64, uint64_t>(row_idx);
    	Float32 var1;
    	var1.valid = true;
    	var1.value = (float)2.5f;
    	Float64 var2;
    	var2.valid = var1.valid;
    	if (var2.valid) {
    	var2.value = (double)(var1.value);
    	}
    	Float64 var3 = math::mul<false>(ctx, var0, var2);
    	Int64 var4 = input[1].load<TypeKind::Int64, uint64_t>(row_idx);
    	Float32 var5;
    	var5.valid = var4.valid;
    	if (var5.valid) {
    	var5.value = (float)(var4.value);
    	}
    	Float64 var6;
    	var6.valid = var5.valid;
    	if (var6.valid) {
    	var6.value = (double)(var5.value);
    	}
    	Float64 var7 = math::add<false>(ctx, var3, var6);
    	output[0].store<TypeKind::Float64, uint64_t>(row_idx, var7);"#;
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
        let schema = SchemaContext::new().with_column("flag", DataType::Bool);
        let e = Expr::binary(Operator::And, Expr::col("flag"), Expr::bool_lit(true));
        let ty = e.infer_type(&schema).unwrap();
        assert_eq!(ty, DataType::Bool);
    }
}
