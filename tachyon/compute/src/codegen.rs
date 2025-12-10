/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use crate::data_type::DataType;
use crate::error::ErrorMode;
use crate::expr::{Expr, Literal, SchemaContext, TypeError};
use crate::operator::Operator;

#[derive(Debug, Default)]
pub struct CodeBlock {
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

    pub fn add_load_column<'a>(
        &'a mut self, col_name: &str, col_idx: u16, col_type: &DataType,
    ) -> &'a str {
        if !self.column_var_map.contains_key(col_name) {
            let var = self.next_var();
            let kernel_type = col_type.kernel_type();
            let code = format!(
                "\t{kernel_type} {var} = input[{col_idx}].load<TypeKind::{kernel_type}>(row_idx);\n"
            );
            self.add_code(&code);
            self.column_var_map.insert(col_name.to_string(), var);
        }

        self.column_var_map.get(col_name).unwrap()
    }

    pub fn add_store_column(&mut self, col_idx: u16, col_type: &DataType, var: &str) {
        let kernel_type = col_type.kernel_type();
        let code = format!("\toutput[{col_idx}].store<TypeKind::{kernel_type}>(row_idx, {var});\n");
        self.add_code(&code);
    }

    pub(crate) fn next_var(&mut self) -> String {
        let var = format!("var{}", self.var_counter);
        self.var_counter += 1;
        var
    }

    pub fn code(&self) -> &str {
        &self.code
    }
}

pub trait CodeGen {
    fn to_nvrtc(&self, schema: &SchemaContext, code_block: &mut CodeBlock)
    -> Result<(), TypeError>;
    fn build_nvrtc_code(
        &self, schema: &SchemaContext, code_block: &mut CodeBlock,
    ) -> Result<String, TypeError>;
}

impl CodeGen for Expr {
    fn to_nvrtc(
        &self, schema: &SchemaContext, code_block: &mut CodeBlock,
    ) -> Result<(), TypeError> {
        let result_type = self.infer_type(schema)?;
        let res = self.build_nvrtc_code(schema, code_block)?;
        code_block.add_store_column(0, &result_type, &res);
        Ok(())
    }

    fn build_nvrtc_code(
        &self, schema: &SchemaContext, code_block: &mut CodeBlock,
    ) -> Result<String, TypeError> {
        let result_type = self.infer_type(schema)?;
        let error_mode = schema.error_mode() == ErrorMode::Ansi;

        let var = match self {
            Expr::Column(col_name) => {
                let (col_idx, col_type) = match schema.lookup(col_name) {
                    Some(pair) => pair,
                    None => Err(TypeError::Unsupported(col_name.to_string()))?,
                };
                let var = code_block.add_load_column(col_name, *col_idx, col_type);
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
                let e_var = expr.build_nvrtc_code(schema, code_block)?;
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
                let l_var = left.build_nvrtc_code(schema, code_block)?;
                let r_var = right.build_nvrtc_code(schema, code_block)?;
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
                    arg_strs.push(a.build_nvrtc_code(schema, code_block)?);
                }
                let var = code_block.next_var();
                code_block.add_code(&format!("{}({})", name, arg_strs.join(", ")));
                var
            }
            Expr::Cast { expr, to } => {
                let e_var = expr.build_nvrtc_code(schema, code_block)?;
                if to.kernel_type() == expr.infer_type(schema)?.kernel_type() {
                    return Ok(e_var);
                }
                let var = code_block.next_var();
                code_block
                    .add_variable_decl(result_type.kernel_type(), &var)
                    .add_validity_check(&var, &[&format!("{}.valid", e_var)])
                    .add_conditional(&var, |block| {
                        block.add_code(&format!(
                            "\t{}.value = ({})({}.value)\n",
                            var,
                            to.c_type(),
                            e_var,
                        ));
                    });
                var
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
pub fn float_literal_to_str<T: Into<f64> + Copy + PartialEq>(f: T) -> String {
    let f64_val = f.into();
    if f64_val.fract() == 0.0 { format!("{}.0", f64_val) } else { format!("{}", f64_val) }
}
