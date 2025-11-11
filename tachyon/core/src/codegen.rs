/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;

use crate::data_type::DataType;
use crate::expr::{Expr, Literal, SchemaContext, TypeError};
use crate::operator::Operator;

#[derive(Debug, Default)]
pub struct CodeBlock {
    code: String,
    var_counter: u16,
    column_var_map: HashMap<String, (String, String)>,
}

impl CodeBlock {
    fn add_code(&mut self, code: &str) -> &mut Self {
        self.code.push_str(code);
        self
    }

    fn add_variable_decl(&mut self, ty: &str, var: &str) -> &mut Self {
        self.add_code(&format!("{} {};\n", ty, var))
    }

    fn add_validity_check(&mut self, valid: &str, operands: &[&str]) -> &mut Self {
        assert!(!operands.is_empty(), "add_validity_check requires at least one operand");

        let expression = match operands.len() {
            1 => operands[0].to_string(),
            _ => operands.join(" & "),
        };

        self.add_code(&format!("bool {} = {};\n", valid, expression))
    }

    fn add_conditional<F>(&mut self, condition: &str, body: F) -> &mut Self
    where
        F: FnOnce(&mut Self),
    {
        self.add_code(&format!("if ({}) {{\n", condition));
        body(self);
        self.add_code("}\n")
    }

    pub fn add_column<'a>(
        &'a mut self, col_name: &str, col_idx: u16, col_type: &DataType,
    ) -> (&'a str, &'a str) {
        if !self.column_var_map.contains_key(col_name) {
            let (valid, var) = self.next_var_pair();
            let kernel_type = col_type.kernel_type();
            let code = format!(
                "{kernel_type} {var};\nbool {valid} = columns[{col_idx}].load(row_idx, &{var});\n"
            );
            self.add_code(&code);
            self.column_var_map.insert(col_name.to_string(), (valid, var));
        }

        let var_pair = self.column_var_map.get(col_name).unwrap();
        (&var_pair.0, &var_pair.1)
    }

    pub(crate) fn next_var_pair(&mut self) -> (String, String) {
        let pair = (format!("valid{}", self.var_counter), format!("value{}", self.var_counter));
        self.var_counter += 1;
        pair
    }

    pub fn code(&self) -> &str {
        &self.code
    }
}

pub trait CodeGen {
    fn to_nvrtc(
        &self, schema: &SchemaContext, code_block: &mut CodeBlock,
    ) -> Result<(String, String), TypeError>;
}

impl CodeGen for Expr {
    fn to_nvrtc(
        &self, schema: &SchemaContext, code_block: &mut CodeBlock,
    ) -> Result<(String, String), TypeError> {
        let result_type = self.infer_type(schema)?;

        let res = match self {
            Expr::Column(col_name) => {
                let (col_idx, col_type) = match schema.lookup(col_name) {
                    Some(pair) => pair,
                    None => Err(TypeError::Unsupported(col_name.to_string()))?,
                };
                let (valid, var) = code_block.add_column(col_name, *col_idx, col_type);
                (valid.to_string(), var.to_string())
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
                let (valid, var) = code_block.next_var_pair();
                let ty_c = result_type.c_type();
                code_block
                    .add_validity_check(&valid, &["true"])
                    .add_variable_decl(result_type.kernel_type(), &var)
                    .add_code(&format!("{var}.value = ({ty_c}){value};\n"));
                (valid, var)
            }
            Expr::Unary { op, expr } => {
                let (e_valid, e_var) = expr.to_nvrtc(schema, code_block)?;
                let value = match op {
                    Operator::Neg => format!("(-({}))", e_var),
                    Operator::Not => format!("(!({}))", e_var),
                    _ => Err(TypeError::Unsupported(format!("Not supported unary op {}", op)))?,
                };
                let (valid, var) = code_block.next_var_pair();
                code_block.add_validity_check(&valid, &[&e_valid]).add_conditional(
                    &valid,
                    |block| {
                        block.add_code(&format!(
                            "\t{}.value = ({})({}.value)\n",
                            var,
                            result_type.c_type(),
                            value
                        ));
                    },
                );
                (valid, var)
            }
            Expr::Binary { op, left, right } => {
                let (l_valid, l_var) = left.to_nvrtc(schema, code_block)?;
                let (r_valid, r_var) = right.to_nvrtc(schema, code_block)?;
                if op.is_binary() {
                    let (valid, var) = code_block.next_var_pair();
                    code_block
                        .add_validity_check(&valid, &[&l_valid, &r_valid])
                        .add_variable_decl(result_type.kernel_type(), &var)
                        .add_conditional(&valid, |block| {
                            block.add_code(&format!(
                                "\t{}.value = ({})({}.value {} {}.value)\n",
                                var,
                                result_type.c_type(),
                                l_var,
                                op,
                                r_var
                            ));
                        });

                    (valid, var)
                } else {
                    Err(TypeError::Unsupported(format!("Not supported binary op {}", op)))?
                }
            }
            Expr::Nary { op: _, args: _ } => unimplemented!(),
            Expr::Call { name, args } => {
                let mut arg_strs = Vec::with_capacity(args.len());
                for a in args {
                    arg_strs.push(a.to_nvrtc(schema, code_block)?.1);
                }
                let (valid, var) = code_block.next_var_pair();
                code_block.add_code(&format!("{}({})", name, arg_strs.join(", ")));
                (valid, var)
            }
            Expr::Cast { expr, to } => {
                let (e_valid, e_var) = expr.to_nvrtc(schema, code_block)?;
                if to.kernel_type() == expr.infer_type(schema)?.kernel_type() {
                    return Ok((e_valid, e_var));
                }
                let (valid, var) = code_block.next_var_pair();
                code_block
                    .add_validity_check(&valid, &[&e_valid])
                    .add_variable_decl(result_type.kernel_type(), &var)
                    .add_conditional(&valid, |block| {
                        block.add_code(&format!(
                            "\t{}.value = ({})({}.value)\n",
                            var,
                            to.c_type(),
                            e_var,
                        ));
                    });
                (valid, var)
            }
        };
        Ok(res)
    }
}

fn escape_c_string(s: &str) -> String {
    s.replace('"', "\\\"")
}
pub fn float_literal_to_str<T: Into<f64> + Copy + PartialEq>(f: T) -> String {
    let f64_val = f.into();
    if f64_val.fract() == 0.0 { format!("{}.0", f64_val) } else { format!("{}", f64_val) }
}
