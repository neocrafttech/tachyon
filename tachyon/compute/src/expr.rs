/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

use std::collections::HashMap;
use std::fmt;

use half::{bf16, f16};

use crate::data_type::DataType;
use crate::error::ErrorMode;
use crate::operator::Operator;

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    BF16(bf16),
    F16(f16),
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Column(String),

    Literal(Literal),

    Unary { op: Operator, expr: Box<Expr> },

    Binary { op: Operator, left: Box<Expr>, right: Box<Expr> },

    Nary { op: Operator, args: Vec<Box<Expr>> },

    Call { name: String, args: Vec<Expr> },

    Cast { expr: Box<Expr>, to: DataType },
}

impl Expr {
    pub fn col<S: Into<String>>(name: S) -> Self {
        Expr::Column(name.into())
    }

    pub fn lit(l: Literal) -> Self {
        Expr::Literal(l)
    }

    pub fn i8(i: i8) -> Self {
        Expr::Literal(Literal::I8(i))
    }

    pub fn i16(i: i16) -> Self {
        Expr::Literal(Literal::I16(i))
    }

    pub fn i32(i: i32) -> Self {
        Expr::Literal(Literal::I32(i))
    }

    pub fn i64(i: i64) -> Self {
        Expr::Literal(Literal::I64(i))
    }

    pub fn u8(i: u8) -> Self {
        Expr::Literal(Literal::U8(i))
    }

    pub fn u16(i: u16) -> Self {
        Expr::Literal(Literal::U16(i))
    }

    pub fn u32(i: u32) -> Self {
        Expr::Literal(Literal::U32(i))
    }

    pub fn u64(i: u64) -> Self {
        Expr::Literal(Literal::U64(i))
    }

    pub fn bf16(f: bf16) -> Self {
        Expr::Literal(Literal::BF16(f))
    }

    pub fn f16(f: f16) -> Self {
        Expr::Literal(Literal::F16(f))
    }

    pub fn f32(f: f32) -> Self {
        Expr::Literal(Literal::F32(f))
    }

    pub fn f64(f: f64) -> Self {
        Expr::Literal(Literal::F64(f))
    }

    pub fn bool_lit(b: bool) -> Self {
        Expr::Literal(Literal::Bool(b))
    }

    pub fn binary(op: Operator, left: Expr, right: Expr) -> Self {
        Expr::Binary { op, left: Box::new(left), right: Box::new(right) }
    }

    pub fn unary(op: Operator, expr: Expr) -> Self {
        Expr::Unary { op, expr: Box::new(expr) }
    }

    pub fn call<N: Into<String>>(name: N, args: Vec<Expr>) -> Self {
        Expr::Call { name: name.into(), args }
    }

    pub fn cast(self, to: DataType) -> Self {
        Expr::Cast { expr: Box::new(self), to }
    }

    pub fn children(&self) -> Vec<&Expr> {
        match self {
            Expr::Column(_) | Expr::Literal(_) => vec![],
            Expr::Unary { expr, .. } => vec![expr.as_ref()],
            Expr::Binary { left, right, .. } => vec![left.as_ref(), right.as_ref()],
            Expr::Nary { args, .. } => args.iter().map(|x| x.as_ref()).collect(),
            Expr::Call { args, .. } => args.iter().collect(),
            Expr::Cast { expr, .. } => vec![expr.as_ref()],
        }
    }
}

pub trait ExprVisitor {
    fn enter(&mut self, _expr: &Expr) -> bool {
        true
    }

    fn exit(&mut self, _expr: &Expr) {}
}

pub fn walk_expr<V: ExprVisitor + ?Sized>(expr: &Expr, visitor: &mut V) {
    if !visitor.enter(expr) {
        return;
    }
    for c in expr.children() {
        walk_expr(c, visitor);
    }
    visitor.exit(expr);
}

#[derive(Debug, thiserror::Error)]
pub enum TypeError {
    #[error("unknown column: {0}")]
    UnknownColumn(String),

    #[error("type mismatch: expected {expected:?}, got {got:?}")]
    TypeMismatch { expected: DataType, got: DataType },

    #[error("unsupported operation: {0}")]
    Unsupported(String),
}

#[derive(Debug, Clone)]
pub struct SchemaContext {
    pub columns: HashMap<String, (u16, DataType)>,
    pub error_mode: ErrorMode,
}

impl SchemaContext {
    pub fn new() -> Self {
        Self { columns: Default::default(), error_mode: ErrorMode::Tachyon }
    }

    pub fn with_columns(mut self, columns: &HashMap<String, (u16, DataType)>) -> Self {
        self.columns = columns.clone();
        self
    }

    pub fn with_error_mode(mut self, error_mode: ErrorMode) -> Self {
        self.error_mode = error_mode;
        self
    }

    pub fn with_column<S: Into<String>>(mut self, name: S, dt: DataType) -> Self {
        let size = self.columns.len();
        self.columns.insert(name.into(), (size.try_into().unwrap(), dt));
        self
    }

    pub fn lookup(&self, name: &str) -> Option<&(u16, DataType)> {
        self.columns.get(name)
    }

    pub fn error_mode(&self) -> ErrorMode {
        self.error_mode
    }
}

impl Default for SchemaContext {
    fn default() -> Self {
        Self::new()
    }
}

impl Expr {
    pub fn infer_type(&self, schema: &SchemaContext) -> Result<DataType, TypeError> {
        match self {
            Expr::Column(name) => match schema.lookup(name) {
                Some(pair) => Ok(pair.1),
                _ => Err(TypeError::UnknownColumn(name.clone()))?,
            },
            Expr::Literal(l) => match l {
                Literal::I8(_) => Ok(DataType::I8),
                Literal::I16(_) => Ok(DataType::I16),
                Literal::I32(_) => Ok(DataType::I32),
                Literal::I64(_) => Ok(DataType::I64),
                Literal::U8(_) => Ok(DataType::U8),
                Literal::U16(_) => Ok(DataType::U16),
                Literal::U32(_) => Ok(DataType::U32),
                Literal::U64(_) => Ok(DataType::U64),
                Literal::BF16(_) => Ok(DataType::BF16),
                Literal::F16(_) => Ok(DataType::F16),
                Literal::F32(_) => Ok(DataType::F32),
                Literal::F64(_) => Ok(DataType::F64),
                Literal::Bool(_) => Ok(DataType::Bool),
                Literal::Str(_) => Ok(DataType::Str),
            },

            Expr::Unary { op, expr } => {
                let t = expr.infer_type(schema)?;
                match op {
                    Operator::Neg => match t {
                        DataType::I8
                        | DataType::I16
                        | DataType::I32
                        | DataType::I64
                        | DataType::F32
                        | DataType::F64 => Ok(t),
                        _ => Err(TypeError::Unsupported(format!("neg on {:?}", t))),
                    },
                    Operator::Not => match t {
                        DataType::Bool => Ok(DataType::Bool),
                        _ => Err(TypeError::Unsupported(format!("not on {:?}", t))),
                    },
                    _ => Err(TypeError::Unsupported(format!("Not supported unary op {}", op)))?,
                }
            }

            Expr::Binary { op, left, right } => {
                let lt = left.infer_type(schema)?;
                let rt = right.infer_type(schema)?;

                match op {
                    Operator::Add | Operator::Sub | Operator::Mul | Operator::Div => {
                        match (&lt, &rt) {
                            (lt_val, rt_val) if lt_val == rt_val => Ok(*lt_val),

                            //Integer Promotion: Promote to the wider integer type. (e.g., I32 + I64 -> I64)
                            (lt_val, rt_val) if lt_val.is_integer() && rt_val.is_integer() => {
                                if lt_val.native_size() > rt_val.native_size() {
                                    Ok(*lt_val) // e.g., I64 + I32 -> I64
                                } else {
                                    Ok(*rt_val) // e.g., I32 + I64 -> I64
                                }
                            }
                            // Float Promotion: Promote to the wider float type. (e.g., F32 + F64 -> F64)
                            (lt_val, rt_val) if lt_val.is_float() && rt_val.is_float() => {
                                if lt_val.native_size() > rt_val.native_size() {
                                    Ok(*lt_val) // e.g., F64 + F32 -> F64
                                } else {
                                    Ok(*rt_val) // e.g., F32 + F64 -> F64
                                }
                            }

                            // Integer/Float Promotion (Left is Float): Promote to the wider float type. (e.g., F32 + I64 -> F64)
                            (lt_val, rt_val) if lt_val.is_float() && rt_val.is_integer() => {
                                match (lt_val, rt_val.native_size()) {
                                    (DataType::F64, _) => Ok(DataType::F64), // F64 is always the widest
                                    (DataType::F32, size) if size > DataType::F32.native_size() => {
                                        Ok(DataType::F64)
                                    } // I64 is larger than F32
                                    _ => Ok(*lt_val), // F32 + smaller integer => F32
                                }
                            }

                            // 5. Integer/Float Promotion (Right is Float): Same logic, reversed. (e.g., I64 + F32 -> F64)
                            (lt_val, rt_val) if lt_val.is_integer() && rt_val.is_float() => {
                                match (rt_val, lt_val.native_size()) {
                                    (DataType::F64, _) => Ok(DataType::F64), // F64 is always the widest
                                    (DataType::F32, size) if size > DataType::F32.native_size() => {
                                        Ok(DataType::F64)
                                    } // I64 is larger than F32
                                    _ => Ok(*rt_val), // F32 + smaller integer => F32
                                }
                            }

                            (lt_val, rt_val) => {
                                Err(TypeError::TypeMismatch { expected: *lt_val, got: *rt_val })
                            }
                        }
                    }
                    Operator::Eq
                    | Operator::NotEq
                    | Operator::Lt
                    | Operator::LtEq
                    | Operator::Gt
                    | Operator::GtEq => Ok(DataType::Bool),
                    Operator::And | Operator::Or => {
                        if lt == rt {
                            Ok(DataType::Bool)
                        } else {
                            Err(TypeError::TypeMismatch {
                                expected: DataType::Bool,
                                got: if lt != DataType::Bool { lt } else { rt },
                            })
                        }
                    }
                    _ => Err(TypeError::Unsupported(format!("Unsuported Binary Op {}", op)))?,
                }
            }
            Expr::Nary { op: _, args: _ } => unimplemented!(),
            Expr::Call { name, args } => match name.as_str() {
                "sqrt" => {
                    if args.len() != 1 {
                        Err(TypeError::Unsupported("sqrt arity".into()))?;
                    }
                    let t = args[0].infer_type(schema)?;
                    match t {
                        DataType::F32 | DataType::F64 => Ok(t),
                        DataType::I32 | DataType::I64 => Ok(DataType::F64),
                        _ => Err(TypeError::Unsupported(format!("sqrt on {:?}", t))),
                    }
                }
                "lower" => {
                    if args.len() != 1 {
                        Err(TypeError::Unsupported("lower arity".into()))?;
                    }
                    Ok(DataType::Str)
                }
                "upper" => {
                    if args.len() != 1 {
                        Err(TypeError::Unsupported("upper arity".into()))?;
                    }
                    Ok(DataType::Str)
                }
                _ => Err(TypeError::Unsupported(format!("unknown function {}", name))),
            },

            Expr::Cast { expr, to } => {
                let _ = expr.infer_type(schema)?;
                Ok(*to)
            }
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Column(name) => write!(f, "col({})", name),
            Expr::Literal(l) => write!(f, "lit({:?})", l),
            Expr::Unary { op, expr } => write!(f, "un({:?} {})", op, expr),
            Expr::Binary { op, left, right } => write!(f, "({} {:?} {})", left, op, right),
            Expr::Nary { op, args } => write!(f, "{}({:?})", op, args),
            Expr::Call { name, args } => write!(f, "{}({:?})", name, args),
            Expr::Cast { expr, to } => write!(f, "cast({} as {:?})", expr, to),
        }
    }
}
