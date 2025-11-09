use crate::data_type::DataType;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    NotEq,
    Lt,
    LtEq,
    Gt,
    GtEq,
    And,
    Or,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

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
    F32(f32),
    F64(f64),
    Bool(bool),
    Str(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Column(String),

    Literal(Literal),

    Binary { op: BinaryOp, left: Box<Expr>, right: Box<Expr> },

    Unary { op: UnaryOp, expr: Box<Expr> },

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

    pub fn f32(f: f32) -> Self {
        Expr::Literal(Literal::F32(f))
    }

    pub fn f64(f: f64) -> Self {
        Expr::Literal(Literal::F64(f))
    }

    pub fn bool_lit(b: bool) -> Self {
        Expr::Literal(Literal::Bool(b))
    }

    pub fn bin(op: BinaryOp, left: Expr, right: Expr) -> Self {
        Expr::Binary { op, left: Box::new(left), right: Box::new(right) }
    }

    pub fn unary(op: UnaryOp, expr: Expr) -> Self {
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
            Expr::Binary { left, right, .. } => vec![left.as_ref(), right.as_ref()],
            Expr::Unary { expr, .. } => vec![expr.as_ref()],
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
    pub columns: std::collections::HashMap<String, DataType>,
}

impl SchemaContext {
    pub fn new() -> Self {
        Self { columns: Default::default() }
    }

    pub fn with_column<S: Into<String>>(mut self, name: S, dt: DataType) -> Self {
        self.columns.insert(name.into(), dt);
        self
    }

    pub fn lookup(&self, name: &str) -> Option<&DataType> {
        self.columns.get(name)
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
            Expr::Column(name) => {
                schema.lookup(name).cloned().ok_or_else(|| TypeError::UnknownColumn(name.clone()))
            }

            Expr::Literal(l) => match l {
                Literal::I8(_) => Ok(DataType::I8),
                Literal::I16(_) => Ok(DataType::I16),
                Literal::I32(_) => Ok(DataType::I32),
                Literal::I64(_) => Ok(DataType::I64),
                Literal::U8(_) => Ok(DataType::U8),
                Literal::U16(_) => Ok(DataType::U16),
                Literal::U32(_) => Ok(DataType::U32),
                Literal::U64(_) => Ok(DataType::U64),
                Literal::F32(_) => Ok(DataType::F32),
                Literal::F64(_) => Ok(DataType::F64),
                Literal::Bool(_) => Ok(DataType::Bool),
                Literal::Str(_) => Ok(DataType::Utf8),
            },

            Expr::Unary { op, expr } => {
                let t = expr.infer_type(schema)?;
                match op {
                    UnaryOp::Neg => match t {
                        DataType::I8
                        | DataType::I16
                        | DataType::I32
                        | DataType::I64
                        | DataType::F32
                        | DataType::F64 => Ok(t),
                        _ => Err(TypeError::Unsupported(format!("neg on {:?}", t))),
                    },
                    UnaryOp::Not => match t {
                        DataType::Bool => Ok(DataType::Bool),
                        _ => Err(TypeError::Unsupported(format!("not on {:?}", t))),
                    },
                }
            }

            Expr::Binary { op, left, right } => {
                let lt = left.infer_type(schema)?;
                let rt = right.infer_type(schema)?;

                use BinaryOp::*;
                match op {
                    Add | Sub | Mul | Div => match (&lt, &rt) {
                        (DataType::F64, _) | (_, DataType::F64) => Ok(DataType::F64),
                        (DataType::F32, _) | (_, DataType::F32) => Ok(DataType::F32),
                        (DataType::I64, DataType::I64) => Ok(DataType::I64),
                        (DataType::I32, DataType::I32) => Ok(DataType::I32),
                        _ => Err(TypeError::TypeMismatch { expected: lt, got: rt }),
                    },
                    Eq | NotEq | Lt | LtEq | Gt | GtEq => Ok(DataType::Bool),
                    And | Or => {
                        if lt == DataType::Bool && rt == DataType::Bool {
                            Ok(DataType::Bool)
                        } else {
                            Err(TypeError::TypeMismatch {
                                expected: DataType::Bool,
                                got: if lt != DataType::Bool { lt } else { rt },
                            })
                        }
                    }
                }
            }

            Expr::Call { name, args } => {
                // primitive builtin handling (extend as needed)
                match name.as_str() {
                    "sqrt" => {
                        if args.len() != 1 {
                            return Err(TypeError::Unsupported("sqrt arity".into()));
                        }
                        let t = args[0].infer_type(schema)?;
                        match t {
                            DataType::F32 | DataType::F64 => Ok(t),
                            DataType::I32 | DataType::I64 => Ok(DataType::F64),
                            _ => Err(TypeError::Unsupported(format!("sqrt on {:?}", t))),
                        }
                    }
                    "upper" => {
                        if args.len() != 1 {
                            return Err(TypeError::Unsupported("upper arity".into()));
                        }
                        Ok(DataType::Utf8)
                    }
                    _ => Err(TypeError::Unsupported(format!("unknown function {}", name))),
                }
            }

            Expr::Cast { expr, to } => {
                // very permissive cast for now
                let _ = expr.infer_type(schema)?;
                Ok(to.clone())
            }
        }
    }
}

pub trait ToNvrtc {
    fn to_nvrtc(&self, schema: &SchemaContext) -> Result<String, TypeError>;
}

impl ToNvrtc for Expr {
    fn to_nvrtc(&self, schema: &SchemaContext) -> Result<String, TypeError> {
        // Ensure type correctness early
        let _t = self.infer_type(schema)?;

        let res = match self {
            Expr::Column(name) => {
                // assume columns are available as arrays like `col_<name>[i]`
                format!("col_{}[i]", sanitize_ident(name))
            }
            Expr::Literal(l) => match l {
                Literal::I8(i) => format!("{}ll", i),
                Literal::I16(i) => format!("{}ll", i),
                Literal::I32(i) => format!("{}ll", i),
                Literal::I64(i) => format!("{}ll", i),
                Literal::U8(i) => format!("{}ull", i),
                Literal::U16(i) => format!("{}ull", i),
                Literal::U32(i) => format!("{}ull", i),
                Literal::U64(i) => format!("{}ull", i),
                Literal::F32(f) => float_literal_to_str(*f).to_string(),
                Literal::F64(f) => float_literal_to_str(*f).to_string(),
                Literal::Bool(b) => format!("{}", if *b { 1 } else { 0 }),
                Literal::Str(s) => format!("\"{}\"", escape_c_string(s)),
            },
            Expr::Unary { op, expr } => {
                let e = expr.to_nvrtc(schema)?;
                match op {
                    UnaryOp::Neg => format!("(-({}))", e),
                    UnaryOp::Not => format!("(!({}))", e),
                }
            }
            Expr::Binary { op, left, right } => {
                let l = left.to_nvrtc(schema)?;
                let r = right.to_nvrtc(schema)?;
                let op_s = match op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub => "-",
                    BinaryOp::Mul => "*",
                    BinaryOp::Div => "/",
                    BinaryOp::Eq => "==",
                    BinaryOp::NotEq => "!=",
                    BinaryOp::Lt => "<",
                    BinaryOp::LtEq => "<=",
                    BinaryOp::Gt => ">",
                    BinaryOp::GtEq => ">=",
                    BinaryOp::And => "&&",
                    BinaryOp::Or => "||",
                };
                format!("(({}) {} ({}))", l, op_s, r)
            }
            Expr::Call { name, args } => {
                let mut arg_strs = Vec::with_capacity(args.len());
                for a in args {
                    arg_strs.push(a.to_nvrtc(schema)?);
                }
                format!("{}({})", name, arg_strs.join(", "))
            }
            Expr::Cast { expr, to } => {
                let e = expr.to_nvrtc(schema)?;
                let ty = DataType::c_type(to);
                format!("(({})({}))", ty, e)
            }
        };
        Ok(res)
    }
}

/// Helpers
fn sanitize_ident(s: &str) -> String {
    s.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect()
}

fn escape_c_string(s: &str) -> String {
    s.replace('"', "\\\"")
}

pub fn float_literal_to_str<T: Into<f64> + Copy + PartialEq>(f: T) -> String {
    let f64_val = f.into();
    if f64_val.fract() == 0.0 { format!("{}.0", f64_val) } else { format!("{}", f64_val) }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Column(name) => write!(f, "col({})", name),
            Expr::Literal(l) => write!(f, "lit({:?})", l),
            Expr::Unary { op, expr } => write!(f, "un({:?} {})", op, expr),
            Expr::Binary { op, left, right } => write!(f, "({} {:?} {})", left, op, right),
            Expr::Call { name, args } => write!(f, "{}({:?})", name, args),
            Expr::Cast { expr, to } => write!(f, "cast({} as {:?})", expr, to),
        }
    }
}
