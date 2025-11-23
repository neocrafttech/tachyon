/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd, Eq, Hash)]
pub enum Operator {
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
    Neg,
    Not,
    Cast,
    Call,
}

impl Operator {
    pub(crate) fn is_binary(&self) -> bool {
        matches!(
            self,
            Operator::Add
                | Operator::Sub
                | Operator::Mul
                | Operator::Div
                | Operator::Eq
                | Operator::NotEq
                | Operator::Lt
                | Operator::LtEq
                | Operator::Gt
                | Operator::GtEq
                | Operator::And
                | Operator::Or
        )
    }
}

impl From<&str> for Operator {
    fn from(op: &str) -> Self {
        match op.to_lowercase().as_str() {
            "+" | "add" => Operator::Add,
            "-" | "sub" => Operator::Sub,
            "*" | "mul" => Operator::Mul,
            "/" | "div" => Operator::Div,
            "==" | "=" | "eq" => Operator::Eq,
            "<" | "lt" => Operator::Lt,
            "!=" => Operator::NotEq,
            "<=" | "lte" => Operator::LtEq,
            ">" | "gt" => Operator::Gt,
            ">=" | "gte" => Operator::GtEq,
            "&&" | "and" => Operator::And,
            "||" | "or" => Operator::Or,
            "neg" => Operator::Neg,
            "!" | "not" => Operator::Not,
            "cast" => Operator::Cast,
            _ => Operator::Call,
        }
    }
}

impl Operator {
    pub fn as_symbol(&self) -> &str {
        match self {
            Operator::Add => "+",
            Operator::Sub => "-",
            Operator::Mul => "*",
            Operator::Div => "/",
            Operator::Eq => "==",
            Operator::Lt => "<",
            Operator::NotEq => "!=",
            Operator::LtEq => "<=",
            Operator::Gt => ">",
            Operator::GtEq => ">=",
            Operator::And => "&&",
            Operator::Or => "||",
            Operator::Neg => "neg",
            Operator::Not => "!",
            Operator::Cast => "cast",
            Operator::Call => "call",
        }
    }
}

impl std::fmt::Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_symbol())
    }
}
