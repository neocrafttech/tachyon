/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

use crate::data_type::DataType;
use crate::expr::{Expr, Literal};
use crate::operator::Operator;

#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("unexpected end of input")]
    UnexpectedEof,

    #[error("expected '(' but got: {0}")]
    ExpectedOpenParen(String),

    #[error("expected ')' but got: {0}")]
    ExpectedCloseParen(String),

    #[error("expected ',' but got: {0}")]
    ExpectedComma(String),

    #[error("unknown operator: {0}")]
    UnknownOperator(String),

    #[error("invalid literal: {0}")]
    InvalidLiteral(String),

    #[error("wrong number of arguments for {op}: expected {expected}, got {got}")]
    WrongArity { op: String, expected: usize, got: usize },

    #[error("invalid token: {0}")]
    InvalidToken(String),
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    OpenParen,
    CloseParen,
    Comma,
    Ident(String),
    Number(String),
    StringLit(String),
}

struct Lexer {
    input: Vec<char>,
    pos: usize,
}

impl Lexer {
    fn new(input: &str) -> Self {
        Self { input: input.chars().collect(), pos: 0 }
    }

    fn peek(&self) -> Option<char> {
        self.input.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.peek()?;
        self.pos += 1;
        Some(ch)
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn read_string(&mut self) -> Result<String, ParseError> {
        self.advance();
        let mut s = String::new();

        loop {
            match self.peek() {
                Some('"') => {
                    self.advance();
                    return Ok(s);
                }
                Some('\\') => {
                    self.advance();
                    match self.advance() {
                        Some('n') => s.push('\n'),
                        Some('t') => s.push('\t'),
                        Some('r') => s.push('\r'),
                        Some('\\') => s.push('\\'),
                        Some('"') => s.push('"'),
                        Some(ch) => s.push(ch),
                        None => return Err(ParseError::UnexpectedEof),
                    }
                }
                Some(ch) => {
                    s.push(ch);
                    self.advance();
                }
                None => return Err(ParseError::UnexpectedEof),
            }
        }
    }

    fn next_token(&mut self) -> Result<Option<Token>, ParseError> {
        self.skip_whitespace();

        match self.peek() {
            None => Ok(None),
            Some('(') => {
                self.advance();
                Ok(Some(Token::OpenParen))
            }
            Some(')') => {
                self.advance();
                Ok(Some(Token::CloseParen))
            }
            Some(',') => {
                self.advance();
                Ok(Some(Token::Comma))
            }
            Some('"') => {
                let s = self.read_string()?;
                Ok(Some(Token::StringLit(s)))
            }
            Some(ch) if ch.is_ascii_digit() => {
                let mut num = String::new();

                while let Some(ch) = self.peek() {
                    if ch.is_ascii_digit() || ch == '.' || ch == 'e' || ch == 'E' {
                        num.push(ch);
                        self.advance();
                    } else {
                        break;
                    }
                }

                Ok(Some(Token::Number(num)))
            }
            Some(ch) if ch == '-' || ch == '+' => {
                self.advance();

                if let Some(next_ch) = self.peek() {
                    if next_ch.is_ascii_digit() {
                        let mut num = String::new();
                        num.push(ch);

                        while let Some(ch) = self.peek() {
                            if ch.is_ascii_digit() || ch == '.' || ch == 'e' || ch == 'E' {
                                num.push(ch);
                                self.advance();
                            } else {
                                break;
                            }
                        }

                        Ok(Some(Token::Number(num)))
                    } else {
                        Ok(Some(Token::Ident(ch.to_string())))
                    }
                } else {
                    Ok(Some(Token::Ident(ch.to_string())))
                }
            }
            Some(ch) if ch.is_alphabetic() || ch == '_' => {
                let mut ident = String::new();

                while let Some(ch) = self.peek() {
                    if ch.is_alphanumeric() || ch == '_' {
                        ident.push(ch);
                        self.advance();
                    } else {
                        break;
                    }
                }

                Ok(Some(Token::Ident(ident)))
            }
            Some(ch) if "*/=!<>&|".contains(ch) => {
                let mut op = String::new();
                op.push(ch);
                self.advance();

                if let Some(next_ch) = self.peek() {
                    match (ch, next_ch) {
                        ('=', '=')
                        | ('!', '=')
                        | ('<', '=')
                        | ('>', '=')
                        | ('&', '&')
                        | ('|', '|') => {
                            op.push(next_ch);
                            self.advance();
                        }
                        _ => {}
                    }
                }

                Ok(Some(Token::Ident(op)))
            }
            Some(ch) => Err(ParseError::InvalidToken(ch.to_string())),
        }
    }
}

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(input: &str) -> Result<Self, ParseError> {
        let mut lexer = Lexer::new(input);
        let mut tokens = Vec::new();

        while let Some(token) = lexer.next_token()? {
            tokens.push(token);
        }

        Ok(Self { tokens, pos: 0 })
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.pos).cloned();
        if token.is_some() {
            self.pos += 1;
        }
        token
    }

    fn expect(&mut self, expected: Token) -> Result<(), ParseError> {
        match self.advance() {
            Some(token) if token == expected => Ok(()),
            Some(token) => match expected {
                Token::OpenParen => Err(ParseError::ExpectedOpenParen(format!("{:?}", token))),
                Token::CloseParen => Err(ParseError::ExpectedCloseParen(format!("{:?}", token))),
                Token::Comma => Err(ParseError::ExpectedComma(format!("{:?}", token))),
                _ => Err(ParseError::InvalidToken(format!("{:?}", token))),
            },
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn parse_literal(s: &str) -> Result<Literal, ParseError> {
        if s == "true" {
            return Ok(Literal::Bool(true));
        }
        if s == "false" {
            return Ok(Literal::Bool(false));
        }

        if s.contains('.') {
            if let Ok(f) = s.parse::<f64>() {
                return Ok(Literal::F64(f));
            }
            if let Ok(f) = s.parse::<f32>() {
                return Ok(Literal::F32(f));
            }
        }

        if let Ok(i) = s.parse::<i32>() {
            return Ok(Literal::I32(i));
        }
        if let Ok(i) = s.parse::<i64>() {
            return Ok(Literal::I64(i));
        }

        Err(ParseError::InvalidLiteral(s.to_string()))
    }

    pub fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        match self.peek() {
            Some(Token::OpenParen) => self.parse_call(),
            Some(Token::Ident(name)) => {
                let name = name.clone();
                self.advance();

                Ok(Expr::col(name))
            }
            Some(Token::Number(num)) => {
                let num = num.clone();
                self.advance();
                Ok(Expr::lit(Self::parse_literal(&num)?))
            }
            Some(Token::StringLit(s)) => {
                let s = s.clone();
                self.advance();
                Ok(Expr::lit(Literal::Str(s)))
            }
            Some(token) => Err(ParseError::InvalidToken(format!("{:?}", token))),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    fn parse_call(&mut self) -> Result<Expr, ParseError> {
        self.expect(Token::OpenParen)?;

        let op = match self.advance() {
            Some(Token::Ident(name)) => name,
            Some(token) => return Err(ParseError::InvalidToken(format!("{:?}", token))),
            None => return Err(ParseError::UnexpectedEof),
        };

        self.expect(Token::Comma)?;

        let mut args = vec![self.parse_expr()?];

        while let Some(Token::Comma) = self.peek() {
            self.advance();
            args.push(self.parse_expr()?);
        }

        self.expect(Token::CloseParen)?;
        let op_type = Operator::from(op.as_str());

        match op_type {
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
            | Operator::Or => {
                if args.len() != 2 {
                    Err(ParseError::WrongArity { op: op.clone(), expected: 2, got: args.len() })?;
                }

                Ok(Expr::binary(op_type, args.remove(0), args.remove(0)))
            }
            Operator::Neg | Operator::Not => {
                if args.len() != 1 {
                    Err(ParseError::WrongArity { op: op.clone(), expected: 1, got: args.len() })?;
                }
                Ok(Expr::unary(op_type, args.remove(0)))
            }

            Operator::Cast => {
                if args.len() != 2 {
                    return Err(ParseError::WrongArity { op, expected: 2, got: args.len() });
                }
                let type_expr = args.remove(1);
                if let Expr::Column(type_name) = type_expr {
                    let data_type = match type_name.as_str() {
                        "i8" => DataType::I8,
                        "i16" => DataType::I16,
                        "i32" => DataType::I32,
                        "i64" => DataType::I64,
                        "u8" => DataType::U8,
                        "u16" => DataType::U16,
                        "u32" => DataType::U32,
                        "u64" => DataType::U64,
                        "f32" => DataType::F32,
                        "f64" => DataType::F64,
                        "bool" => DataType::BOOL,
                        "utf8" | "string" => DataType::STR,
                        _ => return Err(ParseError::InvalidLiteral(type_name)),
                    };
                    Ok(args.remove(0).cast(data_type))
                } else {
                    Err(ParseError::InvalidToken("expected type name".to_string()))
                }
            }

            _ => Ok(Expr::call(op, args)),
        }
    }

    pub fn parse(&mut self) -> Result<Expr, ParseError> {
        self.parse_expr()
    }
}

pub fn parse_scheme_expr(input: &str) -> Result<Expr, ParseError> {
    let mut parser = Parser::new(input)?;
    parser.parse()
}
