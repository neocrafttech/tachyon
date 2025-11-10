use core::expr::{BinaryOp, Expr, UnaryOp};
use core::parser::parse_scheme_expr;

macro_rules! test_parser_matrix {
    (
        Binary,
        [
            $(
                ( $test_name:ident, $expr:expr, $child:expr )
            ),* $(,)?
        ]
    ) => {
        $(
            #[test]
            fn $test_name() {
                let result = parse_scheme_expr($expr).unwrap();
                match result {
                    Expr::Binary { op, .. } => assert_eq!(op, $child),
                    _ => panic!("Expected Binary expression, got {:?}", result),
                }
            }
        )*
    };
    (
        Unary,
        [
            $(
                ( $test_name:ident, $expr:expr, $child:expr )
            ),* $(,)?
        ]
    ) => {
        $(
            #[test]
            fn $test_name() {
                let result = parse_scheme_expr($expr).unwrap();
                match result {
                    Expr::Unary { op, .. } => assert_eq!(op, $child),
                    _ => panic!("Expected Unary expression, got {:?}", result),
                }
            }
        )*
    };
    (
        Call,
        [
            $(
                ( $test_name:ident, $expr:expr, $child:expr )
            ),* $(,)?
        ]
    ) => {
        $(
            #[test]
            fn $test_name() {
                let result = parse_scheme_expr($expr).unwrap();
                match result {
                    Expr::Call { name, .. } => assert_eq!(name, $child),
                    _ => panic!("Expected Call expression, got {:?}", result),
                }
            }
        )*
    };
}

test_parser_matrix!(
    Binary,
    [
        (test_parser_add, "(+, i0, i1)", BinaryOp::Add),
        (test_parser_sub, "(-, i0, i1)", BinaryOp::Sub),
        (test_parser_mul, "(*, i0, i1)", BinaryOp::Mul),
        (test_parser_div, "(/, i0, i1)", BinaryOp::Div),
        (test_parse_eq, "(==, i0, i1)", BinaryOp::Eq),
        (test_parse_neq, "(!=, i0, i1)", BinaryOp::NotEq),
        (test_parse_lt, "(<, i0, i1)", BinaryOp::Lt),
        (test_parse_lte, "(<=, i0, i1)", BinaryOp::LtEq),
        (test_parse_gt, "(>, i0, i1)", BinaryOp::Gt),
        (test_parse_gte, "(>=, i0, i1)", BinaryOp::GtEq),
        (test_parse_and, "(&&, i0, i1)", BinaryOp::And),
        (test_parse_or, "(||, i0, i1)", BinaryOp::Or),
    ]
);

test_parser_matrix!(
    Unary,
    [(test_parser_neg, "(neg, i0)", UnaryOp::Neg), (test_parser_not, "(not, i0)", UnaryOp::Not),]
);

test_parser_matrix!(
    Call,
    [(test_parser_sqrt, "(sqrt, i0)", "sqrt"), (test_parser_upper, "(upper, i0)", "upper"),]
);

#[test]
fn test_nested() {
    let expr = parse_scheme_expr("(*, (+ , i0, 1), 2.5)").unwrap();
    assert!(matches!(expr, Expr::Binary { op: BinaryOp::Mul, .. }));
}

#[test]
fn test_function_call() {
    let expr = parse_scheme_expr("(sqrt, (*, col_x, 2.5))").unwrap();
    if let Expr::Call { name, args } = expr {
        assert_eq!(name, "sqrt");
        assert_eq!(args.len(), 1);
    } else {
        panic!("Expected Call expr");
    }
}

#[test]
fn test_cast() {
    let expr = parse_scheme_expr("(cast, i0, f64)").unwrap();
    assert!(matches!(expr, Expr::Cast { .. }));
}
