use core::expr::Expr;
use core::operator::Operator;
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
        (test_parse_add, "(+, i0, i1)", Operator::Add),
        (test_parse_sub, "(-, i0, i1)", Operator::Sub),
        (test_parse_mul, "(*, i0, i1)", Operator::Mul),
        (test_parse_div, "(/, i0, i1)", Operator::Div),
        (test_parse_eq, "(==, i0, i1)", Operator::Eq),
        (test_parse_neq, "(!=, i0, i1)", Operator::NotEq),
        (test_parse_lt, "(<, i0, i1)", Operator::Lt),
        (test_parse_lte, "(<=, i0, i1)", Operator::LtEq),
        (test_parse_gt, "(>, i0, i1)", Operator::Gt),
        (test_parse_gte, "(>=, i0, i1)", Operator::GtEq),
        (test_parse_and, "(&&, i0, i1)", Operator::And),
        (test_parse_or, "(||, i0, i1)", Operator::Or),
    ]
);

test_parser_matrix!(
    Unary,
    [(test_parse_neg, "(neg, i0)", Operator::Neg), (test_parse_not, "(not, i0)", Operator::Not),]
);

test_parser_matrix!(
    Call,
    [(test_parse_sqrt, "(sqrt, i0)", "sqrt"), (test_parse_upper, "(upper, i0)", "upper"),]
);

#[test]
fn test_nested() {
    let expr = parse_scheme_expr("(*, (+ , i0, 1), 2.5)").unwrap();
    assert!(matches!(expr, Expr::Binary { op: Operator::Mul, .. }));
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
