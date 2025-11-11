use core::data_type::DataType;

macro_rules! matrix_test {
    ($test_fn:ident) => {{
        let all_types = [
            DataType::I8,
            DataType::I16,
            DataType::I32,
            DataType::I64,
            DataType::U8,
            DataType::U16,
            DataType::U32,
            DataType::U64,
            DataType::BF16,
            DataType::F16,
            DataType::F32,
            DataType::F64,
            DataType::Bool,
            DataType::Utf8,
        ];
        for dtype in all_types {
            $test_fn(dtype);
        }
    }};
}

#[test]
fn test_c_type_mapping() {
    fn check_c_type(dt: DataType) {
        let ctype = dt.c_type();
        match dt {
            DataType::I8 => assert_eq!(ctype, "int8_t"),
            DataType::I16 => assert_eq!(ctype, "int16_t"),
            DataType::I32 => assert_eq!(ctype, "int32_t"),
            DataType::I64 => assert_eq!(ctype, "int64_t"),
            DataType::U8 => assert_eq!(ctype, "uint8_t"),
            DataType::U16 => assert_eq!(ctype, "uint16_t"),
            DataType::U32 => assert_eq!(ctype, "uint32_t"),
            DataType::U64 => assert_eq!(ctype, "uint64_t"),
            DataType::BF16 => assert_eq!(ctype, "bfloat16"),
            DataType::F16 => assert_eq!(ctype, "float16"),
            DataType::F32 => assert_eq!(ctype, "float"),
            DataType::F64 => assert_eq!(ctype, "double"),
            DataType::Bool => assert_eq!(ctype, "bool"),
            DataType::Utf8 => assert_eq!(ctype, "uint8_t"),
        }
    }
    matrix_test!(check_c_type);
}

#[test]
fn test_is_signed_unsigned_numeric_flags() {
    fn check_flags(dt: DataType) {
        match dt {
            DataType::I8 | DataType::I16 | DataType::I32 | DataType::I64 => {
                assert!(dt.is_signed());
                assert!(!dt.is_unsigned());
                assert!(dt.is_integer());
                assert!(dt.is_numeric());
            }
            DataType::U8 | DataType::U16 | DataType::U32 | DataType::U64 => {
                assert!(!dt.is_signed());
                assert!(dt.is_unsigned());
                assert!(dt.is_integer());
                assert!(dt.is_numeric());
            }
            DataType::BF16 | DataType::F16 | DataType::F32 | DataType::F64 => {
                assert!(!dt.is_signed());
                assert!(!dt.is_unsigned());
                assert!(!dt.is_integer());
                assert!(dt.is_float());
                assert!(dt.is_numeric());
            }
            DataType::Bool => {
                assert!(!dt.is_signed());
                assert!(!dt.is_unsigned());
                assert!(!dt.is_integer());
                assert!(!dt.is_float());
                assert!(!dt.is_numeric());
                assert!(dt.is_boolean());
            }
            DataType::Utf8 => {
                assert!(!dt.is_signed());
                assert!(!dt.is_unsigned());
                assert!(!dt.is_integer());
                assert!(!dt.is_float());
                assert!(!dt.is_numeric());
                assert!(dt.is_string());
            }
        }
    }
    matrix_test!(check_flags);
}

#[test]
fn test_boolean_and_string_methods() {
    assert!(DataType::Bool.is_boolean());
    assert!(!DataType::Utf8.is_boolean());
    assert!(DataType::Utf8.is_string());
    assert!(!DataType::Bool.is_string());
}
