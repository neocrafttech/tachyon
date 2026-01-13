use std::fmt::Debug;
use std::sync::Once;

use arrow::datatypes::{
    ArrowPrimitiveType, Float16Type, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use compute::operator::Operator;
use half::f16;
use tracing_subscriber;

static TRACING: Once = Once::new();

pub fn init_tracing() {
    TRACING.call_once(|| {
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::DEBUG)
            .with_test_writer()
            .try_init()
            .ok();
    });
}

pub trait ArrowMapper {
    type ArrowType: ArrowPrimitiveType;
}

macro_rules! arrow_mapper {
    ($t:ty, $arrow_type:ty) => {
        impl ArrowMapper for $t {
            type ArrowType = $arrow_type;
        }
    };
}

arrow_mapper!(i8, Int8Type);
arrow_mapper!(i16, Int16Type);
arrow_mapper!(i32, Int32Type);
arrow_mapper!(i64, Int64Type);
arrow_mapper!(u8, UInt8Type);
arrow_mapper!(u16, UInt16Type);
arrow_mapper!(u32, UInt32Type);
arrow_mapper!(u64, UInt64Type);
arrow_mapper!(f16, Float16Type);
arrow_mapper!(f32, Float32Type);
arrow_mapper!(f64, Float64Type);

macro_rules! impl_numeric_cast {
    ($target:ty, $($source:ty)*) => {
        $(
            impl CastTo<$target> for $source {
                fn cast(self) -> $target {
                    self as $target
                }
            }
        )*
    };
}

pub trait CastTo<T> {
    fn cast(self) -> T;
}

impl_numeric_cast!(f64, u8 u16 u32 u64 usize i8 i16 i32 i64 isize f32 f64);
impl_numeric_cast!(f32, u8 u16 u32 u64 usize i8 i16 i32 i64 isize f32);
impl_numeric_cast!(u64, u8 u16 u32 u64 i8 i16 i32);
impl_numeric_cast!(u32, u8 u16 u32 i8 i16);
impl_numeric_cast!(u16, u8 u16 i8);
impl_numeric_cast!(u8, u8);
impl_numeric_cast!(i64, i8 i16 i32 i64 isize u8 u16 u32);
impl_numeric_cast!(i32, i8 i16 i32 u8 u16);
impl_numeric_cast!(i16, i8 i16 u8);
impl_numeric_cast!(i8, i8);

impl CastTo<f64> for f16 {
    fn cast(self) -> f64 {
        f64::from(f32::from(self))
    }
}
impl CastTo<f32> for f16 {
    fn cast(self) -> f32 {
        f32::from(self)
    }
}

impl CastTo<f16> for f16 {
    fn cast(self) -> f16 {
        self
    }
}

impl CastTo<f16> for u8 {
    fn cast(self) -> f16 {
        f16::from_f32(self as f32)
    }
}
impl CastTo<f16> for u16 {
    fn cast(self) -> f16 {
        f16::from_f32(self as f32)
    }
}

impl CastTo<f16> for u32 {
    fn cast(self) -> f16 {
        f16::from_f32(self as f32)
    }
}

impl CastTo<f16> for u64 {
    fn cast(self) -> f16 {
        f16::from_f32(self as f32)
    }
}

impl CastTo<f16> for i8 {
    fn cast(self) -> f16 {
        f16::from_f32(self as f32)
    }
}

impl CastTo<f16> for i16 {
    fn cast(self) -> f16 {
        f16::from_f32(self as f32)
    }
}

impl CastTo<f16> for i32 {
    fn cast(self) -> f16 {
        f16::from_f32(self as f32)
    }
}

impl CastTo<f16> for i64 {
    fn cast(self) -> f16 {
        f16::from_f32(self as f32)
    }
}

pub trait TypeTestRange: Sized + Copy + Debug {
    fn test_range() -> (Self, Self);
}

macro_rules! impl_test_range {
    ($($t:ty)*) => {
        $(
            impl TypeTestRange for $t {
                fn test_range() -> (Self, Self) {
                    (<$t>::MIN, <$t>::MAX)
                }
            }
        )*
    };
}
impl_test_range!(u8 u16 u32 u64 usize i8 i16 i32 i64 isize f16);

impl TypeTestRange for f32 {
    fn test_range() -> (Self, Self) {
        (f32::MIN / 2.0, f32::MAX / 2.0)
    }
}

impl TypeTestRange for f64 {
    fn test_range() -> (Self, Self) {
        (f64::MIN / 2.0, f64::MAX / 2.0)
    }
}

#[macro_export]
macro_rules! random_num {
    ($min:expr, $max:expr) => {{
        use rand;
        use rand::Rng;
        let mut rng = rand::rng();
        let num: usize = rng.random_range($min..$max);
        num
    }};
}

#[macro_export]
macro_rules! random_vec {
    ($size:expr, $ty:ty, $min:expr, $max:expr) => {{
        use rand::Rng;
        let mut rng = rand::rng();
        (0..$size).map(|_| rng.random_range($min..$max)).collect::<Vec<$ty>>()
    }};
}

#[macro_export]
macro_rules! random_bit_vec {
    ($size:expr, $ty:ty) => {{
        use compute::bit_vector::BitVector;
        use rand;
        use rand::Rng;
        let mut rng = rand::rng();
        const BITS: usize = std::mem::size_of::<$ty>() * 8;
        let num_blocks = $size.div_ceil(BITS);
        let mut bits: Vec<$ty> = Vec::with_capacity(num_blocks);

        for _ in 0..(num_blocks.saturating_sub(1)) {
            let random_block: $ty = rng.random_range(0..=<$ty>::MAX);
            bits.push(random_block);
        }

        if num_blocks > 0 {
            let last_idx = num_blocks - 1;
            let total_used_bits = last_idx * BITS;
            let valid_bits_in_last_block = $size - total_used_bits;
            let last_block: $ty = rng.random_range(0..=<$ty>::MAX);

            if valid_bits_in_last_block < BITS {
                let low_bits_mask = !(<$ty>::MAX << valid_bits_in_last_block);
                bits.push(last_block & low_bits_mask);
            } else {
                bits.push(last_block);
            }
        }

        BitVector::new(bits, $size)
    }};
}

#[macro_export]
macro_rules! create_arrow_array {
    ($vec:expr, $bit_vec:expr, $native_type:ty) => {{
        use arrow::array::PrimitiveArray;
        let arrow_vec: Vec<Option<$native_type>> = $vec
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                if $bit_vec.is_valid(i) {
                    let y: $native_type = x.cast();
                    Some(y)
                } else {
                    None
                }
            })
            .collect();
        PrimitiveArray::<<$native_type as ArrowMapper>::ArrowType>::from(arrow_vec)
    }};
}

#[macro_export]
macro_rules! create_column {
    ($vec:expr, $bit_vec:expr, $name:expr, $data_type:expr) => {{
        use std::sync::Arc;

        use compute::column::{Column, VecArray};
        let arr = Arc::new(VecArray { data: $vec.clone(), datatype: $data_type });
        Column::new($name, arr, $bit_vec)
    }};
}

use arrow::array::{Array, BooleanArray};
use arrow::compute::cast;
use arrow::compute::kernels::cmp;
use arrow::datatypes::DataType;
use arrow::error::{ArrowError, Result};

pub fn compare_numeric_arrays(a: &dyn Array, b: &dyn Array, op: Operator) -> Result<BooleanArray> {
    let target_type = get_common_numeric_type(a.data_type(), b.data_type())?;
    println!("Target type: {:?}", target_type);
    let a_cast = cast(a, &target_type)?;
    let b_cast = cast(b, &target_type)?;

    match op {
        Operator::Eq => cmp::eq(&a_cast, &b_cast),
        Operator::NotEq => cmp::neq(&a_cast, &b_cast),
        Operator::Lt => cmp::lt(&a_cast, &b_cast),
        Operator::LtEq => cmp::lt_eq(&a_cast, &b_cast),
        Operator::Gt => cmp::gt(&a_cast, &b_cast),
        Operator::GtEq => cmp::gt_eq(&a_cast, &b_cast),
        _ => Err(ArrowError::NotYetImplemented(format!(
            "Comparison operator {:?} not supported",
            op
        ))),
    }
}

fn get_common_numeric_type(a: &DataType, b: &DataType) -> Result<DataType> {
    use DataType::*;

    match (a, b) {
        // If either is float, prioritize float types
        (Float64, _) | (_, Float64) => Ok(Float64),
        (Float32, _) | (_, Float32) => Ok(Float32),
        (Float16, _) | (_, Float16) => Ok(Float16),

        (a_int, b_int) if is_integer(a_int) && is_integer(b_int) => {
            get_widest_integer_type(a_int, b_int)
        }

        _ => Err(ArrowError::ComputeError(format!("Cannot compare types {:?} and {:?}", a, b))),
    }
}

fn is_integer(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
    )
}

fn get_widest_integer_type(a: &DataType, b: &DataType) -> Result<DataType> {
    use DataType::*;

    // Check if we're mixing signed and unsigned
    let a_signed = is_signed_int(a);
    let b_signed = is_signed_int(b);

    // If mixing signed/unsigned, use Float64 to avoid comparison issues
    if a_signed != b_signed {
        return Ok(Float64);
    }

    // Get the bit width for each type
    let a_bits = get_int_bit_width(a);
    let b_bits = get_int_bit_width(b);

    let max_bits = a_bits.max(b_bits);

    if a_signed {
        Ok(match max_bits {
            8 => Int8,
            16 => Int16,
            32 => Int32,
            _ => Int64,
        })
    } else {
        Ok(match max_bits {
            8 => UInt8,
            16 => UInt16,
            32 => UInt32,
            _ => UInt64,
        })
    }
}

fn is_signed_int(dt: &DataType) -> bool {
    matches!(dt, DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64)
}

fn get_int_bit_width(dt: &DataType) -> u32 {
    match dt {
        DataType::Int8 | DataType::UInt8 => 8,
        DataType::Int16 | DataType::UInt16 => 16,
        DataType::Int32 | DataType::UInt32 => 32,
        DataType::Int64 | DataType::UInt64 => 64,
        _ => 64,
    }
}
