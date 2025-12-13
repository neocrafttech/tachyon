/*
 * Copyright (c) NeoCraft Technologies.
 *
 * This source code is licensed under the Apache License, Version 2.0,
 * as found in the LICENSE file in the root directory of this source tree.
 */

pub trait BitBlock:
    Copy
    + Sized
    + PartialEq
    + std::ops::BitAnd<Output = Self>
    + std::ops::BitOr<Output = Self>
    + std::ops::BitOrAssign
    + std::ops::BitAndAssign
    + std::ops::Not<Output = Self>
    + std::ops::Shl<usize, Output = Self>
{
    const BITS: usize = std::mem::size_of::<Self>() * 8;

    const MAX: Self;

    const ONE: Self;

    const ZERO: Self;

    const C_TYPE: &'static str;

    /// Count the number of 1 bits
    fn count_ones(self) -> u32;
}

macro_rules! bit_block {
    ($ty : ty, $c_type: expr) => {
        impl BitBlock for $ty {
            const MAX: Self = <$ty>::MAX;
            const ONE: Self = 1;
            const ZERO: Self = 0;
            const C_TYPE: &'static str = $c_type;
            #[inline]
            fn count_ones(self) -> u32 {
                self.count_ones()
            }
        }
    };
}

bit_block!(u8, "uint8_t");
bit_block!(u16, "uint16_t");
bit_block!(u32, "uint32_t");
bit_block!(u64, "uint64_t");

#[derive(Debug, Clone)]
pub struct BitVector<T: BitBlock> {
    bits: Vec<T>,
    num_bits: usize,
}

impl<T: BitBlock> BitVector<T> {
    pub fn new(bits: Vec<T>, num_bits: usize) -> Self {
        Self { bits, num_bits }
    }
    pub fn new_all_null(num_bits: usize) -> Self {
        Self { bits: vec![T::ZERO; Self::num_blocks(num_bits)], num_bits }
    }

    pub fn new_all_valid(num_bits: usize) -> Self {
        let num_blocks = Self::num_blocks(num_bits);
        let mut bits = vec![T::MAX; num_blocks];

        if !num_bits.is_multiple_of(T::BITS) {
            let last_idx = num_blocks - 1;
            let valid_bits = num_bits % T::BITS;

            let low_bits_mask = !(T::MAX << valid_bits);

            bits[last_idx] &= low_bits_mask;
        }

        Self { bits, num_bits }
    }

    #[inline(always)]
    fn num_blocks(num_bits: usize) -> usize {
        num_bits.div_ceil(T::BITS)
    }

    #[inline]
    fn index_mask(idx: usize) -> (usize, T) {
        let unit = idx / T::BITS;
        let offset = idx % T::BITS;
        let mask = T::ONE << offset;

        (unit, mask)
    }

    pub fn set_valid(&mut self, idx: usize) {
        if idx >= self.num_bits {
            panic!("Index out of bounds: set_valid");
        }
        let (unit, mask) = Self::index_mask(idx);
        self.bits[unit] |= mask;
    }

    pub fn set_null(&mut self, idx: usize) {
        if idx >= self.num_bits {
            panic!("Index out of bounds: set_null");
        }
        let (unit, mask) = Self::index_mask(idx);
        self.bits[unit] &= !mask
    }

    pub fn is_valid(&self, idx: usize) -> bool {
        if idx >= self.num_bits {
            panic!("Index out of bounds: is_valid");
        }
        let (unit, mask) = Self::index_mask(idx);
        (self.bits[unit] & mask) != T::ZERO
    }

    pub fn count_valid(&self) -> usize {
        self.bits.iter().map(|&block| block.count_ones() as usize).sum()
    }

    pub fn count_null(&self) -> usize {
        self.num_bits - self.count_valid()
    }

    pub fn is_null(&self, idx: usize) -> bool {
        !self.is_valid(idx)
    }

    pub fn as_slice(&self) -> &[T] {
        self.bits.as_slice()
    }

    pub fn as_vec(&self) -> &Vec<T> {
        &self.bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_valid_u8() {
        let vec = vec![127u8, 5];
        let bv = BitVector::<u8>::new(vec, 11);

        assert!(bv.is_valid(0));
        assert!(bv.is_valid(1));
        assert!(bv.is_valid(2));
        assert!(bv.is_valid(3));
        assert!(bv.is_valid(4));
        assert!(bv.is_valid(5));
        assert!(bv.is_valid(6));
        assert!(!bv.is_valid(7));
        assert!(bv.is_valid(8));
        assert!(!bv.is_valid(9));
        assert!(bv.is_valid(10));

        let vec = vec![255u8, 0];
        let bv = BitVector::<u8>::new(vec, 9);
        assert!(bv.is_valid(0));
        assert!(bv.is_valid(7));
        assert!(!bv.is_valid(8));
    }

    #[test]
    fn test_set_and_get_valid_u8() {
        let mut bv = BitVector::<u8>::new_all_null(16);

        for i in 0..16 {
            assert!(!bv.is_valid(i));
        }

        bv.set_valid(3);
        bv.set_valid(7);

        assert!(bv.is_valid(3));
        assert!(bv.is_valid(7));

        assert!(!bv.is_valid(0));
        assert!(!bv.is_valid(15));
    }

    #[test]
    fn test_set_and_get_valid_u64() {
        let mut bv = BitVector::<u64>::new_all_null(128);

        for i in 0..128 {
            assert!(!bv.is_valid(i));
        }

        bv.set_valid(10);
        bv.set_valid(63);
        bv.set_valid(64);
        bv.set_valid(127);

        assert!(bv.is_valid(10));
        assert!(bv.is_valid(63));
        assert!(bv.is_valid(64));
        assert!(bv.is_valid(127));
        assert!(!bv.is_valid(5));
        assert!(!bv.is_valid(100));
    }

    #[test]
    fn test_set_null() {
        let mut bv = BitVector::<u8>::new_all_null(16);

        bv.set_valid(5);
        assert!(bv.is_valid(5));

        bv.set_null(5);
        assert!(!bv.is_valid(5));
    }

    #[test]
    fn test_multiple_bits_updates() {
        let mut bv = BitVector::<u64>::new_all_null(100);

        bv.set_valid(0);
        bv.set_valid(50);
        bv.set_valid(99);

        assert!(bv.is_valid(0));
        assert!(bv.is_valid(50));
        assert!(bv.is_valid(99));

        bv.set_null(50);
        assert!(!bv.is_valid(50));

        assert!(bv.is_valid(0));
        assert!(bv.is_valid(99));
    }

    #[test]
    fn test_boundary_bits() {
        let mut bv = BitVector::<u64>::new_all_null(64);

        bv.set_valid(0);
        bv.set_valid(63);

        assert!(bv.is_valid(0));
        assert!(bv.is_valid(63));

        bv.set_null(63);
        assert!(!bv.is_valid(63));
    }

    #[test]
    fn test_count_valid() {
        let mut bv = BitVector::<u8>::new_all_null(16);

        bv.set_valid(5);
        bv.set_valid(15);

        assert_eq!(bv.count_valid(), 2);
    }

    #[test]
    fn test_count_null() {
        let mut bv = BitVector::<u8>::new_all_null(11);

        bv.set_valid(5);
        bv.set_valid(10);

        assert_eq!(bv.count_null(), 9);
    }
}
