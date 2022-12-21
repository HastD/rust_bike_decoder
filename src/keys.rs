use crate::vectors::{Index, SparseVector, InvalidSupport};
use crate::parameters::*;
use rand::Rng;
use serde::{Serialize, Deserialize};
use std::{convert::TryFrom, fmt};
use thiserror::Error;

pub type CyclicBlock = SparseVector<BLOCK_WEIGHT, BLOCK_LENGTH>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct Key {
    h0: CyclicBlock,
    h1: CyclicBlock,
}

impl From<(CyclicBlock, CyclicBlock)> for Key {
    fn from((h0, h1): (CyclicBlock, CyclicBlock)) -> Self {
        Self { h0, h1 }
    }
}

impl TryFrom<([Index; BLOCK_WEIGHT], [Index; BLOCK_WEIGHT])> for Key {
    type Error = InvalidSupport;

    fn try_from((h0_supp, h1_supp): ([Index; BLOCK_WEIGHT], [Index; BLOCK_WEIGHT])) -> Result<Self, Self::Error> {
        Key::from_support(h0_supp, h1_supp)
    }
}

impl Key {
    pub fn from_support(h0_supp: [Index; BLOCK_WEIGHT], h1_supp: [Index; BLOCK_WEIGHT]) -> Result<Self, InvalidSupport> {
        Ok(Self {
            h0: CyclicBlock::from_support(h0_supp)?,
            h1: CyclicBlock::from_support(h1_supp)?
        })
    }

    #[inline]
    pub fn h0(&self) -> &CyclicBlock {
        &self.h0
    }

    #[inline]
    pub fn h1(&self) -> &CyclicBlock {
        &self.h1
    }

    #[inline]
    pub fn take_blocks(self) -> (CyclicBlock, CyclicBlock) {
        (self.h0, self.h1)
    }

    #[inline]
    pub fn h0_supp(&self) -> &[Index; BLOCK_WEIGHT] {
        self.h0.support()
    }

    #[inline]
    pub fn h1_supp(&self) -> &[Index; BLOCK_WEIGHT] {
        self.h1.support()
    }

    #[inline]
    pub fn block_weight(&self) -> usize {
        self.h0.weight()
    }

    #[inline]
    pub fn block_length(&self) -> Index {
        self.h0.length()
    }

    #[inline]
    pub fn validate(&self) -> Result<(), InvalidSupport> {
        self.h0.validate()?;
        self.h1.validate()?;
        Ok(())
    }

    #[inline]
    pub fn sort(&mut self) {
        self.h0.sort();
        self.h1.sort();
    }

    #[inline]
    pub fn sorted(mut self) -> Self {
        self.sort();
        self
    }

    pub fn matches_filter(&self, key_filter: KeyFilter) -> bool {
        match key_filter {
            KeyFilter::Any => true,
            KeyFilter::NonWeak(threshold) => !self.is_weak(threshold),
            KeyFilter::Weak(weak_type, threshold) => match weak_type {
                WeakType::Type1 => self.is_weak_type1(threshold),
                WeakType::Type2 => self.is_weak_type2(threshold),
                WeakType::Type3 => self.is_weak_type3(threshold),
            }
        }
    }

    pub fn is_weak_type1(&self, _threshold: usize) -> bool {
        unimplemented!();
    }

    pub fn is_weak_type2(&self, threshold: usize) -> bool {
        self.h0.shifts_above_threshold(threshold) || self.h1.shifts_above_threshold(threshold)
    }

    pub fn is_weak_type3(&self, threshold: usize) -> bool {
        self.h0.max_shifted_product_weight_geq(&self.h1, threshold)
    }

    pub fn is_weak(&self, threshold: usize) -> bool {
        self.is_weak_type2(threshold) || self.is_weak_type3(threshold)
    }

    pub fn random_filtered<R: Rng + ?Sized>(key_filter: KeyFilter, rng: &mut R) -> Self {
        match key_filter {
            KeyFilter::Any => Self::random(rng),
            KeyFilter::NonWeak(threshold) => Self::random_non_weak(threshold, rng),
            KeyFilter::Weak(weak_type, threshold) => match weak_type {
                WeakType::Type1 => Self::random_weak_type1(threshold, rng),
                WeakType::Type2 => Self::random_weak_type2(threshold, rng),
                WeakType::Type3 => Self::random_weak_type3(threshold, rng),
            },
        }
    }

    #[inline]
    pub fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self {
            h0: CyclicBlock::random(rng),
            h1: CyclicBlock::random(rng)
        }
    }

    pub fn random_non_weak<R>(threshold: usize, rng: &mut R) -> Self
        where R: Rng + ?Sized
    {
        loop {
            let h0 = CyclicBlock::random_non_weak_type2(threshold, rng);
            let h1 = CyclicBlock::random_non_weak_type2(threshold, rng);
            if !h0.max_shifted_product_weight_geq(&h1, threshold) {
                return Self { h0, h1 };
            }
        }
    }

    pub fn random_weak_type1<R>(thresh: usize, rng: &mut R) -> Self
        where R: Rng + ?Sized
    {
        let random_block = CyclicBlock::random(rng);
        let weak_block = CyclicBlock::random_weak_type1(thresh, rng);
        if rng.gen::<bool>() {
            Self { h0: weak_block, h1: random_block }
        } else {
            Self { h0: random_block, h1: weak_block }
        }
    }

    pub fn random_weak_type2<R>(thresh: usize, rng: &mut R) -> Self
        where R: Rng + ?Sized
    {
        let random_block = CyclicBlock::random(rng);
        let weak_block = CyclicBlock::random_weak_type2(thresh, rng);
        if rng.gen::<bool>() {
            Self { h0: weak_block, h1: random_block }
        } else {
            Self { h0: random_block, h1: weak_block }
        }
    }

    pub fn random_weak_type3<R>(thresh: usize, rng: &mut R) -> Self
        where R: Rng + ?Sized
    {
        let (h0, h1) = CyclicBlock::random_weak_type3(thresh, rng);
        Self { h0, h1 }
    }
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{h0: {}, h1: {}}}", self.h0(), self.h1())
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum WeakType {
    Type1, Type2, Type3
}

impl WeakType {
    pub fn number(&self) -> u8 {
        match self {
            Self::Type1 => 1,
            Self::Type2 => 2,
            Self::Type3 => 3,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum KeyFilter {
    Any,
    NonWeak(usize),
    Weak(WeakType, usize),
}

impl Default for KeyFilter {
    #[inline]
    fn default() -> Self {
        Self::Any
    }
}

impl TryFrom<(i8, usize)> for KeyFilter {
    type Error = KeyFilterError;

    fn try_from((filter, threshold): (i8, usize)) -> Result<Self, KeyFilterError> {
        if filter != 0 && threshold < 2 {
            return Err(KeyFilterError::InvalidThreshold);
        } else if threshold >= BLOCK_WEIGHT {
            // Thresholds >= BLOCK_WEIGHT are tautological and impose no conditions
            return Ok(Self::Any);
        }
        match filter {
            0 => Ok(Self::Any),
            -1 => Ok(Self::NonWeak(threshold)),
            1 => Ok(Self::Weak(WeakType::Type1, threshold)),
            2 => Ok(Self::Weak(WeakType::Type2, threshold)),
            3 => Ok(Self::Weak(WeakType::Type3, threshold)),
            _ => Err(KeyFilterError::InvalidFilter)
        }
    }
}

#[derive(Clone, Copy, Debug, Error)]
pub enum KeyFilterError {
    #[error("weak key filter must be in {{-1, 0, 1, 2, 3}}")]
    InvalidFilter,
    #[error("weak key threshold must be >= 2")]
    InvalidThreshold,
}

#[cfg(test)]
mod tests {
    use super::*;
    const TRIALS: usize = 1000;

    #[test]
    fn non_weak_keys() {
        let mut rng = rand::thread_rng();
        let weak_key_threshold = 3;
        for _ in 0..TRIALS {
            let key = Key::random_non_weak(weak_key_threshold, &mut rng);
            assert!(!key.is_weak(weak_key_threshold), "Non-weak key was actually weak: {:?}", key);
        }
    }

    #[test]
    fn weak_keys_type1() {
        let mut rng = rand::thread_rng();
        let weak_key_threshold = 7;
        for _ in 0 .. TRIALS {
            let key = Key::random_weak_type1(weak_key_threshold, &mut rng);
            assert!(key.is_weak(weak_key_threshold), "Type 1 weak key was not actually weak: {:?}", key);
        }
    }

    #[test]
    fn weak_keys_type2() {
        let mut rng = rand::thread_rng();
        let weak_key_threshold = 7;
        for _ in 0 .. TRIALS {
            let key = Key::random_weak_type2(weak_key_threshold, &mut rng);
            assert!(key.is_weak(weak_key_threshold), "Type 2 weak key was not actually weak: {:?}", key);
        }
    }

    #[test]
    fn weak_keys_type3() {
        let mut rng = rand::thread_rng();
        let weak_key_threshold = 7;
        for _ in 0 .. TRIALS {
            let key = Key::random_weak_type3(weak_key_threshold, &mut rng);
            assert!(key.is_weak(weak_key_threshold), "Type 3 weak key was not actually weak: {:?}", key);
        }
    }
}
