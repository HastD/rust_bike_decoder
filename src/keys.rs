use crate::vectors::{Index, SparseVector, InvalidSupport};
use crate::parameters::*;
use rand::Rng;
use serde::{Serialize, Deserialize};
use std::fmt;

pub type CyclicBlock = SparseVector<BLOCK_WEIGHT, BLOCK_LENGTH>;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq, Eq)]
pub struct Key {
    h0: CyclicBlock,
    h1: CyclicBlock,
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
    pub fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self {
            h0: CyclicBlock::random(rng),
            h1: CyclicBlock::random(rng)
        }
    }

    pub fn is_weak(&self, threshold: usize) -> bool {
        // type I or II weak key
        self.h0.shifts_above_threshold(threshold)
        || self.h1.shifts_above_threshold(threshold)
        // type III weak key
        || self.h0.max_shifted_product_weight_geq(&self.h1, threshold)
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
