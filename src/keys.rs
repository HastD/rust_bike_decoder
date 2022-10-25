use crate::vectors::{Index, SparseVector};
use crate::constants::*;
use rand::{Rng, distributions::Uniform};

pub type CyclicBlock = SparseVector<BLOCK_WEIGHT, BLOCK_LENGTH>;

pub struct Key {
    h0: CyclicBlock,
    h1: CyclicBlock,
}

impl Key {
    pub fn from(h0: CyclicBlock, h1: CyclicBlock) -> Self {
        Self {
            h0,
            h1
        }
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
    pub fn h0_supp(&self) -> [Index; BLOCK_WEIGHT] {
        self.h0.support()
    }

    #[inline]
    pub fn h1_supp(&self) -> [Index; BLOCK_WEIGHT] {
        self.h1.support()
    }

    pub fn block_weight(&self) -> usize {
        self.h0.weight()
    }

    pub fn block_length(&self) -> Index {
        self.h0.length()
    }

    pub fn random<R: Rng + ?Sized>(rng: &mut R, dist: &Uniform<Index>) -> Self {
        Self {
            h0: CyclicBlock::random(rng, dist),
            h1: CyclicBlock::random(rng, dist)
        }
    }

    pub fn is_weak(&self) -> bool {
        // type I or II weak key
        self.h0.shifts_above_threshold(WEAK_KEY_THRESHOLD)
        || self.h1.shifts_above_threshold(WEAK_KEY_THRESHOLD)
        // type III weak key
        || self.h0.max_shifted_product_weight_geq(&self.h1, WEAK_KEY_THRESHOLD)
    }

    pub fn random_non_weak<R: Rng + ?Sized>(rng: &mut R, dist: &Uniform<Index>) -> Self {
        loop {
            let key = Self::random(rng, dist);
            if !key.is_weak() {
                return key;
            }
        }
    }
}
