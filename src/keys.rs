use crate::vectors::{Index, SparseVector, DenseVector};
use crate::constants::*;
use rand::Rng;

type CyclicBlock = SparseVector<BLOCK_WEIGHT, BLOCK_LENGTH>;

pub struct Key {
    pub h0: CyclicBlock,
    pub h1: CyclicBlock,
}
impl Key {
    pub fn from(h0: CyclicBlock, h1: CyclicBlock) -> Self {
        Self {
            h0,
            h1
        }
    }

    pub fn block_weight(&self) -> usize {
        self.h0.weight()
    }

    pub fn block_length(&self) -> Index {
        self.h0.length()
    }

    pub fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Self {
            h0: CyclicBlock::random(rng),
            h1: CyclicBlock::random(rng)
        }
    }

    pub fn is_weak(&self) -> bool {
        self.h0.shifts_above_threshold(WEAK_KEY_THRESHOLD) // type I or II weak key
        || self.h1.shifts_above_threshold(WEAK_KEY_THRESHOLD) // type I or II weak key
        || self.h0.max_shifted_product_weight_geq(&self.h1, WEAK_KEY_THRESHOLD) // type III weak key
    }

    pub fn random_non_weak<R: Rng + ?Sized>(rng: &mut R) -> Self {
        loop {
            let key = Self::random(rng);
            if !key.is_weak() {
                return key;
            }
        }
    }
}

pub type ErrorVector = SparseVector<ERROR_WEIGHT, ROW_LENGTH>;
pub type Syndrome = DenseVector<BLOCK_LENGTH>;
pub type DecoderOutput = DenseVector<ROW_LENGTH>;

pub fn syndrome(key: &Key, err: &ErrorVector) -> Syndrome {
    let mut s = Syndrome::new();
    for i in err.support() {
        let i = i as usize;
        if i < BLOCK_LENGTH {
            for j in key.h0.support() {
                s.flip((i + j as usize) % BLOCK_LENGTH);
            }
        } else {
            for j in key.h1.support() {
                s.flip((i + j as usize) % BLOCK_LENGTH);
            }
        }
    }
    s
}
