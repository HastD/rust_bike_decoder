use crate::parameters::*;
use crate::vectors::{DenseVector, SparseErrorVector, ErrorVector};
use crate::keys::Key;

// Note: syndromes are padded out to 2*SIZE_AVX so they can be passed to
// code in decoder.rs that uses AVX2 instructions.
// However, only bits up to BLOCK_LENGTH are ever used outside of that.
pub type Syndrome = DenseVector<DOUBLE_SIZE_AVX>;

impl Syndrome {
    pub fn hamming_weight(&self) -> u32 {
        let mut wt = 0;
        for i in 0..BLOCK_LENGTH {
            if self.get(i) == 1 {
                wt += 1;
            }
        }
        wt
    }

    pub fn from_sparse(key: &Key, err: &SparseErrorVector) -> Self {
        let mut s = [0u8; DOUBLE_SIZE_AVX];
        for &i in err.support() {
            if i < BLOCK_LENGTH as u32 {
                for &j in key.h0().support() {
                    s[(i + j) as usize % BLOCK_LENGTH] ^= 1;
                }
            } else {
                for &j in key.h1().support() {
                    s[(i + j) as usize % BLOCK_LENGTH] ^= 1;
                }
            }
        }
        Self::from(s)
    }

    pub fn recompute_from(&mut self, key: &Key, err: &ErrorVector) {
        self.set_all_zero();
        for i in 0..BLOCK_LENGTH {
            if err.get(i) == 1 {
                for &j in key.h0().support() {
                    self.flip((i + j as usize) % BLOCK_LENGTH);
                }
            }
        }
        for i in 0..BLOCK_LENGTH {
            if err.get(BLOCK_LENGTH + i) == 1 {
                for &j in key.h1().support() {
                    self.flip((i + j as usize) % BLOCK_LENGTH);
                }
            }
        }
    }

    pub fn recompute_flipped_bit(&mut self, key: &Key, block_idx: usize, pos: usize) {
        if block_idx == 0 {
            for &j in key.h0().support() {
                self.flip((pos + j as usize) % BLOCK_LENGTH);
            }
        } else {
            for &j in key.h1().support() {
                self.flip((pos + j as usize) % BLOCK_LENGTH);
            }
        }
    }
}
