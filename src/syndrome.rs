use crate::parameters::*;
use crate::vectors::{DenseVector, SparseErrorVector, ErrorVector};
use crate::keys::Key;
use std::fmt;

// Note: syndromes are padded out to 2*SIZE_AVX so they can be passed to
// code in decoder.rs that uses AVX2 instructions.
// However, only bits up to BLOCK_LENGTH are ever used outside of that.
pub type Syndrome = DenseVector<DOUBLE_SIZE_AVX>;

impl Syndrome {
    pub fn hamming_weight(&self) -> usize {
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

    pub fn from_dense(key: &Key, err: &ErrorVector) -> Self {
        let mut s = [0u8; DOUBLE_SIZE_AVX];
        for i in 0..BLOCK_LENGTH {
            if err.get(i) == 1 {
                for &j in key.h0().support() {
                    s[(i + j as usize) % BLOCK_LENGTH] ^= 1;
                }
            }
        }
        for i in 0..BLOCK_LENGTH {
            if err.get(BLOCK_LENGTH + i) == 1 {
                for &j in key.h1().support() {
                    s[(i + j as usize) % BLOCK_LENGTH] ^= 1;
                }
            }
        }
        Self::from(s)
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

impl fmt::Display for Syndrome {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut str_vec = Vec::new();
        for bit in self.contents()[..BLOCK_LENGTH].iter() {
            str_vec.push(bit.to_string());
        }
        write!(f, "[{}]", str_vec.join(", "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn syndrome_weight() {
        let mut syn = Syndrome::zero();
        let supp = [2, 3, 5, 7, 11, 13, 17, 19, BLOCK_LENGTH, BLOCK_LENGTH + 4];
        for idx in supp {
            syn.flip(idx);
        }
        assert_eq!(syn.hamming_weight(), supp.len() - 2);
    }
}
