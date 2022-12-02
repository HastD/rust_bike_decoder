use crate::parameters::*;
use crate::vectors::{DenseVector, SparseErrorVector, ErrorVector};
use crate::keys::Key;
use std::{cmp, fmt, ops::Add};

// Note: syndromes are padded out to 2*SIZE_AVX so they can be passed to
// code in decoder.rs that uses AVX2 instructions.
// However, only bits up to BLOCK_LENGTH are ever used outside of that.
#[derive(Debug, Clone)]
pub struct Syndrome(DenseVector<DOUBLE_SIZE_AVX>);

impl Syndrome {
    pub fn zero() -> Self {
        Self(DenseVector::from([0u8; DOUBLE_SIZE_AVX]))
    }

    pub fn from(list: [u8; BLOCK_LENGTH]) -> Self {
        let mut v = [0; DOUBLE_SIZE_AVX];
        v[..BLOCK_LENGTH].copy_from_slice(&list);
        Self(DenseVector::from(v))
    }

    pub fn from_sparse(key: &Key, err: &SparseErrorVector) -> Self {
        let mut s = [0u8; BLOCK_LENGTH];
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
        let mut s = [0u8; BLOCK_LENGTH];
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

    #[inline]
    pub fn get(&self, i: usize) -> u8 {
        self.0.get(i)
    }

    #[inline]
    pub fn contents(&self) -> &[u8] {
        &self.0.contents()[..BLOCK_LENGTH]
    }

    #[inline]
    pub fn contents_with_buffer(&self) -> &[u8] {
        self.0.contents()
    }

    #[inline]
    pub fn flip(&mut self, i: usize) {
        self.0.flip(i);
    }

    #[inline]
    pub fn set_zero(&mut self, i: usize) {
        self.0.set_zero(i);
    }

    #[inline]
    pub fn set_one(&mut self, i: usize) {
        self.0.set_one(i);
    }

    #[inline]
    pub fn set_all_zero(&mut self) {
        self.0.set_all_zero();
    }

    pub fn hamming_weight(&self) -> usize {
        let mut wt = 0;
        for i in 0..BLOCK_LENGTH {
            if self.get(i) == 1 {
                wt += 1;
            }
        }
        wt
    }

    pub fn duplicate_up_to(&mut self, length: usize) {
        self.0.duplicate_up_to(length);
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

impl Add for Syndrome {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self(self.0.add_mod2(other.0))
    }
}

impl cmp::PartialEq for Syndrome {
    // Equality ignores the extra buffer space
    fn eq(&self, other: &Self) -> bool {
        self.contents() == other.contents()
    }
}

impl cmp::Eq for Syndrome { }

impl fmt::Display for Syndrome {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut str_vec = Vec::new();
        for bit in self.contents().iter() {
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
