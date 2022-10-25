use crate::constants::*;
use rand::{Rng, distributions::{Distribution, Uniform}};
use std::cmp;
use std::fmt;

pub type Index = u32;

pub type SparseErrorVector = SparseVector<ERROR_WEIGHT, ROW_LENGTH>;
pub type ErrorVector = DenseVector<ROW_LENGTH>;

// Sparse vector of fixed weight and length over GF(2)
#[derive(Clone)]
pub struct SparseVector<const WEIGHT: usize, const LENGTH: usize>([Index; WEIGHT]);

impl<const WEIGHT: usize, const LENGTH: usize> SparseVector<WEIGHT, LENGTH> {
    #[inline]
    pub fn weight(&self) -> usize {
        WEIGHT
    }

    #[inline]
    pub fn length(&self) -> Index {
        LENGTH as Index
    }

    #[inline]
    pub fn get(&self, i: usize) -> Index {
        self.0[i]
    }

    #[inline]
    pub fn support(&self) -> [Index; WEIGHT] {
        self.0
    }

    #[inline]
    pub fn contains(&self, index: &Index) -> bool {
        self.0.contains(index)
    }

    pub fn random<R: Rng + ?Sized>(rng: &mut R, dist: &Uniform<Index>) -> Self {
        let mut supp = [0 as Index; WEIGHT];
        let mut ctr = 0;
        while ctr < WEIGHT {
            // Randomly generate element in the appropriate range
            supp[ctr] = dist.sample(rng);
            let mut is_new = 1;
            for i in 0..ctr {
                if supp[i] == supp[ctr] {
                    is_new = 0;
                    break;
                }
            }
            ctr += is_new;
        }
        Self(supp)
    }

    pub fn dense(&self) -> DenseVector<LENGTH> {
        let mut v = DenseVector::zero();
        for i in self.support() {
            v.flip(i as usize);
        }
        v
    }

    pub fn relative_shifts(&self, other: &Self) -> [[Index; WEIGHT]; WEIGHT] {
        let length = self.length();
        let mut shifts = [[0 as Index; WEIGHT]; WEIGHT];
        for i in 0..WEIGHT {
            let self_i = self.get(i);
            let length_plus_self_i = length + self_i;
            for j in 0..WEIGHT {
                let other_j = other.get(j);
                shifts[i][j] = if self_i < other_j {
                    length_plus_self_i - other_j
                } else {
                    self_i - other_j
                };  // this equals (self_i - other_j) % length
                    // since 0 <= self_i, other_j < N.
            }
        }
        shifts
    }

    pub fn max_shifted_product_weight_geq(&self, other: &Self, threshold: u8) -> bool {
        let shifts = self.relative_shifts(other);
        let mut shift_counts = [0; LENGTH];
        for i in 0..WEIGHT {
            for j in 0..WEIGHT {
                let count = &mut shift_counts[shifts[i][j] as usize];
                *count += 1;
                if *count >= threshold {
                    return true;
                }
            }
        }
        false
    }

    pub fn shifts_above_threshold(&self, threshold: u8) -> bool {
        let length = self.length();
        let mut shift_counts = [0; LENGTH];
        for i in 0..WEIGHT {
            for j in i+1..WEIGHT {
                let diff = self.get(j).abs_diff(self.get(i));
                let count = &mut shift_counts[cmp::min(diff, length - diff) as usize];
                *count += 1;
                if *count >= threshold {
                    return true;
                }
            }
        }
        false
    }
}

impl<const W: usize, const L: usize> cmp::PartialEq for SparseVector<W, L> {
    // Supports may or may not be sorted, so we have to sort to test equality
    fn eq(&self, other: &Self) -> bool {
        let mut supp_self = self.support();
        let mut supp_other = other.support();
        supp_self.sort();
        supp_other.sort();
        supp_self == supp_other
    }
}

impl<const W: usize, const L: usize> cmp::Eq for SparseVector<W, L> { }

impl<const W: usize, const L: usize> fmt::Display for SparseVector<W, L> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut supp = self.support();
        supp.sort();
        let mut str_supp = Vec::new();
        for index in supp {
            str_supp.push(index.to_string());
        }
        write!(f, "[{}]", str_supp.join(", "))
    }
}

// Dense vectors of fixed length over GF(2)
#[derive(Clone, cmp::PartialEq, cmp::Eq)]
pub struct DenseVector<const LENGTH: usize>([u8; LENGTH]);

impl<const LENGTH: usize> DenseVector<LENGTH> {
    pub fn zero() -> Self {
        Self([0u8; LENGTH])
    }

    pub fn from(list: [u8; LENGTH]) -> Self {
        Self(list)
    }

    #[inline]
    pub fn get(&self, i: usize) -> u8 {
        self.0[i]
    }

    #[inline]
    pub fn flip(&mut self, i: usize) {
        self.0[i] ^= 1;
    }

    #[inline]
    pub fn set_zero(&mut self, i: usize) {
        self.0[i] = 0;
    }

    #[inline]
    pub fn set_one(&mut self, i: usize) {
        self.0[i] = 1;
    }

    #[inline]
    pub fn set_all_zero(&mut self) {
        self.0.iter_mut().for_each(|entry| *entry = 0);
    }

    pub fn support(&self) -> Vec<Index> {
        let mut supp: Vec<Index> = Vec::new();
        for i in 0..LENGTH {
            if self.0[i] == 1 {
                supp.push(i as Index);
            }
        }
        supp
    }

    pub fn hamming_weight(&self) -> u32 {
        let mut count = 0;
        for i in self.0 {
            if i == 1 {
                count += 1;
            }
        }
        count
    }
}
