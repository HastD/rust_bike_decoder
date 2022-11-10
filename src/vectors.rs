use crate::parameters::*;
use rand::{Rng, distributions::{Distribution, Uniform}};
use serde::{Serialize, Deserialize};
use std::{cmp, fmt};
use thiserror::Error;

pub type Index = u32;

pub type SparseErrorVector = SparseVector<ERROR_WEIGHT, ROW_LENGTH>;
pub type ErrorVector = DenseVector<ROW_LENGTH>;

#[derive(Error, Debug)]
pub struct InvalidSupport(String);

impl fmt::Display for InvalidSupport {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "invalid support for sparse vector: {}", self.0)
    }
}

// Sparse vector of fixed weight and length over GF(2)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVector<const WEIGHT: usize, const LENGTH: usize>(
    #[serde(with = "serde_arrays")]
    [Index; WEIGHT]
);

impl<const WEIGHT: usize, const LENGTH: usize> SparseVector<WEIGHT, LENGTH> {
    // Ensure that the support represents a valid vector of the specified weight and length
    pub fn validate(&self) -> Result<(), InvalidSupport> {
        for idx in self.0 {
            if idx >= self.length() {
                return Err(InvalidSupport(format!("support indices must be in range 0..{}", LENGTH)));
            }
        }
        for i in 0..WEIGHT {
            for j in (i+1)..WEIGHT {
                if self.get(i) == self.get(j) {
                    return Err(InvalidSupport(String::from("support indices must all be distinct")));
                }
            }
        }
        Ok(())
    }

    pub fn from_support(supp: [Index; WEIGHT]) -> Result<Self, InvalidSupport> {
        let v = Self(supp);
        v.validate()?;
        Ok(v)
    }

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
    pub fn support(&self) -> &[Index; WEIGHT] {
        &self.0
    }

    #[inline]
    pub fn contains(&self, index: &Index) -> bool {
        self.0.contains(index)
    }

    pub fn random<R>(rng: &mut R) -> Self
        where R: Rng + ?Sized
    {
        let mut supp = [0 as Index; WEIGHT];
        let mut ctr = 0;
        let dist = Uniform::new(0, LENGTH as Index);
        'outer: while ctr < WEIGHT {
            // Randomly generate element in the appropriate range
            supp[ctr] = dist.sample(rng);
            for i in 0..ctr {
                if supp[i] == supp[ctr] {
                    continue 'outer;
                }
            }
            ctr += 1;
        }
        Self(supp)
    }

    pub fn random_sorted<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let mut supp = [0 as Index; WEIGHT];
        for i in 0..WEIGHT as usize {
            // Randomly generate element in the appropriate range
            let rand = rng.gen_range(0..LENGTH-i);
            // Insert in sorted order
            insert_sorted_inc(&mut supp, rand as Index, i);
        }
        Self(supp)
    }

    pub fn random_weak_type1<R>(thresh: usize, rng: &mut R) -> Self
        where R: Rng + ?Sized
    {
        let r = LENGTH as Index;
        let delta = rng.gen_range(1..=r/2);
        let shift = rng.gen_range(0..r);
        let mut supp = [0 as Index; WEIGHT];
        for j in 0..=thresh {
            insert_sorted_noinc(&mut supp, (delta * (shift + j as Index)) % r, j);
        }
        for j in thresh+1 .. WEIGHT {
            let rand = rng.gen_range(0..r-j as Index);
            insert_sorted_inc(&mut supp, rand, j);
        }
        Self(supp)
    }

    pub fn random_weak_type2<R>(thresh: usize, rng: &mut R) -> Self
        where R: Rng + ?Sized
    {
        let s = WEIGHT - thresh - 1;
        let (r, d) = (LENGTH as Index, WEIGHT as Index);
        let mut o = [0 as Index; WEIGHT];
        let mut z = [0 as Index; WEIGHT];
        for j in 1..s {
            // Randomly generate elements in the appropriate ranges
            let rand_o = rng.gen_range(0..d-j as Index);
            let rand_z = rng.gen_range(0..r-d-j as Index);
            // Insert in sorted order
            insert_sorted_inc(&mut o, rand_o, j);
            insert_sorted_inc(&mut z, rand_z, j);
        }
        o[s] = d;
        z[s] = r - d;
        for j in 0..s {
            o[j] = o[j+1] - o[j];
            z[j] = z[j+1] - z[j];
        }
        let delta = rng.gen_range(1..=r/2);
        let shift = rng.gen_range(0..z[0]+o[0]);
        let mut supp = [0 as Index; WEIGHT];
        let mut idx = 0;
        let mut pos = r - shift;
        for j in 0..s {
            pos = (pos + z[j]) % r;
            for k in 0..o[j] {
                insert_sorted_noinc(&mut supp, (delta * (pos + k)) % r, idx);
                idx += 1;
            }
            pos += o[j];
        }
        Self(supp)
    }

    pub fn random_weak_type3<R>(thresh: usize, rng: &mut R) -> (Self, Self)
        where R: Rng + ?Sized
    {
        let r = LENGTH as Index;
        let shift = rng.gen_range(0..r);
        let mut h0 = [0 as Index; WEIGHT];
        let mut h1 = [0 as Index; WEIGHT];
        // Generate entries that overlap after a shift
        for j in 0..thresh {
            let rand = rng.gen_range(0..r-j as Index);
            insert_sorted_inc(&mut h0, rand, j);
            insert_sorted_noinc(&mut h1, (rand + shift) % r, j);
        }
        // Generate other entries
        for j in thresh..WEIGHT {
            let rand = rng.gen_range(0..r-j as Index);
            insert_sorted_inc(&mut h0, rand, j);
            let rand = rng.gen_range(0..r-j as Index);
            insert_sorted_inc(&mut h1, rand, j);
        }
        (Self(h0), Self(h1))
    }

    pub fn dense(&self) -> DenseVector<LENGTH> {
        let mut v = DenseVector::zero();
        for &i in self.support() {
            v.set_one(i as usize);
        }
        v
    }

    pub fn relative_shifts(&self, other: &Self) -> [[Index; WEIGHT]; WEIGHT] {
        let length = self.length();
        let mut shifts = [[0 as Index; WEIGHT]; WEIGHT];
        for i in 0..WEIGHT {
            let self_i = self.get(i);
            let length_plus_self_i = length.wrapping_add(self_i);
            for j in 0..WEIGHT {
                let other_j = other.get(j);
                shifts[i][j] = if self_i < other_j {
                    length_plus_self_i.wrapping_sub(other_j)
                } else {
                    self_i.wrapping_sub(other_j)
                };  // this equals (self_i - other_j) % length
                    // since 0 <= self_i, other_j < N.
            }
        }
        shifts
    }

    pub fn max_shifted_product_weight_geq(&self, other: &Self, threshold: usize) -> bool {
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

    pub fn shifts_above_threshold(&self, threshold: usize) -> bool {
        let length = self.length();
        let mut shift_counts = [0; LENGTH];
        for i in 0..WEIGHT {
            for j in i+1..WEIGHT {
                let diff = self.get(j).abs_diff(self.get(i));
                let delta = cmp::min(diff, length.wrapping_sub(diff));
                let count = &mut shift_counts[delta as usize];
                *count += 1;
                if *count >= threshold {
                    return true;
                }
            }
        }
        false
    }

    pub fn random_non_weak_type2<R>(thresh: usize, rng: &mut R) -> Self
        where R: Rng + ?Sized
    {
        loop {
            let block = Self::random(rng);
            if !block.shifts_above_threshold(thresh) {
                return block;
            }
        }
    }
}

impl<const W: usize, const L: usize> cmp::PartialEq for SparseVector<W, L> {
    // Supports may or may not be sorted, so we have to sort to test equality
    fn eq(&self, other: &Self) -> bool {
        let mut supp_self = self.support().clone();
        let mut supp_other = other.support().clone();
        supp_self.sort();
        supp_other.sort();
        supp_self == supp_other
    }
}

impl<const W: usize, const L: usize> cmp::Eq for SparseVector<W, L> { }

impl<const W: usize, const L: usize> fmt::Display for SparseVector<W, L> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut supp = self.support().clone();
        supp.sort();
        let mut str_supp = Vec::new();
        for index in supp {
            str_supp.push(index.to_string());
        }
        write!(f, "[{}]", str_supp.join(", "))
    }
}

// Dense vectors of fixed length over GF(2)
#[derive(Debug, Clone, cmp::PartialEq, cmp::Eq)]
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
    pub fn contents(&self) -> &[u8; LENGTH] {
        &self.0
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

    pub fn duplicate_up_to(&mut self, length: usize) {
        let (left, right) = self.0.split_at_mut(length);
        right[..length].copy_from_slice(left);
    }
}

fn insert_sorted_noinc<T: Ord + Copy>(array: &mut [T], value: T, max_i: usize) {
    // Find index to insert the element in order
    let mut idx = 0;
    while idx < max_i && array[idx] <= value {
        idx += 1;
    }
    // Move larger elements to make space
    let mut j = max_i;
    while j > idx {
        array[j] = array[j - 1];
        j -= 1;
    }
    // Insert the element
    array[idx] = value;
}

fn insert_sorted_inc(array: &mut [Index], mut value: Index, max_i: usize) {
    // Find index to insert the element in order
    let mut idx = 0;
    while idx < max_i && array[idx] <= value {
        idx += 1;
        // Element gets incremented so it's uniformly distributed
        // over numbers not already in list
        value += 1;
    }
    // Move larger elements to make space
    let mut j = max_i;
    while j > idx {
        array[j] = array[j - 1];
        j -= 1;
    }
    // Insert the element
    array[idx] = value;
}
