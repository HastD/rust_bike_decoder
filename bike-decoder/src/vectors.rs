use crate::parameters::*;
use rand::{
    distributions::{Distribution, Uniform},
    Rng,
};
use serde::{Deserialize, Serialize, Serializer};
use serde_with::serde_as;
use std::{
    fmt,
    ops::{Add, Sub},
};
use thiserror::Error;

pub type Index = u32;

pub type SparseErrorVector = SparseVector<ERROR_WEIGHT, ROW_LENGTH>;
pub type ErrorVector = DenseVector<ROW_LENGTH>;

#[derive(Copy, Clone, Debug, Error)]
pub enum InvalidSupport {
    #[error("support indices must be in range 0..{0}")]
    OutOfBounds(usize),
    #[error("support indices must all be distinct")]
    RepeatedIndex,
    #[error("support must be of length {0}")]
    WrongLength(usize),
}

// Sparse vector of fixed weight and length over GF(2)
#[serde_as]
#[derive(Debug, Clone, Deserialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct SparseVector<const WEIGHT: usize, const LENGTH: usize>(
    #[serde_as(as = "[_; WEIGHT]")] [Index; WEIGHT],
);

impl<const WEIGHT: usize, const LENGTH: usize> TryFrom<&[Index]> for SparseVector<WEIGHT, LENGTH> {
    type Error = InvalidSupport;
    fn try_from(supp: &[Index]) -> Result<Self, Self::Error> {
        let supp =
            <[Index; WEIGHT]>::try_from(supp).map_err(|_| InvalidSupport::WrongLength(WEIGHT))?;
        Self::from_support(supp)
    }
}

impl<const WEIGHT: usize, const LENGTH: usize> SparseVector<WEIGHT, LENGTH> {
    pub fn from_support(supp: [Index; WEIGHT]) -> Result<Self, InvalidSupport> {
        let v = Self(supp);
        v.validate()?;
        Ok(v)
    }

    // Ensure that the support represents a valid vector of the specified weight and length
    pub fn validate(&self) -> Result<(), InvalidSupport> {
        for idx in self.0 {
            if idx >= self.length() {
                return Err(InvalidSupport::OutOfBounds(LENGTH));
            }
        }
        for i in 0..WEIGHT {
            for j in (i + 1)..WEIGHT {
                if self.get(i) == self.get(j) {
                    return Err(InvalidSupport::RepeatedIndex);
                }
            }
        }
        Ok(())
    }

    #[inline]
    pub fn sort(&mut self) {
        self.0.sort_unstable()
    }

    #[inline]
    pub fn sorted(mut self) -> Self {
        self.sort();
        self
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
    where
        R: Rng + ?Sized,
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
        for i in 0..WEIGHT {
            // Randomly generate element in the appropriate range
            let rand = rng.gen_range(0..LENGTH - i);
            // Insert in sorted order
            insert_sorted_inc(&mut supp, rand as Index, i);
        }
        Self(supp)
    }

    pub fn random_weak_type1<R>(thresh: u8, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
    {
        let thresh = usize::from(thresh);
        if thresh >= WEIGHT {
            return Self::random(rng);
        }
        let r = LENGTH as Index;
        let delta = rng.gen_range(1..=r / 2);
        let shift = rng.gen_range(0..r);
        let mut supp = [0 as Index; WEIGHT];
        for j in 0..=thresh {
            insert_sorted_noinc(&mut supp, (delta * (shift + j as Index)) % r, j);
        }
        for j in thresh + 1..WEIGHT {
            let rand = rng.gen_range(0..r - j as Index);
            insert_sorted_inc(&mut supp, rand, j);
        }
        Self(supp)
    }

    pub fn random_weak_type2<R>(thresh: u8, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
    {
        let thresh = usize::from(thresh);
        if thresh >= WEIGHT {
            return Self::random(rng);
        }
        let s = WEIGHT - thresh - 1;
        let (r, d) = (LENGTH as Index, WEIGHT as Index);
        let mut o = [0 as Index; WEIGHT];
        let mut z = [0 as Index; WEIGHT];
        for j in 1..s {
            // Randomly generate elements in the appropriate ranges
            let rand_o = rng.gen_range(0..d - j as Index);
            let rand_z = rng.gen_range(0..r - d - j as Index);
            // Insert in sorted order
            insert_sorted_inc(&mut o, rand_o, j);
            insert_sorted_inc(&mut z, rand_z, j);
        }
        o[s] = d;
        z[s] = r - d;
        for j in 0..s {
            o[j] = o[j + 1] - o[j];
            z[j] = z[j + 1] - z[j];
        }
        let delta = rng.gen_range(1..=r / 2);
        let shift = rng.gen_range(0..z[0] + o[0]);
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

    pub fn random_weak_type3<R>(thresh: u8, rng: &mut R) -> (Self, Self)
    where
        R: Rng + ?Sized,
    {
        let thresh = usize::from(thresh);
        if thresh >= WEIGHT {
            return (Self::random(rng), Self::random(rng));
        }
        let r = LENGTH as Index;
        let shift = rng.gen_range(0..r);
        let mut h0 = [0 as Index; WEIGHT];
        let mut h1 = [0 as Index; WEIGHT];
        // Generate entries that overlap after a shift
        for j in 0..thresh {
            let rand = rng.gen_range(0..r - j as Index);
            let value = insert_sorted_inc(&mut h0, rand, j);
            insert_sorted_noinc(&mut h1, (value + shift) % r, j);
        }
        // Generate other entries
        for j in thresh..WEIGHT {
            let rand = rng.gen_range(0..r - j as Index);
            insert_sorted_inc(&mut h0, rand, j);
            let rand = rng.gen_range(0..r - j as Index);
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
        for (i, &self_i) in self.0.iter().enumerate() {
            let length_plus_self_i = length + self_i;
            for (j, &other_j) in other.0.iter().enumerate() {
                shifts[i][j] = if self_i < other_j {
                    length_plus_self_i - other_j
                } else {
                    self_i - other_j
                }; // this equals (self_i - other_j) % length
                   // since 0 <= self_i, other_j < N.
            }
        }
        shifts
    }

    pub fn max_shifted_product_weight_geq(&self, other: &Self, threshold: u8) -> bool {
        let shifts = self.relative_shifts(other);
        let mut shift_counts = [0; LENGTH];
        for shift in shifts.into_iter().flatten() {
            let count = &mut shift_counts[shift as usize];
            *count += 1;
            if *count >= threshold {
                return true;
            }
        }
        false
    }

    pub fn shifts_above_threshold(&self, threshold: u8) -> bool {
        let length = self.length();
        let mut shift_counts = [0; LENGTH];
        for (i, &self_i) in self.0.iter().enumerate() {
            for &self_j in self.0[i + 1..].iter() {
                let diff = self_j.abs_diff(self_i);
                let delta = diff.min(length - diff);
                let count = &mut shift_counts[delta as usize];
                *count += 1;
                if *count >= threshold {
                    return true;
                }
            }
        }
        false
    }

    pub fn random_non_weak_type2<R>(thresh: u8, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
    {
        loop {
            let block = Self::random(rng);
            if !block.shifts_above_threshold(thresh) {
                return block;
            }
        }
    }
}

// Sort support lists before serializing
impl<const W: usize, const L: usize> Serialize for SparseVector<W, L> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.clone().sorted().0.serialize(serializer)
    }
}

impl<const W: usize, const L: usize> PartialEq for SparseVector<W, L> {
    // Supports may or may not be sorted, so we have to sort to test equality
    fn eq(&self, other: &Self) -> bool {
        self.clone().sorted().0 == other.clone().sorted().0
    }
}

impl<const W: usize, const L: usize> Eq for SparseVector<W, L> {}

impl<const W: usize, const L: usize> fmt::Display for SparseVector<W, L> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let str_supp = self
            .support()
            .iter()
            .map(|idx| idx.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        write!(f, "[{str_supp}]")
    }
}

// Dense vectors of fixed length over GF(2)
#[derive(Debug, Clone, PartialEq, Eq)]
#[repr(transparent)]
pub struct DenseVector<const LENGTH: usize>([bool; LENGTH]);

impl<const LENGTH: usize> Default for DenseVector<LENGTH> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const LENGTH: usize> DenseVector<LENGTH> {
    pub fn zero() -> Self {
        Self([false; LENGTH])
    }

    pub fn new(list: [bool; LENGTH]) -> Self {
        Self(list)
    }

    #[inline]
    pub fn get(&self, i: usize) -> bool {
        self.0[i]
    }

    #[inline]
    pub fn contents(&self) -> &[bool] {
        &self.0
    }

    #[inline]
    pub fn flip(&mut self, i: usize) {
        self.0[i] ^= true;
    }

    #[inline]
    pub fn set_zero(&mut self, i: usize) {
        self.0[i] = false;
    }

    #[inline]
    pub fn set_one(&mut self, i: usize) {
        self.0[i] = true;
    }

    #[inline]
    pub fn set_all_zero(&mut self) {
        self.0.iter_mut().for_each(|entry| *entry = false);
    }

    pub fn support(&self) -> Vec<Index> {
        self.0
            .iter()
            .enumerate()
            .filter_map(|(idx, bit)| bit.then_some(idx as Index))
            .collect()
    }

    pub fn duplicate_up_to(&mut self, length: usize) {
        let (left, right) = self.0.split_at_mut(length);
        right[..length].copy_from_slice(left);
    }

    #[inline]
    pub fn xor_with<M>(&mut self, mask: M)
    where
        M: IntoIterator<Item = bool>,
    {
        self.0
            .iter_mut()
            .zip(mask)
            .for_each(|(bit, mask)| *bit ^= mask);
    }

    #[inline]
    pub fn add_mod2(mut self, other: Self) -> Self {
        self.xor_with(other.0);
        self
    }
}

impl<const L: usize> Add for DenseVector<L> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        self.add_mod2(other)
    }
}

impl<const L: usize> Sub for DenseVector<L> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.add_mod2(other)
    }
}

fn insert_sorted_noinc<T: Ord + Copy>(array: &mut [T], value: T, max_i: usize) -> T {
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
    value
}

fn insert_sorted_inc(array: &mut [Index], mut value: Index, max_i: usize) -> Index {
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
    value
}

#[cfg(test)]
mod tests {
    use super::*;

    const TRIALS: usize = 1000;

    #[test]
    fn validate_random() {
        let mut rng = rand::thread_rng();
        for _ in 0..TRIALS {
            let v = SparseVector::<ERROR_WEIGHT, BLOCK_LENGTH>::random(&mut rng);
            v.validate().expect("Random vector should validate");
        }
    }

    // Checks that the support of the associated dense vector is equal to the original sparse vector
    #[test]
    fn dense_support() {
        let mut rng = rand::thread_rng();
        for _ in 0..TRIALS {
            let sparse = SparseVector::<ERROR_WEIGHT, BLOCK_LENGTH>::random(&mut rng);
            let mut sorted_supp = sparse.0;
            sorted_supp.sort_unstable();
            assert_eq!(sorted_supp.to_vec(), sparse.dense().support());
        }
    }

    #[test]
    fn weak_type1() {
        let mut rng = rand::thread_rng();
        let thresh = 5;
        for _ in 0..TRIALS {
            let v = SparseVector::<BLOCK_WEIGHT, BLOCK_LENGTH>::random_weak_type1(thresh, &mut rng);
            assert!(
                v.shifts_above_threshold(thresh),
                "Type 1 weak block was not actually weak of type 1/2: {v:?}"
            );
        }
    }

    #[test]
    fn weak_type2() {
        let mut rng = rand::thread_rng();
        let thresh = 5;
        for _ in 0..TRIALS {
            let v = SparseVector::<BLOCK_WEIGHT, BLOCK_LENGTH>::random_weak_type2(thresh, &mut rng);
            assert!(
                v.shifts_above_threshold(thresh),
                "Type 2 weak block was not actually weak of type 1/2: {v:?}"
            );
        }
    }

    #[test]
    fn weak_type3() {
        let mut rng = rand::thread_rng();
        let thresh = 5;
        for _ in 0..TRIALS {
            let (v1, v2) =
                SparseVector::<BLOCK_WEIGHT, BLOCK_LENGTH>::random_weak_type3(thresh, &mut rng);
            assert!(
                v1.max_shifted_product_weight_geq(&v2, thresh),
                "Pair of type 3 weak blocks was not actually weak of type 3: {:?}",
                (v1, v2)
            );
        }
    }
}
