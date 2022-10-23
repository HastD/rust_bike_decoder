use rand::{Rng, distributions::Uniform};
use std::cmp::{self, Ord, Ordering};

pub type Index = u32;

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

fn sorted_intersection_count<const D: usize>(a: &[Index; D], b: &[Index; D]) -> usize {
    let mut i = 0;
    let mut j = 0;
    let mut count = 0;
    while i < D && j < D {
        match Ord::cmp(&a[i], &b[j]) {
            Ordering::Less => {
                i += 1;
            }
            Ordering::Greater => {
                j += 1;
            }
            Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }
    count
}

// Sparse vector of fixed weight and length over GF(2)
pub struct SparseVector<const WEIGHT: usize, const LENGTH: usize>([Index; WEIGHT]);

impl<const WEIGHT: usize, const LENGTH: usize> SparseVector<WEIGHT, LENGTH> {
    pub fn weight(&self) -> usize {
        WEIGHT
    }

    pub fn length(&self) -> Index {
        LENGTH as Index
    }

    #[inline]
    pub fn get(&self, i: usize) -> Index {
        self.0[i]
    }

    pub fn support(&self) -> [Index; WEIGHT] {
        self.0
    }

    pub fn contains(&self, index: &Index) -> bool {
        self.0.contains(index)
    }

    pub fn random<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let mut supp = [0 as Index; WEIGHT];
        for i in 0..WEIGHT as usize {
            // Randomly generate element in the appropriate range
            let rand = rng.gen_range(0..LENGTH-i);
            // Insert in sorted order
            insert_sorted_inc(&mut supp, rand as Index, i);
        }
        Self(supp)
    }

    pub fn dense(&self) -> DenseVector<LENGTH> {
        let mut v = DenseVector::new();
        for i in self.support() {
            v.flip(i as usize);
        }
        v
    }

    pub fn cyclic_shift(&self, shift: Index) -> Self {
        let mut supp = [0 as Index; WEIGHT];
        for j in 0..WEIGHT {
            insert_sorted_noinc(&mut supp, (self.get(j) + shift) % self.length(), j);
        }
        Self(supp)
    }

    pub fn product_weight(&self, other: &Self) -> usize {
        sorted_intersection_count(&self.support(), &other.support())
    }

    pub fn relative_shifts(&self, other: &Self) -> [[Index; WEIGHT]; WEIGHT] {
        let length = self.length();
        let mut shifts = [[0 as Index; WEIGHT]; WEIGHT];
        for i in 0..WEIGHT {
            for j in 0..WEIGHT {
                let self_i = self.get(i);
                let other_j = other.get(j);
                shifts[i][j] = if self_i < other_j {
                    length + self_i - other_j
                } else {
                    self_i - other_j
                };
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
        let mut shift_counts = [0; LENGTH];
        for i in 0..WEIGHT {
            for j in i+1..WEIGHT {
                let diff = self.get(j) - self.get(i);
                let delta = cmp::min(diff, self.length() - diff) as usize;
                shift_counts[delta] += 1;
                if shift_counts[delta] >= threshold {
                    return true;
                }
            }
        }
        false
    }
}

// Dense vectors of fixed length over GF(2)
#[derive(Copy, Clone)]
pub struct DenseVector<const LENGTH: usize>([u8; LENGTH]);

impl<const LENGTH: usize> DenseVector<LENGTH> {
    pub fn get(&self, i: Index) -> u8 {
        self.0[i as usize]
    }

    pub fn support(&self) -> Vec<Index> {
        let mut supp: Vec<Index> = Vec::new();
        for i in 0..LENGTH {
            if self.0[i] != 0 {
                supp.push(i as Index);
            }
        }
        supp
    }

    pub fn new() -> Self {
        Self([0u8; LENGTH])
    }

    pub fn flip(&mut self, i: usize) {
        self.0[i] ^= 1;
    }

    pub fn hamming_weight(&self) -> usize {
        let mut count = 0;
        for i in self.support() {
            if i != 0 {
                count += 1;
            }
        }
        count
    }
}

impl<const LENGTH: usize> Default for DenseVector<LENGTH> {
    fn default() -> Self {
        Self::new()
    }
}
