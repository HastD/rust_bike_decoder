use crate::parameters::*;
use crate::vectors::{Index, SparseErrorVector};
use crate::keys::Key;
use rand::{Rng, seq::SliceRandom, distributions::{Distribution, Uniform}};
use serde::{Serialize, Deserialize};
use std::fmt;

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum NearCodewordSet {
    C, N, TwoN
}

// Set up NearCodewordSet for use in command-line arguments
impl clap::ValueEnum for NearCodewordSet {
    fn value_variants<'a>() -> &'a [Self] {
        &[Self::C, Self::N, Self::TwoN]
    }
    fn to_possible_value<'a>(&self) -> Option<clap::builder::PossibleValue> {
        match self {
            Self::C => Some(clap::builder::PossibleValue::new("C")),
            Self::N => Some(clap::builder::PossibleValue::new("N")),
            Self::TwoN => Some(clap::builder::PossibleValue::new("2N")),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ElementOfAtlS {
    supp: SparseErrorVector,
    l: usize,
    delta: usize
}

impl ElementOfAtlS {
    pub fn random_from<R: Rng + ?Sized>(
        key: &Key,
        sample_set: NearCodewordSet,
        l: usize,
        rng: &mut R
    ) -> Self {
        let r = BLOCK_LENGTH as Index;
        let sample = match sample_set {
            NearCodewordSet::C => Self::sample_c(key),
            NearCodewordSet::N => Self::sample_n(key, rng.gen_range(0..2)),
            NearCodewordSet::TwoN => {
                loop {
                    let sample = Self::sample_2n(key, rng.gen_range(0..r), rng.gen_range(0..4));
                    if sample.len() >= l {
                        break sample;
                    }
                }
            }
        };
        assert!(sample.len() >= l);
        let mut supp = [0 as Index; ERROR_WEIGHT];
        // Fill out first l entries from sample
        for (idx, slot) in sample.choose_multiple(rng, l).zip(&mut supp[0..l]) {
            *slot = *idx;
        }
        // Fill remaining elements from complement of sample
        let mut ctr = l;
        let dist = Uniform::new(0, ROW_LENGTH as Index);
        while ctr < ERROR_WEIGHT {
            supp[ctr] = dist.sample(rng);
            if sample.contains(&supp[ctr]) || supp[l..ctr].contains(&supp[ctr]) {
                continue;
            } else {
                ctr += 1;
            }
        }
        let shift = rng.gen_range(0..r);
        shift_blockwise(&mut supp, shift, r);
        Self {
            supp: SparseErrorVector::from_support(supp),
            l,
            delta: sample.len() + ERROR_WEIGHT - 2*l,
        }
    }

    pub fn supp(&self) -> &SparseErrorVector {
        &self.supp
    }

    pub fn l(&self) -> &usize {
        &self.l
    }

    pub fn delta(&self) -> &usize {
        &self.delta
    }

    fn sample_c(key: &Key) -> Vec<Index> {
        let mut supp: Vec<Index> = Vec::with_capacity(ROW_WEIGHT);
        for i in 0..BLOCK_WEIGHT {
            supp.push(key.h1().get(i));
            supp.push(key.h0().get(i) + BLOCK_LENGTH as Index);
        }
        supp
    }
    
    fn sample_n(key: &Key, block_flag: u8) -> Vec<Index>
    {
        if block_flag % 2 == 0 {
            key.h0().support().to_vec()
        } else {
            let mut supp: Vec<Index> = Vec::with_capacity(BLOCK_WEIGHT);
            for &idx in key.h1().support() {
                supp.push(idx + BLOCK_WEIGHT as Index);
            }
            supp
        }
    }
    
    fn sample_2n(key: &Key, shift: Index, block_flag: u8) -> Vec<Index>
    {
        let supp1 = Self::sample_n(key, block_flag % 2);
        let mut supp2 = Self::sample_n(key, (block_flag >> 1) % 2);
        shift_blockwise(&mut supp2, shift, BLOCK_LENGTH as Index);
        let mut sum_n = supp1.clone();
        // Symmetric difference of supp1 and supp2
        for idx in &supp2 {
            if let Some(pos) = sum_n.iter().position(|x| *x == *idx) {
                // If sum_n contains idx at position pos, remove it from sum_n
                sum_n.swap_remove(pos);
            } else {
                // If sum_n doesn't already contain idx, add it to sum_n
                sum_n.push(*idx);
            }
        }
        sum_n
    }    
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TaggedErrorVector {
    Random(SparseErrorVector),
    NearCodeword(ElementOfAtlS)
}

impl TaggedErrorVector {
    pub fn unpack(&self) -> &SparseErrorVector {
        match self {
            Self::Random(e_supp) => e_supp,
            Self::NearCodeword(elt) => elt.supp()
        }
    }
}

impl fmt::Display for TaggedErrorVector {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Random(e_supp) => write!(f, "{}", e_supp),
            Self::NearCodeword(elt) => write!(f, "{} (l = {}, δ = {})",
                elt.supp(), elt.l(), elt.delta()),
        }
    }
}

// Cyclically shift support of vector by shift in blocks of length block_length
pub fn shift_blockwise(supp: &mut [Index], shift: Index, block_length: Index) {
    for idx in supp.iter_mut() {
        *idx = ((*idx + shift) % block_length) + (*idx / block_length) * block_length;
    }
}
