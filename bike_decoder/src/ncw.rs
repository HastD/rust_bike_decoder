use crate::parameters::*;
use crate::vectors::{Index, SparseErrorVector};
use crate::keys::Key;
use getset::{CopyGetters, Getters};
use rand::{Rng, seq::SliceRandom, distributions::{Distribution, Uniform}};
use serde::{Serialize, Deserialize};
use std::fmt;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum NearCodewordClass {
    C,
    N,
    #[serde(rename = "2N")]
    TwoN
}

impl NearCodewordClass {
    pub fn max_l(&self) -> usize {
        match self {
            Self::C => ERROR_WEIGHT,
            Self::N => BLOCK_WEIGHT,
            Self::TwoN => ERROR_WEIGHT,
        }
    }

    pub fn label(&self) -> &'static str {
        match self {
            Self::C => "C",
            Self::N => "N",
            Self::TwoN => "2N",
        }
    }
}

impl fmt::Display for NearCodewordClass {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(self.label())
    }
}

// Set up NearCodewordClass for use in command-line arguments
impl clap::ValueEnum for NearCodewordClass {
    fn value_variants<'a>() -> &'a [Self] {
        &[Self::C, Self::N, Self::TwoN]
    }
    fn to_possible_value<'a>(&self) -> Option<clap::builder::PossibleValue> {
        Some(clap::builder::PossibleValue::new(self.label()))
    }
}

#[derive(Copy, Clone, CopyGetters, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[getset(get_copy="pub")]
pub struct NearCodewordSet {
    class: NearCodewordClass,
    l: usize,
    delta: usize
}

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorVectorSource {
    Random,
    NearCodeword(NearCodewordSet),
    Other,
    #[default]
    Unknown,
}

#[derive(Clone, Debug, Getters, Serialize, PartialEq, Eq, Deserialize)]
#[getset(get="pub")]
pub struct TaggedErrorVector {
    #[serde(rename = "e_supp")]
    vector: SparseErrorVector,
    #[serde(default, rename = "e_source")]
    source: ErrorVectorSource,
}

impl TaggedErrorVector {
    #[inline]
    pub fn take_vector(self) -> (SparseErrorVector, ErrorVectorSource) {
        (self.vector, self.source)
    }

    #[inline]
    pub fn sorted(self) -> Self {
        Self {
            vector: self.vector.sorted(),
            source: self.source,
        }
    }

    #[inline]
    pub fn from_other(vector: SparseErrorVector) -> Self {
        Self {
            vector,
            source: ErrorVectorSource::Other,
        }
    }

    #[inline]
    pub fn random<R>(rng: &mut R) -> Self
        where R: Rng + ?Sized
    {
        Self {
            vector: SparseErrorVector::random(rng),
            source: ErrorVectorSource::Random,
        }
    }

    pub fn near_codeword<R>(key: &Key, class: NearCodewordClass, l: usize, rng: &mut R)
        -> TaggedErrorVector
        where R: Rng + ?Sized
    {
        let r = BLOCK_LENGTH as Index;
        let sample = match class {
            NearCodewordClass::C => sample_c(key),
            NearCodewordClass::N => sample_n(key, rng.gen_range(0..2)),
            NearCodewordClass::TwoN => {
                loop {
                    let sample = sample_2n(key, rng.gen_range(0..r), rng.gen_range(0..4));
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
            if !sample.contains(&supp[ctr]) && !supp[l..ctr].contains(&supp[ctr]) {
                ctr += 1;
            }
        }
        let shift = rng.gen_range(0..r);
        shift_blockwise(&mut supp, shift, r);
        Self {
            vector: SparseErrorVector::from_support(supp)
                .expect("near_codeword should always produce valid vector support"),
            source: ErrorVectorSource::NearCodeword(NearCodewordSet {
                class,
                l,
                delta: sample.len() + ERROR_WEIGHT - 2*l,
            })
        }
    }
}

impl fmt::Display for TaggedErrorVector {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.source() {
            ErrorVectorSource::NearCodeword(source) => write!(f, "{} [element of A_{{t,{}}}({})]",
                self.vector(), source.l(), source.class()),
            _ => write!(f, "{}", self.vector()),
            }
    }
}


fn sample_c(key: &Key) -> Vec<Index> {
    key.h0().support().iter().map(|idx| *idx + BLOCK_LENGTH as Index)
        .chain(key.h1().support().iter().copied())
        .collect()
}

fn sample_n(key: &Key, block_flag: u8) -> Vec<Index>
{
    if block_flag % 2 == 0 {
        key.h0().support().to_vec()
    } else {
        key.h1().support().iter().map(|idx| *idx + BLOCK_LENGTH as Index).collect()
    }
}

fn sample_2n(key: &Key, shift: Index, block_flag: u8) -> Vec<Index>
{
    let mut sum_n = sample_n(key, block_flag % 2);
    let mut supp2 = sample_n(key, (block_flag >> 1) % 2);
    shift_blockwise(&mut supp2, shift, BLOCK_LENGTH as Index);
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

// Cyclically shift support of vector by shift in blocks of length block_length
pub fn shift_blockwise(supp: &mut [Index], shift: Index, block_length: Index) {
    for idx in supp.iter_mut() {
        *idx = ((*idx + shift) % block_length) + (*idx / block_length) * block_length;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blockwise_shift() {
        let mut supp = [2, 3, 5, 7, 11, 13, 17, 19];
        shift_blockwise(&mut supp, 4, 7);
        assert_eq!(supp, [6, 0, 2, 11, 8, 10, 14, 16]);
    }
}
