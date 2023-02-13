use crate::{
    keys::{Key, QuasiCyclic},
    parameters::*,
    random::custom_thread_rng,
    vectors::{Index, SparseErrorVector},
};
use getset::{CopyGetters, Getters};
use rand::{
    distributions::{Distribution, Uniform},
    seq::{IteratorRandom, SliceRandom},
    Rng,
};
use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum NearCodewordClass {
    C,
    N,
    #[serde(rename = "2N")]
    TwoN,
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
#[getset(get_copy = "pub")]
pub struct NearCodewordSet {
    class: NearCodewordClass,
    l: usize,
    delta: usize,
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
#[getset(get = "pub")]
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
    where
        R: Rng + ?Sized,
    {
        Self {
            vector: SparseErrorVector::random(rng),
            source: ErrorVectorSource::Random,
        }
    }

    pub fn near_codeword<R>(key: &Key, class: NearCodewordClass, l: usize, rng: &mut R) -> Self
    where
        R: Rng + ?Sized,
    {
        let r = BLOCK_LENGTH as Index;
        let sample = match class {
            NearCodewordClass::C => sample_c(key),
            NearCodewordClass::N => sample_n(key, rng.gen_range(0..2)),
            NearCodewordClass::TwoN => loop {
                let sample = sample_2n(key, rng.gen_range(0..r), rng.gen_range(0..4));
                if sample.len() >= l {
                    break sample;
                }
            },
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
        shift_blockwise::<BLOCK_LENGTH>(&mut supp, shift);
        Self {
            vector: SparseErrorVector::from_support(supp)
                .expect("near_codeword should always produce valid vector support"),
            source: ErrorVectorSource::NearCodeword(NearCodewordSet {
                class,
                l,
                delta: sample.len() + ERROR_WEIGHT - 2 * l,
            }),
        }
    }
}

impl fmt::Display for TaggedErrorVector {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.source() {
            ErrorVectorSource::NearCodeword(source) => write!(
                f,
                "{} [element of A_{{t,{}}}({})]",
                self.vector(),
                source.l(),
                source.class()
            ),
            _ => write!(f, "{}", self.vector()),
        }
    }
}

fn sample_c<const WEIGHT: usize, const LENGTH: usize>(
    key: &QuasiCyclic<WEIGHT, LENGTH>,
) -> Vec<Index> {
    key.h0()
        .support()
        .iter()
        .map(|idx| *idx + LENGTH as Index)
        .chain(key.h1().support().iter().copied())
        .collect()
}

fn sample_n<const WEIGHT: usize, const LENGTH: usize>(
    key: &QuasiCyclic<WEIGHT, LENGTH>,
    block_flag: u8,
) -> Vec<Index> {
    if block_flag % 2 == 0 {
        key.h0().support().to_vec()
    } else {
        key.h1()
            .support()
            .iter()
            .map(|idx| *idx + LENGTH as Index)
            .collect()
    }
}

fn sample_2n<const WEIGHT: usize, const LENGTH: usize>(
    key: &QuasiCyclic<WEIGHT, LENGTH>,
    shift: Index,
    block_flag: u8,
) -> Vec<Index> {
    let mut sum_n = sample_n(key, block_flag % 2);
    let mut supp2 = sample_n(key, (block_flag >> 1) % 2);
    shift_blockwise::<LENGTH>(&mut supp2, shift);
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

fn patterns_c<const WT: usize, const LEN: usize>(key: &QuasiCyclic<WT, LEN>) -> Vec<Vec<Index>> {
    let mut codeword = key.h1().support().to_vec();
    let mut h0 = key.h0().support().to_vec();
    let r = BLOCK_LENGTH as Index;
    for entry in h0.iter_mut() {
        *entry += r;
    }
    codeword.append(&mut h0);
    vec![codeword]
}

fn patterns_n<const WT: usize, const LEN: usize>(key: &QuasiCyclic<WT, LEN>) -> Vec<Vec<Index>> {
    let vec1 = key.h0().support().to_vec();
    let mut vec2 = key.h1().support().to_vec();
    let r = LEN as Index;
    for entry in vec2.iter_mut() {
        *entry += r;
    }
    vec![vec1, vec2]
}

fn patterns_2n<const WT: usize, const LEN: usize>(key: &QuasiCyclic<WT, LEN>) -> Vec<Vec<Index>> {
    let n_patterns = patterns_n(key);
    let mut patterns = Vec::with_capacity(4 * LEN);
    for supp1 in n_patterns.iter() {
        for supp2 in n_patterns.iter() {
            let mut supp2 = supp2.clone();
            for _ in 0..LEN {
                shift_blockwise::<LEN>(&mut supp2, 1);
                let mut v = supp1.clone();
                for entry in supp2.iter() {
                    if let Some(index) = v.iter().position(|item| *item == *entry) {
                        v.swap_remove(index);
                    } else {
                        v.push(*entry);
                    }
                }
                patterns.push(v);
            }
        }
    }
    patterns
}

pub fn ncw_patterns<const WT: usize, const LEN: usize>(
    key: &QuasiCyclic<WT, LEN>,
    ncw_class: NearCodewordClass,
) -> Vec<Vec<Index>> {
    match ncw_class {
        NearCodewordClass::C => patterns_c(key),
        NearCodewordClass::N => patterns_n(key),
        NearCodewordClass::TwoN => patterns_2n(key),
    }
}

pub fn near_codeword_max_overlap<const LEN: usize>(
    supp: &[Index],
    patterns: &[Vec<Index>],
) -> usize {
    patterns
        .iter()
        .map(|pattern| max_shifted_overlap_blockwise::<LEN>(supp, pattern))
        .max()
        .unwrap_or(0)
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct NcwOverlaps {
    pub c: usize,
    pub n: usize,
    pub two_n: usize,
}

impl NcwOverlaps {
    pub fn new<const WT: usize, const LEN: usize>(
        key: &QuasiCyclic<WT, LEN>,
        supp: &[Index],
    ) -> Self {
        let patterns_c = ncw_patterns(key, NearCodewordClass::C);
        let patterns_n = ncw_patterns(key, NearCodewordClass::N);
        let patterns_2n = ncw_patterns(key, NearCodewordClass::TwoN);
        Self {
            c: near_codeword_max_overlap::<LEN>(supp, &patterns_c),
            n: near_codeword_max_overlap::<LEN>(supp, &patterns_n),
            two_n: near_codeword_max_overlap::<LEN>(supp, &patterns_2n),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClassifiedVector<const WT: usize, const LEN: usize> {
    key: QuasiCyclic<WT, LEN>,
    supp: Vec<Index>,
    overlaps: NcwOverlaps,
}

impl<const WT: usize, const LEN: usize> ClassifiedVector<WT, LEN> {
    pub fn new(key: QuasiCyclic<WT, LEN>, supp: &[Index]) -> Self {
        let overlaps = NcwOverlaps::new(&key, supp);
        let mut supp: Vec<Index> = supp.to_vec();
        supp.sort_unstable();
        Self {
            key,
            supp,
            overlaps,
        }
    }

    pub fn random(key: &QuasiCyclic<WT, LEN>, supp_weight: usize) -> Self {
        let n = 2 * LEN as Index;
        let mut supp = (0..n).choose_multiple(&mut custom_thread_rng(), supp_weight);
        supp.sort_unstable();
        let overlaps = NcwOverlaps::new(key, &supp);
        Self {
            key: key.clone(),
            supp,
            overlaps,
        }
    }
}

/// Cyclically shifts support of vector by shift in blocks of length block_length.
pub fn shift_blockwise<const LEN: usize>(supp: &mut [Index], shift: Index) {
    let block_length = LEN as Index;
    for idx in supp.iter_mut() {
        *idx = ((*idx + shift) % block_length) + (*idx / block_length) * block_length;
    }
}

pub fn relative_shifts_blockwise<const LEN: usize>(supp1: &[Index], supp2: &[Index]) -> Vec<Index> {
    let block_length = LEN as Index;
    let mut shifts = Vec::with_capacity(supp1.len() * supp2.len());
    for idx1 in supp1 {
        let block = idx1 / block_length;
        let length_plus_idx1 = block_length + idx1;
        for idx2 in supp2 {
            if idx2 / block_length == block {
                shifts.push(if idx1 < idx2 {
                    length_plus_idx1 - idx2
                } else {
                    idx1 - idx2
                });
            }
        }
    }
    shifts
}

pub fn max_shifted_overlap_blockwise<const LEN: usize>(supp1: &[Index], supp2: &[Index]) -> usize {
    let shifts = relative_shifts_blockwise::<LEN>(supp1, supp2);
    let mut shift_counts = [0; LEN];
    let mut max_shift_count = 0;
    for shift in shifts {
        let entry = &mut shift_counts[shift as usize];
        *entry += 1;
        max_shift_count = max_shift_count.max(*entry);
    }
    max_shift_count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blockwise_shift() {
        let mut supp = [2, 3, 5, 7, 11, 13, 17, 19];
        shift_blockwise::<7>(&mut supp, 4);
        assert_eq!(supp, [6, 0, 2, 11, 8, 10, 14, 16]);
    }

    #[test]
    fn blockwise_shifted_overlap() {
        let supp = [130, 351, 527, 541];
        let key = Key::from_support(
            [
                0, 11, 14, 53, 69, 134, 190, 213, 218, 245, 378, 408, 411, 480, 545,
            ],
            [
                26, 104, 110, 137, 207, 252, 258, 310, 326, 351, 367, 459, 461, 506, 570,
            ],
        )
        .unwrap();
        let patterns = ncw_patterns(&key, NearCodewordClass::C);
        let shifts = relative_shifts_blockwise::<587>(&supp, &patterns[0]);
        assert_eq!(
            shifts,
            vec![
                104, 26, 20, 580, 510, 465, 459, 407, 391, 366, 350, 258, 256, 211, 147, 325, 247,
                241, 214, 144, 99, 93, 41, 25, 0, 571, 479, 477, 432, 368, 501, 423, 417, 390, 320,
                275, 269, 217, 201, 176, 160, 68, 66, 21, 544, 515, 437, 431, 404, 334, 289, 283,
                231, 215, 190, 174, 82, 80, 35, 558
            ]
        );
        let max_overlap = near_codeword_max_overlap::<587>(&supp, &patterns);
        assert_eq!(max_overlap, 1);
    }

    #[test]
    fn classify_example() {
        let key = Key::from_support(
            [
                13, 26, 58, 68, 69, 73, 117, 133, 190, 239, 346, 483, 508, 545, 576,
            ],
            [
                10, 103, 108, 141, 273, 337, 342, 343, 377, 451, 465, 473, 496, 546, 556,
            ],
        )
        .unwrap();
        let supp = [
            7, 42, 99, 107, 114, 159, 181, 235, 274, 325, 432, 569, 575, 770, 887, 900, 945, 955,
        ];
        assert_eq!(
            NcwOverlaps::new(&key, &supp),
            NcwOverlaps {
                c: 4,
                n: 6,
                two_n: 8
            }
        );
    }
}
