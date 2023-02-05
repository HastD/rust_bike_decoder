use bike_decoder::{
    decoder::DecodingFailure,
    keys::QuasiCyclic,
    ncw::ClassifiedVector,
    parameters::{BLOCK_LENGTH, BLOCK_WEIGHT},
    vectors::Index,
};
use itertools::Itertools;
use rayon::prelude::*;

/// Classify decoding failures in the given list into near-codeword sets
pub fn ncw_classify(
    decoding_failures: Vec<DecodingFailure>,
    parallel: bool,
) -> Vec<ClassifiedVector<BLOCK_WEIGHT, BLOCK_LENGTH>> {
    if parallel {
        decoding_failures
            .into_par_iter()
            .map(|df| ClassifiedVector::new(df.key().clone(), df.vector().vector().support()))
            .collect()
    } else {
        decoding_failures
            .into_iter()
            .map(|df| ClassifiedVector::new(df.key().clone(), df.vector().vector().support()))
            .collect()
    }
}

/// Randomly samples vectors and classifies them into near-codeword sets
pub fn classify_sample<const WT: usize, const LEN: usize>(
    key: &QuasiCyclic<WT, LEN>,
    supp_weight: usize,
    samples: usize,
    parallel: bool,
) -> Vec<ClassifiedVector<WT, LEN>> {
    if parallel {
        (0..samples)
            .into_par_iter()
            .map(|_| ClassifiedVector::random(key, supp_weight))
            .collect()
    } else {
        (0..samples)
            .map(|_| ClassifiedVector::random(key, supp_weight))
            .collect()
    }
}

/// Exhaustively enumerates vectors of a given weight and classifies them into near-codeword sets
pub fn classify_enumerate<const WT: usize, const LEN: usize>(
    key: &QuasiCyclic<WT, LEN>,
    supp_weight: usize,
    parallel: bool,
) -> Vec<ClassifiedVector<WT, LEN>> {
    let n = 2 * LEN as Index;
    let combinations = (0..n).combinations(supp_weight);
    if parallel {
        combinations
            .par_bridge()
            .map(|_| ClassifiedVector::random(key, supp_weight))
            .collect()
    } else {
        combinations
            .map(|_| ClassifiedVector::random(key, supp_weight))
            .collect()
    }
}
