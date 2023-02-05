use bike_decoder::{
    decoder::DecodingFailure,
    graphs::{self, AbsorbingDecodingFailure, TannerGraphEdges},
    keys::QuasiCyclic,
    random::custom_thread_rng,
    vectors::Index,
};
use itertools::Itertools;
use rand::seq::IteratorRandom;
use rayon::prelude::*;

/// Finds absorbing sets from the given list of decoding failures
pub fn find_absorbing(
    decoding_failures: Vec<DecodingFailure>,
    parallel: bool,
    ncw: bool,
) -> Vec<AbsorbingDecodingFailure> {
    if parallel {
        decoding_failures
            .into_par_iter()
            .filter_map(|df| AbsorbingDecodingFailure::new(df, ncw))
            .collect()
    } else {
        decoding_failures
            .into_iter()
            .filter_map(|df| AbsorbingDecodingFailure::new(df, ncw))
            .collect()
    }
}

/// Searches for absorbing sets of a given weight for `key`.
pub fn sample_absorbing_sets<const WEIGHT: usize, const LENGTH: usize>(
    key: &QuasiCyclic<WEIGHT, LENGTH>,
    supp_weight: usize,
    samples: usize,
    parallel: bool,
) -> Vec<Vec<Index>> {
    let n = 2 * LENGTH as Index;
    let edges = TannerGraphEdges::new(key);
    if parallel {
        (0..samples)
            .into_par_iter()
            .map(|_| (0..n).choose_multiple(&mut custom_thread_rng(), supp_weight))
            .filter(|supp| graphs::is_absorbing_subgraph(&edges, supp))
            .collect()
    } else {
        (0..samples)
            .map(|_| (0..n).choose_multiple(&mut custom_thread_rng(), supp_weight))
            .filter(|supp| graphs::is_absorbing_subgraph(&edges, supp))
            .collect()
    }
}

/// Enumerates all absorbing sets of a given weight for `key`.
pub fn enumerate_absorbing_sets<const WEIGHT: usize, const LENGTH: usize>(
    key: &QuasiCyclic<WEIGHT, LENGTH>,
    supp_weight: usize,
    parallel: bool,
) -> Vec<Vec<Index>> {
    let n = 2 * LENGTH as Index;
    let edges = TannerGraphEdges::new(key);
    let combinations = (0..n).combinations(supp_weight);
    if parallel {
        combinations
            .par_bridge()
            .filter(|supp| graphs::is_absorbing_subgraph(&edges, supp))
            .collect()
    } else {
        combinations
            .filter(|supp| graphs::is_absorbing_subgraph(&edges, supp))
            .collect()
    }
}
