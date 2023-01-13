use bike_decoder::{
    keys::QuasiCyclic, ncw::NcwOverlaps, random::custom_thread_rng, vectors::Index,
};
use rand::seq::IteratorRandom;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

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
        supp.sort();
        Self {
            key,
            supp,
            overlaps,
        }
    }

    pub fn random(key: &QuasiCyclic<WT, LEN>, supp_weight: usize) -> Self {
        let n = 2 * LEN as Index;
        let mut supp = (0..n).choose_multiple(&mut custom_thread_rng(), supp_weight);
        supp.sort();
        let overlaps = NcwOverlaps::new(key, &supp);
        Self {
            key: key.clone(),
            supp,
            overlaps,
        }
    }

    pub fn sample(
        key: &QuasiCyclic<WT, LEN>,
        supp_weight: usize,
        samples: usize,
        parallel: bool,
    ) -> Vec<Self> {
        if parallel {
            (0..samples)
                .into_par_iter()
                .map(|_| Self::random(key, supp_weight))
                .collect()
        } else {
            (0..samples)
                .map(|_| Self::random(key, supp_weight))
                .collect()
        }
    }
}
