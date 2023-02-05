use bike_decoder::{
    decoder::DecoderCycle,
    graphs::AbsorbingDecodingFailure,
    keys::QuasiCyclic,
    ncw::ClassifiedVector,
    parameters::GRAY_THRESHOLD_DIFF,
    random::{global_seed, Seed},
    threshold::{bf_masked_threshold, bf_threshold_min},
    vectors::Index,
};
use serde::{Deserialize, Serialize};
use serde_with::{formats::Flexible, serde_as, DurationSecondsWithFrac};
use std::time::Duration;

#[serde_as]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct AnalysisRecord<const WT: usize, const LEN: usize> {
    r: usize,
    d: usize,
    weight: usize,
    gray_threshold_diff: u8,
    bf_threshold_min: u8,
    bf_masked_threshold: u8,
    fixed_key: Option<QuasiCyclic<WT, LEN>>,
    num_processed: usize,
    #[serde(flatten)]
    results: AnalysisResults<WT, LEN>,
    seed: Option<Seed>,
    #[serde_as(as = "DurationSecondsWithFrac<f64, Flexible>")]
    runtime: Duration,
}

impl<const WT: usize, const LEN: usize> AnalysisRecord<WT, LEN> {
    pub fn new(
        fixed_key: Option<QuasiCyclic<WT, LEN>>,
        weight: usize,
        num_processed: usize,
        results: AnalysisResults<WT, LEN>,
        runtime: Duration,
    ) -> Self {
        Self {
            r: LEN,
            d: WT,
            weight,
            gray_threshold_diff: GRAY_THRESHOLD_DIFF,
            bf_threshold_min: bf_threshold_min(WT),
            bf_masked_threshold: bf_masked_threshold(WT),
            fixed_key,
            num_processed,
            results,
            seed: global_seed(),
            runtime,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AnalysisResults<const WT: usize, const LEN: usize> {
    AbsorbingSets {
        data: Vec<Vec<Index>>,
    },
    AbsorbingDecodingFailures {
        data: Vec<AbsorbingDecodingFailure>,
    },
    NcwClassified {
        data: Vec<ClassifiedVector<WT, LEN>>,
    },
    DecoderCycles {
        data: Vec<DecoderCycle>,
        iterations: usize,
    },
}
