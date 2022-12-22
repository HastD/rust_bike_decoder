use crate::{
    decoder::DecodingFailure,
    keys::{CyclicBlock, Key, KeyFilter},
    ncw::ErrorVectorSource,
    parameters::*,
    random::Seed,
    vectors::SparseErrorVector,
};
use std::{fmt, ops::AddAssign, time::Duration};
use serde::{Serialize, Deserialize};
use thiserror::Error;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RecordedDecodingFailure {
    h0: CyclicBlock,
    h1: CyclicBlock,
    e_supp: SparseErrorVector,
    e_source: ErrorVectorSource,
    thread: usize,
}

impl From<(DecodingFailure, usize)> for RecordedDecodingFailure {
    fn from((df, thread_id): (DecodingFailure, usize)) -> Self {
        Self::from(df, thread_id)
    }
}

impl RecordedDecodingFailure {
    pub fn from(df: DecodingFailure, thread: usize) -> Self {
        let (key, e) = df.take_key_vector();
        let (h0, h1) = key.take_blocks();
        let (e_supp, e_source) = e.take_vector();
        Self {
            h0: h0.sorted(),
            h1: h1.sorted(),
            e_supp: e_supp.sorted(),
            e_source,
            thread,
        }
    }

    #[inline]
    pub fn h0(&self) -> &CyclicBlock {
        &self.h0
    }

    #[inline]
    pub fn h1(&self) -> &CyclicBlock {
        &self.h1
    }

    #[inline]
    pub fn e_supp(&self) -> &SparseErrorVector {
        &self.e_supp
    }

    #[inline]
    pub fn e_source(&self) -> ErrorVectorSource {
        self.e_source
    }

    #[inline]
    pub fn thread(&self) -> usize {
        self.thread
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DataRecord {
    r: usize,
    d: usize,
    t: usize,
    iterations: usize,
    gray_threshold_diff: u8,
    bf_threshold_min: u8,
    bf_masked_threshold: u8,
    key_filter: KeyFilter,
    fixed_key: Option<Key>,
    #[serde(flatten)] decoding_failure_ratio: DecodingFailureRatio,
    decoding_failures: Vec<RecordedDecodingFailure>,
    seed: Seed,
    runtime: Duration,
    thread_count: Option<usize>,
}

impl DataRecord {
    pub fn new(key_filter: KeyFilter, fixed_key: Option<Key>, seed: Seed) -> Self {
        Self {
            r: BLOCK_LENGTH,
            d: BLOCK_WEIGHT,
            t: ERROR_WEIGHT,
            iterations: NB_ITER,
            gray_threshold_diff: GRAY_THRESHOLD_DIFF,
            bf_threshold_min: BF_THRESHOLD_MIN,
            bf_masked_threshold: BF_MASKED_THRESHOLD,
            key_filter,
            fixed_key,
            decoding_failure_ratio: DecodingFailureRatio::default(),
            decoding_failures: Vec::new(),
            seed,
            runtime: Duration::new(0, 0),
            thread_count: None,
        }
    }

    #[inline]
    pub fn seed(&self) -> Seed {
        self.seed
    }

    #[inline]
    pub fn decoding_failures(&self) -> &Vec<RecordedDecodingFailure> {
        &self.decoding_failures
    }

    #[inline]
    pub fn push_decoding_failure(&mut self, df: RecordedDecodingFailure) {
        self.decoding_failures.push(df);
    }

    #[inline]
    pub fn failure_count(&self) -> usize {
        self.decoding_failure_ratio.failure_count()
    }

    #[inline]
    pub fn trials(&self) -> usize {
        self.decoding_failure_ratio.trials()
    }

    #[inline]
    pub fn add_to_failure_count(&mut self, dfr: DecodingFailureRatio) {
        self.decoding_failure_ratio += dfr;
    }

    #[inline]
    pub fn runtime(&self) -> Duration {
        self.runtime
    }

    #[inline]
    pub fn set_runtime(&mut self, runtime: Duration) {
        self.runtime = runtime;
    }

    #[inline]
    pub fn thread_count(&self) -> Option<usize> {
        self.thread_count
    }

    #[inline]
    pub fn set_thread_count(&mut self, count: usize) {
        self.thread_count = Some(count);
    }
}

impl fmt::Display for DataRecord {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", serde_json::to_string(self).or(Err(fmt::Error))?)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DecodingFailureRatio {
    failure_count: usize,
    trials: usize,
}

impl TryFrom<(usize, usize)> for DecodingFailureRatio {
    type Error = InvalidDFRError;

    fn try_from((failure_count, trials): (usize, usize)) -> Result<Self, InvalidDFRError> {
        Self::from(failure_count, trials)
    }
}

impl AddAssign for DecodingFailureRatio {
    fn add_assign(&mut self, other: Self) {
        self.failure_count += other.failure_count;
        self.trials += other.trials;
    }
}

impl DecodingFailureRatio {
    #[inline]
    pub fn from(failure_count: usize, trials: usize) -> Result<Self, InvalidDFRError> {
        if failure_count <= trials {
            Ok(Self { failure_count, trials })
        } else {
            Err(InvalidDFRError)
        }
    }

    #[inline]
    pub fn failure_count(&self) -> usize {
        self.failure_count
    }

    #[inline]
    pub fn trials(&self) -> usize {
        self.trials
    }
}

#[derive(Clone, Copy, Debug, Error)]
#[error("invalid decoding failure ratio: number of failures must be <= number of trials")]
pub struct InvalidDFRError;
