use crate::{
    keys::{CyclicBlock, Key, KeyFilter},
    ncw::{ErrorVectorSource, TaggedErrorVector},
    parameters::*,
    random::Seed,
    vectors::SparseErrorVector,
};
use std::{fmt, time::Duration};
use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DecodingResult {
    key: Key,
    vector: TaggedErrorVector,
    success: bool
}

impl DecodingResult {
    #[inline]
    pub fn from(key: Key, vector: TaggedErrorVector, success: bool) -> Self {
        Self { key, vector, success }
    }

    #[inline]
    pub fn key(&self) -> &Key {
        &self.key
    }

    #[inline]
    pub fn vector(&self) -> &TaggedErrorVector {
        &self.vector
    }

    #[inline]
    pub fn success(&self) -> bool {
        self.success
    }

    #[inline]
    pub fn take_key_vector(self) -> (Key, TaggedErrorVector) {
        (self.key, self.vector)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct DecodingFailureRecord {
    h0: CyclicBlock,
    h1: CyclicBlock,
    e_supp: SparseErrorVector,
    e_source: ErrorVectorSource,
    thread: usize,
}

impl DecodingFailureRecord {
    pub fn from(result: DecodingResult, thread: usize) -> Self {
        let (key, e) = result.take_key_vector();
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
    trials: usize,
    failure_count: usize,
    decoding_failures: Vec<DecodingFailureRecord>,
    seed: Option<Seed>,
    runtime: Duration,
    thread_count: usize,
}

impl DataRecord {
    pub fn new(thread_count: usize, key_filter: KeyFilter, fixed_key: Option<Key>) -> Self {
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
            trials: 0,
            failure_count: 0,
            decoding_failures: Vec::new(),
            seed: None,
            runtime: Duration::new(0, 0),
            thread_count,
        }
    }

    #[inline]
    pub fn set_seed(&mut self, seed: Seed) {
        assert!(self.seed.is_none(), "Can't set seed twice");
        self.seed = Some(seed);
    }

    #[inline]
    pub fn decoding_failures(&self) -> &Vec<DecodingFailureRecord> {
        &self.decoding_failures
    }

    #[inline]
    pub fn failure_count(&self) -> usize {
        self.failure_count
    }

    #[inline]
    pub fn add_to_failure_count(&mut self, count: usize) {
        self.failure_count += count;
    }

    #[inline]
    pub fn trials(&self) -> usize {
        self.trials
    }

    #[inline]
    pub fn set_trials(&mut self, count: usize) {
        assert!(count >= self.trials, "Number of trials cannot decrease");
        self.trials = count;
    }

    #[inline]
    pub fn add_to_trials(&mut self, new_trials: usize) {
        self.trials += new_trials;
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
    pub fn push_decoding_failure(&mut self, df: DecodingFailureRecord) {
        self.decoding_failures.push(df);
    }
}

impl fmt::Display for DataRecord {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", serde_json::to_string(self).or(Err(fmt::Error))?)
    }
}
