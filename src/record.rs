use crate::{
    keys::{CyclicBlock, Key, KeyFilter},
    ncw::{ErrorVectorSource, TaggedErrorVector},
    parameters::*,
    random::Seed,
    vectors::SparseErrorVector,
};
use std::{fmt, time::Duration};
use serde::{Serialize, Deserialize};
use serde_json::json;

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
}

impl From<DecodingResult> for DecodingFailureRecord {
    fn from(result: DecodingResult) -> Self {
        let (key, e) = result.take_key_vector();
        let (mut h0, mut h1) = key.take_blocks();
        h0.sort();
        h1.sort();
        let (mut e_supp, e_source) = e.take_vector();
        e_supp.sort();
        Self { h0, h1, e_supp, e_source }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ThreadStats {
    thread_id: usize,
    seed: Option<Seed>,
    failure_count: usize,
    #[serde(skip)] cached_failure_count: usize,
    trials: usize,
    runtime: Duration,
    done: bool,
}

impl ThreadStats {
    pub fn new(thread_id: usize) -> Self {
        Self {
            thread_id,
            seed: None,
            failure_count: 0,
            cached_failure_count: 0,
            trials: 0,
            runtime: Duration::new(0, 0),
            done: false,
        }
    }

    #[inline]
    pub fn done(&self) -> bool {
        self.done
    }

    #[inline]
    pub fn set_done(&mut self, done: bool) {
        self.done = done;
    }

    #[inline]
    pub fn id(&self) -> usize {
        self.thread_id
    }

    #[inline]
    pub fn set_seed(&mut self, seed: Seed) {
        assert!(self.seed.is_none(), "Can't set seed twice");
        self.seed = Some(seed);
    }

    #[inline]
    pub fn failure_count(&self) -> usize {
        self.failure_count
    }

    #[inline]
    pub fn increment_failure_count(&mut self, recorded: bool) {
        self.failure_count += 1;
        if !recorded {
            self.cached_failure_count += 1;
        }
    }

    #[inline]
    pub fn reset_cached_failure_count(&mut self) {
        self.cached_failure_count = 0;
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
    pub fn runtime(&self) -> Duration {
        self.runtime
    }

    #[inline]
    pub fn set_runtime(&mut self, runtime: Duration) {
        self.runtime = runtime;
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
    thread_stats: Option<Vec<ThreadStats>>,
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
            thread_stats: if thread_count > 1 {
                let mut stats = Vec::with_capacity(thread_count);
                for i in 0..thread_count {
                    stats.push(ThreadStats::new(i));
                }
                Some(stats)
            } else { None },
        }
    }

    #[inline]
    pub fn record_seed(&mut self, seed: Seed) {
        assert!(self.seed.is_none(), "Can't set seed twice");
        self.seed = Some(seed);
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
        assert_eq!(self.thread_count, 1, "Use thread stats to set trials in multithreaded mode");
        assert!(count >= self.trials, "Number of trials cannot decrease");
        self.trials = count;
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

    #[inline]
    pub fn update_thread_stats(&mut self, mut stats: ThreadStats) {
        let thread_id = stats.id();
        self.add_to_failure_count(stats.cached_failure_count);
        stats.reset_cached_failure_count();
        let thread_stats = self.thread_stats.as_mut()
            .expect("Can't record thread stats, not in multithreaded mode");
        thread_stats[thread_id] = stats;
        self.trials = thread_stats.iter().map(|stats| stats.trials).sum();
    }
}

impl fmt::Display for DataRecord {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", json!(self))
    }
}
