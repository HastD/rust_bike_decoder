use crate::{
    decoder::DecodingFailure,
    keys::{CyclicBlock, Key, KeyFilter},
    ncw::ErrorVectorSource,
    parameters::*,
    random::Seed,
    vectors::SparseErrorVector,
};
use std::{fmt, ops::AddAssign, time::Duration};
use serde::{de::Error, Deserialize, Deserializer, Serialize, Serializer};
use thiserror::Error;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RecordedDecodingFailure {
    h0: CyclicBlock,
    h1: CyclicBlock,
    e_supp: SparseErrorVector,
    e_source: ErrorVectorSource,
    thread: usize,
}

impl RecordedDecodingFailure {
    pub fn new(df: DecodingFailure, thread: usize) -> Self {
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
    #[serde(flatten)]
    decoding_failure_ratio: DecodingFailureRatio,
    decoding_failures: Vec<RecordedDecodingFailure>,
    seed: Seed,
    #[serde(serialize_with = "serialize_duration",
        deserialize_with = "deserialize_duration")]
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
    pub fn num_failures(&self) -> u64 {
        self.decoding_failure_ratio.num_failures()
    }

    #[inline]
    pub fn num_trials(&self) -> u64 {
        self.decoding_failure_ratio.num_trials()
    }

    #[inline]
    pub fn decoding_failure_ratio(&self) -> &DecodingFailureRatio {
        &self.decoding_failure_ratio
    }

    #[inline]
    pub fn add_results(&mut self, dfr: DecodingFailureRatio) {
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
    num_failures: u64,
    num_trials: u64,
}

impl AddAssign for DecodingFailureRatio {
    fn add_assign(&mut self, other: Self) {
        self.num_failures += other.num_failures;
        self.num_trials += other.num_trials;
    }
}

impl DecodingFailureRatio {
    #[inline]
    pub fn new(num_failures: u64, num_trials: u64) -> Result<Self, InvalidDFRError> {
        if num_failures <= num_trials {
            Ok(Self { num_failures, num_trials })
        } else {
            Err(InvalidDFRError)
        }
    }

    #[inline]
    pub fn num_failures(&self) -> u64 {
        self.num_failures
    }

    #[inline]
    pub fn num_trials(&self) -> u64 {
        self.num_trials
    }

    #[inline]
    pub fn as_f64(&self) -> f64 {
        self.num_failures as f64 / self.num_trials as f64
    }
}

#[derive(Clone, Copy, Debug, Error)]
#[error("invalid decoding failure ratio: number of failures must be <= number of trials")]
pub struct InvalidDFRError;

fn serialize_duration<S>(duration: &Duration, ser: S) -> Result<S::Ok, S::Error>
    where S: Serializer,
{
    let secs_str = format!("{}.{:09}", duration.as_secs(), duration.subsec_nanos());
    ser.serialize_str(&secs_str)
}

fn deserialize_duration<'de, D>(de: D) -> Result<Duration, D::Error>
    where D: Deserializer<'de>,
{
    let secs_str = <&str>::deserialize(de)?;
    let secs = secs_str.parse::<f64>().map_err(D::Error::custom)?;
    Duration::try_from_secs_f64(secs).map_err(D::Error::custom)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn data_record_serde() {
        let json_str = r#"{"r":587,"d":15,"t":18,"iterations":7,"gray_threshold_diff":3,
            "bf_threshold_min":8,"bf_masked_threshold":9,"key_filter":"Any","fixed_key":null,
            "num_failures":1,"num_trials":1000000,"decoding_failures":[{"h0":[11,21,100,124,229,
            271,284,307,380,397,420,438,445,495,555],"h1":[10,41,50,59,62,119,153,164,179,208,284,
            384,438,513,554],"e_supp":[42,187,189,336,409,445,464,485,524,532,617,804,877,892,
            1085,1099,1117,1150],"e_source":"Random","thread":2}],"seed":
            "52e19bb7d8474289f86caee35a11ac16dd09902d84fa01173ad83d7b1c376109","runtime":
            "1.478772912","thread_count":8}"#.replace(&[' ', '\n'], "");
        let data_record: DataRecord = serde_json::from_str(&json_str).unwrap();
        assert_eq!(json_str, serde_json::to_string(&data_record).unwrap());
    }
}
