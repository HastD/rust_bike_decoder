use bike_decoder::{
    decoder::DecodingFailure,
    keys::{Key, KeyFilter},
    parameters::*,
    random::Seed,
    threshold::{bf_masked_threshold, bf_threshold_min},
};
use getset::{CopyGetters, Getters, Setters};
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use std::{fmt, ops::AddAssign, time::Duration};
use thiserror::Error;

#[derive(Clone, CopyGetters, Debug, Deserialize, Getters, Serialize, Setters)]
pub struct DataRecord {
    #[getset(get_copy = "pub")]
    r: usize,
    #[getset(get_copy = "pub")]
    d: usize,
    #[getset(get_copy = "pub")]
    t: usize,
    #[getset(get_copy = "pub")]
    iterations: usize,
    #[getset(get_copy = "pub")]
    gray_threshold_diff: u8,
    #[getset(get_copy = "pub")]
    bf_threshold_min: u8,
    #[getset(get_copy = "pub")]
    bf_masked_threshold: u8,
    #[getset(get_copy = "pub")]
    key_filter: KeyFilter,
    #[getset(get = "pub")]
    fixed_key: Option<Key>,
    #[getset(get = "pub")]
    #[serde(flatten)]
    decoding_failure_ratio: DecodingFailureRatio,
    #[getset(get = "pub")]
    decoding_failures: Vec<DecodingFailure>,
    #[getset(get_copy = "pub")]
    seed: Seed,
    #[getset(get_copy = "pub", set = "pub")]
    #[serde(
        serialize_with = "serialize_duration",
        deserialize_with = "deserialize_duration"
    )]
    runtime: Duration,
    #[getset(get_copy = "pub", set = "pub")]
    thread_count: Option<u32>,
}

impl DataRecord {
    pub fn new(key_filter: KeyFilter, fixed_key: Option<Key>, seed: Seed) -> Self {
        Self {
            r: BLOCK_LENGTH,
            d: BLOCK_WEIGHT,
            t: ERROR_WEIGHT,
            iterations: NB_ITER,
            gray_threshold_diff: GRAY_THRESHOLD_DIFF,
            bf_threshold_min: bf_threshold_min(BLOCK_WEIGHT),
            bf_masked_threshold: bf_masked_threshold(BLOCK_WEIGHT),
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
    pub fn push_decoding_failure(&mut self, df: DecodingFailure) {
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
    pub fn add_results(&mut self, dfr: DecodingFailureRatio) {
        self.decoding_failure_ratio += dfr;
    }
}

impl fmt::Display for DataRecord {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", serde_json::to_string(self).or(Err(fmt::Error))?)
    }
}

#[derive(Clone, CopyGetters, Debug, Default, Serialize, Deserialize)]
#[getset(get_copy = "pub")]
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
            Ok(Self {
                num_failures,
                num_trials,
            })
        } else {
            Err(InvalidDFRError)
        }
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
where
    S: Serializer,
{
    let secs_str = format!("{}.{:09}", duration.as_secs(), duration.subsec_nanos());
    ser.serialize_str(&secs_str)
}

struct DurationVisitor;

impl<'de> de::Visitor<'de> for DurationVisitor {
    type Value = Duration;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("a duration in seconds (as string or floating-point)")
    }

    fn visit_f64<E>(self, secs: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Duration::try_from_secs_f64(secs).map_err(E::custom)
    }

    fn visit_str<E>(self, secs_str: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        let secs = secs_str.parse::<f64>().map_err(|_| {
            E::invalid_type(
                de::Unexpected::Str(secs_str),
                &"a string containing a valid float literal",
            )
        })?;
        self.visit_f64(secs)
    }

    fn visit_u64<E>(self, secs: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Duration::from_secs(secs))
    }
}

fn deserialize_duration<'de, D>(deserializer: D) -> Result<Duration, D::Error>
where
    D: Deserializer<'de>,
{
    deserializer.deserialize_any(DurationVisitor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn json_test_string() -> String {
        r#"{"r":587,"d":15,"t":18,"iterations":7,"gray_threshold_diff":3,"bf_threshold_min":8,
        "bf_masked_threshold":9,"key_filter":"Any","fixed_key":null,"num_failures":1,"num_trials":
        1000000,"decoding_failures":[{"h0":[11,21,100,124,229,271,284,307,380,397,420,438,445,495,
        555],"h1":[10,41,50,59,62,119,153,164,179,208,284,384,438,513,554],"e_supp":[42,187,189,
        336,409,445,464,485,524,532,617,804,877,892,1085,1099,1117,1150],"e_source":"Random",
        "thread":2}],"seed":"52e19bb7d8474289f86caee35a11ac16dd09902d84fa01173ad83d7b1c376109",
        "runtime":"1.478772912","thread_count":8}"#
            .split_whitespace()
            .collect()
    }

    #[test]
    fn data_record_serde_str() {
        let json_str = json_test_string();
        let data_record: DataRecord = serde_json::from_str(&json_str).unwrap();
        assert_eq!(json_str, serde_json::to_string(&data_record).unwrap());
    }

    #[test]
    fn data_record_serde_value() {
        let json_data: Value = serde_json::from_str(&json_test_string()).unwrap();
        let data_record: DataRecord = serde_json::from_value(json_data.clone()).unwrap();
        assert_eq!(json_data, serde_json::to_value(data_record).unwrap());
    }
}
