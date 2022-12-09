use crate::{
    atls::{self, NearCodewordClass, ErrorVectorSource, TaggedErrorVector},
    decoder,
    keys::{Key, KeyFilter, WeakType, CyclicBlock},
    parameters::*,
    syndrome::Syndrome,
    threshold::ThresholdCache,
    vectors::{SparseErrorVector, InvalidSupport},
};
use std::{
    cmp,
    collections::HashMap,
    fmt::Display,
    fs::File,
    io::Write,
    sync::mpsc,
    time::{Duration, Instant},
    thread,
};
use clap::Parser;
use rand::Rng;
use serde::{Serialize, Deserialize};
use serde_json::json;
use thiserror::Error;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(short='N',long,help="Number of trials (required)")]
    number: f64, // parsed as scientific notation to u64
    #[arg(short, long, default_value_t=0, value_parser=clap::value_parser!(i8).range(-1..=3),
        help="Weak key filter (-1: non-weak only; 0: no filter; 1-3: type 1-3 only)")]
    weak_keys: i8,
    #[arg(short='T',long,default_value_t=3,requires="weak_keys",help="Weak key threshold")]
    weak_key_threshold: usize,
    #[arg(long, help="Always use the specified key (in JSON format)")]
    fixed_key: Option<String>,
    #[arg(short='S',long,help="Use error vectors from near-codeword set A_{t,l}(S)")]
    atls: Option<NearCodewordClass>,
    #[arg(short='l',long,help="Overlap parameter l in A_{t,l}(S)",requires="atls")]
    atls_overlap: Option<usize>,
    #[arg(short,long,help="Output file [default stdout]")]
    output: Option<String>,
    #[arg(short,long,default_value_t=10000.0,help="Max number of decoding failures recorded")]
    recordmax: f64, // parsed as scientific notation to u64
    #[arg(short,long,help="Save to disk frequency [default only at end]")]
    savefreq: Option<f64>, // parsed as scientific notation to u64
    #[arg(long,default_value_t=1,help="Number of threads")]
    threads: u64,
    #[arg(short, long, action = clap::ArgAction::Count,
        help="Print statistics and/or decoding failures [repeat for more verbose, max 3]")]
    verbose: u8,
}

#[derive(Clone, Debug)]
pub struct Settings {
    number_of_trials: u64,
    key_filter: KeyFilter,
    fixed_key: Option<Key>,
    atls: Option<NearCodewordClass>,
    atls_overlap: Option<usize>,
    save_frequency: u64,
    record_max: u64,
    verbose: u8,
    thread_count: u64,
    output_file: Option<String>,
}

impl Settings {
    const MIN_SAVE_FREQUENCY: u64 = 10000;
    const MAX_THREAD_COUNT: u64 = 1024;

    pub fn validate(&self) -> Result<(), SettingsError> {
        if let Some(l) = self.atls_overlap {
            // unwrap() is safe here because atls_overlap requires atls when arguments are parsed
            let sample_class = self.atls.ok_or(SettingsError::DependencyError(
                "atls_overlap requires atls to be set".to_string()))?;
            let l_max = sample_class.max_l();
            if l > l_max {
                return Err(SettingsError::RangeError(
                    format!("l must be in range 0..{} in A_{{t,l}}({})", l_max, sample_class)));
            }
        }
        if self.save_frequency < Self::MIN_SAVE_FREQUENCY {
            return Err(SettingsError::RangeError(
                format!("save_frequency must be >= {}", Self::MIN_SAVE_FREQUENCY)));
        } else if self.thread_count > Self::MAX_THREAD_COUNT {
            return Err(SettingsError::RangeError(
                format!("thread_count must be >= {}", Self::MAX_THREAD_COUNT)));
        }
        if let Some(fixed_key) = &self.fixed_key {
            fixed_key.validate()?;
            if !fixed_key.matches_filter(self.key_filter) {
                return Err(SettingsError::DataError(InvalidSupport(
                    "fixed_key does not match key filter".to_string())));
            }
        }
        Ok(())
    }

    pub fn from_args(args: Args) -> Result<Self, SettingsError> {
        let settings = Self {
            number_of_trials: args.number as u64,
            key_filter: match args.weak_keys {
                0 => KeyFilter::Any,
                -1 => KeyFilter::NonWeak(args.weak_key_threshold),
                1 => KeyFilter::Weak(WeakType::Type1, args.weak_key_threshold),
                2 => KeyFilter::Weak(WeakType::Type2, args.weak_key_threshold),
                3 => KeyFilter::Weak(WeakType::Type3, args.weak_key_threshold),
                _ => {
                    return Err(SettingsError::RangeError(
                        "weak_key_filter must be in {-1, 0, 1, 2, 3}".to_string()));
                }
            },
            fixed_key: if let Some(fixed_key_str) = args.fixed_key {
                let mut key: Key = serde_json::from_str(&fixed_key_str)?;
                key.validate()?;
                key.sort();
                Some(key)
            } else { None },
            atls: args.atls,
            atls_overlap: args.atls_overlap,
            save_frequency: cmp::max(Self::MIN_SAVE_FREQUENCY, args.savefreq.unwrap_or(args.number) as u64),
            record_max: args.recordmax as u64,
            verbose: args.verbose,
            thread_count: cmp::min(cmp::max(args.threads, 1), Self::MAX_THREAD_COUNT),
            output_file: args.output,
        };
        settings.validate()?;
        Ok(settings)
    }
}

#[derive(Error, Debug)]
pub enum SettingsError {
    #[error("fixed_key format must be: {{\"h0\": [...], \"h1\": [...]}}")]
    JsonError(serde_json::Error),
    #[error("blocks of fixed_key must have {} distinct entries in range 0..{}",
        BLOCK_WEIGHT, BLOCK_LENGTH)]
    DataError(InvalidSupport),
    #[error("argument outside of valid range")]
    RangeError(String),
    #[error("broken argument dependency")]
    DependencyError(String),
}

impl From<serde_json::Error> for SettingsError {
    fn from(err: serde_json::Error) -> Self {
        Self::JsonError(err)
    }
}

impl From<InvalidSupport> for SettingsError {
    fn from(err: InvalidSupport) -> Self {
        Self::DataError(err)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecodingResult {
    key: Key,
    vector: TaggedErrorVector,
    success: bool
}

impl DecodingResult {
    #[inline]
    pub fn key(&self) -> &Key {
        &self.key
    }
    #[inline]
    pub fn vector(&self) -> &TaggedErrorVector {
        &self.vector
    }
    #[inline]
    pub fn success(&self) -> &bool {
        &self.success
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DecodingFailureRecord {
    h0: CyclicBlock,
    h1: CyclicBlock,
    e_supp: SparseErrorVector,
    e_source: ErrorVectorSource,
}

impl DecodingFailureRecord {
    pub fn from(result: &DecodingResult) -> Self {
        Self {
            h0: result.key().h0().clone(),
            h1: result.key().h1().clone(),
            e_supp: result.vector().vector().clone(),
            e_source: *result.vector().source(),
        }
    }
}

#[derive(Copy,Clone,Debug,Serialize,Deserialize)]
pub struct ThreadStats {
    thread_id: u64,
    failure_count: u64,
    cached_failure_count: u64,
    trials_completed: u64,
    runtime: Duration,
    done: bool
}

#[derive(Debug)]
pub enum DecoderMessage {
    TrialResult(DecodingResult),
    Stats(ThreadStats)
}

pub fn decoding_trial<R: Rng + ?Sized>(
    settings: &Settings,
    rng: &mut R,
    threshold_cache: &mut ThresholdCache
) -> DecodingResult {
    // Use fixed_key if provided, otherwise generate random key using specified filter
    let key = settings.fixed_key.clone()
        .unwrap_or_else(|| Key::random_filtered(settings.key_filter, rng));
    let tagged_error_vector = if let Some(sample_class) = settings.atls {
        let l = settings.atls_overlap.unwrap_or_else(|| rng.gen_range(0 ..= sample_class.max_l()));
        atls::element_of_atls(&key, sample_class, l, rng)
    } else {
        TaggedErrorVector::from_random(SparseErrorVector::random(rng))
    };
    let e_supp = tagged_error_vector.vector();
    let e_in = e_supp.dense();
    let mut syn = Syndrome::from_sparse(&key, tagged_error_vector.vector());
    let (e_out, same_syndrome) = decoder::bgf_decoder(&key, &mut syn, threshold_cache);
    let success = e_in == e_out;
    assert!(same_syndrome || !success);
    DecodingResult {
        key,
        vector: tagged_error_vector,
        success
    }
}

// Runs decoding_trial in a loop, sending decoding failures (as they occur) and trial
// statistics (periodically) via an asynchronous mpsc sender.
pub fn trial_loop_async(
    thread_id: u64,
    settings: Settings,
    tx: mpsc::Sender<DecoderMessage>
) {
    let start_time = Instant::now();
    let mut rng = crate::random::get_rng();
    let mut cache = ThresholdCache::with_parameters(BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT);
    let mut failure_count = 0;
    let mut cached_failure_count = 0;
    for i in 0..settings.number_of_trials {
        let result = decoding_trial(&settings, &mut rng, &mut cache);
        if !result.success {
            failure_count += 1;
            if failure_count <= settings.record_max {
                let message = DecoderMessage::TrialResult(result);
                tx.send(message).expect("Error transmitting decoding failure");
            } else {
                // When many decoding failures are found, cache decoding failure counts.
                // This prevents the main thread from being flooded with messages,
                // which can be a bottleneck in cases with a very high decoding failure rate.
                cached_failure_count += 1;
            }
        }
        if i != 0 && i % settings.save_frequency == 0 {
            let message = DecoderMessage::Stats(ThreadStats {
                thread_id,
                failure_count,
                cached_failure_count,
                trials_completed: i,
                runtime: start_time.elapsed(),
                done: false
            });
            tx.send(message).expect("Error transmitting thread stats");
            cached_failure_count = 0;
        }
    }
    let message = DecoderMessage::Stats(ThreadStats {
        thread_id,
        failure_count,
        cached_failure_count,
        trials_completed: settings.number_of_trials,
        runtime: start_time.elapsed(),
        done: true
    });
    tx.send(message).expect("Error transmitting thread stats");
}

fn build_json(
    failure_count: u64,
    total_trials: u64,
    decoding_failures: &[DecodingFailureRecord],
    runtime: Duration,
    settings: &Settings,
    thread_stats: Option<serde_json::Value>
) -> serde_json::Value {
    json!({
        "r": BLOCK_LENGTH,
        "d": BLOCK_WEIGHT,
        "t": ERROR_WEIGHT,
        "iterations": NB_ITER,
        "gray_threshold_diff": GRAY_THRESHOLD_DIFF,
        "bf_threshold_min": BF_THRESHOLD_MIN,
        "bf_masked_threshold": BF_MASKED_THRESHOLD,
        "key_filter": settings.key_filter,
        "fixed_key": settings.fixed_key,
        "trials": total_trials,
        "failure_count": failure_count,
        "decoding_failures": decoding_failures,
        "runtime": runtime.as_secs_f64(),
        "thread_stats": thread_stats
    })
}

fn write_to_file_or_stdout(path: &Option<String>, data: &impl Display) {
    match path {
        Some(filename) => {
            let mut file = File::create(filename).expect("Must be able to open file");
            file.write_all(&data.to_string().into_bytes()).expect("Must be able to write to file");
        }
        None => {
            println!("{}", data);
        }
    };
}

fn start_message(settings: &Settings) -> String
{
    let parameter_message = format!("    r = {}, d = {}, t = {}, iterations = {}, tau = {}\n",
        BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT, NB_ITER, GRAY_THRESHOLD_DIFF);
    let weak_key_message = match settings.key_filter {
        KeyFilter::Any => String::new(),
        KeyFilter::NonWeak(threshold) => format!("    Testing only non-weak keys (T = {})\n", threshold),
        KeyFilter::Weak(weak_type, threshold) => {
            format!("    Testing only weak keys of type {} (T = {})\n",
                weak_type.number(), threshold)
        }
    };
    let atls_message = if let Some(atls_set) = settings.atls {
        let l_str = if let Some(l) = settings.atls_overlap {
            l.to_string()
        } else {
            String::from("l")
        };
        format!("    Sampling error vectors from A_{{t,{}}}({})\n", l_str, atls_set)
    } else {
        String::new()
    };
    let thread_message = if settings.thread_count > 1 {
        format!("[running with {} threads]\n", settings.thread_count)
    } else {
        String::new()
    };
    format!("Starting decoding trials (N = {}) with parameters:\n{}{}{}{}",
        settings.number_of_trials, parameter_message, weak_key_message, atls_message, thread_message)
}

fn end_message(failure_count: u64, number_of_trials: u64, runtime: Duration) -> String {
    let dfr = failure_count as f64 / number_of_trials as f64;
    let avg_nanos = runtime.as_nanos() / number_of_trials as u128;
    let (avg_mcs, ns_rem) = (avg_nanos / 1000, avg_nanos % 1000);
    let avg_text = if avg_mcs >= 100 {
        format!("{} μs", avg_mcs)
    } else if avg_mcs >= 10 {
        format!("{}.{} μs", avg_mcs, ns_rem / 100)
    } else if avg_mcs >= 1 {
        format!("{}.{:0width$} μs", avg_mcs, ns_rem / 10, width=2)
    } else {
        format!("{}.{:0width$} μs", avg_mcs, ns_rem, width=3)
    };
    format!("Trials: {}\n\
        Decoding failures: {}\n\
        log2(DFR): {:.2}\n\
        Runtime: {:.3} s\n\
        Average: {}",
        number_of_trials, failure_count, dfr.log2(), runtime.as_secs_f64(), avg_text)
}

pub fn avx2_warning() {
    // Warn if non-AVX2 fallback is used
    if !std::arch::is_x86_feature_detected!("avx2") {
        eprintln!("Warning: AVX2 not supported; falling back to slower method.");
    }
    #[cfg(not(all(
        any(target_arch = "x86", target_arch = "x86_64"),
        target_feature = "avx2"
    )))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            eprintln!("Warning: binary compiled without AVX2 feature; falling back to slower method.");
        }
    }
}

pub fn run_cli_single_threaded(settings: Settings) {
    let mut failure_count = 0;
    let mut decoding_failures = Vec::new();
    let start_time = Instant::now();
    let mut rng = crate::random::get_rng();
    let mut cache = ThresholdCache::with_parameters(BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT);    
    for i in 0..settings.number_of_trials {
        let result = decoding_trial(&settings, &mut rng, &mut cache);
        if !result.success {
            failure_count += 1;
            if failure_count <= settings.record_max {
                if settings.verbose >= 3 {
                    println!("Decoding failure found!");
                    println!("Key: {}\nError vector: {}", result.key, result.vector);
                    if failure_count == settings.record_max {
                        println!("Maximum number of decoding failures recorded.");
                    }
                }
                decoding_failures.push(DecodingFailureRecord::from(&result));
            }
        }
        if i != 0 && i % settings.save_frequency == 0 {
            let runtime = start_time.elapsed();
            let json_output = build_json(failure_count, i, &decoding_failures, runtime, &settings, None);
            write_to_file_or_stdout(&settings.output_file, &json_output);
            if settings.verbose >= 2 {
                println!("Found {} decoding failures in {} trials (runtime: {:.3} s)",
                    failure_count, i, runtime.as_secs_f64());
            }
        }
    }
    // Write final data
    let runtime = start_time.elapsed();
    let json_output = build_json(failure_count, settings.number_of_trials, &decoding_failures,
        runtime, &settings, None);
    write_to_file_or_stdout(&settings.output_file, &json_output);
    if settings.verbose >= 1 {
        println!("{}", end_message(failure_count, settings.number_of_trials, start_time.elapsed()));
    }
}

pub fn run_cli_multithreaded(settings: Settings) {
    let mut failure_count = 0;
    let mut decoding_failures = Vec::new();
    let start_time = Instant::now();
    // Set up (transmitter, receiver) pair and divide trials among threads
    let (tx, rx) = mpsc::channel();
    let trials_per_thread = settings.number_of_trials / settings.thread_count;
    let trials_remainder = settings.number_of_trials % settings.thread_count;
    for thread_id in 0..settings.thread_count {
        // Start the threads, passing them each a copy of the transmitter
        let tx_clone = tx.clone();
        let mut settings = settings.clone();
        settings.number_of_trials = trials_per_thread + if thread_id == 0 { trials_remainder } else { 0 };
        thread::spawn(move || {
            trial_loop_async(thread_id, settings, tx_clone);
        });
    }
    // Drop original transmitter so rx will close when all threads finish
    drop(tx);
    // Track thread stats and how many threads are still in progress
    let mut thread_stats = HashMap::with_capacity(settings.thread_count as usize);
    // Wait for messages
    for received in rx {
        match received {
            // If we receive a decoding failure, record it and increment the failure count
            DecoderMessage::TrialResult(result) => {
                if !result.success {
                    failure_count += 1;
                    if failure_count <= settings.record_max {
                        if settings.verbose >= 3 {
                            println!("Decoding failure found!");
                            println!("Key: {}\nError vector: {}", result.key, result.vector);
                            if failure_count == settings.record_max {
                                println!("Maximum number of decoding failures recorded.");
                            }
                        }
                        decoding_failures.push(DecodingFailureRecord::from(&result));
                    }
                }
            }
            // If we receive updated thread statistics, record and save those
            DecoderMessage::Stats(stats) => {
                failure_count += stats.cached_failure_count;
                thread_stats.insert(stats.thread_id, stats);
                let total_trials = thread_stats.values().map(|st| st.trials_completed).sum();
                let runtime = start_time.elapsed();
                let json_output = build_json(failure_count, total_trials, &decoding_failures, runtime,
                    &settings, Some(json!(thread_stats)));
                write_to_file_or_stdout(&settings.output_file, &json_output);
                if settings.verbose >= 2 {
                    println!("Found {} decoding failures in {} trials (runtime: {:.3} s)",
                        failure_count, total_trials, runtime.as_secs_f64());
                    if stats.done {
                        println!("\nThread {} done. Statistics:", stats.thread_id);
                        println!("    failure count: {}, trials: {}, runtime: {:.3}\n",
                            stats.failure_count, stats.trials_completed,
                            stats.runtime.as_secs_f64());
                    }
                }
            }
        }
    }
    if settings.verbose >= 1 {
        println!("{}", end_message(failure_count, settings.number_of_trials, start_time.elapsed()));
    }
}

pub fn run_cli(settings: Settings) {
    avx2_warning();
    if settings.verbose >= 1 {
        println!("{}", start_message(&settings));
    }
    if settings.thread_count > 1 {
        run_cli_multithreaded(settings);
    } else {
        run_cli_single_threaded(settings);
    }
}
