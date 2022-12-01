use crate::{
    atls::{self, NearCodewordClass, ErrorVectorSource, TaggedErrorVector},
    decoder,
    keys::{Key, CyclicBlock},
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
    #[arg(long,conflicts_with_all=["weak_keys","weak_key_threshold"],
        help="Always use the specified key (in JSON format)")]
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

#[derive(Error, Debug)]
pub enum UserInputError {
    #[error("fixed_key format must be: {{\"h0\": [...], \"h1\": [...]}}")]
    JsonError(serde_json::Error),
    #[error("blocks of fixed_key must have {} distinct entries in range 0..{}",
        BLOCK_WEIGHT, BLOCK_LENGTH)]
    DataError(InvalidSupport),
    #[error("argument outside of valid range")]
    RangeError(String),
}

impl From<serde_json::Error> for UserInputError {
    fn from(err: serde_json::Error) -> Self {
        Self::JsonError(err)
    }
}

impl From<InvalidSupport> for UserInputError {
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
    weak_key_filter: i8,
    weak_key_threshold: usize,
    fixed_key: Option<&Key>,
    atls: Option<NearCodewordClass>,
    atls_overlap: Option<usize>,
    rng: &mut R,
    threshold_cache: &mut ThresholdCache
) -> DecodingResult {
    // Use fixed_key if provided, otherwise generate random key using specified filter
    let key = if let Some(key) = fixed_key {
        key.clone()
    } else {
        match weak_key_filter {
            0 => Key::random(rng),
            -1 => Key::random_non_weak(weak_key_threshold, rng),
            1 => Key::random_weak_type1(weak_key_threshold, rng),
            2 => Key::random_weak_type2(weak_key_threshold, rng),
            3 => Key::random_weak_type3(weak_key_threshold, rng),
            _ => panic!("Invalid value for weak key filter (must be -1, 0 (default), 1, 2, or 3)")
        }
    };
    let tagged_error_vector = if let Some(sample_class) = atls {
        let l = atls_overlap.unwrap_or_else(|| rng.gen_range(0 ..= sample_class.max_l()));
        atls::element_of_atls(&key, sample_class, l, rng)
    } else {
        TaggedErrorVector::from_random(SparseErrorVector::random(rng))
    };
    let mut syn = Syndrome::from_sparse(&key, tagged_error_vector.vector());
    let (_e_out, success) = decoder::bgf_decoder(&key, &mut syn, threshold_cache);
    DecodingResult { key, vector: tagged_error_vector, success }
}

// Runs decoding_trial in a loop, sending decoding failures (as they occur) and trial
// statistics (periodically) via an asynchronous mpsc sender.
pub fn trial_loop_async(
    thread_id: u64,
    number_of_trials: u64,
    weak_key_filter: i8,
    weak_key_threshold: usize,
    fixed_key: Option<&Key>,
    atls: Option<NearCodewordClass>,
    atls_overlap: Option<usize>,
    save_frequency: u64,
    record_max: u64,
    tx: mpsc::Sender<DecoderMessage>
) {
    let start_time = Instant::now();
    let mut rng = crate::random::get_rng();
    let (r, d, t) = (BLOCK_LENGTH as u32, BLOCK_WEIGHT as u32, ERROR_WEIGHT as u32);
    let mut cache = ThresholdCache::with_parameters(r, d, t);
    let mut failure_count = 0;
    let mut cached_failure_count = 0;
    for i in 0..number_of_trials {
        let result = decoding_trial(weak_key_filter, weak_key_threshold, fixed_key, atls, atls_overlap,
            &mut rng, &mut cache);
        if !result.success {
            failure_count += 1;
            if failure_count <= record_max {
                let message = DecoderMessage::TrialResult(result);
                tx.send(message).expect("Error transmitting decoding failure");
            } else {
                // When many decoding failures are found, cache decoding failure counts.
                // This prevents the main thread from being flooded with messages,
                // which can be a bottleneck in cases with a very high decoding failure rate.
                cached_failure_count += 1;
            }
        }
        if i != 0 && i % save_frequency == 0 {
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
        trials_completed: number_of_trials,
        runtime: start_time.elapsed(),
        done: true
    });
    tx.send(message).expect("Error transmitting thread stats");
}

fn build_json(
    failure_count: u64,
    number_of_trials: u64,
    decoding_failures: &[DecodingFailureRecord],
    weak_key_filter: i8,
    weak_key_threshold: usize,
    fixed_key: Option<&Key>,
    runtime: Duration,
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
        "weak_key_filter": weak_key_filter,
        "weak_key_threshold": weak_key_threshold,
        "fixed_key": fixed_key,
        "trials": number_of_trials,
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

pub fn run_cli(args: Args) -> Result<(), UserInputError> {
    let number_of_trials = args.number as u64;
    let weak_key_filter = args.weak_keys;
    let weak_key_threshold = if weak_key_filter == 0 { 0 } else { args.weak_key_threshold };
    // If set, this key will be used for all decoding trials
    let fixed_key: Option<Key> = if let Some(fixed_key_str) = args.fixed_key {
        let key: Key = serde_json::from_str(&fixed_key_str)?;
        key.validate()?;
        Some(key)
    } else { None };
    if let Some(l) = args.atls_overlap {
        // unwrap() is safe here because atls_overlap requires atls when arguments are parsed
        let sample_class = args.atls.unwrap();
        let l_max = sample_class.max_l();
        if l > l_max {
            return Err(UserInputError::RangeError(format!("l must be in range 0..{} in A_{{t,l}}({})", l_max, sample_class)));
        }
    }
    let thread_count = cmp::min(cmp::max(args.threads, 1), 1024);
    let record_max = args.recordmax as u64;
    let save_frequency = cmp::max(10000, args.savefreq.unwrap_or(args.number) as u64);
    let (r, d, t) = (BLOCK_LENGTH as u32, BLOCK_WEIGHT as u32, ERROR_WEIGHT as u32);
    let mut failure_count = 0;
    let mut decoding_failures: Vec<DecodingFailureRecord> = Vec::new();
    if args.verbose >= 1 {
        println!("Starting decoding trials (N = {}) with parameters:", number_of_trials);
        println!("    r = {}, d = {}, t = {}, iterations = {}, tau = {}",
            r, d, t, NB_ITER, GRAY_THRESHOLD_DIFF);
        match weak_key_filter {
            -1 => {
                println!("    Testing only non-weak keys (T = {})", weak_key_threshold);
            }
            0 => {}
            filter => {
                println!("    Testing only weak keys of type {} (T = {})", filter, weak_key_threshold);
            }
        }
        if let Some(atls_set) = args.atls {
            let l_str = if let Some(l) = args.atls_overlap {
                l.to_string()
            } else {
                String::from("l")
            };
            println!("    Sampling error vectors from A_{{t,{}}}({})", l_str, atls_set);
        }
        if thread_count > 1 {
            println!("[running with {} threads]\n", thread_count);
        }
    }
    let start_time = Instant::now();
    if thread_count > 1 { // multi-threaded mode
        // Set up (transmitter, receiver) pair and divide trials among threads
        let (tx, rx) = mpsc::channel();
        let trials_per_thread = number_of_trials / thread_count;
        let trials_remainder = number_of_trials % thread_count;
        for thread_id in 0..thread_count {
            // Start the threads, passing them each a copy of the transmitter
            let tx_clone = tx.clone();
            let fixed_key_clone = fixed_key.clone();
            thread::spawn(move || {
                let number_of_trials = trials_per_thread + if thread_id == 0
                    { trials_remainder } else { 0 };
                trial_loop_async(thread_id, number_of_trials, weak_key_filter, weak_key_threshold,
                    fixed_key_clone.as_ref(), args.atls, args.atls_overlap, save_frequency, record_max, tx_clone);
            });
        }
        // Drop original transmitter so rx will close when all threads finish
        std::mem::drop(tx);
        // Track thread stats and how many threads are still in progress
        let mut thread_stats: HashMap<u64, ThreadStats> = HashMap::with_capacity(thread_count as usize);
        // Wait for messages
        for received in rx {
            match received {
                // If we receive a decoding failure, record it and increment the failure count
                DecoderMessage::TrialResult(result) => {
                    if !result.success {
                        failure_count += 1;
                        if failure_count <= record_max {
                            if args.verbose >= 3 {
                                println!("Decoding failure found!");
                                println!("Key: {}\nError vector: {}", result.key, result.vector);
                                if failure_count == record_max {
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
                    let json_output = build_json(failure_count, total_trials, &decoding_failures,
                        weak_key_filter, weak_key_threshold, fixed_key.as_ref(), runtime,
                        Some(json!(thread_stats)));
                    write_to_file_or_stdout(&args.output, &json_output);
                    if args.verbose >= 2 {
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
    } else { // synchronous decoding trial loop
        let mut rng = crate::random::get_rng();
        let mut cache = ThresholdCache::with_parameters(r, d, t);    
        for i in 0..number_of_trials {
            let result = decoding_trial(weak_key_filter, weak_key_threshold, fixed_key.as_ref(),
                args.atls, args.atls_overlap, &mut rng, &mut cache);
            if !result.success {
                failure_count += 1;
                if failure_count <= record_max {
                    if args.verbose >= 3 {
                        println!("Decoding failure found!");
                        println!("Key: {}\nError vector: {}", result.key, result.vector);
                        if failure_count == record_max {
                            println!("Maximum number of decoding failures recorded.");
                        }
                    }
                    decoding_failures.push(DecodingFailureRecord::from(&result));
                }
            }
            if i != 0 && i % save_frequency == 0 {
                let runtime = start_time.elapsed();
                let json_output = build_json(failure_count, i, &decoding_failures, weak_key_filter,
                    weak_key_threshold, fixed_key.as_ref(), runtime, None);
                write_to_file_or_stdout(&args.output, &json_output);
                if args.verbose >= 2 {
                    println!("Found {} decoding failures in {} trials (runtime: {:.3} s)",
                        failure_count, i, runtime.as_secs_f64());
                }
            }
        }
        // Write final data
        let runtime = start_time.elapsed();
        let json_output = build_json(failure_count, number_of_trials, &decoding_failures,
            weak_key_filter, weak_key_threshold, fixed_key.as_ref(), runtime, None);
        write_to_file_or_stdout(&args.output, &json_output);
    }
    if args.verbose >= 1 {
        println!("{}", end_message(failure_count, number_of_trials, start_time.elapsed()));
    }
    Ok(())
}
