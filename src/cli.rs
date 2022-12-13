use crate::{
    decoder,
    keys::{Key, KeyFilter, WeakType, CyclicBlock},
    ncw::{NearCodewordClass, ErrorVectorSource, TaggedErrorVector},
    parameters::*,
    syndrome::Syndrome,
    threshold::ThresholdCache,
    vectors::{SparseErrorVector, InvalidSupport},
};
use std::{
    cmp,
    fmt,
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::mpsc,
    time::{Duration, Instant},
    thread,
};
use clap::Parser;
use rand::Rng;
use serde::{Serialize, Deserialize};
use thiserror::Error;
use uuid::Uuid;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    #[arg(short='N',long,help="Number of trials (required)")]
    number: f64, // parsed as scientific notation to usize
    #[arg(short, long, default_value_t=0, value_parser=clap::value_parser!(i8).range(-1..=3),
        help="Weak key filter (-1: non-weak only; 0: no filter; 1-3: type 1-3 only)")]
    weak_keys: i8,
    #[arg(short='T',long,default_value_t=3,requires="weak_keys",help="Weak key threshold")]
    weak_key_threshold: usize,
    #[arg(long, help="Always use the specified key (in JSON format)")]
    fixed_key: Option<String>,
    #[arg(short='S',long,help="Use error vectors from near-codeword set A_{t,l}(S)")]
    ncw: Option<NearCodewordClass>,
    #[arg(short='l',long,help="Overlap parameter l in A_{t,l}(S)",requires="ncw")]
    ncw_overlap: Option<usize>,
    #[arg(short,long,help="Output file [default stdout]")]
    output: Option<String>,
    #[arg(long, help="If output file already exists, overwrite without creating backup")]
    overwrite: bool,
    #[arg(short,long,default_value_t=10000.0,help="Max number of decoding failures recorded")]
    recordmax: f64, // parsed as scientific notation to usize
    #[arg(short,long,help="Save to disk frequency [default only at end]")]
    savefreq: Option<f64>, // parsed as scientific notation to usize
    #[arg(long,default_value_t=1,help="Number of threads")]
    threads: usize,
    #[arg(short, long, action = clap::ArgAction::Count,
        help="Print statistics and/or decoding failures [repeat for more verbose, max 3]")]
    verbose: u8,
}

#[derive(Clone, Debug)]
pub struct Settings {
    number_of_trials: usize,
    key_filter: KeyFilter,
    fixed_key: Option<Key>,
    ncw_class: Option<NearCodewordClass>,
    ncw_overlap: Option<usize>,
    save_frequency: usize,
    record_max: usize,
    verbose: u8,
    thread_count: usize,
    output_file: Option<PathBuf>,
    overwrite: bool,
}

impl Settings {
    const MIN_SAVE_FREQUENCY: usize = 10000;
    const MAX_THREAD_COUNT: usize = 1024;

    pub fn validate(&self) -> Result<(), RuntimeError> {
        if let Some(l) = self.ncw_overlap {
            let sample_class = self.ncw_class.ok_or(RuntimeError::DependencyError(
                "ncw_overlap requires ncw_class to be set".to_string()))?;
            let l_max = sample_class.max_l();
            if l > l_max {
                return Err(RuntimeError::RangeError(
                    format!("l must be in range 0..{} in A_{{t,l}}({})", l_max, sample_class)));
            }
        }
        if self.save_frequency < Self::MIN_SAVE_FREQUENCY {
            return Err(RuntimeError::RangeError(
                format!("save_frequency must be >= {}", Self::MIN_SAVE_FREQUENCY)));
        } else if self.thread_count > Self::MAX_THREAD_COUNT {
            return Err(RuntimeError::RangeError(
                format!("thread_count must be >= {}", Self::MAX_THREAD_COUNT)));
        }
        if let Some(fixed_key) = &self.fixed_key {
            fixed_key.validate()?;
            if !fixed_key.matches_filter(self.key_filter) {
                return Err(RuntimeError::DataError(InvalidSupport(
                    "fixed_key does not match key filter".to_string())));
            }
        }
        Ok(())
    }

    pub fn from_args(args: Args) -> Result<Self, RuntimeError> {
        let settings = Self {
            number_of_trials: args.number as usize,
            key_filter: match args.weak_keys {
                0 => KeyFilter::Any,
                -1 => KeyFilter::NonWeak(args.weak_key_threshold),
                1 => KeyFilter::Weak(WeakType::Type1, args.weak_key_threshold),
                2 => KeyFilter::Weak(WeakType::Type2, args.weak_key_threshold),
                3 => KeyFilter::Weak(WeakType::Type3, args.weak_key_threshold),
                _ => {
                    return Err(RuntimeError::RangeError(
                        "weak_key_filter must be in {-1, 0, 1, 2, 3}".to_string()));
                }
            },
            fixed_key: if let Some(fixed_key_str) = args.fixed_key {
                let mut key: Key = serde_json::from_str(&fixed_key_str)?;
                key.validate()?;
                key.sort();
                Some(key)
            } else { None },
            ncw_class: args.ncw,
            ncw_overlap: args.ncw_overlap,
            save_frequency: cmp::max(Self::MIN_SAVE_FREQUENCY, args.savefreq.unwrap_or(args.number) as usize),
            record_max: args.recordmax as usize,
            verbose: args.verbose,
            thread_count: cmp::min(cmp::max(args.threads, 1), Self::MAX_THREAD_COUNT),
            output_file: args.output.map(PathBuf::from),
            overwrite: args.overwrite,
        };
        settings.validate()?;
        Ok(settings)
    }
}

#[derive(Error, Debug)]
pub enum RuntimeError {
    #[error("error parsing JSON for fixed_key argument: {0}")]
    JsonError(#[from] serde_json::Error),
    #[error("invalid support for vector or key: {0}")]
    DataError(#[from] InvalidSupport),
    #[error("argument outside of valid range: {0}")]
    RangeError(String),
    #[error("broken argument dependency: {0}")]
    DependencyError(String),
    #[error("error writing to file: {0}")]
    IOError(#[from] io::Error),
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

    #[inline]
    pub fn take_key_vector(self) -> (Key, TaggedErrorVector) {
        (self.key, self.vector)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Copy,Clone,Debug,Serialize,Deserialize)]
pub struct ThreadStats {
    thread_id: usize,
    failure_count: usize,
    #[serde(skip)] cached_failure_count: usize,
    trials: usize,
    runtime: Duration,
    done: bool,
}

impl ThreadStats {
    pub fn initial(thread_id: usize) -> Self {
        Self {
            thread_id,
            failure_count: 0,
            cached_failure_count: 0,
            trials: 0,
            runtime: Duration::new(0, 0),
            done: false,
        }
    }
}

#[derive(Debug)]
pub enum DecoderMessage {
    TrialResult(DecodingResult),
    Stats(ThreadStats)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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
    runtime: Duration,
    thread_count: usize,
    thread_stats: Option<Vec<ThreadStats>>,
}

impl DataRecord {
    pub fn new(settings: &Settings) -> Self {
        Self {
            r: BLOCK_LENGTH,
            d: BLOCK_WEIGHT,
            t: ERROR_WEIGHT,
            iterations: NB_ITER,
            gray_threshold_diff: GRAY_THRESHOLD_DIFF,
            bf_threshold_min: BF_THRESHOLD_MIN,
            bf_masked_threshold: BF_MASKED_THRESHOLD,
            key_filter: settings.key_filter,
            fixed_key: settings.fixed_key.clone(),
            trials: 0,
            failure_count: 0,
            decoding_failures: Vec::new(),
            runtime: Duration::new(0, 0),
            thread_count: settings.thread_count,
            thread_stats: if settings.thread_count > 1 {
                let mut stats = Vec::with_capacity(settings.thread_count);
                for i in 0..settings.thread_count {
                    stats.push(ThreadStats::initial(i));
                }
                Some(stats)
            } else { None },
        }
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
    pub fn update_thread_stats(&mut self, stats: ThreadStats) {
        let thread_stats = self.thread_stats.as_mut()
            .expect("Can't record thread stats, not in multithreaded mode");
        thread_stats[stats.thread_id] = stats;
        self.trials = thread_stats.iter().map(|stats| stats.trials).sum();
    }
}

impl fmt::Display for DataRecord {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", serde_json::json!(self))
    }
}

pub fn decoding_trial<R>(settings: &Settings, rng: &mut R, cache: &mut ThresholdCache)
    -> DecodingResult
    where R: Rng + ?Sized
{
    // Use fixed_key if provided, otherwise generate random key using specified filter
    let key = settings.fixed_key.clone()
        .unwrap_or_else(|| Key::random_filtered(settings.key_filter, rng));
    let tagged_error_vector = if let Some(ncw_class) = settings.ncw_class {
        let l = settings.ncw_overlap.unwrap_or_else(|| rng.gen_range(0 ..= ncw_class.max_l()));
        TaggedErrorVector::near_codeword(&key, ncw_class, l, rng)
    } else {
        TaggedErrorVector::random(rng)
    };
    let e_supp = tagged_error_vector.vector();
    let e_in = e_supp.dense();
    let mut syn = Syndrome::from_sparse(&key, tagged_error_vector.vector());
    let (e_out, same_syndrome) = decoder::bgf_decoder(&key, &mut syn, cache);
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
    thread_id: usize,
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
                trials: i,
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
        trials: settings.number_of_trials,
        runtime: start_time.elapsed(),
        done: true
    });
    tx.send(message).expect("Error transmitting thread stats");
}

fn check_file_writable(output: Option<&Path>, overwrite: bool) -> Result<(), RuntimeError> {
    if let Some(filename) = output {
        if !overwrite && filename.try_exists()? && fs::metadata(filename)?.len() > 0 {
            // If file already exists and is nonempty, copy its contents to a backup file
            fs::copy(filename, &format!("{}-backup-{}", filename.display(), Uuid::new_v4()))?;
        }
        let mut file = File::create(filename)?;
        file.write_all(b"")?;
    }
    Ok(())
}

fn write_to_file_or_stdout(output: Option<&Path>, data: &impl fmt::Display) -> Result<(), RuntimeError> {
    if let Some(filename) = output {
        let mut file = File::create(filename)?;
        file.write_all(&data.to_string().into_bytes())?;
    } else {
        println!("{}", data);
    }
    Ok(())
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
    let ncw_message = if let Some(ncw_class) = settings.ncw_class {
        let l_str = if let Some(l) = settings.ncw_overlap {
            l.to_string()
        } else {
            String::from("l")
        };
        format!("    Sampling error vectors from A_{{t,{}}}({})\n", l_str, ncw_class)
    } else {
        String::new()
    };
    let thread_message = if settings.thread_count > 1 {
        format!("[running with {} threads]\n", settings.thread_count)
    } else {
        String::new()
    };
    format!("Starting decoding trials (N = {}) with parameters:\n{}{}{}{}",
        settings.number_of_trials, parameter_message, weak_key_message, ncw_message, thread_message)
}

fn end_message(failure_count: usize, number_of_trials: usize, runtime: Duration) -> String {
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

pub fn run_cli_single_threaded(settings: Settings) -> Result<(), RuntimeError> {
    let mut data = DataRecord::new(&settings);
    let start_time = Instant::now();
    let mut rng = crate::random::get_rng();
    let mut cache = ThresholdCache::with_parameters(BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT);    
    for i in 0..settings.number_of_trials {
        let result = decoding_trial(&settings, &mut rng, &mut cache);
        if !result.success {
            data.add_to_failure_count(1);
            if data.failure_count() <= settings.record_max {
                if settings.verbose >= 3 {
                    println!("Decoding failure found!");
                    println!("Key: {}\nError vector: {}", result.key, result.vector);
                    if data.failure_count() == settings.record_max {
                        println!("Maximum number of decoding failures recorded.");
                    }
                }
                data.push_decoding_failure(result.into());
            }
        }
        if i != 0 && i % settings.save_frequency == 0 {
            data.set_runtime(start_time.elapsed());
            data.set_trials(i);
            write_to_file_or_stdout(settings.output_file.as_deref(), &data)?;
            if settings.verbose >= 2 {
                println!("Found {} decoding failures in {} trials (runtime: {:.3} s)",
                    data.failure_count(), i, data.runtime().as_secs_f64());
            }
        }
    }
    // Write final data
    data.set_runtime(start_time.elapsed());
    data.set_trials(settings.number_of_trials);
    write_to_file_or_stdout(settings.output_file.as_deref(), &data)?;
    if settings.verbose >= 1 {
        println!("{}", end_message(data.failure_count(), data.trials(), start_time.elapsed()));
    }
    Ok(())
}

pub fn run_cli_multithreaded(settings: Settings) -> Result<(), RuntimeError> {
    let mut data = DataRecord::new(&settings);
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
    // Wait for messages
    for received in rx {
        match received {
            // If we receive a decoding failure, record it and increment the failure count
            DecoderMessage::TrialResult(result) => {
                if !result.success {
                    data.add_to_failure_count(1);
                    if data.failure_count() <= settings.record_max {
                        if settings.verbose >= 3 {
                            println!("Decoding failure found!");
                            println!("Key: {}\nError vector: {}", result.key, result.vector);
                            if data.failure_count() == settings.record_max {
                                println!("Maximum number of decoding failures recorded.");
                            }
                        }
                        data.push_decoding_failure(result.into());
                    }
                }
            }
            // If we receive updated thread statistics, record and save those
            DecoderMessage::Stats(stats) => {
                data.add_to_failure_count(stats.cached_failure_count);
                data.update_thread_stats(stats);
                data.set_runtime(start_time.elapsed());
                write_to_file_or_stdout(settings.output_file.as_deref(), &data)?;
                if settings.verbose >= 2 {
                    println!("Found {} decoding failures in {} trials (runtime: {:.3} s)",
                        data.failure_count(), data.trials(), data.runtime().as_secs_f64());
                    if stats.done {
                        println!("\nThread {} done. Statistics:", stats.thread_id);
                        println!("    failure count: {}, trials: {}, runtime: {:.3}\n",
                            stats.failure_count, stats.trials,
                            stats.runtime.as_secs_f64());
                    }
                }
            }
        }
    }
    if settings.verbose >= 1 {
        println!("{}", end_message(data.failure_count(), data.trials(), start_time.elapsed()));
    }
    Ok(())
}

pub fn run_cli(settings: Settings) -> Result<(), RuntimeError> {
    check_file_writable(settings.output_file.as_deref(), settings.overwrite)?;
    avx2_warning();
    if settings.verbose >= 1 {
        println!("{}", start_message(&settings));
    }
    if settings.thread_count > 1 {
        run_cli_multithreaded(settings)?;
    } else {
        run_cli_single_threaded(settings)?;
    }
    Ok(())
}
