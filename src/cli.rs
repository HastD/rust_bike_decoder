use crate::{
    error::RuntimeError,
    keys::{Key, KeyFilter},
    ncw::TaggedErrorVector,
    parameters::*,
    record::{DecodingResult, ThreadStats, DataRecord},
    settings::{Settings, TrialSettings},
    syndrome::Syndrome,
    threshold::ThresholdCache,
};
use std::{
    convert::AsRef,
    fs::{self, File},
    io::{self, Write},
    path::Path,
    sync::mpsc,
    time::{Duration, Instant},
    thread,
};
use rand::Rng;
use serde::Serialize;
use uuid::Uuid;

#[derive(Debug)]
pub enum TrialMessage {
    TrialResult(DecodingResult),
    Stats(ThreadStats),
    CachedFailureCount(usize),
}

pub fn decoding_trial<R>(settings: &TrialSettings, rng: &mut R, cache: &mut ThresholdCache)
    -> DecodingResult
    where R: Rng + ?Sized
{
    // Use fixed_key if provided, otherwise generate random key using specified filter
    let key = settings.fixed_key().cloned()
        .unwrap_or_else(|| Key::random_filtered(settings.key_filter(), rng));
    let tagged_error_vector = if let Some(ncw_class) = settings.ncw_class() {
        let l = settings.ncw_overlap().unwrap_or_else(|| rng.gen_range(0 ..= ncw_class.max_l()));
        TaggedErrorVector::near_codeword(&key, ncw_class, l, rng)
    } else {
        TaggedErrorVector::random(rng)
    };
    let e_supp = tagged_error_vector.vector();
    let e_in = e_supp.dense();
    let mut syn = Syndrome::from_sparse(&key, tagged_error_vector.vector());
    let (e_out, same_syndrome) = crate::decoder::bgf_decoder(&key, &mut syn, cache);
    let success = e_in == e_out;
    assert!(same_syndrome || !success);
    DecodingResult::from(key, tagged_error_vector, success)
}

// Runs decoding_trial in a loop, sending decoding failures (as they occur) and trial
// statistics (periodically) via an asynchronous mpsc sender.
pub fn trial_loop_async(
    thread_id: usize,
    settings: Settings,
    tx: mpsc::Sender<TrialMessage>
) {
    let start_time = Instant::now();
    let (mut rng, seed) = crate::random::get_rng(settings.seed());
    let mut cache = ThresholdCache::with_parameters(BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT);
    let mut stats = ThreadStats::new(thread_id);
    stats.set_seed(seed);
    let mut cached_failure_count = 0;
    for i in 0..settings.number_of_trials() {
        let result = decoding_trial(&settings.trial_settings(), &mut rng, &mut cache);
        if !result.success() {
            // When many decoding failures are found, we cache decoding failure counts.
            // This prevents the main thread from being flooded with messages,
            // which can be a bottleneck in cases with a very high decoding failure rate.
            stats.increment_failure_count();
            if stats.failure_count() <= settings.record_max() {
                let message = TrialMessage::TrialResult(result);
                tx.send(message).expect("Must be able to transmit decoding failure");
            } else {
                cached_failure_count += 1;
            }
        }
        if i % settings.save_frequency() == 0 && i != 0 {
            if cached_failure_count > 0 {
                let message = TrialMessage::CachedFailureCount(cached_failure_count);
                tx.send(message).expect("Must be able to transmit cached failure count");
                cached_failure_count = 0;
            }
            // Transmit thread statistics
            stats.set_trials(i);
            stats.set_runtime(start_time.elapsed());
            let message = TrialMessage::Stats(stats.clone());
            tx.send(message).expect("Must be able to transmit thread stats");
        }
    }
    if cached_failure_count > 0 {
        let message = TrialMessage::CachedFailureCount(cached_failure_count);
        tx.send(message).expect("Must be able to transmit cached failure count");
    }
    stats.set_trials(settings.number_of_trials());
    stats.set_runtime(start_time.elapsed());
    stats.set_done(true);
    let message = TrialMessage::Stats(stats);
    tx.send(message).expect("Must be able to transmit thread stats");
}

pub fn handle_decoding_failure(result: DecodingResult, data: &mut DataRecord, settings: &Settings) {
    assert!(!result.success(), "handle_decoding_failure should only be called for decoding failures");
    data.add_to_failure_count(1);
    if data.failure_count() <= settings.record_max() {
        if settings.verbose() >= 3 {
            println!("Decoding failure found!");
            println!("Key: {}\nError vector: {}", result.key(), result.vector());
            if data.failure_count() == settings.record_max() {
                println!("Maximum number of decoding failures recorded.");
            }
        }
        data.push_decoding_failure(result.into());
    }
}

pub fn handle_trial_message(
    trial_result: TrialMessage,
    data: &mut DataRecord,
    settings: &Settings,
    start_time: &Instant
) -> Result<(), RuntimeError> {
    match trial_result {
        // If we receive a decoding failure, record it and increment the failure count
        TrialMessage::TrialResult(result) => {
            if !result.success() {
                handle_decoding_failure(result, data, settings);
            }
        }
        // If we receive a count of cached decoding failures, count those
        TrialMessage::CachedFailureCount(count) => {
            data.add_to_failure_count(count);
        }
        // If we receive updated thread statistics, record and save those
        TrialMessage::Stats(stats) => {
            if settings.verbose() >= 2 {
                println!("Found {} decoding failures in {} trials (runtime: {:.3} s)",
                    data.failure_count(), data.trials(), data.runtime().as_secs_f64());
                if stats.done() {
                    println!("\nThread {} done. Statistics:", stats.id());
                    println!("    failure count: {}, trials: {}, runtime: {:.3}\n",
                        stats.failure_count(), stats.trials(),
                        stats.runtime().as_secs_f64());
                }
            }
            data.update_thread_stats(stats);
            data.set_runtime(start_time.elapsed());
            if settings.output_file().is_some() || settings.verbose() >= 2 {
                write_json(settings.output_file(), &data)?;
            }
        }
    }
    Ok(())
}

fn check_file_writable(output: Option<&Path>, overwrite: bool) -> Result<(), RuntimeError> {
    if let Some(filename) = output {
        if !overwrite && filename.try_exists()? && fs::metadata(filename)?.len() > 0 {
            // If file already exists and is nonempty, copy its contents to a backup file
            fs::copy(filename, &format!("{}-backup-{}", filename.display(), Uuid::new_v4()))?;
        }
        File::create(filename)?.write_all(b"")?;
    }
    Ok(())
}

/// Serializes data in JSON format to the specified path, or to standard output if path not provided.
fn write_json<P>(output: Option<P>, data: &impl Serialize) -> Result<(), RuntimeError>
    where P: AsRef<Path> + Copy
{
    if let Some(filename) = output {
        serde_json::to_writer(File::create(filename)?, data)?;
        File::options().append(true).open(filename)?.write_all(b"\n")?;
    } else {
        serde_json::to_writer(io::stdout(), data)?;
        io::stdout().write_all(b"\n")?;
    }
    Ok(())
}

fn start_message(settings: &Settings) -> String {
    let parameter_message = format!("    r = {}, d = {}, t = {}, iterations = {}, tau = {}\n",
        BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT, NB_ITER, GRAY_THRESHOLD_DIFF);
    let weak_key_message = match settings.key_filter() {
        KeyFilter::Any => String::new(),
        KeyFilter::NonWeak(threshold) => format!("    Testing only non-weak keys (T = {})\n", threshold),
        KeyFilter::Weak(weak_type, threshold) => {
            format!("    Testing only weak keys of type {} (T = {})\n",
                weak_type.number(), threshold)
        }
    };
    let ncw_message = settings.ncw_class().map_or(String::new(), |ncw_class| {
        let l_str = settings.ncw_overlap().map_or_else(|| "l".to_string(), |l| l.to_string());
        format!("    Sampling error vectors from A_{{t,{}}}({})\n", l_str, ncw_class)
    });
    let thread_message = if settings.parallel() {
        format!("[running with {} threads]\n", settings.threads())
    } else {
        String::new()
    };
    format!("Starting decoding trials (N = {}) with parameters:\n{}{}{}{}",
        settings.number_of_trials(), parameter_message, weak_key_message, ncw_message, thread_message)
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

pub fn run_cli_single_threaded(settings: Settings) -> Result<(), RuntimeError> {
    let start_time = Instant::now();
    let mut data = DataRecord::new(settings.threads(), settings.key_filter(), settings.fixed_key().cloned());
    let (mut rng, seed) = crate::random::get_rng(settings.seed());
    data.record_seed(seed);
    let mut cache = ThresholdCache::with_parameters(BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT);    
    for i in 0..settings.number_of_trials() {
        let result = decoding_trial(settings.trial_settings(), &mut rng, &mut cache);
        if !result.success() {
            handle_decoding_failure(result, &mut data, &settings);
        }
        if i % settings.save_frequency() == 0 && i != 0 {
            data.set_runtime(start_time.elapsed());
            data.set_trials(i);
            if settings.output_file().is_some() || settings.verbose() >= 2 {
                write_json(settings.output_file(), &data)?;
            }
            if settings.verbose() >= 2 {
                println!("Found {} decoding failures in {} trials (runtime: {:.3} s)",
                    data.failure_count(), i, data.runtime().as_secs_f64());
            }
        }
    }
    // Write final data
    data.set_trials(settings.number_of_trials());
    data.set_runtime(start_time.elapsed());
    write_json(settings.output_file(), &data)?;
    if settings.verbose() >= 1 {
        println!("{}", end_message(data.failure_count(), data.trials(), data.runtime()));
    }
    Ok(())
}

pub fn record_trial_results(rx: mpsc::Receiver<TrialMessage>, settings: Settings, start_time: Instant)
-> Result<(), RuntimeError> {
    let mut data = DataRecord::new(settings.threads(), settings.key_filter(), settings.fixed_key().cloned());
    for received in rx {
        handle_trial_message(received, &mut data, &settings, &start_time)?;
    }
    data.set_runtime(start_time.elapsed());
    write_json(settings.output_file(), &data)?;
    if settings.verbose() >= 1 {
        println!("{}", end_message(data.failure_count(), data.trials(), start_time.elapsed()));
    }
    Ok(())
}

pub fn run_cli_multithreaded(settings: Settings) -> Result<(), RuntimeError> {
    let start_time = Instant::now();
    // Set up (transmitter, receiver) pair and divide trials among threads
    let (tx, rx) = mpsc::channel();
    let settings_clone = settings.clone();
    let handler_thread = thread::spawn(move ||
        record_trial_results(rx, settings_clone, start_time)
    );
    let trials_per_thread = settings.number_of_trials() / settings.threads();
    let trials_remainder = settings.number_of_trials() % settings.threads();
    for thread_id in 0..settings.threads() {
        // Start the threads, passing them each a copy of the transmitter
        let tx = tx.clone();
        let mut settings = settings.clone();
        settings.set_number_of_trials(trials_per_thread + if thread_id == 0 { trials_remainder } else { 0 });
        thread::spawn(move || {
            trial_loop_async(thread_id, settings, tx);
        });
    }
    // Drop original transmitter so rx will close when all threads finish
    drop(tx);
    // Wait for data processing to finish
    handler_thread.join().expect("Recorder thread should not panic")
}

pub fn run_cli(settings: Settings) -> Result<(), RuntimeError> {
    check_file_writable(settings.output_file(), settings.overwrite())?;
    if settings.verbose() >= 1 {
        println!("{}", start_message(&settings));
    }
    if settings.parallel() {
        run_cli_multithreaded(settings)?;
    } else {
        run_cli_single_threaded(settings)?;
    }
    Ok(())
}
