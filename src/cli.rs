use crate::{
    error::RuntimeError,
    keys::{Key, KeyFilter},
    ncw::TaggedErrorVector,
    parameters::*,
    random::JUMPS,
    record::{DecodingResult, DecodingFailureRecord, DataRecord},
    settings::{Settings, TrialSettings},
    syndrome::Syndrome,
};
use std::{
    cmp,
    convert::AsRef,
    fs::{self, File},
    io::{self, Write},
    path::Path,
    sync::mpsc,
    time::{Duration, Instant},
    thread,
};
use rand::Rng;
use rayon::prelude::*;
use serde::Serialize;
use uuid::Uuid;

pub fn decoding_trial(settings: &TrialSettings) -> DecodingResult {
    let mut rng = crate::random::custom_thread_rng();
    // Use fixed_key if provided, otherwise generate random key using specified filter
    let key = settings.fixed_key().cloned()
        .unwrap_or_else(|| Key::random_filtered(settings.key_filter(), &mut rng));
    let tagged_error_vector = if let Some(ncw_class) = settings.ncw_class() {
        let l = settings.ncw_overlap().unwrap_or_else(|| rng.gen_range(0 ..= ncw_class.max_l()));
        TaggedErrorVector::near_codeword(&key, ncw_class, l, &mut rng)
    } else {
        TaggedErrorVector::random(&mut rng)
    };
    let e_supp = tagged_error_vector.vector();
    let e_in = e_supp.dense();
    let mut syn = Syndrome::from_sparse(&key, tagged_error_vector.vector());
    let (e_out, same_syndrome) = crate::decoder::bgf_decoder(&key, &mut syn);
    let success = e_in == e_out;
    assert!(same_syndrome || !success);
    DecodingResult::from(key, tagged_error_vector, success)
}

pub fn trial_iteration(settings: &TrialSettings, tx: &mpsc::Sender<(DecodingResult, usize)>) -> usize {
    let result = decoding_trial(settings);
    if result.success() {
        0
    } else {
        // Attempt to send decoding failure, but ignore errors, as the receiver may
        // choose to hang up after receiving the maximum number of decoding failures.
        tx.send((result, JUMPS.with(|x| *x))).ok();
        1
    }
}

// Runs decoding_trial in a loop, sending decoding failures via tx_results and
// progress updates (counts of decoding failures and trials run) via tx_progress.
pub fn trial_loop(
    settings: &Settings,
    tx_progress: mpsc::Sender<(usize, usize)>,
    tx_results: mpsc::Sender<(DecodingResult, usize)>,
) {
    let mut trials_remaining = settings.number_of_trials();
    while trials_remaining > 0 {
        let tx_results = tx_results.clone();
        let new_trials = cmp::min(trials_remaining, settings.save_frequency());
        let new_failure_count = (0..new_trials).into_par_iter().map_with(
            (settings.trial_settings(), tx_results),
            |(settings, tx), _| trial_iteration(&settings, &tx)
        ).sum();
        tx_progress.send((new_failure_count, new_trials)).expect("Must be able to transmit progress");
        trials_remaining -= new_trials;
    }
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
        let thread_count = if settings.threads() == 0 { num_cpus::get() } else { settings.threads() };
        format!("[running with {} threads]\n", thread_count)
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

pub fn handle_decoding_failure(result: DecodingResult, thread: usize,
        data: &mut DataRecord, settings: &Settings) {
    assert!(!result.success(), "handle_decoding_failure should only be called for decoding failures");
    let recorded_failure_count = data.decoding_failures().len();
    let verbose = settings.verbose() >= 3;
    if recorded_failure_count < settings.record_max() {
        if verbose {
            println!("Decoding failure found!");
            println!("Key: {}\nError vector: {}", result.key(), result.vector());
            if recorded_failure_count + 1 == settings.record_max() {
                println!("Maximum number of decoding failures recorded.");
            }    
        }
        data.push_decoding_failure(DecodingFailureRecord::from(result, thread));
    }
}

fn handle_progress(new_failure_count: usize, new_trials: usize, data: &mut DataRecord,
        settings: &Settings, start_time: Instant) -> Result<(), RuntimeError> {
    data.add_to_failure_count(new_failure_count);
    data.add_to_trials(new_trials);
    data.set_runtime(start_time.elapsed());
    if settings.output_file().is_some() || settings.verbose() >= 2 {
        write_json(settings.output_file(), &data)?;
    }    
    if settings.verbose() >= 2 {
        println!("Found {} decoding failures in {} trials (runtime: {:.3} s)",
            data.failure_count(), data.trials(), data.runtime().as_secs_f64());
    }
    Ok(())
}

pub fn record_trial_results(
    rx_progress: mpsc::Receiver<(usize, usize)>,
    rx_results: mpsc::Receiver<(DecodingResult, usize)>,
    settings: Settings,
    start_time: Instant
) -> Result<(usize, usize, Duration), RuntimeError> {
    let mut data = DataRecord::new(settings.key_filter(), settings.fixed_key().cloned());
    data.set_seed(crate::random::global_seed().expect("Global seed should be set"));
    'outer: loop {
        for (result, thread) in rx_results.try_iter() {
            if !result.success() {
                handle_decoding_failure(result, thread, &mut data, &settings);
                if data.decoding_failures().len() == settings.record_max() {
                    break 'outer;
                }
            }
        }
        match rx_progress.try_recv() {
            Ok((new_failure_count, new_trials)) => {
                handle_progress(new_failure_count, new_trials, &mut data, &settings, start_time)?;
            }
            Err(mpsc::TryRecvError::Empty) => { thread::sleep(Duration::from_millis(100)); }
            Err(mpsc::TryRecvError::Disconnected) => { break; }
        }
    }
    // Drops the results receiver so no more decoding failures are handled
    drop(rx_results);
    for (new_failure_count, new_trials) in rx_progress {
        handle_progress(new_failure_count, new_trials, &mut data, &settings, start_time)?;
    }
    data.update_thread_count();
    data.set_runtime(start_time.elapsed());
    write_json(settings.output_file(), &data)?;
    Ok((data.failure_count(), data.trials(), start_time.elapsed()))
}

pub fn run_cli_multithreaded(settings: Settings) -> Result<(), RuntimeError> {
    let start_time = Instant::now();
    rayon::ThreadPoolBuilder::new().num_threads(settings.threads()).build_global()
        .expect("Should be able to construct thread pool");
    // Set up (transmitter, receiver) pair and divide trials among threads
    let (tx_progress, rx_progress) = mpsc::channel();
    let (tx_results, rx_results) = mpsc::channel();
    let settings_clone = settings.clone();
    let handler_thread = thread::spawn(move ||
        record_trial_results(rx_progress, rx_results, settings_clone, start_time)
    );
    trial_loop(&settings, tx_progress, tx_results);
    // Wait for data processing to finish
    let (failure_count, trials, runtime) = handler_thread.join()
        .expect("Recorder thread should not panic")?;
    if settings.verbose() >= 1 {
        println!("{}", end_message(failure_count, trials, runtime));
    }
    Ok(())
}

pub fn run_cli_single_threaded(settings: Settings) -> Result<(), RuntimeError> {
    let start_time = Instant::now();
    let mut data = DataRecord::new(settings.key_filter(), settings.fixed_key().cloned());
    data.set_seed(crate::random::global_seed().expect("Global seed should be set"));
    for i in 0..settings.number_of_trials() {
        let result = decoding_trial(settings.trial_settings());
        if !result.success() {
            data.add_to_failure_count(1);
            handle_decoding_failure(result, JUMPS.with(|x| *x), &mut data, &settings);
        }
        if i % settings.save_frequency() == 0 && i != 0 {
            data.set_trials(i);
            data.set_runtime(start_time.elapsed());
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

pub fn run_cli(settings: Settings) -> Result<(), RuntimeError> {
    crate::random::get_or_insert_global_seed(settings.seed());
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
