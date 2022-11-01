#![forbid(unsafe_code)]

pub mod decoder;
//pub mod graphs;
pub mod keys;
pub mod parameters;
pub mod random;
pub mod syndrome;
pub mod threshold;
pub mod vectors;

use crate::{
    parameters::*,
    keys::Key,
    threshold::ThresholdCache,
    vectors::SparseErrorVector,
    syndrome::Syndrome
};
use std::{
    cmp,
    collections::HashMap,
    fmt::Display,
    fs::File,
    io::Write,
    sync::mpsc,
    time::{Duration, Instant},
    thread
};
use clap::Parser;
use rand::Rng;
use serde::{Serialize, Deserialize};
use serde_json::json;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short='N',long,help="Number of trials (required)")]
    number: f64, // parsed as scientific notation to u64
    #[arg(short,long,default_value_t=0,
        help="Weak key filter (-1: non-weak only; 0: no filter; 1-3: type 1-3 only)")]
    weak_keys: i8,
    #[arg(short='T',long,default_value_t=3,help="Weak key threshold")]
    weak_key_threshold: usize,
    #[arg(short,long,help="Output file [default stdout]")]
    output: Option<String>,
    #[arg(short,long,help="Max number of decoding failures recorded [default all]")]
    recordmax: Option<f64>, // parsed as scientific notation to u64
    #[arg(short,long,help="Save to disk frequency [default only at end]")]
    savefreq: Option<f64>, // parsed as scientific notation to u64
    #[arg(long,default_value_t=1,help="Number of threads")]
    threads: u64,
    #[arg(short,long,help="Print decoding failures as they are found")]
    verbose: bool,
}

fn decoding_trial<R: Rng + ?Sized>(
    weak_key_filter: i8,
    weak_key_threshold: usize,
    rng: &mut R,
    threshold_cache: &mut ThresholdCache
) -> DecodingResult {
    let key = match weak_key_filter {
        0 => Key::random(rng),
        -1 => Key::random_non_weak(weak_key_threshold, rng),
        1 => Key::random_weak_type1(weak_key_threshold, rng),
        2 => Key::random_weak_type2(weak_key_threshold, rng),
        3 => Key::random_weak_type3(weak_key_threshold, rng),
        _ => panic!("Invalid value for weak key filter (must be -1, 0 (default), 1, 2, or 3)")
    };
    let e_supp = SparseErrorVector::random(rng);
    let mut syn = Syndrome::from_sparse(&key, &e_supp);
    let (_e_out, success) = decoder::bgf_decoder(&key, &mut syn, threshold_cache);
    DecodingResult { key, e_supp, success }
}

#[derive(Debug)]
struct DecodingResult {
    key: Key,
    e_supp: SparseErrorVector,
    success: bool
}

#[derive(Copy,Clone,Debug,Serialize,Deserialize)]
struct ThreadStats {
    thread_id: u64,
    failure_count: u64,
    trials_completed: u64,
    runtime: Duration,
    done: bool
}

#[derive(Debug)]
enum DecoderMessage {
    TrialResult(DecodingResult),
    Stats(ThreadStats)
}

// Runs decoding_trial in a loop, sending decoding failures (as they occur) and trial
// statistics (periodically) via an asynchronous mpsc sender.
fn trial_loop_async(
    thread_id: u64,
    number_of_trials: u64,
    weak_key_filter: i8,
    weak_key_threshold: usize,
    save_frequency: u64,
    tx: mpsc::Sender<DecoderMessage>
) {
    let start_time = Instant::now();
    let mut rng = crate::random::get_rng();
    let (r, d, t) = (BLOCK_LENGTH as u32, BLOCK_WEIGHT as u32, ERROR_WEIGHT as u32);
    let mut cache = ThresholdCache::with_parameters(r, d, t);
    let mut failure_count = 0;
    for i in 0..number_of_trials {
        let result = decoding_trial(weak_key_filter, weak_key_threshold, &mut rng, &mut cache);
        if !result.success {
            failure_count += 1;
            let message = DecoderMessage::TrialResult(result);
            tx.send(message).expect("Error transmitting decoding failure");
        }
        if i != 0 && i % save_frequency == 0 {
            let message = DecoderMessage::Stats(ThreadStats {
                thread_id,
                failure_count,
                trials_completed: i,
                runtime: start_time.elapsed(),
                done: false
            });
            tx.send(message).expect("Error transmitting thread stats");
        }
    }
    let message = DecoderMessage::Stats(ThreadStats {
        thread_id,
        failure_count,
        trials_completed: number_of_trials,
        runtime: start_time.elapsed(),
        done: true
    });
    tx.send(message).expect("Error transmitting thread stats");
}

fn build_json(
    failure_count: u64,
    number_of_trials: u64,
    decoding_failures: &Vec<(Key, SparseErrorVector)>,
    weak_key_filter: i8,
    weak_key_threshold: usize,
    runtime: Duration,
    thread_stats: Option<serde_json::Value>
) -> serde_json::Value {
    json!({
        "r": BLOCK_LENGTH,
        "d": BLOCK_WEIGHT,
        "t": ERROR_WEIGHT,
        "iterations": NB_ITER,
        "bgf_threshold": BGF_THRESHOLD,
        "weak_key_filter": weak_key_filter,
        "weak_key_threshold": weak_key_threshold,
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

fn print_end_message(failure_count: u64, number_of_trials: u64, runtime: Duration) {
    println!("Trials: {}", number_of_trials);
    println!("Decoding failures: {}", failure_count);
    let dfr = failure_count as f64 / number_of_trials as f64;
    println!("log2(DFR): {:.2}", dfr.log2());
    println!("Runtime: {:.3} s", runtime.as_secs_f64());
    let avg_nanos = runtime.as_nanos() / number_of_trials as u128;
    let (avg_mcs, ns_rem) = (avg_nanos / 1000, avg_nanos % 1000);
    if avg_mcs >= 100 {
        println!("Average: {} μs", avg_mcs);
    } else if avg_mcs >= 10 {
        println!("Average: {}.{} μs", avg_mcs, ns_rem / 100);
    } else if avg_mcs >= 1 {
        println!("Average: {}.{:0width$} μs", avg_mcs, ns_rem / 10, width=2);
    } else {
        println!("Average: {}.{:0width$} μs", avg_mcs, ns_rem, width=3);
    }
}

fn main() {
    let args = Args::parse();
    let number_of_trials = args.number as u64;
    let weak_key_filter = args.weak_keys;
    let weak_key_threshold = if weak_key_filter == 0 { 0 } else { args.weak_key_threshold };
    let thread_count = cmp::min(cmp::max(args.threads, 1), 1024);
    let record_max = args.recordmax.unwrap_or(args.number) as u64;
    let save_frequency = cmp::max(10000, args.savefreq.unwrap_or(args.number) as u64);
    let (r, d, t) = (BLOCK_LENGTH as u32, BLOCK_WEIGHT as u32, ERROR_WEIGHT as u32);
    let mut failure_count = 0;
    let mut decoding_failures: Vec<(Key, SparseErrorVector)> = Vec::new();
    if args.verbose {
        println!("Starting decoding trials (N = {}) with parameters:", number_of_trials);
        println!("    r = {}, d = {}, t = {}, iterations = {}, tau = {}",
            r, d, t, NB_ITER, BGF_THRESHOLD);
        match weak_key_filter {
            -1 => {
                println!("    Testing only non-weak keys (T = {})", weak_key_threshold);
            }
            0 => {}
            filter => {
                println!("    Testing only weak keys of type {} (T = {})", filter, weak_key_threshold);
            }
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
            thread::spawn(move || {
                let number_of_trials = trials_per_thread + if thread_id == 0
                    { trials_remainder } else { 0 };
                trial_loop_async(thread_id, number_of_trials, weak_key_filter,
                    weak_key_threshold, save_frequency, tx_clone);
            });
        }
        // Track thread stats and how many threads are still in progress
        let mut open_thread_count = thread_count;
        let mut thread_stats: HashMap<u64, ThreadStats> = HashMap::with_capacity(thread_count as usize);
        // Wait for messages
        for received in rx {
            match received {
                // If we receive a decoding failure, record it and increment the failure count
                DecoderMessage::TrialResult(result) => {
                    if !result.success {
                        failure_count += 1;
                        if failure_count <= record_max {
                            if args.verbose {
                                println!("Decoding failure found!");
                                println!("Key: {}\nError vector: {}", result.key, result.e_supp);
                                if failure_count == record_max {
                                    println!("Maximum number of decoding failures recorded.");
                                }
                            }
                            decoding_failures.push((result.key, result.e_supp));
                        }
                    }
                }
                // If we receive updated thread statistics, record and save those
                DecoderMessage::Stats(stats) => {
                    thread_stats.insert(stats.thread_id, stats);
                    let total_trials = thread_stats.values().map(|st| st.trials_completed).sum();
                    let runtime = start_time.elapsed();
                    let json_output = build_json(failure_count, total_trials, &decoding_failures,
                        weak_key_filter, weak_key_threshold, runtime, Some(json!(thread_stats)));
                    write_to_file_or_stdout(&args.output, &json_output);
                    if stats.done {
                        open_thread_count -= 1;
                    }
                    if args.verbose {
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
            if open_thread_count == 0 {
                break; // all threads completed successfully, so wrap up
            }
        }
    } else { // synchronous decoding trial loop
        let mut rng = crate::random::get_rng();
        let mut cache = ThresholdCache::with_parameters(r, d, t);    
        for i in 0..number_of_trials {
            let result = decoding_trial(weak_key_filter, weak_key_threshold, &mut rng, &mut cache);
            if !result.success {
                failure_count += 1;
                if failure_count <= record_max {
                    if args.verbose {
                        println!("Decoding failure found!");
                        println!("Key: {}\nError vector: {}", result.key, result.e_supp);
                        if failure_count == record_max {
                            println!("Maximum number of decoding failures recorded.");
                        }
                    }
                    decoding_failures.push((result.key, result.e_supp));
                }
            }
            if i != 0 && i % save_frequency == 0 {
                let runtime = start_time.elapsed();
                let json_output = build_json(failure_count, i, &decoding_failures, weak_key_filter,
                    weak_key_threshold, runtime, None);
                write_to_file_or_stdout(&args.output, &json_output);
                if args.verbose {
                    println!("Found {} decoding failures in {} trials (runtime: {:.3} s)",
                        failure_count, i, runtime.as_secs_f64());
                }
            }
        }
        // Write final data
        let runtime = start_time.elapsed();
        let json_output = build_json(failure_count, number_of_trials, &decoding_failures,
            weak_key_filter, weak_key_threshold, runtime, None);
        write_to_file_or_stdout(&args.output, &json_output);
    }
    if args.verbose {
        print_end_message(failure_count, number_of_trials, start_time.elapsed());
    }
}
