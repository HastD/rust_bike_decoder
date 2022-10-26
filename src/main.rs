pub mod decoder;
pub mod graphs;
pub mod keys;
pub mod parameters;
pub mod random;
pub mod syndrome;
pub mod threshold;
pub mod vectors;

use crate::parameters::*;
use crate::keys::Key;
use crate::threshold::ThresholdCache;
use crate::vectors::{Index, SparseErrorVector};
use crate::syndrome::Syndrome;
use clap::Parser;
use rand::{Rng, distributions::Uniform};
use std::cmp;
use std::{fmt::Display, fs::File, io::Write};
use std::time::{Duration, Instant};
use serde_json::json;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short='N',long,help="Number of trials (required)")]
    number: u32,
    #[arg(short,long,help="Suppress weak key filtering")]
    nofilter: bool,
    #[arg(short,long,help="Output file (default stdout)")]
    output: Option<String>,
    #[arg(short,long,help="Max number of decoding failures recorded (default all)")]
    recordmax: Option<u32>,
    #[arg(short='f',long,help="Write frequency (default only at end)")]
    writefreq: Option<u32>,
    #[arg(short,long)]
    verbose: bool,
}

fn decoding_trial<R: Rng + ?Sized>(
    filtered: bool,
    rng: &mut R,
    key_dist: &Uniform<Index>,
    err_dist: &Uniform<Index>,
    threshold_cache: &mut ThresholdCache
) -> (Key, SparseErrorVector, bool) {
    let key: Key;
    if filtered {
        key = Key::random_non_weak(rng, key_dist);
    } else {
        key = Key::random(rng, key_dist);
    }
    let e_supp = SparseErrorVector::random(rng, err_dist);
    let mut syn = Syndrome::from_sparse(&key, &e_supp);
    let (_e_out, success) = decoder::bgf_decoder(&key, &mut syn, threshold_cache);
    (key, e_supp, success)
}

fn build_json(
    failure_count: u32,
    number_of_trials: u32,
    decoding_failures: &Vec<(Key, SparseErrorVector)>,
    runtime: Duration
) -> serde_json::Value {
    json!({
        "r": BLOCK_LENGTH,
        "d": BLOCK_WEIGHT,
        "t": ERROR_WEIGHT,
        "iterations": NB_ITER,
        "bgf_threshold": BGF_THRESHOLD,
        "weak_key_threshold": WEAK_KEY_THRESHOLD,
        "trials": number_of_trials,
        "failure_count": failure_count,
        "decoding_failures": decoding_failures,
        "runtime": runtime.as_secs_f64()
    })
}

fn write_to_file_or_stdout(path: &Option<String>, data: &impl Display) {
    match path {
        Some(filename) => {
            let mut file = File::create(filename).expect("Must be able to open file");
            file.write(&data.to_string().into_bytes()).expect("Must be able to write to file");
        }
        None => {
            println!("{}", data);
        }
    };
}

fn main() {
    let args = Args::parse();
    let number_of_trials = args.number;
    let record_max = args.recordmax.unwrap_or(number_of_trials);
    let write_frequency = cmp::max(10000, args.writefreq.unwrap_or(number_of_trials));

    let (r, d, t) = (BLOCK_LENGTH as u32, BLOCK_WEIGHT as u32, ERROR_WEIGHT as u32);
    let key_dist = crate::random::get_key_dist();
    let err_dist = crate::random::get_err_dist();
    let mut rng = crate::random::get_rng();
    let mut threshold_cache = ThresholdCache::with_parameters(r, d, t);
    let mut failure_count = 0;
    let mut decoding_failures: Vec<(Key, SparseErrorVector)> = Vec::new();
    if args.verbose {
        println!("Starting decoding trials (N = {}) with parameters:", number_of_trials);
        println!(
            "    r = {}, d = {}, t = {}, iterations = {}, tau = {}, T = {}",
            r, d, t, NB_ITER, BGF_THRESHOLD, WEAK_KEY_THRESHOLD
        );
    }
    let start_time = Instant::now();
    for i in 0..number_of_trials {
        let (key, e_supp, success) = decoding_trial(
            !args.nofilter,
            &mut rng, &key_dist, &err_dist, &mut threshold_cache
        );
        if !success {
            failure_count += 1;
            if failure_count <= record_max {
                if args.verbose {
                    println!("Decoding failure found!");
                    println!("Key: {}\nError vector: {}", key, e_supp);
                    if failure_count == record_max {
                        println!("Maximum number of decoding failures recorded.");
                    }
                }
                decoding_failures.push((key, e_supp));
            }
        }
        if i != 0 && i % write_frequency == 0 {
            let runtime = start_time.elapsed();
            let json_output = build_json(failure_count, i, &decoding_failures, runtime);
            write_to_file_or_stdout(&args.output, &json_output);
            if args.verbose {
                println!(
                    "Found {} decoding failures in {} trials (runtime: {:.3} s)",
                    failure_count, i, runtime.as_secs_f64()
                );
            }
        }
    }
    let runtime = start_time.elapsed();
    let json_output = build_json(failure_count, number_of_trials, &decoding_failures, runtime);
    write_to_file_or_stdout(&args.output, &json_output);
    if args.verbose {
        println!("Trials: {}", number_of_trials);
        println!("Decoding failures: {}", failure_count);
        let dfr = (failure_count as f64) / number_of_trials as f64;
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
}
