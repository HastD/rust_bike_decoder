use crate::{
    decoder::bgf_decoder,
    keys::{Key, KeyFilter},
    ncw::TaggedErrorVector,
    parameters::*,
    random::{Seed, current_thread_id, get_rng_from_seed, global_thread_count},
    record::{DecodingResult, DecodingFailureRecord, DataRecord},
    settings::{Settings, TrialSettings},
    syndrome::Syndrome,
};
use std::{
    convert::AsRef,
    fs::{self, File},
    io::{self, Write},
    path::Path,
    time::{Duration, Instant},
};
use anyhow::{Context, Result};
use rand::Rng;
use serde::Serialize;
use uuid::Uuid;

pub fn decoding_trial<R>(settings: &TrialSettings, rng: &mut R) -> DecodingResult
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
    let (e_out, same_syndrome) = bgf_decoder(&key, &mut syn);
    let success = e_in == e_out;
    assert!(same_syndrome || !success);
    DecodingResult::from(key, tagged_error_vector, success)
}

pub fn check_file_writable(output: Option<&Path>, overwrite: bool) -> Result<()> {
    if let Some(filename) = output {
        if !overwrite && filename.try_exists().context("Should be able to check existence of output file")?
                && fs::metadata(filename).context("Should be able to access output file metadata")?.len() > 0 {
            // If file already exists and is nonempty, copy its contents to a backup file
            fs::copy(filename, format!("{}-backup-{}", filename.display(), Uuid::new_v4()))
                .with_context(|| format!("Should be able to back up existing file at {}", filename.display()))?;
        }
        File::create(filename).context("Should be able to create output file")?
            .write_all(b"").context("Should be able to write to output file")?;
    }
    Ok(())
}

/// Serializes data in JSON format to the specified path, or to standard output if path not provided.
pub fn write_json<P>(output: Option<P>, data: &impl Serialize) -> Result<()>
    where P: AsRef<Path> + Copy
{
    if let Some(filename) = output {
        serde_json::to_writer(File::create(filename).context("Should be able to open output file")?, data)
            .context("Should be able to serialize data to output file as JSON")?;
        File::options().append(true).open(filename)?.write_all(b"\n")?;
    } else {
        serde_json::to_writer(io::stdout(), data)
            .context("Should be able to serialize data to standard output as JSON")?;
        io::stdout().write_all(b"\n")?;
    }
    Ok(())
}

pub fn start_message(settings: &Settings) -> String {
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

pub fn end_message(failure_count: usize, number_of_trials: usize, runtime: Duration) -> String {
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

pub fn handle_decoding_failure(result: DecodingResult, thread_id: usize,
        data: &mut DataRecord, settings: &Settings) {
    assert!(!result.success(), "handle_decoding_failure should only be called for decoding failures");
    if data.decoding_failures().len() < settings.record_max() {
        if settings.verbose() >= 3 {
            println!("Decoding failure found!");
            println!("Key: {}\nError vector: {}", result.key(), result.vector());
            if data.decoding_failures().len() + 1 == settings.record_max() {
                println!("Maximum number of decoding failures recorded.");
            }    
        }
        data.push_decoding_failure(DecodingFailureRecord::from(result, thread_id));
    }
}

pub fn handle_progress(new_failure_count: usize, new_trials: usize, data: &mut DataRecord,
        settings: &Settings, runtime: Duration) -> Result<()> {
    data.add_to_failure_count(new_failure_count);
    data.add_to_trials(new_trials);
    if settings.parallel() {
        data.set_thread_count(global_thread_count());
    }
    data.set_runtime(runtime);
    if !settings.silent() && (settings.output_file().is_some() || settings.verbose() >= 2) {
        write_json(settings.output_file(), &data)?;
    }    
    if settings.verbose() >= 2 {
        println!("Found {} decoding failures in {} trials (runtime: {:.3} s)",
            data.failure_count(), data.trials(), runtime.as_secs_f64());
    }
    Ok(())
}

pub fn run(settings: Settings) -> Result<DataRecord> {
    let start_time = Instant::now();
    if settings.verbose() >= 1 {
        println!("{}", start_message(&settings));
    }
    check_file_writable(settings.output_file(), settings.overwrite())?;
    // Initialize object storing data to be recorded
    let mut data = DataRecord::new(settings.key_filter(), settings.fixed_key().cloned());
    let seed = settings.seed().unwrap_or_else(Seed::from_entropy);
    // Set PRNG seed used for generating data
    data.set_seed(seed);
    let seed_index = settings.seed_index().unwrap_or_else(current_thread_id);
    let mut rng = get_rng_from_seed(seed, seed_index);
    let mut trials_remaining = settings.number_of_trials();
    while trials_remaining > 0 {
        let mut new_failure_count = 0;
        let new_trials = settings.save_frequency().min(trials_remaining);
        for _ in 0..new_trials {
            let result = decoding_trial(settings.trial_settings(), &mut rng);
            if !result.success() {
                new_failure_count += 1;
                handle_decoding_failure(result, seed_index, &mut data, &settings);
            }
        }
        handle_progress(new_failure_count, new_trials, &mut data, &settings, start_time.elapsed())?;
        trials_remaining -= new_trials;
    }
    // Write final data
    data.set_runtime(start_time.elapsed());
    if !settings.silent() {
        write_json(settings.output_file(), &data)?;
    }
    if settings.verbose() >= 1 {
        println!("{}", end_message(data.failure_count(), data.trials(), data.runtime()));
    }
    Ok(data)
}
