use crate::{
    decoder::{DecodingResult, DecodingFailure},
    keys::{Key, KeyFilter},
    ncw::TaggedErrorVector,
    parameters::*,
    random::{Seed, current_thread_id, get_rng_from_seed, global_thread_count},
    record::{DataRecord, RecordedDecodingFailure, DecodingFailureRatio},
    settings::{Settings, TrialSettings, OutputTo},
};
use std::{
    fs::{self, File},
    io::{self, Write},
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
    DecodingResult::from(key, tagged_error_vector)
}

pub fn decoding_failure_trial<R>(settings: &TrialSettings, rng: &mut R) -> Option<DecodingFailure>
    where R: Rng + ?Sized
{
    let result = decoding_trial(settings, rng);
    DecodingFailure::try_from(result).ok()
}

pub fn check_writable(output: &OutputTo) -> Result<()> {
    if let OutputTo::File(path, overwrite) = output {
        if !overwrite
            && path.try_exists()
                .context("Output file path should be accessible")?
            && fs::metadata(path)
                .context("Output file metadata should be readable")?.len() > 0
        {
            // If file already exists and is nonempty, copy its contents to a backup file
            fs::copy(path, format!("{}-backup-{}", path.display(), Uuid::new_v4()))
                .with_context(|| format!("Should be able to back up existing file at {}",
                    path.display()))?;
        }
        File::create(path).context("Output file should be openable")?
            .write_all(b"").context("Output file should be writable")?;
    }
    Ok(())
}

/// Serializes data in JSON format to specified output location
pub fn write_json(output: &OutputTo, data: &impl Serialize) -> Result<()> {
    let mut writer: Box<dyn Write> = match output {
        OutputTo::Stdout => Box::new(io::stdout()),
        OutputTo::File(filename, _) => {
            let file = File::create(filename).context("Output file should be writable")?;
            Box::new(file)
        }
        OutputTo::Void => return Ok(()),
    };
    let mut ser = serde_json::Serializer::new(&mut writer);
    data.serialize(&mut ser).context("data should be writable as JSON")?;
    writer.write_all(b"\n")?;
    Ok(())
}

pub fn start_message(settings: &Settings) -> String {
    let parameter_message = format!("    r = {}, d = {}, t = {}, iterations = {}, tau = {}\n",
        BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT, NB_ITER, GRAY_THRESHOLD_DIFF);
    let weak_key_message = match settings.key_filter() {
        KeyFilter::Any => String::new(),
        KeyFilter::NonWeak(threshold) =>
            format!("    Testing only non-weak keys (T = {})\n", threshold),
        KeyFilter::Weak(weak_type, threshold) =>
            format!("    Testing only weak keys of type {} (T = {})\n",
                weak_type as u8, threshold),
    };
    let ncw_message = settings.ncw_class().map_or(String::new(), |ncw_class| {
        let l_str = settings.ncw_overlap().map_or_else(|| "l".to_string(), |l| l.to_string());
        format!("    Sampling error vectors from A_{{t,{}}}({})\n", l_str, ncw_class)
    });
    let thread_message = if settings.parallel() {
        let thread_count = if settings.threads() == 0 {
            num_cpus::get()
        } else {
            settings.threads()
        };
        format!("[running with {} threads]\n", thread_count)
    } else {
        String::new()
    };
    format!("Starting decoding trials (N = {}) with parameters:\n{}{}{}{}",
        settings.num_trials(), parameter_message, weak_key_message, ncw_message,
        thread_message)
}

pub fn end_message(dfr: &DecodingFailureRatio, runtime: Duration) -> String {
    let avg_nanos = runtime.as_nanos() / dfr.num_trials() as u128;
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
        dfr.num_trials(), dfr.num_failures(), dfr.as_f64().log2(),
        runtime.as_secs_f64(), avg_text)
}

pub fn handle_decoding_failure(df: DecodingFailure, thread_id: usize,
        data: &mut DataRecord, settings: &Settings) {
    if data.decoding_failures().len() < settings.record_max() {
        if settings.verbose() >= 3 {
            println!("Decoding failure found!");
            println!("Key: {}\nError vector: {}", df.key(), df.vector());
            if data.decoding_failures().len() + 1 == settings.record_max() {
                println!("Maximum number of decoding failures recorded.");
            }    
        }
        data.push_decoding_failure(RecordedDecodingFailure::new(df, thread_id));
    }
}

pub fn handle_progress(dfr: DecodingFailureRatio, data: &mut DataRecord,
        settings: &Settings, runtime: Duration) -> Result<()> {
    data.add_results(dfr);
    data.set_runtime(runtime);
    if settings.parallel() {
        data.set_thread_count(global_thread_count());
    }
    if settings.output().is_file() || settings.verbose() >= 2 {
        write_json(settings.output(), &data)?;
    }    
    if settings.verbose() >= 2 {
        println!("Found {} decoding failures in {} trials (runtime: {:.3} s)",
            data.num_failures(), data.num_trials(), runtime.as_secs_f64());
    }
    Ok(())
}

pub fn run(settings: &Settings) -> Result<DataRecord> {
    let start_time = Instant::now();
    if settings.verbose() >= 1 {
        println!("{}", start_message(settings));
    }
    check_writable(settings.output())?;
    // Set PRNG seed used for generating data
    let seed = settings.seed().unwrap_or_else(Seed::from_entropy);
    // Initialize object storing data to be recorded
    let mut data = DataRecord::new(settings.key_filter(), settings.fixed_key().cloned(), seed);
    let seed_index = settings.seed_index().unwrap_or_else(current_thread_id);
    let mut rng = get_rng_from_seed(seed, seed_index);
    let mut trials_remaining = settings.num_trials();
    while trials_remaining > 0 {
        let mut new_failure_count = 0;
        let new_trials = settings.save_frequency().min(trials_remaining);
        for _ in 0..new_trials {
            let result = decoding_failure_trial(settings.trial_settings(), &mut rng);
            if let Some(df) = result {
                new_failure_count += 1;
                handle_decoding_failure(df, seed_index, &mut data, settings);
            }
        }
        let dfr = DecodingFailureRatio::new(new_failure_count, new_trials)
            .expect("Number of decoding failures should be <= number of trials");
        handle_progress(dfr, &mut data, settings, start_time.elapsed())?;
        trials_remaining -= new_trials;
    }
    // Write final data
    data.set_runtime(start_time.elapsed());
    write_json(settings.output(), &data)?;
    if settings.verbose() >= 1 {
        println!("{}", end_message(data.decoding_failure_ratio(), data.runtime()));
    }
    Ok(data)
}
