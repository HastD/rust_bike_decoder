use crate::{
    output,
    record::{DataRecord, DecodingFailureRatio},
    settings::{Settings, TrialSettings},
};
use bike_decoder::{
    decoder::{DecodingFailure, DecodingResult},
    keys::{Key, KeyFilter},
    ncw::TaggedErrorVector,
    parameters::*,
    random::{current_thread_id, get_rng_from_seed, global_thread_count, Seed},
};
use rand::Rng;
use std::time::{Duration, Instant};

pub fn decoding_trial<R>(settings: &TrialSettings, rng: &mut R) -> DecodingResult
where
    R: Rng + ?Sized,
{
    // Use fixed_key if provided, otherwise generate random key using specified filter
    let key = settings
        .fixed_key()
        .cloned()
        .unwrap_or_else(|| Key::random_filtered(settings.key_filter(), rng));
    let tagged_error_vector = if let Some(ncw_class) = settings.ncw_class() {
        let l = settings
            .ncw_overlap()
            .unwrap_or_else(|| rng.gen_range(0..=ncw_class.max_l()));
        TaggedErrorVector::near_codeword(&key, ncw_class, l, rng)
    } else {
        TaggedErrorVector::random(rng)
    };
    DecodingResult::from_key_vector(key, tagged_error_vector)
}

#[inline]
pub fn decoding_failure_trial<R>(settings: &TrialSettings, rng: &mut R) -> Option<DecodingFailure>
where
    R: Rng + ?Sized,
{
    decoding_trial(settings, rng).try_into().ok()
}

pub(crate) fn start_message(settings: &Settings) -> String {
    let num_trials = settings.num_trials();
    let parameter_message = format!(
        "    r = {BLOCK_LENGTH}, d = {BLOCK_WEIGHT}, t = {ERROR_WEIGHT}, \
        iterations = {NB_ITER}, tau = {GRAY_THRESHOLD_DIFF}\n"
    );
    let weak_key_message = match settings.key_filter() {
        KeyFilter::Any => String::new(),
        KeyFilter::NonWeak(threshold) => {
            format!("    Testing only non-weak keys (T = {threshold})\n")
        }
        KeyFilter::Weak(weak_type, threshold) => format!(
            "    Testing only weak keys of type {} (T = {})\n",
            weak_type as u8, threshold
        ),
    };
    let ncw_message = settings.ncw_class().map_or(String::new(), |ncw_class| {
        let l_str = settings
            .ncw_overlap()
            .map_or_else(|| "l".to_string(), |l| l.to_string());
        format!("    Sampling error vectors from A_{{t,{l_str}}}({ncw_class})\n")
    });
    let thread_message = if settings.parallel() {
        let thread_count = if settings.threads() == 0 {
            num_cpus::get()
        } else {
            settings.threads()
        };
        format!("[running with {thread_count} threads]\n")
    } else {
        String::new()
    };
    format!(
        "Starting decoding trials (N = {num_trials}) with parameters:\n\
        {parameter_message}\
        {weak_key_message}\
        {ncw_message}\
        {thread_message}"
    )
}

pub(crate) fn end_message(dfr: &DecodingFailureRatio, runtime: Duration) -> String {
    let avg_nanos = runtime.as_nanos() / u128::from(dfr.num_trials());
    let (avg_mcs, ns_rem) = (avg_nanos / 1000, avg_nanos % 1000);
    let avg_text = if avg_mcs >= 100 {
        format!("{avg_mcs} μs")
    } else if avg_mcs >= 10 {
        format!("{}.{} μs", avg_mcs, ns_rem / 100)
    } else if avg_mcs >= 1 {
        format!("{}.{:0width$} μs", avg_mcs, ns_rem / 10, width = 2)
    } else {
        format!("{}.{:0width$} μs", avg_mcs, ns_rem, width = 3)
    };
    format!(
        "Trials: {}\n\
        Decoding failures: {}\n\
        log2(DFR): {:.2}\n\
        Runtime: {:.3} s\n\
        Average: {}",
        dfr.num_trials(),
        dfr.num_failures(),
        dfr.as_f64().log2(),
        runtime.as_secs_f64(),
        avg_text
    )
}

pub fn handle_decoding_failure(df: DecodingFailure, data: &mut DataRecord, settings: &Settings) {
    if data.decoding_failures().len() < settings.record_max() {
        if settings.verbose() >= 3 {
            eprintln!("Decoding failure found!");
            eprintln!("Key: {}\nError vector: {}", df.key(), df.vector());
            if data.decoding_failures().len() + 1 == settings.record_max() {
                eprintln!("Maximum number of decoding failures recorded.");
            }
        }
        data.push_decoding_failure(df);
    }
}

pub fn handle_progress(
    dfr: DecodingFailureRatio,
    data: &mut DataRecord,
    settings: &Settings,
    runtime: Duration,
) {
    data.add_results(dfr);
    data.set_runtime(runtime);
    if settings.parallel() {
        data.set_thread_count(Some(global_thread_count()));
    }
    if settings.verbose() >= 2 {
        eprintln!(
            "Found {} decoding failures in {} trials (runtime: {:.3} s)",
            data.num_failures(),
            data.num_trials(),
            runtime.as_secs_f64()
        );
    }
}

pub fn run(settings: &Settings) -> Result<DataRecord, anyhow::Error> {
    let start_time = Instant::now();
    if settings.verbose() >= 1 {
        eprintln!("{}", start_message(settings));
    }
    output::check_writable(settings.output(), settings.overwrite())?;
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
            if let Some(mut df) = result {
                new_failure_count += 1;
                df.thread = Some(seed_index);
                handle_decoding_failure(df, &mut data, settings);
            }
        }
        let dfr = DecodingFailureRatio::new(new_failure_count, new_trials)
            .expect("Number of decoding failures should be <= number of trials");
        handle_progress(dfr, &mut data, settings, start_time.elapsed());
        output::write_json(settings.output(), &data)?;
        trials_remaining -= new_trials;
    }
    if settings.verbose() >= 1 {
        eprintln!(
            "{}",
            end_message(data.decoding_failure_ratio(), data.runtime())
        );
    }
    Ok(data)
}
