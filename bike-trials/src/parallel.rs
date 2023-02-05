use crate::{
    application,
    output::{self, OutputError},
    record::{DataRecord, DecodingFailureRatio},
    settings::{Settings, TrialSettings},
};
use anyhow::Context;
use bike_decoder::{
    decoder::DecodingFailure,
    random::{
        current_thread_id, custom_thread_rng, get_or_insert_global_seed, try_insert_global_seed,
    },
};
use crossbeam_channel::{unbounded as channel, Receiver, Select, Sender};
use rand::Rng;
use rayon::prelude::*;
use std::time::Instant;

pub fn trial_iteration<R: Rng + ?Sized>(
    settings: &TrialSettings,
    tx: &Sender<DecodingFailure>,
    rng: &mut R,
) -> u64 {
    let result = application::decoding_failure_trial(settings, rng);
    if let Some(mut df) = result {
        df.thread = Some(current_thread_id());
        // Attempt to send decoding failure, but ignore errors, as the receiver may
        // choose to hang up after receiving the maximum number of decoding failures.
        tx.send(df).ok();
        1
    } else {
        0
    }
}

// Runs decoding_trial in a loop, sending decoding failures via tx_results and
// progress updates (counts of decoding failures and trials run) via tx_progress.
pub fn trial_loop(
    settings: &TrialSettings,
    num_trials: u64,
    save_frequency: u64,
    tx_results: &Sender<DecodingFailure>,
    tx_progress: &Sender<DecodingFailureRatio>,
) -> Result<(), anyhow::Error> {
    let mut trials_remaining = num_trials;
    while trials_remaining > 0 {
        let new_trials = save_frequency.min(trials_remaining);
        let new_failure_count = (0..new_trials)
            .into_par_iter()
            .map_with((settings, tx_results), |(settings, tx), _| {
                trial_iteration(settings, tx, &mut custom_thread_rng())
            })
            .sum();
        let dfr = DecodingFailureRatio::new(new_failure_count, new_trials)
            .expect("Number of decoding failures should be <= number of trials");
        tx_progress
            .send(dfr)
            .context("Progress receiver should not be closed")?;
        trials_remaining -= new_trials;
    }
    Ok(())
}

pub fn record_trial_results(
    settings: &Settings,
    rx_results: Receiver<DecodingFailure>,
    rx_progress: Receiver<DecodingFailureRatio>,
    start_time: Instant,
) -> Result<DataRecord, OutputError> {
    let seed = get_or_insert_global_seed(settings.seed());
    let mut data = DataRecord::new(settings.key_filter(), settings.fixed_key().cloned(), seed);
    const CONSECUTIVE_RESULTS_MAX: usize = 10_000;
    let mut unwritten_data = false;
    let mut selector = Select::new();
    let rx_results_idx = selector.recv(&rx_results);
    let rx_progress_idx = selector.recv(&rx_progress);
    // Receive and handle messages from rx_results and rx_progress until rx_results
    // closes or the maximum number of decoding failures have been recorded.
    while data.decoding_failures().len() < settings.record_max() {
        let oper = selector.select();
        match oper.index() {
            i if i == rx_results_idx => match oper.recv(&rx_results) {
                Ok(df) => {
                    application::handle_decoding_failure(df, &mut data, settings);
                    unwritten_data = true;
                    for df in rx_results.try_iter().take(CONSECUTIVE_RESULTS_MAX) {
                        application::handle_decoding_failure(df, &mut data, settings);
                    }
                }
                Err(_) => break,
            },
            i if i == rx_progress_idx => match oper.recv(&rx_progress) {
                Ok(dfr) => {
                    application::handle_progress(dfr, &mut data, settings, start_time.elapsed());
                    if data.num_trials() == settings.num_trials() {
                        // Defer final write to make sure all decoding failures have been recorded
                        unwritten_data = true;
                    } else {
                        output::write_json(settings.output(), &data)?;
                        unwritten_data = false;
                    }
                }
                Err(_) => selector.remove(rx_progress_idx),
            },
            _ => unreachable!(),
        }
    }
    // Drops the results receiver so no more decoding failures are handled
    drop(rx_results);
    // Receive and handle all remaining progress updates
    for dfr in rx_progress {
        application::handle_progress(dfr, &mut data, settings, start_time.elapsed());
        output::write_json(settings.output(), &data)?;
        unwritten_data = false;
    }
    // Failsafe to ensure any remaining data is written
    if unwritten_data {
        output::write_json(settings.output(), &data)?;
    }
    Ok(data)
}

pub fn run_parallel(settings: &Settings) -> Result<DataRecord, anyhow::Error> {
    let start_time = Instant::now();
    if settings.verbose() >= 1 {
        eprintln!("{}", application::start_message(settings));
    }
    output::check_writable(settings.output(), settings.overwrite())?;
    // Set global PRNG seed used for generating data
    let seed = try_insert_global_seed(settings.seed())
        .context("Must be able to set global seed to user-specified seed")?;
    // Set up channels to receive decoding results and progress updates
    let (tx_results, rx_results) = channel();
    let (tx_progress, rx_progress) = channel();
    let settings_clone = settings.clone();
    // Start main trial loop in separate thread
    let trial_thread = std::thread::spawn(move || {
        let settings = settings_clone;
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(settings.threads())
            .build()?;
        pool.install(|| {
            trial_loop(
                settings.trial_settings(),
                settings.num_trials(),
                settings.save_frequency(),
                &tx_results,
                &tx_progress,
            )
        })
    });
    // Process messages from trial_loop
    let data = record_trial_results(settings, rx_results, rx_progress, start_time)
        .with_context(|| format!("Data processing error [seed = {seed}]"))?;
    // Propagate any errors or panics from thread
    trial_thread
        .join()
        .unwrap_or_else(|err| std::panic::resume_unwind(err))?;
    if settings.verbose() >= 1 {
        eprintln!(
            "{}",
            application::end_message(data.decoding_failure_ratio(), data.runtime())
        );
    }
    Ok(data)
}
