use crate::{
    application,
    record::{DataRecord, DecodingFailureRatio},
    settings::{Settings, TrialSettings},
};
use anyhow::{Context, Result};
use bike_decoder::{
    decoder::DecodingFailure,
    random::{
        current_thread_id, custom_thread_rng, get_or_insert_global_seed, try_insert_global_seed,
    },
};
use crossbeam_channel::{unbounded as channel, Receiver, RecvTimeoutError, Sender, TryRecvError};
use rand::Rng;
use rayon::prelude::*;
use std::time::{Duration, Instant};

pub fn trial_iteration<R: Rng + ?Sized>(
    settings: &TrialSettings,
    tx: &Sender<(DecodingFailure, usize)>,
    rng: &mut R,
) -> u64 {
    let result = application::decoding_failure_trial(settings, rng);
    if let Some(df) = result {
        // Attempt to send decoding failure, but ignore errors, as the receiver may
        // choose to hang up after receiving the maximum number of decoding failures.
        tx.send((df, current_thread_id())).ok();
        1
    } else {
        0
    }
}

// Runs decoding_trial in a loop, sending decoding failures via tx_results and
// progress updates (counts of decoding failures and trials run) via tx_progress.
pub fn trial_loop(
    settings: &Settings,
    tx_results: &Sender<(DecodingFailure, usize)>,
    tx_progress: &Sender<DecodingFailureRatio>,
    pool: rayon::ThreadPool,
) -> Result<()> {
    let mut trials_remaining = settings.num_trials();
    while trials_remaining > 0 {
        let tx_results = tx_results.clone();
        let new_trials = settings.save_frequency().min(trials_remaining);
        let new_failure_count = pool.install(|| {
            (0..new_trials)
                .into_par_iter()
                .map_with(
                    (settings.trial_settings(), tx_results),
                    |(settings, tx), _| trial_iteration(settings, tx, &mut custom_thread_rng()),
                )
                .sum()
        });
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
    rx_results: Receiver<(DecodingFailure, usize)>,
    rx_progress: Receiver<DecodingFailureRatio>,
    start_time: Instant,
) -> Result<DataRecord> {
    let seed = get_or_insert_global_seed(settings.seed());
    let mut data = DataRecord::new(settings.key_filter(), settings.fixed_key().cloned(), seed);
    let mut rx_results_open = true;
    let mut rx_progress_open = true;
    // Alternate between handling decoding failures and handling progress updates
    'outer: while rx_results_open || rx_progress_open {
        // Handle all decoding failures currently in channel, then continue
        while rx_results_open {
            match rx_results.try_recv() {
                Ok((df, thread)) => {
                    application::handle_decoding_failure(df, thread, &mut data, settings);
                    if data.decoding_failures().len() == settings.record_max() {
                        // Max number of decoding failures recorded, short-circuit outer loop
                        break 'outer;
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    // results channel closed, flag this loop to be skipped
                    rx_results_open = false;
                }
            }
        }
        // Handle all progress updates currently in channel, then continue (w/ timeout delay)
        while rx_progress_open {
            match rx_progress.recv_timeout(Duration::from_millis(100)) {
                Ok(dfr) => {
                    application::handle_progress(dfr, &mut data, settings, start_time.elapsed())?;
                    application::write_json(settings.output(), &data)?;
                }
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => {
                    // progress channel closed, flag this loop to be skipped
                    rx_progress_open = false;
                }
            }
        }
    }
    // Drops the results receiver so no more decoding failures are handled
    drop(rx_results);
    // Receive and handle all remaining progress updates
    for dfr in rx_progress {
        application::handle_progress(dfr, &mut data, settings, start_time.elapsed())?;
        application::write_json(settings.output(), &data)?;
    }
    Ok(data)
}

pub fn run_parallel(settings: &Settings) -> Result<DataRecord> {
    let start_time = Instant::now();
    if settings.verbose() >= 1 {
        eprintln!("{}", application::start_message(settings));
    }
    application::check_writable(settings.output(), settings.overwrite())?;
    // Set global PRNG seed used for generating data
    try_insert_global_seed(settings.seed())
        .context("Must be able to set global seed to user-specified seed")?;
    // Set up channels to receive decoding results and progress updates
    let (tx_results, rx_results) = channel();
    let (tx_progress, rx_progress) = channel();
    let settings_clone = settings.clone();
    // Start main trial loop in separate thread
    let trial_thread = std::thread::spawn(move || -> Result<()> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(settings_clone.threads())
            .build()?;
        trial_loop(&settings_clone, &tx_results, &tx_progress, pool)?;
        Ok(())
    });
    // Process messages from trial_loop
    let data = record_trial_results(settings, rx_results, rx_progress, start_time)?;
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
