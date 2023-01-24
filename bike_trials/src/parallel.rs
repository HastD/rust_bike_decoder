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
) -> Result<()> {
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
) -> Result<DataRecord> {
    const CONSECUTIVE_RESULTS_MAX: u32 = 10_000;
    const TIMEOUT: Duration = Duration::from_millis(100);
    const CONSECUTIVE_PROGRESS_MAX: u32 = 100;
    let seed = get_or_insert_global_seed(settings.seed());
    let mut data = DataRecord::new(settings.key_filter(), settings.fixed_key().cloned(), seed);
    let mut rx_results_open = true;
    let mut rx_progress_open = true;
    let mut consecutive_results_handled;
    let mut consecutive_progress_handled;
    // Alternate between handling decoding failures and handling progress updates
    'outer: while rx_results_open || rx_progress_open {
        consecutive_results_handled = 0;
        // Handle decoding failures currently in channel, then continue
        while rx_results_open && consecutive_results_handled < CONSECUTIVE_RESULTS_MAX {
            match rx_results.try_recv() {
                Ok(df) => {
                    application::handle_decoding_failure(df, &mut data, settings);
                    if data.decoding_failures().len() == settings.record_max() {
                        // Max number of decoding failures recorded, short-circuit outer loop
                        break 'outer;
                    }
                    consecutive_results_handled += 1;
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    // results channel closed, flag this loop to be skipped
                    rx_results_open = false;
                }
            }
        }
        let timeout = if consecutive_results_handled >= CONSECUTIVE_RESULTS_MAX {
            // Likely still decoding failures waiting to be handled, so skip delay
            Duration::ZERO
        } else {
            TIMEOUT
        };
        consecutive_progress_handled = 0;
        // Handle all progress updates currently in channel, then continue (w/ timeout delay)
        while rx_progress_open && consecutive_progress_handled < CONSECUTIVE_PROGRESS_MAX {
            match rx_progress.recv_timeout(timeout) {
                Ok(dfr) => {
                    application::handle_progress(dfr, &mut data, settings, start_time.elapsed());
                    consecutive_progress_handled += 1;
                }
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => {
                    // progress channel closed, flag this loop to be skipped
                    rx_progress_open = false;
                }
            }
        }
        if consecutive_progress_handled >= 1 {
            application::write_json(settings.output(), &data)?;
        }
    }
    // Drops the results receiver so no more decoding failures are handled
    drop(rx_results);
    // Receive and handle all remaining progress updates
    while let Ok(dfr) = rx_progress.recv() {
        application::handle_progress(dfr, &mut data, settings, start_time.elapsed());
        consecutive_progress_handled = 1;
        // Handle any backlog of progress updates before writing
        while let Ok(dfr) = rx_progress.recv_timeout(TIMEOUT) {
            application::handle_progress(dfr, &mut data, settings, start_time.elapsed());
            consecutive_progress_handled += 1;
            if consecutive_progress_handled >= CONSECUTIVE_PROGRESS_MAX {
                break;
            }
        }
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
