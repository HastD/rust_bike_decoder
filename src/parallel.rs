use crate::{
    application,
    random::{get_or_insert_global_seed, current_thread_id,
        custom_thread_rng, global_thread_count},
    record::{DecodingResult, DataRecord},
    settings::{Settings, TrialSettings},
};
use std::{
    cmp,
    sync::mpsc,
    time::{Duration, Instant},
    thread,
};
use anyhow::{Context, Result};
use rand::Rng;
use rayon::prelude::*;

pub fn trial_iteration<R: Rng + ?Sized>(
    settings: &TrialSettings,
    tx: &mpsc::Sender<(DecodingResult, usize)>,
    rng: &mut R
) -> usize {
    let result = application::decoding_trial(settings, rng);
    if result.success() {
        0
    } else {
        // Attempt to send decoding failure, but ignore errors, as the receiver may
        // choose to hang up after receiving the maximum number of decoding failures.
        tx.send((result, current_thread_id())).ok();
        1
    }
}

// Runs decoding_trial in a loop, sending decoding failures via tx_results and
// progress updates (counts of decoding failures and trials run) via tx_progress.
pub fn trial_loop(
    settings: &Settings,
    tx_progress: mpsc::Sender<(usize, usize)>,
    tx_results: mpsc::Sender<(DecodingResult, usize)>,
    pool: rayon::ThreadPool,
) -> Result<()> {
    let mut trials_remaining = settings.number_of_trials();
    while trials_remaining > 0 {
        let tx_results = tx_results.clone();
        let new_trials = cmp::min(trials_remaining, settings.save_frequency());
        let new_failure_count = pool.install(|| (0..new_trials).into_par_iter().map_with(
            (settings.trial_settings(), tx_results),
            |(settings, tx), _| trial_iteration(&settings, &tx, &mut custom_thread_rng())
        ).sum());
        tx_progress.send((new_failure_count, new_trials))
            .context("Progress receiver should not be closed")?;
        trials_remaining -= new_trials;
    }
    Ok(())
}

pub fn record_trial_results(
    rx_progress: mpsc::Receiver<(usize, usize)>,
    rx_results: mpsc::Receiver<(DecodingResult, usize)>,
    settings: Settings,
    start_time: Instant
) -> Result<DataRecord> {
    let mut data = DataRecord::new(settings.key_filter(), settings.fixed_key().cloned());
    data.set_seed(get_or_insert_global_seed(settings.seed()));
    let mut rx_results_open = true;
    let mut rx_progress_open = true;
    'outer: while rx_results_open || rx_progress_open {
        while rx_results_open {
            match rx_results.try_recv() {
                Ok((result, thread)) => {
                    application::handle_decoding_failure(result, thread, &mut data, &settings);
                    if data.decoding_failures().len() == settings.record_max() {
                        break 'outer;
                    }        
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    rx_results_open = false;
                }
            }
        }
        while rx_progress_open {
            match rx_progress.recv_timeout(Duration::from_millis(100)) {
                Ok((new_fc, new_trials)) =>
                    application::handle_progress(new_fc, new_trials, &mut data,
                        &settings, start_time.elapsed())?,
                Err(mpsc::RecvTimeoutError::Timeout) => break,
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    rx_progress_open = false;
                },
            }
        }
    }
    // Drops the results receiver so no more decoding failures are handled
    drop(rx_results);
    for (new_fc, new_trials) in rx_progress {
        application::handle_progress(new_fc, new_trials, &mut data,
            &settings, start_time.elapsed())?;
    }
    data.set_thread_count(global_thread_count());
    data.set_runtime(start_time.elapsed());
    if !settings.silent() {
        application::write_json(settings.output_file(), &data)?;
    }
    Ok(data)
}

pub fn run_multithreaded(settings: Settings) -> Result<DataRecord> {
    let start_time = Instant::now();
    if settings.verbose() >= 1 {
        println!("{}", application::start_message(&settings));
    }
    application::check_file_writable(settings.output_file(), settings.overwrite())?;
    // Set global PRNG seed used for generating data
    get_or_insert_global_seed(settings.seed());
    let pool = rayon::ThreadPoolBuilder::new().num_threads(settings.threads()).build()?;
    // Set up channels to receive decoding results and progress updates
    let (tx_results, rx_results) = mpsc::channel();
    let (tx_progress, rx_progress) = mpsc::channel();
    let settings_clone = settings.clone();
    let handler_thread = thread::spawn(move ||
        record_trial_results(rx_progress, rx_results, settings_clone, start_time)
    );
    trial_loop(&settings, tx_progress, tx_results, pool)?;
    // Wait for data processing to finish
    let data = handler_thread.join()
        // If join() failed, propagate the panic from the thread
        .unwrap_or_else(|err| std::panic::resume_unwind(err))?;
    if settings.verbose() >= 1 {
        println!("{}", application::end_message(data.failure_count(), data.trials(), data.runtime()));
    }
    Ok(data)
}
