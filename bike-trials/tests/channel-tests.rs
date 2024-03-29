use bike_decoder::{keys::KeyFilter, ncw::NearCodewordClass, parameters::*, random};
use bike_trials::{
    parallel,
    settings::{SettingsBuilder, TrialSettings},
};
use crossbeam_channel::unbounded as channel;
use std::time::Duration;

fn guaranteed_failure_settings() -> TrialSettings {
    TrialSettings::new(
        KeyFilter::Any,
        None,
        Some(NearCodewordClass::N),
        Some(BLOCK_WEIGHT),
    )
    .unwrap()
}

#[test]
fn receive_decoding_failure() {
    let settings = guaranteed_failure_settings();
    let (tx, rx) = channel();
    let mut rng = random::custom_thread_rng();
    parallel::trial_iteration(&settings, &tx, &mut rng);
    let result = rx
        .recv_timeout(Duration::from_secs(1))
        .expect("Should receive decoding failure in under 1 second");
    assert_eq!(result.thread.unwrap(), random::current_thread_id());
}

#[test]
fn receive_progress_message() {
    let settings = SettingsBuilder::default()
        .num_trials(10)
        .threads(4)
        .build()
        .unwrap();
    let (tx_results, _) = channel();
    let (tx_progress, rx) = channel();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(settings.threads())
        .build()
        .unwrap();
    pool.install(|| {
        parallel::trial_loop(
            settings.trial_settings(),
            settings.num_trials(),
            settings.save_frequency(),
            &tx_results,
            &tx_progress,
        )
        .unwrap()
    });
    let dfr = rx
        .recv_timeout(Duration::from_secs(1))
        .expect("Should receive progress update in under 1 second");
    assert_eq!(dfr.num_failures(), 0);
    assert_eq!(dfr.num_trials(), 10);
}
