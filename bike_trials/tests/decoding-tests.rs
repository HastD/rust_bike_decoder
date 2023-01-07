use bike_decoder::{
    keys::{Key, KeyFilter},
    ncw::{ErrorVectorSource, NearCodewordClass},
    parameters::*,
    random,
    vectors::SparseErrorVector,
};
use bike_trials::{application, settings::TrialSettings};
use rand::{rngs::StdRng, SeedableRng};

const TRIALS: usize = 10000;

#[test]
fn decoding_trial_example() {
    let settings = TrialSettings::default();
    let mut rng = StdRng::seed_from_u64(15912031812057470983);
    let result = application::decoding_trial(&settings, &mut rng);
    let key = Key::from_support(
        [
            203, 396, 303, 540, 109, 508, 149, 15, 161, 332, 511, 243, 367, 305, 103,
        ],
        [
            389, 255, 270, 131, 555, 562, 175, 223, 273, 576, 449, 106, 116, 8, 120,
        ],
    )
    .unwrap();
    let vector = SparseErrorVector::from_support([
        533, 450, 441, 491, 1039, 130, 180, 1086, 97, 23, 1169, 67, 619, 596, 759, 120, 157, 958,
    ])
    .unwrap();
    assert_eq!(result.key().clone(), key);
    assert_eq!(
        result.vector().clone().take_vector(),
        (vector, ErrorVectorSource::Random)
    );
    assert!(result.success());
}

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
fn guaranteed_decoding_failure() {
    let settings = guaranteed_failure_settings();
    let mut rng = random::custom_thread_rng();
    for _ in 0..TRIALS {
        let result = application::decoding_trial(&settings, &mut rng);
        assert!(!result.success());
    }
}
