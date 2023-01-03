use bike_decoder::{
    application, decoder,
    keys::{Key, KeyFilter},
    ncw::{ErrorVectorSource, NearCodewordClass},
    parameters::*,
    random,
    settings::TrialSettings,
    syndrome::Syndrome,
    vectors::{ErrorVector, SparseErrorVector},
};
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

#[test]
fn decoding_failure_example() {
    assert_eq!((BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT), (587, 15, 18));
    let key = Key::from_support(
        [
            41, 57, 63, 158, 163, 180, 194, 213, 234, 276, 337, 428, 451, 485, 573,
        ],
        [
            55, 84, 127, 185, 194, 218, 260, 374, 382, 394, 404, 509, 528, 537, 580,
        ],
    )
    .unwrap();
    let e_in = SparseErrorVector::from_support([
        10, 62, 157, 283, 460, 503, 533, 564, 715, 806, 849, 858, 916, 991, 996, 1004, 1078, 1096,
    ])
    .unwrap();
    let mut syn = Syndrome::from_sparse(&key, &e_in);
    let known_syn = [
        0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1,
        1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1,
        0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1,
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
    ];
    assert_eq!(bytemuck::cast_slice::<bool, u8>(syn.contents()), &known_syn);
    let (e_out, success) = decoder::bgf_decoder(&key, &mut syn);
    assert_ne!(syn.hamming_weight(), 0);
    assert!(!success);
    assert_eq!(
        e_out.support(),
        vec![
            10, 62, 157, 283, 460, 503, 533, 564, 644, 663, 672, 777, 858, 907, 940, 982, 991, 996,
            1004, 1078, 1104, 1116, 1126
        ]
    );
}

#[test]
fn syndrome_e_out_consistent() {
    let mut rng = rand::thread_rng();
    for _ in 0..TRIALS {
        let key = Key::random(&mut rng);
        let e_in = SparseErrorVector::random(&mut rng);
        let mut s = Syndrome::from_sparse(&key, &e_in);
        let s_original = s.clone();
        let mut e_out = ErrorVector::zero();
        let (black, _) = decoder::bf_iter(&key, &mut s, &mut e_out, BF_THRESHOLD_MIN);
        assert_eq!(s, s_original.clone() + Syndrome::from_dense(&key, &e_out));
        decoder::bf_masked_iter(&key, &mut s, &mut e_out, black, BF_MASKED_THRESHOLD);
        assert_eq!(s, s_original.clone() + Syndrome::from_dense(&key, &e_out));
    }
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
