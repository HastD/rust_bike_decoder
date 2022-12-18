use bike_decoder::{
    cli,
    decoder,
    keys::{Key, KeyFilter},
    ncw::{ErrorVectorSource, NearCodewordClass},
    parameters::*,
    random::{self, Seed},
    settings::{SettingsBuilder, TrialSettings},
    syndrome::Syndrome,
    vectors::{ErrorVector, SparseErrorVector},
};
use std::{sync::mpsc, time::Duration};

const TRIALS: usize = 10000;

#[test]
fn decoding_failure_example() {
    assert_eq!((BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT), (587, 15, 18));
    let key = Key::from_support(
        [41, 57, 63, 158, 163, 180, 194, 213, 234, 276, 337, 428, 451, 485, 573],
        [55, 84, 127, 185, 194, 218, 260, 374, 382, 394, 404, 509, 528, 537, 580],
    ).unwrap();
    let e_in = SparseErrorVector::from_support([10, 62, 157, 283, 460, 503, 533, 564, 715, 806, 849, 858, 916, 991, 996, 1004, 1078, 1096]).unwrap();
    let mut syn = Syndrome::from_sparse(&key, &e_in);
    assert_eq!(syn.contents(), [0u8, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]);
    println!("e_in = {}", e_in);
    println!("syn = {}", syn);
    let (e_out, success) = decoder::bgf_decoder(&key, &mut syn);
    println!("syn (after decoding) = {}\nsuccess = {}\ne_out = {:?}", syn, success, e_out.support());
    assert!(!success);
    assert_eq!(e_out.support(), vec![10, 62, 157, 283, 460, 503, 533, 564, 644, 663, 672, 777, 858, 907, 940, 982, 991, 996, 1004, 1078, 1104, 1116, 1126]);
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
        let mut black = [[false; BLOCK_LENGTH]; 2];
        let mut gray = [[false; BLOCK_LENGTH]; 2];
        decoder::bf_iter(&key, &mut s, &mut e_out, &mut black, &mut gray, BF_THRESHOLD_MIN as u8);
        assert_eq!(s, s_original.clone() + Syndrome::from_dense(&key, &e_out));
        decoder::bf_masked_iter(&key, &mut s, &mut e_out, black, BF_MASKED_THRESHOLD);
        assert_eq!(s, s_original.clone() + Syndrome::from_dense(&key, &e_out));
    }
}

fn guaranteed_failure_settings() -> TrialSettings {
    TrialSettings::new(KeyFilter::Any, None, Some(NearCodewordClass::N), Some(BLOCK_WEIGHT)).unwrap()
}

#[test]
fn guaranteed_decoding_failure() {
    let settings = guaranteed_failure_settings();
    for _ in 0..TRIALS {
        let result = cli::decoding_trial(&settings);
        assert!(!result.success());
    }
}

#[test]
fn receive_decoding_failure() {
    let settings = guaranteed_failure_settings();
    let (tx, rx) = mpsc::channel();
    cli::trial_iteration(&settings, &tx);
    let (result, thread_id) = rx.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(!result.success());
    assert_eq!(thread_id, random::current_thread_id())
}

#[test]
fn receive_progress_message() {
    let settings = SettingsBuilder::default()
        .number_of_trials(10).threads(4)
        .build().unwrap();
    let (tx_results, _) = mpsc::channel();
    let (tx_progress, rx) = mpsc::channel();
    cli::trial_loop_parallel(&settings, tx_progress, tx_results).unwrap();
    let (failure_count, trials) = rx.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(trials, 10);
    assert_eq!(failure_count, 0);
}

// This test has to run by itself or the global seeding causes problems
#[test]
#[ignore]
fn main_trials_test() {
    let seed = Seed::try_from("052a104710b64326bcfd1ce592b9817552f72e210fa2b0520c64e9c9535606bf".to_string()).unwrap();
    let settings = SettingsBuilder::default()
        .number_of_trials(100_000).silent(true)
        .threads(1)
        .seed(Some(seed))
        .build().unwrap();
    let data = cli::run_cli(settings).unwrap();
    assert_eq!(random::global_seed().unwrap(), seed);
    assert_eq!(data.seed().unwrap(), seed);
    assert_eq!(data.failure_count(), 1);
    assert_eq!(data.decoding_failures().len(), 1);
    let df = &data.decoding_failures()[0];
    assert_eq!(Key::from((df.h0().clone(), df.h1().clone())), Key::from_support(
        [78,107,113,195,230,231,259,265,354,383,412,430,455,501,583],
        [8,26,62,150,204,242,265,312,324,386,437,523,535,547,566]
    ).unwrap());
    assert_eq!(df.e_supp().clone(), SparseErrorVector::from_support(
        [138,276,406,447,489,494,523,553,562,622,630,651,692,733,735,783,951,1158]
    ).unwrap());
    assert_eq!(df.e_source(), ErrorVectorSource::Random);
}
