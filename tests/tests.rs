use bike_decoder::{
    application,
    decoder,
    keys::{Key, KeyFilter},
    ncw::{ErrorVectorSource, NearCodewordClass},
    parallel,
    parameters::*,
    random::{self, Seed},
    settings::{SettingsBuilder, TrialSettings},
    syndrome::Syndrome,
    vectors::{ErrorVector, SparseErrorVector},
};
use std::{sync::mpsc, time::Duration};
use rand::{SeedableRng, rngs::StdRng};

const TRIALS: usize = 10000;

#[test]
fn decoding_trial_example() {
    let settings = TrialSettings::default();
    let mut rng = StdRng::seed_from_u64(15912031812057470983);
    let result = application::decoding_trial(&settings, &mut rng);
    let key = Key::from_support(
        [203, 396, 303, 540, 109, 508, 149, 15, 161, 332, 511, 243, 367, 305, 103],
        [389, 255, 270, 131, 555, 562, 175, 223, 273, 576, 449, 106, 116, 8, 120]
    ).unwrap();
    let vector = SparseErrorVector::from_support([533,450,441,491,1039,130,180,1086,97,23,1169,67,619,596,759,120,157,958]).unwrap();
    assert_eq!(result.key().clone(), key);
    assert_eq!(result.vector().clone().take_vector(), (vector, ErrorVectorSource::Random));
    assert!(result.success());
}

#[test]
fn decoding_failure_example() {
    assert_eq!((BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT), (587, 15, 18));
    let key = Key::from_support(
        [41, 57, 63, 158, 163, 180, 194, 213, 234, 276, 337, 428, 451, 485, 573],
        [55, 84, 127, 185, 194, 218, 260, 374, 382, 394, 404, 509, 528, 537, 580],
    ).unwrap();
    let e_in = SparseErrorVector::from_support([10,62,157,283,460,503,533,564,715,806,849,858,916,991,996,1004,1078,1096]).unwrap();
    let mut syn = Syndrome::from_sparse(&key, &e_in);
    assert_eq!(syn.contents(), [0,0,0,1,0,0,0,1,0,1,0,1,0,0,1,1,1,0,1,0,0,1,0,0,1,0,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,0,1,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,1,0,0,1,0,0,0,0,1,0,0,0,1,0,1,1,1,0,0,0,0,0,1,0,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,1,1,0,0,1,1,0,0,1,0,0,0,0,0,1,1,0,0,0,1,1,0,1,0,0,1,1,0,1,1,0,0,0,0,0,0,1,0,1,1,0,0,0,1,0,0,0,1,1,0,1,0,1,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,1,1,1,0,0,1,0,1,1,0,0,0,1,1,0,0,0,0,0,1,0,1,0,1,0,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,0,1,1,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,1,1,0,0,0,1,0,0,0,0,1,1,1,1,1,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,1,1,1,0,0,0,1,1,0,1,0,0,1,1,0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,1,0,0,0,0,0,0,1,1,0,1,1,0,1,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,1,0,0,0,0,1,0,1,1,0,0,0,1,0,0,0,1,0,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,1,0]);
    let (e_out, success) = decoder::bgf_decoder(&key, &mut syn);
    println!("syn (after decoding) = {}\nsuccess = {}\ne_out = {:?}", syn, success, e_out.support());
    assert!(!success);
    assert_eq!(e_out.support(), vec![10,62,157,283,460,503,533,564,644,663,672,777,858,907,940,982,991,996,1004,1078,1104,1116,1126]);
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
    let mut rng = random::custom_thread_rng();
    for _ in 0..TRIALS {
        let result = application::decoding_trial(&settings, &mut rng);
        assert!(!result.success());
    }
}

#[test]
fn receive_decoding_failure() {
    let settings = guaranteed_failure_settings();
    let (tx, rx) = mpsc::channel();
    let mut rng = random::custom_thread_rng();
    parallel::trial_iteration(&settings, &tx, &mut rng);
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
    parallel::trial_loop(&settings, tx_progress, tx_results).unwrap();
    let (failure_count, trials) = rx.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(trials, 10);
    assert_eq!(failure_count, 0);
}

#[test]
fn main_single_threaded_test() {
    let seed = Seed::try_from("052a104710b64326bcfd1ce592b9817552f72e210fa2b0520c64e9c9535606bf".to_string()).unwrap();
    let settings = SettingsBuilder::default()
        .number_of_trials(100_000).silent(true)
        .threads(1)
        .seed(Some(seed))
        .seed_index(Some(0))
        .build().unwrap();
    let data = application::run_single_threaded(settings).unwrap();
    assert!(data.thread_count().is_none());
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

// This test has to run by itself or the global seeding causes problems
#[test]
#[ignore]
fn main_multithreaded_test() {
    let seed = Seed::try_from("53b3f64c5c1421b41fef9c6485a98f6739ba8cceedbe57cba1770324eb8f3b61".to_string()).unwrap();
    let settings = SettingsBuilder::default()
        .number_of_trials(200_000).silent(true)
        .threads(4)
        .seed(Some(seed))
        .build().unwrap();
    let data = parallel::run_multithreaded(settings).unwrap();
    assert_eq!(random::global_seed().unwrap(), seed);
    assert_eq!(data.seed().unwrap(), seed);
    assert_eq!(data.thread_count(), Some(4));
    assert_eq!(data.failure_count(), 2);
    let mut decoding_failures = data.decoding_failures().clone();
    assert_eq!(data.decoding_failures().len(), 2);
    decoding_failures.sort_by_key(|df| df.thread());
    let df2 = decoding_failures.pop().unwrap();
    let df0 = decoding_failures.pop().unwrap();
    assert_eq!(df0.thread(), 0);
    assert_eq!(Key::from((df0.h0().clone(), df0.h1().clone())), Key::from_support(
        [54,102,112,122,165,169,199,400,468,478,496,533,563,571,581],
        [6,16,36,95,104,181,209,229,259,317,325,363,412,549,576]
    ).unwrap());
    assert_eq!(df0.e_supp().clone(), SparseErrorVector::from_support(
        [55,129,138,196,206,399,407,451,471,486,581,646,791,840,847,1099,1127,1165]
    ).unwrap());
    assert_eq!(df0.e_source(), ErrorVectorSource::Random);
    assert_eq!(df2.thread(), 2);
    assert_eq!(Key::from((df2.h0().clone(), df2.h1().clone())), Key::from_support(
        [34,87,90,134,264,273,299,338,382,390,465,512,529,547,556],
        [61,81,193,253,267,341,358,390,394,447,458,510,557,564,579]
    ).unwrap());
    assert_eq!(df2.e_supp().clone(), SparseErrorVector::from_support(
        [12,44,59,101,109,145,150,237,284,289,672,696,741,769,799,986,1117,1124]
    ).unwrap());
    assert_eq!(df2.e_source(), ErrorVectorSource::Random);
}
