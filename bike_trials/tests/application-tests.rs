use bike_decoder::{
    keys::Key,
    ncw::ErrorVectorSource,
    random::{self, Seed},
    vectors::SparseErrorVector,
};
use bike_trials::{
    application, parallel,
    settings::{OutputTo, SettingsBuilder},
};

#[test]
fn main_single_threaded_test() {
    let seed =
        Seed::try_from("b439d3f5b9f2d127effcc98ed2a70806441de9e5b3bc4f6d32ec2b963af03fee").unwrap();
    let settings = SettingsBuilder::default()
        .num_trials(10_000)
        .output(OutputTo::Void)
        .threads(1)
        .seed(Some(seed))
        .seed_index(Some(0))
        .build()
        .unwrap();
    let data = application::run(&settings).unwrap();
    assert!(data.thread_count().is_none());
    assert_eq!(data.seed(), seed);
    assert_eq!(data.num_failures(), 1);
    assert_eq!(data.decoding_failures().len(), 1);
    let (key, e_supp) = data.decoding_failures()[0].clone().take_key_vector();
    assert_eq!(
        key,
        Key::from_support(
            [114, 156, 192, 208, 285, 304, 323, 399, 418, 443, 491, 505, 535, 540, 541],
            [8, 27, 67, 140, 160, 173, 220, 225, 242, 259, 310, 313, 358, 550, 579]
        )
        .unwrap()
    );
    assert_eq!(
        *e_supp.vector(),
        SparseErrorVector::from_support([
            110, 114, 129, 177, 204, 226, 236, 387, 446, 592, 622, 660, 732, 753, 836, 841, 959,
            1053
        ])
        .unwrap()
    );
    assert_eq!(*e_supp.source(), ErrorVectorSource::Random);
}

#[test]
fn parallel_fail_if_seed_fail() {
    let settings = SettingsBuilder::default()
        .num_trials(100)
        .seed(Some(Seed::from_entropy()))
        .threads(2)
        .build()
        .unwrap();
    // Ensures global seed is already set before run_parallel is called
    random::get_or_insert_global_seed(None);
    assert!(parallel::run_parallel(&settings)
        .unwrap_err()
        .is::<random::TryInsertGlobalSeedError>());
}
