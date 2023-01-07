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
        Seed::try_from("052a104710b64326bcfd1ce592b9817552f72e210fa2b0520c64e9c9535606bf").unwrap();
    let settings = SettingsBuilder::default()
        .num_trials(100_000)
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
            [78, 107, 113, 195, 230, 231, 259, 265, 354, 383, 412, 430, 455, 501, 583],
            [8, 26, 62, 150, 204, 242, 265, 312, 324, 386, 437, 523, 535, 547, 566]
        )
        .unwrap()
    );
    assert_eq!(
        *e_supp.vector(),
        SparseErrorVector::from_support([
            138, 276, 406, 447, 489, 494, 523, 553, 562, 622, 630, 651, 692, 733, 735, 783, 951,
            1158
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
