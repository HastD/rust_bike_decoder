use bike_decoder::{
    keys::Key,
    ncw::ErrorVectorSource,
    random::{self, Seed},
    vectors::SparseErrorVector,
};
use bike_trials::{
    parallel,
    settings::{OutputTo, Settings, SettingsBuilder},
};

fn multithreaded_example_settings() -> Settings {
    let seed =
        Seed::try_from("0a85f4ae8350b3a3034145b19a8d7bfa11f0baeeae106f6812ddfd0e5890b61d").unwrap();
    SettingsBuilder::default()
        .num_trials(10_000)
        .output(OutputTo::Void)
        .threads(3)
        .seed(Some(seed))
        .build()
        .unwrap()
}

#[test]
fn main_multithreaded_test() {
    let settings = multithreaded_example_settings();
    let seed = settings.seed().unwrap();
    let data = parallel::run_parallel(&settings).unwrap();
    assert_eq!(random::global_seed().unwrap(), seed);
    assert_eq!(data.seed(), seed);
    assert_eq!(data.num_failures(), data.decoding_failures().len() as u64);
    assert_eq!(data.thread_count(), Some(3));
    assert_eq!(data.num_failures(), 1, "num_failures() didn't match");
    assert_eq!(data.num_trials(), settings.num_trials());
    assert_eq!(
        data.decoding_failures().len(),
        1,
        "decoding_failures().len() didn't match"
    );
    let df = &data.decoding_failures()[0];
    assert_eq!(df.thread, Some(1));
    assert_eq!(
        df.key(),
        &Key::from_support(
            [78, 96, 136, 206, 231, 307, 378, 443, 475, 493, 498, 512, 535, 560, 576],
            [14, 41, 88, 94, 112, 117, 168, 186, 215, 251, 354, 366, 493, 560, 563]
        )
        .unwrap()
    );
    let e_supp = df.vector();
    assert_eq!(
        e_supp.vector(),
        &SparseErrorVector::from_support([
            29, 47, 308, 334, 338, 375, 491, 579, 665, 681, 760, 765, 789, 821, 875, 900, 957, 1037
        ])
        .unwrap()
    );
    assert_eq!(*e_supp.source(), ErrorVectorSource::Random);
}
