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
        Seed::try_from("b6bcde4d5776464f054200a9d85943c8c66eabf9c9d21de3d0eda6023d38f02a").unwrap();
    SettingsBuilder::default()
        .num_trials(10_000)
        .output(OutputTo::Void)
        .threads(4)
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
    assert_eq!(data.thread_count(), Some(4));
    assert_eq!(data.num_failures(), 1, "num_failures() didn't match");
    assert_eq!(data.num_trials(), settings.num_trials());
    assert_eq!(
        data.decoding_failures().len(),
        1,
        "decoding_failures().len() didn't match"
    );
    let df = &data.decoding_failures()[0];
    assert_eq!(df.thread, Some(3));
    assert_eq!(
        df.key(),
        &Key::from_support(
            [2, 24, 38, 45, 56, 67, 259, 323, 328, 348, 360, 388, 436, 574, 580],
            [26, 55, 173, 231, 316, 324, 347, 354, 436, 439, 453, 459, 494, 532, 579]
        )
        .unwrap()
    );
    let e_supp = df.vector();
    assert_eq!(
        e_supp.vector(),
        &SparseErrorVector::from_support([
            28, 176, 185, 425, 587, 631, 660, 718, 733, 750, 780, 803, 825, 831, 958, 1116, 1132,
            1161
        ])
        .unwrap()
    );
    assert_eq!(*e_supp.source(), ErrorVectorSource::Random);
}
