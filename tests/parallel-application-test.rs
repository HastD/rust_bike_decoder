use bike_decoder::{
    keys::Key,
    ncw::ErrorVectorSource,
    parallel,
    random::{self, Seed},
    settings::{OutputTo, Settings, SettingsBuilder},
    vectors::SparseErrorVector,
};

fn multithreaded_example_settings() -> Settings {
    let seed =
        Seed::try_from("53b3f64c5c1421b41fef9c6485a98f6739ba8cceedbe57cba1770324eb8f3b61").unwrap();
    SettingsBuilder::default()
        .num_trials(200_000)
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
    assert_eq!(data.num_failures(), 2, "num_failures() didn't match");
    assert_eq!(data.num_trials(), settings.num_trials());
    let mut decoding_failures = data.decoding_failures().clone();
    assert_eq!(
        data.decoding_failures().len(),
        2,
        "decoding_failures().len() didn't match"
    );
    decoding_failures.sort_by_key(|df| df.thread);
    let df2 = decoding_failures.pop().unwrap();
    let df0 = decoding_failures.pop().unwrap();
    assert_eq!(df0.thread, Some(0));
    let (key0, e0_supp) = df0.take_key_vector();
    assert_eq!(
        key0,
        Key::from_support(
            [54, 102, 112, 122, 165, 169, 199, 400, 468, 478, 496, 533, 563, 571, 581],
            [6, 16, 36, 95, 104, 181, 209, 229, 259, 317, 325, 363, 412, 549, 576]
        )
        .unwrap()
    );
    assert_eq!(
        *e0_supp.vector(),
        SparseErrorVector::from_support([
            55, 129, 138, 196, 206, 399, 407, 451, 471, 486, 581, 646, 791, 840, 847, 1099, 1127,
            1165
        ])
        .unwrap()
    );
    assert_eq!(*e0_supp.source(), ErrorVectorSource::Random);
    assert_eq!(df2.thread, Some(2));
    let (key2, e2_supp) = df2.take_key_vector();
    assert_eq!(
        key2,
        Key::from_support(
            [34, 87, 90, 134, 264, 273, 299, 338, 382, 390, 465, 512, 529, 547, 556],
            [61, 81, 193, 253, 267, 341, 358, 390, 394, 447, 458, 510, 557, 564, 579]
        )
        .unwrap()
    );
    assert_eq!(
        *e2_supp.vector(),
        SparseErrorVector::from_support([
            12, 44, 59, 101, 109, 145, 150, 237, 284, 289, 672, 696, 741, 769, 799, 986, 1117, 1124
        ])
        .unwrap()
    );
    assert_eq!(*e2_supp.source(), ErrorVectorSource::Random);
}
