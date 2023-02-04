use bike_decoder::{
    keys::KeyFilter,
    ncw::NearCodewordClass,
    parameters::*,
    random::{custom_thread_rng, global_seed},
};
use bike_trials::{
    application::{self, decoding_failure_trial, handle_decoding_failure},
    output::OutputTo,
    parallel,
    record::DataRecord,
    settings::{SettingsBuilder, TrialSettings},
};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use crossbeam_channel::unbounded as channel;
use std::hint::black_box;

pub fn group_application(c: &mut Criterion) {
    c.bench_function("run_application", |b| {
        let settings = SettingsBuilder::default()
            .num_trials(10_000)
            .output(OutputTo::Void)
            .build()
            .unwrap();
        b.iter(|| black_box(application::run(&settings)))
    });

    c.bench_function("run_parallel", |b| {
        let settings = SettingsBuilder::default()
            .num_trials(10_000)
            .threads(0)
            .output(OutputTo::Void)
            .build()
            .unwrap();
        b.iter(|| black_box(parallel::run_parallel(&settings)))
    });

    c.bench_function("decoding_trial", |b| {
        let settings = TrialSettings::default();
        let mut rng = custom_thread_rng();
        b.iter(|| black_box(decoding_failure_trial(&settings, &mut rng)))
    });
}

pub fn group_record(c: &mut Criterion) {
    c.bench_function("record_decoding_failure", |b| {
        let settings = SettingsBuilder::default()
            .num_trials(100)
            .trial_settings(
                TrialSettings::new(
                    KeyFilter::Any,
                    None,
                    Some(NearCodewordClass::N),
                    Some(BLOCK_WEIGHT),
                )
                .unwrap(),
            )
            .output(OutputTo::Void)
            .build()
            .unwrap();
        let mut rng = custom_thread_rng();
        let mut data = DataRecord::new(
            settings.key_filter(),
            settings.fixed_key().cloned(),
            global_seed().unwrap(),
        );
        b.iter_batched(
            || {
                let (tx, rx) = channel();
                for _ in 0..10 {
                    let df = decoding_failure_trial(settings.trial_settings(), &mut rng).unwrap();
                    tx.send(df).ok();
                }
                drop(tx);
                rx
            },
            |rx| {
                rx.iter()
                    .for_each(|result| handle_decoding_failure(result, &mut data, &settings))
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = group_application, group_record
}
criterion_main!(benches);
