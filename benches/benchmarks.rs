use bike_decoder::{
    application::{decoding_failure_trial, handle_decoding_failure},
    decoder::{bgf_decoder, unsatisfied_parity_checks},
    keys::{Key, KeyFilter},
    ncw::{TaggedErrorVector, NearCodewordClass},
    parameters::*,
    random::{custom_thread_rng, global_seed},
    record::DataRecord,
    settings::{SettingsBuilder, TrialSettings},
    syndrome::Syndrome,
    vectors::SparseErrorVector,
    threshold::{compute_x, exact_threshold_ineq},
};
use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion, BatchSize};
use rand::Rng;
use crossbeam_channel::{unbounded as channel};

pub fn group_decoder(c: &mut Criterion) {
    c.bench_function("decoding_trial", |b| {
        let settings = TrialSettings::default();
        let mut rng = custom_thread_rng();
        b.iter(|| black_box(decoding_failure_trial(&settings, &mut rng)))
    });

    c.bench_function("bgf_decoder", |b| {
        let mut rng = custom_thread_rng();
        b.iter_batched_ref(
            || {
                let key = Key::random(&mut rng);
                let e_supp = SparseErrorVector::random(&mut rng);
                let syn = Syndrome::from_sparse(&key, &e_supp);
                (key, syn)
            },
            |inputs| black_box(bgf_decoder(&inputs.0, &mut inputs.1)),
            BatchSize::SmallInput
        )
    });
    c.bench_function("upc", |b| {
        let mut rng = custom_thread_rng();
        b.iter_batched_ref(
            || {
                let key = Key::random(&mut rng);
                let e_supp = SparseErrorVector::random(&mut rng);
                let syn = Syndrome::from_sparse(&key, &e_supp);
                (key, syn)
            },
            |inputs| black_box(unsatisfied_parity_checks(&inputs.0, &mut inputs.1)),
            BatchSize::SmallInput
        )
    });
}

pub fn group_randgen(c: &mut Criterion) {
    c.bench_function("Key::random", |b| {
        let mut rng = custom_thread_rng();
        b.iter(|| black_box(Key::random(&mut rng)))
    });
    c.bench_function("Key::random_non_weak", |b| {
        let mut rng = custom_thread_rng();
        b.iter(|| black_box(Key::random_non_weak(3, &mut rng)))
    });
    c.bench_function("near_codeword", |b| {
        let mut rng = custom_thread_rng();
        b.iter_batched_ref(
            || {
                let mut rng = custom_thread_rng();
                let key = Key::random(&mut rng);
                let l = rng.gen_range(0..=BLOCK_WEIGHT);
                (key, l)
            },
            |inputs| black_box((
                TaggedErrorVector::near_codeword(&inputs.0, NearCodewordClass::C, inputs.1, &mut rng),
                TaggedErrorVector::near_codeword(&inputs.0, NearCodewordClass::N, inputs.1, &mut rng),
                TaggedErrorVector::near_codeword(&inputs.0, NearCodewordClass::TwoN, inputs.1, &mut rng),
            )),
            BatchSize::SmallInput
        )
    });
}

pub fn group_syndrome(c: &mut Criterion) {
    c.bench_function("syndrome", |b| {
        let mut rng = custom_thread_rng();
        b.iter_batched_ref(
            || {
                let key = Key::random(&mut rng);
                let e_supp = SparseErrorVector::random(&mut rng);
                (key, e_supp)
            },
            |inputs| black_box(Syndrome::from_sparse(&inputs.0, &inputs.1)),
            BatchSize::SmallInput
        )
    });
    c.bench_function("weight", |b| {
        let mut rng = custom_thread_rng();
        b.iter_batched_ref(
            || {
                let key = Key::random(&mut rng);
                let e_supp = SparseErrorVector::random(&mut rng);
                Syndrome::from_sparse(&key, &e_supp)
            },
            |syn| black_box(syn.hamming_weight()),
            BatchSize::SmallInput
        )
    });
}

pub fn group_threshold(c: &mut Criterion) {
    c.bench_function("threshold", |b| {
        let (r, d, t) = (BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT);
        b.iter(|| {
            let x = compute_x(r, d, t).unwrap();
            for ws in 0..=r as usize {
                black_box(exact_threshold_ineq(ws, r, d, t, Some(x)).unwrap());
            }
        })
    });
}

pub fn group_record(c: &mut Criterion) {
    c.bench_function("record_decoding_failure", |b| {
        let settings = SettingsBuilder::default().silent(true)
            .number_of_trials(100)
            .trial_settings(TrialSettings::new(KeyFilter::Any, None, Some(NearCodewordClass::N),
                Some(BLOCK_WEIGHT)).unwrap())
            .build().unwrap();
        let mut rng = custom_thread_rng();
        let mut data = DataRecord::new(settings.key_filter(), settings.fixed_key().cloned(),
            global_seed().unwrap());
        b.iter_batched(
            || {
                let (tx, rx) = channel();
                for _ in 0..10 {
                    let df = decoding_failure_trial(&settings.trial_settings(), &mut rng).unwrap();
                    tx.send(df).ok();
                }
                drop(tx);
                rx
            },
            |rx| rx.iter().for_each(|result| handle_decoding_failure(result, 0, &mut data, &settings)),
            BatchSize::SmallInput
        )
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = group_decoder, group_randgen, group_syndrome, group_threshold, group_record
}
criterion_main!(benches);
