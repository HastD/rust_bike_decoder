use bike_decoder::{
    cli,
    decoder,
    keys::Key,
    ncw::{TaggedErrorVector, NearCodewordClass},
    parameters::*,
    random,
    settings::TrialSettings,
    syndrome::Syndrome,
    vectors::SparseErrorVector,
    threshold,
};
use std::hint::black_box;
use criterion::{criterion_group, criterion_main, Criterion, BatchSize};
use rand::Rng;

pub fn decoder_benchmarks(c: &mut Criterion) {
    c.bench_function("decoding_trial", |b| {
        let settings = TrialSettings::default();
        let (mut rng, _) = random::get_rng(None);
        b.iter(|| black_box(cli::decoding_trial(&settings, &mut rng)))
    });
    c.bench_function("Key::random", |b| {
        let (mut rng, _) = random::get_rng(None);
        b.iter(|| black_box(Key::random(&mut rng)))
    });
    c.bench_function("Key::random_non_weak", |b| {
        let (mut rng, _) = random::get_rng(None);
        b.iter(|| black_box(Key::random_non_weak(3, &mut rng)))
    });
    c.bench_function("near_codeword", |b| {
        let (mut rng_key, _) = random::get_rng(None);
        let (mut rng_ncw, _) = random::get_rng(None);
        b.iter_batched_ref(
            || {
                let key = Key::random(&mut rng_key);
                let l = rng_key.gen_range(0..=BLOCK_WEIGHT);
                (key, l)
            },
            |inputs| black_box((
                TaggedErrorVector::near_codeword(&inputs.0, NearCodewordClass::C, inputs.1, &mut rng_ncw),
                TaggedErrorVector::near_codeword(&inputs.0, NearCodewordClass::N, inputs.1, &mut rng_ncw),
                TaggedErrorVector::near_codeword(&inputs.0, NearCodewordClass::TwoN, inputs.1, &mut rng_ncw),
            )),
            BatchSize::SmallInput
        )
    });
    c.bench_function("syndrome", |b| {
        let (mut rng, _) = random::get_rng(None);
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
        let (mut rng, _) = random::get_rng(None);
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
    c.bench_function("bgf_decoder", |b| {
        let (mut rng, _) = random::get_rng(None);
        b.iter_batched_ref(
            || {
                let key = Key::random(&mut rng);
                let e_supp = SparseErrorVector::random(&mut rng);
                let syn = Syndrome::from_sparse(&key, &e_supp);
                (key, syn)
            },
            |inputs| black_box(decoder::bgf_decoder(&inputs.0, &mut inputs.1)),
            BatchSize::SmallInput
        )
    });
    c.bench_function("upc", |b| {
        let (mut rng, _) = random::get_rng(None);
        b.iter_batched_ref(
            || {
                let key = Key::random(&mut rng);
                let e_supp = SparseErrorVector::random(&mut rng);
                let syn = Syndrome::from_sparse(&key, &e_supp);
                (key, syn)
            },
            |inputs| black_box(decoder::unsatisfied_parity_checks(&inputs.0, &mut inputs.1)),
            BatchSize::SmallInput
        )
    });
    c.bench_function("threshold", |b| {
        let (r, d, t) = (BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT);
        b.iter(|| {
            let x = threshold::compute_x(r, d, t);
            for ws in 0..=r as usize {
                black_box(threshold::exact_threshold_ineq(ws, r, d, t, Some(x)).unwrap());
            }
        })
    });
}

criterion_group!(benches, decoder_benchmarks);
criterion_main!(benches);
