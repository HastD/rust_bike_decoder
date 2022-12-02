use criterion::{black_box, criterion_group, criterion_main, Criterion, BatchSize};
use bike_decoder::{
    atls::{self, NearCodewordClass},
    decoder,
    keys::Key,
    random,
    parameters::*,
    syndrome::Syndrome,
    vectors::SparseErrorVector,
    threshold::{self, ThresholdCache},
};
use rand::Rng;

pub fn decoder_benchmarks(c: &mut Criterion) {
    c.bench_function("Key::random", |b| {
        let mut rng = random::get_rng();
        b.iter(|| black_box(Key::random(&mut rng)))
    });
    c.bench_function("Key::random_non_weak", |b| {
        let mut rng = random::get_rng();
        b.iter(|| black_box(Key::random_non_weak(3, &mut rng)))
    });
    c.bench_function("syndrome", |b| {
        let mut rng = random::get_rng();
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
        let mut rng = random::get_rng();
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
        let (r, d, t) = (BLOCK_LENGTH as u32, BLOCK_WEIGHT as u32, ERROR_WEIGHT as u32);
        let mut rng = random::get_rng();
        let mut threshold_cache = ThresholdCache::with_parameters(r, d, t);
        threshold_cache.precompute_all();
        b.iter_batched_ref(
            || {
                let key = Key::random(&mut rng);
                let e_supp = SparseErrorVector::random(&mut rng);
                let syn = Syndrome::from_sparse(&key, &e_supp);
                (key, syn)
            },
            |inputs| black_box(decoder::bgf_decoder(&inputs.0, &mut inputs.1, &mut threshold_cache)),
            BatchSize::SmallInput
        )
    });
    c.bench_function("upc", |b| {
        let (r, d, t) = (BLOCK_LENGTH as u32, BLOCK_WEIGHT as u32, ERROR_WEIGHT as u32);
        let mut rng = random::get_rng();
        let mut threshold_cache = ThresholdCache::with_parameters(r, d, t);
        threshold_cache.precompute_all();
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
        let (r, d, t) = (BLOCK_LENGTH as u32, BLOCK_WEIGHT as u32, ERROR_WEIGHT as u32);
        b.iter(|| black_box(threshold::ThresholdCache::with_parameters(r, d, t).precompute_all()))
    });
    c.bench_function("atls", |b| {
        b.iter_batched_ref(
            || {
                let mut rng = rand::thread_rng();
                let key = Key::random(&mut rng);
                let l = rng.gen_range(0..=BLOCK_WEIGHT);
                (key, l, rng)
            },
            |inputs| black_box((
                atls::element_of_atls(&inputs.0, NearCodewordClass::C, inputs.1, &mut inputs.2),
                atls::element_of_atls(&inputs.0, NearCodewordClass::N, inputs.1, &mut inputs.2),
                atls::element_of_atls(&inputs.0, NearCodewordClass::TwoN, inputs.1, &mut inputs.2),
            )),
            BatchSize::SmallInput
        )
    });
}

criterion_group!(benches, decoder_benchmarks);
criterion_main!(benches);
