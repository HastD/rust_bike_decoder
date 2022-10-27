use criterion::{black_box, criterion_group, criterion_main, Criterion, BatchSize};
use bike_decoder::{
    keys::Key,
    syndrome::Syndrome,
    vectors::SparseErrorVector,
    threshold::ThresholdCache,
    random,
    decoder,
    parameters::*
};

pub fn decoder_benchmarks(c: &mut Criterion) {
    c.bench_function("Key::random", |b| {
        let mut rng = random::get_rng();
        let key_dist = random::get_key_dist();
        b.iter(|| black_box(Key::random(&mut rng, &key_dist)))
    });
    c.bench_function("Key::random_non_weak", |b| {
        let mut rng = random::get_rng();
        let key_dist = random::get_key_dist();
        b.iter(|| black_box(Key::random_non_weak(&mut rng, &key_dist)))
    });
    c.bench_function("syndrome", |b| {
        let mut rng = random::get_rng();
        let key_dist = random::get_key_dist();
        let err_dist = random::get_err_dist();
        b.iter_batched_ref(
            || {
                let key = Key::random(&mut rng, &key_dist);
                let e_supp = SparseErrorVector::random(&mut rng, &err_dist);
                (key, e_supp)
            },
            |inputs| black_box(Syndrome::from_sparse(&inputs.0, &inputs.1)),
            BatchSize::SmallInput
        )
    });
    c.bench_function("bgf_decoder", |b| {
        let (r, d, t) = (BLOCK_LENGTH as u32, BLOCK_WEIGHT as u32, ERROR_WEIGHT as u32);
        let key_dist = random::get_key_dist();
        let err_dist = random::get_err_dist();
        let mut rng = random::get_rng();
        let mut threshold_cache = ThresholdCache::with_parameters(r, d, t);
        threshold_cache.precompute_all();
        b.iter_batched_ref(
            || {
                let key = Key::random_non_weak(&mut rng, &key_dist);
                let e_supp = SparseErrorVector::random(&mut rng, &err_dist);
                let syn = Syndrome::from_sparse(&key, &e_supp);
                (key, syn)
            },
            |inputs| black_box(decoder::bgf_decoder(&inputs.0, &mut inputs.1, &mut threshold_cache)),
            BatchSize::SmallInput
        )
    });
    c.bench_function("upc", |b| {
        let (r, d, t) = (BLOCK_LENGTH as u32, BLOCK_WEIGHT as u32, ERROR_WEIGHT as u32);
        let key_dist = random::get_key_dist();
        let err_dist = random::get_err_dist();
        let mut rng = random::get_rng();
        let mut threshold_cache = ThresholdCache::with_parameters(r, d, t);
        threshold_cache.precompute_all();
        b.iter_batched_ref(
            || {
                let key = Key::random(&mut rng, &key_dist);
                let e_supp = SparseErrorVector::random(&mut rng, &err_dist);
                let syn = Syndrome::from_sparse(&key, &e_supp);
                (key, syn)
            },
            |inputs| black_box(decoder::unsatisfied_parity_checks(&inputs.0, &mut inputs.1)),
            BatchSize::SmallInput
        )
    });
}

criterion_group!(benches, decoder_benchmarks);
criterion_main!(benches);
