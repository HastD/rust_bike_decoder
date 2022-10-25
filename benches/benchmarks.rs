use criterion::{black_box, criterion_group, criterion_main, Criterion, BatchSize};
use bike_decoder::{
    keys::Key,
    syndrome::Syndrome,
    vectors::SparseErrorVector,
    threshold::ThresholdCache,
    random,
    decoder,
    constants::*
};

pub fn decoder_benchmarks(c: &mut Criterion) {
    c.bench_function("Key::random", |b| {
        b.iter_batched_ref(
            || {
                let rng = random::get_rng();
                let key_dist = random::get_key_dist();
                (rng, key_dist)
            },
            |inputs| black_box(Key::random(&mut inputs.0, &inputs.1)),
            BatchSize::SmallInput
        )
    });
    c.bench_function("Key::random_non_weak", |b| {
        b.iter_batched_ref(
            || {
                let rng = random::get_rng();
                let key_dist = random::get_key_dist();
                (rng, key_dist)
            },
            |inputs| black_box(Key::random_non_weak(&mut inputs.0, &inputs.1)),
            BatchSize::SmallInput
        )
    });
    c.bench_function("syndrome", |b| {
        b.iter_batched_ref(
            || {
                let mut rng = random::get_rng();
                let key_dist = random::get_key_dist();
                let err_dist = random::get_err_dist();
                let key = Key::random(&mut rng, &key_dist);
                let e_supp = SparseErrorVector::random(&mut rng, &err_dist);
                (key, e_supp)
            },
            |inputs| black_box(Syndrome::from_sparse(&inputs.0, &inputs.1)),
            BatchSize::SmallInput
        )
    });
    c.bench_function("bgf_decoder", |b| {
        b.iter_batched_ref(
            || {
                let (r, d, t) = (BLOCK_LENGTH as u32, BLOCK_WEIGHT as u32, ERROR_WEIGHT as u32);
                let key_dist = random::get_key_dist();
                let err_dist = random::get_err_dist();
                let mut rng = random::get_rng();
                let threshold_cache = ThresholdCache::with_parameters(r, d, t);
                let key = Key::random_non_weak(&mut rng, &key_dist);
                let e_supp = SparseErrorVector::random(&mut rng, &err_dist);
                let syn = Syndrome::from_sparse(&key, &e_supp);
                (key, syn, threshold_cache)
            },
            |inputs| black_box(decoder::bgf_decoder(&inputs.0, &mut inputs.1, &mut inputs.2)),
            BatchSize::SmallInput
        )
    });
}

criterion_group!(benches, decoder_benchmarks);
criterion_main!(benches);
