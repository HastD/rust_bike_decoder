use bike_decoder::{keys::Key, parameters::ERROR_WEIGHT, random::custom_thread_rng};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use ncw_classify::ClassifiedVector;
use std::hint::black_box;

pub fn group_ncw_classify(c: &mut Criterion) {
    c.bench_function("ncw_classify", |b| {
        let mut rng = custom_thread_rng();
        b.iter_batched_ref(
            || Key::random(&mut rng),
            |key| {
                black_box(ClassifiedVector::random(
                    black_box(key),
                    black_box(ERROR_WEIGHT),
                ));
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = group_ncw_classify,
}
criterion_main!(benches);
