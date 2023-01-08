use absorbing::graphs::{is_absorbing, is_absorbing_subgraph, TannerGraphEdges};
use bike_decoder::{keys::Key, random::custom_thread_rng, vectors::SparseErrorVector};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use std::hint::black_box;

pub fn group_graph(c: &mut Criterion) {
    c.bench_function("is_absorbing", |b| {
        let mut rng = custom_thread_rng();
        b.iter_batched_ref(
            || {
                let key = Key::random(&mut rng);
                let e_supp = SparseErrorVector::random(&mut rng);
                (key, e_supp)
            },
            |(key, e_supp)| {
                black_box(is_absorbing(black_box(key), black_box(e_supp.support())));
            },
            BatchSize::SmallInput,
        )
    });
    c.bench_function("is_absorbing_subgraph", |b| {
        let mut rng = custom_thread_rng();
        let key = Key::random(&mut rng);
        let edges = TannerGraphEdges::new(&key);
        b.iter_batched_ref(
            || SparseErrorVector::random(&mut rng),
            |e_supp| {
                black_box(is_absorbing_subgraph(&edges, black_box(e_supp.support())));
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = group_graph,
}
criterion_main!(benches);
