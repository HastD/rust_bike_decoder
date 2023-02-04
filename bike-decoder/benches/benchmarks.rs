use bike_decoder::{
    decoder::{bgf_decoder, unsatisfied_parity_checks},
    graphs::{self, TannerGraphEdges},
    keys::Key,
    ncw::{ClassifiedVector, NearCodewordClass, TaggedErrorVector},
    parameters::*,
    random::custom_thread_rng,
    syndrome::Syndrome,
    threshold::build_threshold_cache,
    vectors::SparseErrorVector,
};
use criterion::{criterion_group, criterion_main, BatchSize, Criterion};
use rand::Rng;
use std::hint::black_box;

pub fn group_decoder(c: &mut Criterion) {
    c.bench_function("bgf_decoder", |b| {
        let mut rng = custom_thread_rng();
        b.iter_batched_ref(
            || {
                let key = Key::random(&mut rng);
                let e_supp = SparseErrorVector::random(&mut rng);
                let syn = Syndrome::from_sparse(&key, &e_supp);
                (key, syn)
            },
            |(key, syn)| black_box(bgf_decoder(key, syn)),
            BatchSize::SmallInput,
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
            |(key, syn)| black_box(unsatisfied_parity_checks(key, syn)),
            BatchSize::SmallInput,
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
            |(key, l)| {
                black_box((
                    TaggedErrorVector::near_codeword(key, NearCodewordClass::C, *l, &mut rng),
                    TaggedErrorVector::near_codeword(key, NearCodewordClass::N, *l, &mut rng),
                    TaggedErrorVector::near_codeword(key, NearCodewordClass::TwoN, *l, &mut rng),
                ))
            },
            BatchSize::SmallInput,
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
            |(key, e_supp)| black_box(Syndrome::from_sparse(key, e_supp)),
            BatchSize::SmallInput,
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
            BatchSize::SmallInput,
        )
    });
}

pub fn group_threshold(c: &mut Criterion) {
    c.bench_function("threshold", |b| {
        let (r, d, t) = (BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT);
        b.iter(|| black_box(build_threshold_cache(r, d, t)))
    });

    c.bench_function("big threshold", |b| {
        // BIKE security level 5 parameters
        let (r, d, t) = (40_973, 137, 264);
        b.iter(|| black_box(build_threshold_cache(r, d, t)))
    });
}

pub fn group_graphs(c: &mut Criterion) {
    c.bench_function("is_absorbing", |b| {
        let mut rng = custom_thread_rng();
        b.iter_batched_ref(
            || {
                let key = Key::random(&mut rng);
                let e_supp = SparseErrorVector::random(&mut rng);
                (key, e_supp)
            },
            |(key, e_supp)| {
                black_box(graphs::is_absorbing(
                    black_box(key),
                    black_box(e_supp.support()),
                ));
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
                black_box(graphs::is_absorbing_subgraph(
                    &edges,
                    black_box(e_supp.support()),
                ));
            },
            BatchSize::SmallInput,
        )
    });
}

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
    targets = group_decoder, group_randgen, group_syndrome, group_threshold, group_graphs,
        group_ncw_classify
}
criterion_main!(benches);
