use bike_decoder::{
    decoder::{bgf_decoder, unsatisfied_parity_checks},
    keys::Key,
    ncw::{NearCodewordClass, TaggedErrorVector},
    parameters::*,
    random::custom_thread_rng,
    syndrome::Syndrome,
    threshold::{compute_x, exact_threshold_ineq},
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
        b.iter(|| {
            let x = compute_x(r, d, t).unwrap();
            for ws in 0..=r {
                black_box(exact_threshold_ineq(ws, r, d, t, Some(x)).unwrap());
            }
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = group_decoder, group_randgen, group_syndrome, group_threshold,
}
criterion_main!(benches);
