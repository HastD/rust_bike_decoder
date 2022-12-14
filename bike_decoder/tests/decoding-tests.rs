use bike_decoder::{
    decoder,
    keys::Key,
    parameters::*,
    syndrome::Syndrome,
    threshold::{bf_masked_threshold, bf_threshold_min},
    vectors::{ErrorVector, SparseErrorVector},
};

const TRIALS: usize = 10000;

#[test]
fn decoding_failure_example() {
    assert_eq!((BLOCK_LENGTH, BLOCK_WEIGHT, ERROR_WEIGHT), (587, 15, 18));
    let key = Key::from_support(
        [
            41, 57, 63, 158, 163, 180, 194, 213, 234, 276, 337, 428, 451, 485, 573,
        ],
        [
            55, 84, 127, 185, 194, 218, 260, 374, 382, 394, 404, 509, 528, 537, 580,
        ],
    )
    .unwrap();
    let e_in = SparseErrorVector::from_support([
        10, 62, 157, 283, 460, 503, 533, 564, 715, 806, 849, 858, 916, 991, 996, 1004, 1078, 1096,
    ])
    .unwrap();
    let mut syn = Syndrome::from_sparse(&key, &e_in);
    let known_syn = [
        0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0,
        0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1,
        1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
        1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
        1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1,
        0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1,
        1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0,
    ];
    assert_eq!(bytemuck::cast_slice::<bool, u8>(syn.contents()), &known_syn);
    let (e_out, success) = decoder::bgf_decoder(&key, &mut syn);
    assert_ne!(syn.hamming_weight(), 0);
    assert!(!success);
    assert_eq!(
        e_out.support(),
        vec![
            10, 62, 157, 283, 460, 503, 533, 564, 644, 663, 672, 777, 858, 907, 940, 982, 991, 996,
            1004, 1078, 1104, 1116, 1126
        ]
    );
}

#[test]
fn syndrome_e_out_consistent() {
    const BF_THRESHOLD_MIN: u8 = bf_threshold_min(BLOCK_WEIGHT);
    const BF_MASKED_THRESHOLD: u8 = bf_masked_threshold(BLOCK_WEIGHT);
    let mut rng = rand::thread_rng();
    for _ in 0..TRIALS {
        let key = Key::random(&mut rng);
        let e_in = SparseErrorVector::random(&mut rng);
        let mut s = Syndrome::from_sparse(&key, &e_in);
        let s_original = s.clone();
        let mut e_out = ErrorVector::zero();
        let (black, _) = decoder::bf_iter(&key, &mut s, &mut e_out, BF_THRESHOLD_MIN);
        assert_eq!(s, s_original.clone() + Syndrome::from_dense(&key, &e_out));
        decoder::bf_masked_iter(&key, &mut s, &mut e_out, black, BF_MASKED_THRESHOLD);
        assert_eq!(s, s_original.clone() + Syndrome::from_dense(&key, &e_out));
    }
}
